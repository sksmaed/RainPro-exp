from functools import partial
from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange
from torch import Tensor

from rainpro.loss.ordinal_consistent import (
    OrdinalConsistentLoss,
    Threshold,
    _sevir_buckets,
)
from rainpro.modules.utils import EvalOutputs, EvalRequest
from rainpro.network.clt import ConditionalLayerNorm, LeadTimeConditioning
from rainpro.network.maxvit import MaxVitBlocks


class Block(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        norm_layer: Callable[..., nn.Module],
        kernel_size: int = 3,
        activation: nn.Module = nn.ReLU(),
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding="same",
            bias=False,
        )
        self.norm = norm_layer(num_channels=out_channels, attention=False)
        self.activation = activation

    def forward(self, x: Tensor, cond: Optional[Tensor]) -> Tensor:
        x = self.conv(x)
        x = self.norm(x, cond)
        return self.activation(x)


class ResNetBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        norm_layer: Callable[..., nn.Module],
        activation: nn.Module = nn.ReLU(),
        dropout: float = 0.0,
    ):
        super().__init__()
        self.block1 = Block(
            in_channels=in_channels,
            out_channels=out_channels,
            norm_layer=norm_layer,
            activation=activation,
        )
        self.block2 = Block(
            in_channels=out_channels,
            out_channels=out_channels,
            norm_layer=norm_layer,
            activation=nn.Identity(),
        )
        self.residual_conv = (
            nn.Conv2d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )
        self.activation = activation
        self.dropout = nn.Dropout2d(p=dropout)

    def forward(self, x: Tensor, cond: Optional[Tensor]) -> Tensor:
        h = self.dropout(self.block1(x, cond))
        h = self.dropout(self.block2(h, cond))
        return self.activation(h + self.residual_conv(x))


class ResNetBlocks(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        norm_layer: Callable[..., nn.Module],
        dropout: float,
        depth: int = 2,
    ):
        super().__init__()
        blocks = [
            ResNetBlock(
                in_channels if i == 0 else out_channels,
                out_channels,
                norm_layer,
                dropout=dropout,
            )
            for i in range(depth)
        ]
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x: Tensor, cond: Optional[Tensor]) -> Tensor:
        for block in self.blocks:
            x = block(x, cond)
        return x


def Downsample(in_channels: int, out_channels: int):
    # https://arxiv.org/abs/2208.03641 shows this is the most optimal way to downsample
    return nn.Sequential(
        Rearrange("b c (h s1) (w s2) -> b (c s1 s2) h w", s1=2, s2=2),
        nn.Conv2d(in_channels * 4, out_channels, 1),
    )


class DownConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        norm_layer: Callable[..., nn.Module],
        dropout: float,
        depth: int = 2,
    ):
        super().__init__()
        self.down = Downsample(
            in_channels=in_channels,
            out_channels=out_channels,
        )
        self.blocks = ResNetBlocks(
            in_channels=out_channels,
            out_channels=out_channels,
            norm_layer=norm_layer,
            depth=depth,
            dropout=dropout,
        )

    def forward(self, x: Tensor, cond: Optional[Tensor]) -> Tensor:
        x = self.down(x)
        x = self.blocks(x, cond)
        return x


class UpConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        norm_layer: Callable[..., nn.Module],
        dropout: float,
        skip_channels: Optional[int] = None,
        depth: int = 2,
    ):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.blocks = ResNetBlocks(
            in_channels=out_channels + (skip_channels or 0),
            out_channels=out_channels,
            norm_layer=norm_layer,
            depth=depth,
            dropout=dropout,
        )

    def forward(
        self,
        x: Tensor,
        cond: Optional[Tensor],
        skip: Optional[Tensor] = None,
    ) -> Tensor:
        x = self.up(x)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        x = self.blocks(x, cond)
        return x


def StackTimeAndChannels():
    return Rearrange("b t c h w -> b (t c) h w")


def UnstackTimeAndChannels(time: int, channels: int):
    return Rearrange("b (t c) h w -> b t c h w", t=time, c=channels)


class Pad(nn.Module):
    def __init__(self, padding: int):
        super().__init__()
        assert padding >= 0, "Padding must be greater than or equal to 0"
        self.pad = padding

    def forward(self, x: Tensor) -> Tensor:
        return F.pad(
            x, (self.pad, self.pad, self.pad, self.pad), mode="constant", value=0.0
        )


class Unpad(nn.Module):
    def __init__(self, unpadding: int):
        super().__init__()
        assert unpadding >= 0, "Unpadding must be greater than or equal to 0"
        self.unpad = unpadding

    def forward(self, x: Tensor) -> Tensor:
        if self.unpad == 0:
            return x
        return x[..., self.unpad : -self.unpad, self.unpad : -self.unpad]


class RainPro(nn.Module):
    """
    RainPro Model without additional context and skip connections.
    Lead-time conditioning (all zeros) for the single-pass predictions.
    Input: x with shape (B, T_in, C_in, H, W)
    Output: (B, T_out, C_out, H, W)
    """

    def __init__(
        self,
        T_out: int,
        in_dims: tuple[int, int, int],
        context_size_4km,
        context_size_8km,
        skip_padding_4km: int = 32,
        dims: tuple[int, int, int, int] = (128, 256, 512, 128),
        cond_dim: int = 32,
        center_depth: int = 12,
        stochastic_depth_prob: float = 0.2,
        dropout: float = 0.1,
        resnet_depth: int = 2,
        buckets_fn: Callable = _sevir_buckets,
    ):
        super().__init__()
        self.T_out = T_out
        in_dim_4km, in_dim_8km, in_dim_16km = in_dims
        dim_4km, dim_8km, dim_16km, dim_2km = dims
        # A 16km-tier source (e.g. GFS) is optional: with `in_dim_16km == 0`
        # (no such source configured, see `rainpro8_sources.build_taiwan_sources`),
        # the encoder skips that branch entirely instead of embedding a 0-channel
        # input.
        self.has_16km = in_dim_16km > 0

        self.target_crop_4km = Unpad(skip_padding_4km)
        self.skip_crop_4km = Unpad(context_size_4km - skip_padding_4km)
        self.skip_crop_8km = Unpad((context_size_8km - skip_padding_4km // 2))
        self.skip_crop_16km = Unpad((context_size_8km - skip_padding_4km // 2) // 2)
        self.pad_to_8km = Pad(context_size_8km - context_size_4km // 2)

        self.buckets = buckets_fn()
        self.thresholds = Threshold(
            self.buckets,
            0.5,
            num_lead_times=self.T_out,
        )
        self.out_channels = len(self.buckets)
        self.criterion = OrdinalConsistentLoss(self.out_channels, self.T_out)

        norm = partial(ConditionalLayerNorm, cond_dim=cond_dim, hot_enc_dim=self.T_out)

        self.clt_4km = LeadTimeConditioning(
            hot_enc_dim=self.T_out,
            cond_dim=cond_dim,
            dim_out=in_dim_4km,
            attention=False,
        )
        self.clt_8km = LeadTimeConditioning(
            hot_enc_dim=self.T_out,
            cond_dim=cond_dim,
            dim_out=in_dim_8km,
            attention=False,
        )
        if self.has_16km:
            self.clt_16km = LeadTimeConditioning(
                hot_enc_dim=self.T_out,
                cond_dim=cond_dim,
                dim_out=in_dim_16km,
                attention=False,
            )
            self.embed_16km = Block(in_dim_16km, dim_16km, norm, kernel_size=1)
        maxvit_in_channels = dim_8km + dim_16km if self.has_16km else dim_8km

        self.embed_4km = Block(in_dim_4km, dim_4km, norm, kernel_size=1)
        self.down_4km_to_8km = DownConvBlock(
            dim_4km,
            dim_4km,
            norm,
            depth=resnet_depth,
            dropout=dropout,
        )

        self.embed_8km = Block(in_dim_8km, dim_8km, norm, kernel_size=1)
        self.down_8km_to_16km = DownConvBlock(
            dim_4km + dim_8km,
            dim_8km,
            norm,
            depth=resnet_depth,
            dropout=dropout,
        )

        self.maxvit_input = ResNetBlocks(
            in_channels=maxvit_in_channels,
            out_channels=dim_16km,
            norm_layer=norm,
            depth=resnet_depth,
            dropout=dropout,
        )

        self.maxvit = MaxVitBlocks(
            in_channels=dim_16km,
            out_channels=dim_16km,
            n_layers=center_depth,
            norm_layer=norm,
            stochastic_depth_prob=stochastic_depth_prob,
            dropout=dropout,
        )

        self.up_16km_to_8km = UpConvBlock(
            in_channels=dim_16km,
            out_channels=dim_8km,
            norm_layer=norm,
            skip_channels=dim_8km + dim_4km,
            depth=resnet_depth,
            dropout=dropout,
        )
        self.up_8km_to_4km = UpConvBlock(
            in_channels=dim_8km,
            out_channels=dim_4km,
            norm_layer=norm,
            skip_channels=dim_4km,
            depth=resnet_depth,
            dropout=dropout,
        )
        self.up_4km_to_2km = UpConvBlock(
            in_channels=dim_4km,
            out_channels=dim_2km,
            norm_layer=norm,
            skip_channels=None,
            depth=resnet_depth,
            dropout=dropout,
        )

        self.to_outputs = nn.Sequential(
            nn.Conv2d(dim_2km, self.T_out * self.out_channels, 1, 1),
            UnstackTimeAndChannels(self.T_out, self.out_channels),
        )

    def encoder(
        self,
        x_4km: Tensor,
        x_8km: Tensor,
        x_16km: Optional[Tensor],
        cond: Tensor,
    ):
        x_4km = self.clt_4km(x_4km, cond)
        x_8km = self.clt_8km(x_8km, cond)

        # 4 km
        x = self.embed_4km(x_4km, cond=cond)
        skip_4km = self.skip_crop_4km(x)
        x = self.down_4km_to_8km(x, cond=cond)
        x = self.pad_to_8km(x)

        # 8 km
        x = torch.cat([x, self.embed_8km(x_8km, cond=cond)], dim=1)
        skip_8km = self.skip_crop_8km(x)
        x = self.down_8km_to_16km(x, cond=cond)

        # 16 km (optional: only present when a 16km-tier source, e.g. GFS, is
        # configured -- see `self.has_16km`)
        if self.has_16km:
            x_16km = self.clt_16km(x_16km, cond)
            x = torch.cat([x, self.embed_16km(x_16km, cond=cond)], dim=1)
        x = self.maxvit_input(x, cond=cond)
        x = self.maxvit(x, cond=cond)
        x = self.skip_crop_16km(x)

        return x, skip_4km, skip_8km

    def decoder(
        self,
        x: Tensor,
        skip_4km: Tensor,
        skip_8km: Tensor,
        cond: Tensor,
    ):
        x = self.up_16km_to_8km(x, cond=cond, skip=skip_8km)
        x = self.up_8km_to_4km(x, cond=cond, skip=skip_4km)
        x = self.target_crop_4km(x)
        x = self.up_4km_to_2km(x, cond=cond)
        return x

    def forward(
        self,
        x_4km: Tensor,  # B (TC) H W
        x_8km: Tensor,  # B (T'C') H' W'
        x_16km: Optional[Tensor] = None,  # B (T''C'') H'' W'', absent when has_16km=False
    ) -> Tensor:
        b = x_4km.shape[0]
        cond = F.one_hot(
            torch.zeros(b, dtype=torch.long, device=x_4km.device),
            num_classes=self.T_out,
        ).float()

        x, skip_4km, skip_8km = self.encoder(x_4km, x_8km, x_16km, cond)
        output = self.decoder(x, skip_4km, skip_8km, cond)
        return self.to_outputs(output)

    def predict(
        self,
        x_4km: torch.Tensor,
        x_8km: torch.Tensor,
        x_16km: torch.Tensor | None,
        target: torch.Tensor | None,
        eval_request: EvalRequest,
    ) -> EvalOutputs | torch.Tensor:
        outputs = self.forward(x_4km, x_8km, x_16km)
        loss = (
            self.criterion(outputs, target)
            if eval_request.need_loss or eval_request.only_loss
            else None
        )
        if eval_request.only_loss:
            return loss

        preds = torch.cumprod(torch.sigmoid(outputs), dim=2)
        forecast = self.thresholds(preds)
        eval_output = EvalOutputs(forecast=forecast)

        if eval_request.need_target:
            eval_output.target = target
        if eval_request.need_probs:
            eval_output.probs = 1 - preds
        if eval_request.need_loss:
            eval_output.loss = loss

        return eval_output
