from functools import partial
from typing import Callable, List, Optional

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


class RainPro(nn.Module):
    """
    RainPro Model without additional context and skip connections.
    Lead-time conditioning (all zeros) for the single-pass predictions.
    Input: x with shape (B, T_in, C_in, H, W)
    Output: (B, T_out, C_out, H, W)
    """

    def __init__(
        self,
        in_shape: tuple[int, int, int],
        T_in: int,
        T_out: int,
        num_downsample_layers: int = 3,
        dim: int = 256,
        cond_dim: int = 32,
        center_depth: int = 12,
        stochastic_depth_prob: float = 0.2,
        dropout: float = 0.1,
        resnet_depth: int = 2,
    ):
        super().__init__()
        C, H, W = in_shape
        self.T_in, self.T_out = T_in, T_out
        self.input_channels = self.T_in * C  # stacked time+channels

        self.buckets = _sevir_buckets()
        self.thresholds = Threshold(
            self.buckets,
            0.5,
            num_lead_times=self.T_out,
        )
        self.out_channels = len(self.buckets)
        self.criterion = OrdinalConsistentLoss(self.out_channels, self.T_out)

        norm = partial(ConditionalLayerNorm, cond_dim=cond_dim, hot_enc_dim=self.T_out)
        self.stack_time_and_channels = StackTimeAndChannels()
        self.clt = LeadTimeConditioning(
            hot_enc_dim=self.T_out,
            cond_dim=cond_dim,
            dim_out=self.input_channels,
            attention=False,
        )
        self.embed = Block(self.input_channels, dim, norm, kernel_size=1)

        self.down_layers = nn.ModuleList(
            [
                DownConvBlock(
                    in_channels=dim,
                    out_channels=dim,
                    norm_layer=norm,
                    depth=resnet_depth,
                    dropout=dropout,
                )
                for _ in range(num_downsample_layers)
            ]
        )
        self.maxvit = MaxVitBlocks(
            in_channels=dim,
            out_channels=dim,
            n_layers=center_depth,
            norm_layer=norm,
            stochastic_depth_prob=stochastic_depth_prob,
            dropout=dropout,
        )
        self.up_layers = nn.ModuleList(
            [
                UpConvBlock(
                    in_channels=dim,
                    out_channels=dim,
                    norm_layer=norm,
                    skip_channels=dim,
                    depth=resnet_depth,
                    dropout=dropout,
                )
                for _ in range(num_downsample_layers)
            ]
        )

        self.to_outputs = nn.Sequential(
            nn.Conv2d(dim, self.T_out * self.out_channels, 1, 1),
            UnstackTimeAndChannels(self.T_out, self.out_channels),
        )

    def encoder(self, x: Tensor, cond: Optional[Tensor]):
        skips: List[Tensor] = []
        for down in self.down_layers:
            skips.append(x)
            x = down(x, cond)
        return x, skips

    def decoder(self, x: Tensor, skips: List[Tensor], cond: Optional[Tensor]):
        for up in self.up_layers:
            x = up(x, cond=cond, skip=skips.pop())
        return x

    def forward(self, x: Tensor) -> Tensor:
        b = x.shape[0]
        x = self.stack_time_and_channels(x)
        cond = F.one_hot(
            torch.zeros(b, dtype=torch.long, device=x.device), num_classes=self.T_out
        ).float()
        x = self.embed(self.clt(x, cond), cond)
        x, skips = self.encoder(x, cond)
        x = self.maxvit(x, cond)
        x = self.decoder(x, skips, cond)
        return self.to_outputs(x)

    def predict(
        self,
        inputs,
        target: torch.Tensor | None,
        eval_request: EvalRequest,
    ) -> EvalOutputs | torch.Tensor:
        outputs = self.forward(inputs)
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
