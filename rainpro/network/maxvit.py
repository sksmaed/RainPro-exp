"""
This module implements MaxViT with conditioning lead time as in MetNet 3.

The code is based on the implementation of MaxViT in torchvision.models.maxvit.
"""

import math
from collections import OrderedDict
from typing import Callable, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torchvision.models.maxvit import (
    SwapAxes,
    WindowDepartition,
    WindowPartition,
)
from torchvision.ops.misc import SqueezeExcitation
from torchvision.ops.stochastic_depth import StochasticDepth


class Conv2dNormActivation(nn.Module):
    """
    Configurable block used for Convolution2d-Normalization-Activation blocks.

    Args:
        in_channels (int): Number of channels in the input image
        out_channels (int): Number of channels produced by the Convolution-Normalization-Activation block
        kernel_size: (int, optional): Size of the convolving kernel. Default: 3
        stride (int, optional): Stride of the convolution. Default: 1
        padding (int, tuple or str, optional): Padding added to all four sides of the input. Default: None, in which case it will be calculated as ``padding = (kernel_size - 1) // 2 * dilation``
        groups (int, optional): Number of blocked connections from input channels to output channels. Default: 1
        norm_layer (Callable[..., torch.nn.Module], optional): Norm layer that will be stacked on top of the convolution layer. If ``None`` this layer won't be used. Default: ``torch.nn.BatchNorm2d``
        activation_layer (Callable[..., torch.nn.Module], optional): Activation function which will be stacked on top of the normalization layer (if not None), otherwise on top of the conv layer. If ``None`` this layer won't be used. Default: ``torch.nn.ReLU``
        dilation (int): Spacing between kernel elements. Default: 1
        inplace (bool): Parameter for the activation layer, which can optionally do the operation in-place. Default ``True``
        bias (bool, optional): Whether to use bias in the convolution layer. By default, biases are included if ``norm_layer is None``.

    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]] = 3,
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Union[int, Tuple[int, int], str] = 1,
        groups: int = 1,
        norm_layer: Optional[Callable[..., torch.nn.Module]] = torch.nn.BatchNorm2d,
        activation_layer: Optional[Callable[..., torch.nn.Module]] = torch.nn.ReLU,
        inplace: Optional[bool] = True,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=False,
        )
        self.norm = (
            norm_layer(out_channels, attention=False)
            if norm_layer is not None
            else nn.Identity()
        )
        params = {} if inplace is None else {"inplace": inplace}
        self.activation = (
            activation_layer(**params)
            if activation_layer is not None
            else nn.Identity()
        )

    def forward(self, x: Tensor, cond: Tensor) -> Tensor:
        """
        Args:
            x (Tensor): Input tensor with expected layout of [B, C, H, W].
        Returns:
            Tensor: Output tensor with expected layout of [B, C, H, W].
        """
        x = self.conv(x)
        x = self.norm(x, cond)
        x = self.activation(x)
        return x


class MBConv(nn.Module):
    """MBConv: Mobile Inverted Residual Bottleneck.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        expansion_ratio (float): Expansion ratio in the bottleneck.
        squeeze_ratio (float): Squeeze ratio in the SE Layer.
        stride (int): Stride of the depthwise convolution.
        activation_layer (Callable[..., nn.Module]): Activation function.
        norm_layer (Callable[..., nn.Module]): Normalization function.
        p_stochastic_dropout (float): Probability of stochastic depth.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        expansion_ratio: float,
        squeeze_ratio: float,
        stride: int,
        activation_layer: Callable[..., nn.Module],
        norm_layer: Callable[..., nn.Module],
        p_stochastic_dropout: float = 0.0,
    ) -> None:
        super().__init__()

        proj: Sequence[nn.Module]
        self.proj: nn.Module

        should_proj = stride != 1 or in_channels != out_channels
        if should_proj:
            proj = [
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=True)
            ]
            if stride == 2:
                proj = [nn.AvgPool2d(kernel_size=3, stride=stride, padding=1)] + proj  # type: ignore
            self.proj = nn.Sequential(*proj)
        else:
            self.proj = nn.Identity()  # type: ignore

        mid_channels = int(out_channels * expansion_ratio)
        sqz_channels = int(out_channels * squeeze_ratio)

        if p_stochastic_dropout:
            self.stochastic_depth = StochasticDepth(p_stochastic_dropout, mode="row")  # type: ignore
        else:
            self.stochastic_depth = nn.Identity()  # type: ignore

        self.pre_norm = norm_layer(in_channels, attention=False)
        self.conv_a = Conv2dNormActivation(
            in_channels,
            mid_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            activation_layer=activation_layer,
            norm_layer=norm_layer,
            inplace=None,
        )
        self.conv_b = Conv2dNormActivation(
            mid_channels,
            mid_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            activation_layer=activation_layer,
            norm_layer=norm_layer,
            groups=mid_channels,
            inplace=None,
        )

        _layers = OrderedDict()
        _layers["squeeze_excitation"] = SqueezeExcitation(
            mid_channels, sqz_channels, activation=nn.SiLU
        )
        _layers["conv_c"] = nn.Conv2d(
            in_channels=mid_channels,
            out_channels=out_channels,
            kernel_size=1,
            bias=True,
        )

        self.layers = nn.Sequential(_layers)

    def forward(self, x: Tensor, cond: Tensor) -> Tensor:
        """
        Args:
            x (Tensor): Input tensor with expected layout of [B, C, H, W].
        Returns:
            Tensor: Output tensor with expected layout of [B, C, H / stride, W / stride].
        """
        res = self.proj(x)
        x = self.pre_norm(x, cond)
        x = self.conv_a(x, cond)
        x = self.conv_b(x, cond)
        x = self.stochastic_depth(self.layers(x))
        return res + x


def _get_relative_position_index(height: int, width: int) -> torch.Tensor:
    coords = torch.stack(
        torch.meshgrid([torch.arange(height), torch.arange(width)], indexing="ij")
    )
    coords_flat = torch.flatten(coords, 1)
    relative_coords = coords_flat[:, :, None] - coords_flat[:, None, :]
    relative_coords = relative_coords.permute(1, 2, 0).contiguous()
    relative_coords[:, :, 0] += height - 1
    relative_coords[:, :, 1] += width - 1
    relative_coords[:, :, 0] *= 2 * width - 1
    return relative_coords.sum(-1)


class RelativePositionalMultiHeadAttention(nn.Module):
    """Relative Positional Multi-Head Attention.

    Args:
        feat_dim (int): Number of input features.
        head_dim (int): Number of features per head.
        max_seq_len (int): Maximum sequence length.
    """

    def __init__(
        self,
        feat_dim: int,
        head_dim: int,
        max_seq_len: int,
    ) -> None:
        super().__init__()

        if feat_dim % head_dim != 0:
            raise ValueError(
                f"feat_dim: {feat_dim} must be divisible by head_dim: {head_dim}"
            )

        self.n_heads = feat_dim // head_dim
        self.head_dim = head_dim
        self.size = int(math.sqrt(max_seq_len))
        self.max_seq_len = max_seq_len

        self.to_qkv = nn.Linear(feat_dim, self.n_heads * self.head_dim * 3, bias=False)
        self.scale_factor = feat_dim**-0.5

        self.merge = nn.Linear(self.head_dim * self.n_heads, feat_dim, bias=False)
        self.relative_position_bias_table = nn.parameter.Parameter(
            torch.empty(
                ((2 * self.size - 1) * (2 * self.size - 1), self.n_heads),
                dtype=torch.float32,
            ),
        )

        self.register_buffer(
            "relative_position_index",
            _get_relative_position_index(self.size, self.size),
        )
        # initialize with truncated normal the bias
        torch.nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

    def get_relative_positional_bias(self) -> torch.Tensor:
        bias_index = self.relative_position_index.view(-1)  # type: ignore
        relative_bias = self.relative_position_bias_table[bias_index].view(
            self.max_seq_len, self.max_seq_len, -1
        )  # type: ignore
        relative_bias = relative_bias.permute(2, 0, 1).contiguous()
        return relative_bias.unsqueeze(0)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x (Tensor): Input tensor with expected layout of [B, G, P, D].
        Returns:
            Tensor: Output tensor with expected layout of [B, G, P, D].
        """
        B, G, P, D = x.shape
        H, DH = self.n_heads, self.head_dim

        qkv = self.to_qkv(x)
        q, k, v = torch.chunk(qkv, 3, dim=-1)

        # QK norm as used in MetNet-3 (originally from https://arxiv.org/abs/2302.05442)
        q = F.rms_norm(q, (q.size(-1),))
        k = F.rms_norm(k, (k.size(-1),))

        q = q.reshape(B, G, P, H, DH).permute(0, 1, 3, 2, 4)
        k = k.reshape(B, G, P, H, DH).permute(0, 1, 3, 2, 4)
        v = v.reshape(B, G, P, H, DH).permute(0, 1, 3, 2, 4)

        k = k * self.scale_factor
        dot_prod = torch.einsum("B G H I D, B G H J D -> B G H I J", q, k)
        pos_bias = self.get_relative_positional_bias()

        dot_prod = F.softmax(dot_prod + pos_bias, dim=-1)

        out = torch.einsum("B G H I J, B G H J D -> B G H I D", dot_prod, v)
        out = out.permute(0, 1, 3, 2, 4).reshape(B, G, P, D)

        out = self.merge(out)
        return out


class PartitionAttentionLayer(nn.Module):
    """
    Layer for partitioning the input tensor into non-overlapping windows and applying attention to each window.

    Update:
    - remove MLP

    Args:
        in_channels (int): Number of input channels.
        head_dim (int): Dimension of each attention head.
        partition_size (int): Size of the partitions.
        partition_type (str): Type of partitioning to use. Can be either "grid" or "window".
        grid_size (Tuple[int, int]): Size of the grid to partition the input tensor into.
        mlp_ratio (int): Ratio of the  feature size expansion in the MLP layer.
        activation_layer (Callable[..., nn.Module]): Activation function to use.
        norm_layer (Callable[..., nn.Module]): Normalization function to use.
        attention_dropout (float): Dropout probability for the attention layer.
        mlp_dropout (float): Dropout probability for the MLP layer.
        p_stochastic_dropout (float): Probability of dropping out a partition.
    """

    def __init__(
        self,
        in_channels: int,
        head_dim: int,
        # partitioning parameters
        partition_size: int,
        partition_type: str,
        norm_layer: Callable[..., nn.Module],
        attention_dropout: float,
        p_stochastic_dropout: float,
    ) -> None:
        super().__init__()

        self.n_heads = in_channels // head_dim
        self.head_dim = head_dim
        self.partition_type = partition_type
        self.partition_size = partition_size

        if partition_type not in ["grid", "window"]:
            raise ValueError("partition_type must be either 'grid' or 'window'")

        self.partition_type = partition_type

        self.partition_op = WindowPartition()
        self.departition_op = WindowDepartition()
        self.partition_swap = (
            SwapAxes(-2, -3) if partition_type == "grid" else nn.Identity()
        )
        self.departition_swap = (
            SwapAxes(-2, -3) if partition_type == "grid" else nn.Identity()
        )

        self.pre_attn_layer_norm = norm_layer
        self.attn_layer = nn.Sequential(
            RelativePositionalMultiHeadAttention(
                in_channels,
                head_dim,
                partition_size**2,
            ),
            nn.Dropout(attention_dropout),
        )

        # layer scale factors
        self.stochastic_dropout = StochasticDepth(p_stochastic_dropout, mode="row")

    def forward(self, x: Tensor, cond: Tensor) -> Tensor:
        """
        Args:
            x (Tensor): Input tensor with expected layout of [B, C, H, W].
        Returns:
            Tensor: Output tensor with expected layout of [B, C, H, W].
        """
        grid_size = x.shape[2:4]

        if self.partition_type == "window":
            p = self.partition_size
        else:
            p = grid_size[0] // self.partition_size

        # Undefined behavior if H or W are not divisible by p
        # https://github.com/google-research/maxvit/blob/da76cf0d8a6ec668cc31b399c4126186da7da944/maxvit/models/maxvit.py#L766
        gh, gw = grid_size[0] // p, grid_size[1] // p
        torch._assert(
            grid_size[0] % p == 0 and grid_size[1] % p == 0,
            "Grid size must be divisible by partition size. Got grid size of {} and partition size of {}".format(
                grid_size, p
            ),
        )

        x = self.partition_op(x, p)
        x = self.partition_swap(x)
        x = x + self.stochastic_dropout(
            self.attn_layer(self.pre_attn_layer_norm(x, cond))
        )
        x = self.departition_swap(x)
        x = self.departition_op(x, p, gh, gw)

        return x


class MaxVitLayer(nn.Module):
    """
    MaxVit layer consisting of a MBConv layer followed by a PartitionAttentionLayer with `window` and a PartitionAttentionLayer with `grid`.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        expansion_ratio (float): Expansion ratio in the bottleneck.
        squeeze_ratio (float): Squeeze ratio in the SE Layer.
        stride (int): Stride of the depthwise convolution.
        activation_layer (Callable[..., nn.Module]): Activation function.
        norm_layer (Callable[..., nn.Module]): Normalization function.
        head_dim (int): Dimension of the attention heads.
        mlp_ratio (int): Ratio of the MLP layer.
        mlp_dropout (float): Dropout probability for the MLP layer.
        attention_dropout (float): Dropout probability for the attention layer.
        p_stochastic_dropout (float): Probability of stochastic depth.
        partition_size (int): Size of the partitions.
        grid_size (Tuple[int, int]): Size of the input feature grid.
    """

    def __init__(
        self,
        # conv parameters
        in_channels: int,
        out_channels: int,
        squeeze_ratio: float,
        expansion_ratio: float,
        stride: int,
        # conv + transformer parameters
        norm_layer: Callable[..., nn.Module],
        activation_layer: Callable[..., nn.Module],
        # transformer parameters
        head_dim: int,
        attention_dropout: float,
        p_stochastic_dropout: float,
        # partitioning parameters
        partition_size: int,
    ) -> None:
        super().__init__()

        # convolutional layer
        self.mb_conv = MBConv(
            in_channels=in_channels,
            out_channels=out_channels,
            expansion_ratio=expansion_ratio,
            squeeze_ratio=squeeze_ratio,
            stride=stride,
            activation_layer=activation_layer,
            norm_layer=norm_layer,
            p_stochastic_dropout=p_stochastic_dropout,
        )
        # attention layers, block -> grid
        self.window_attention = PartitionAttentionLayer(
            in_channels=out_channels,
            head_dim=head_dim,
            partition_size=partition_size,
            partition_type="window",
            norm_layer=norm_layer(out_channels, attention=True),
            attention_dropout=attention_dropout,
            p_stochastic_dropout=p_stochastic_dropout,
        )
        self.grid_attention = PartitionAttentionLayer(
            in_channels=out_channels,
            head_dim=head_dim,
            partition_size=partition_size,
            partition_type="grid",
            norm_layer=norm_layer(out_channels, attention=True),
            attention_dropout=attention_dropout,
            p_stochastic_dropout=p_stochastic_dropout,
        )

    def forward(self, x: Tensor, cond: Tensor) -> Tensor:
        """
        Args:
            x (Tensor): Input tensor of shape (B, C, H, W).
        Returns:
            Tensor: Output tensor of shape (B, C, H, W).
        """
        x = self.mb_conv(x, cond)
        x = self.window_attention(x, cond)
        x = self.grid_attention(x, cond)
        return x


class ConditionalMaxVitBlock(nn.Module):
    """
    A MaxVit block consisting of `n_layers` MaxVit layers.

     Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        expansion_ratio (float): Expansion ratio in the bottleneck.
        squeeze_ratio (float): Squeeze ratio in the SE Layer.
        activation_layer (Callable[..., nn.Module]): Activation function.
        norm_layer (Callable[..., nn.Module]): Normalization function.
        head_dim (int): Dimension of the attention heads.
        mlp_ratio (int): Ratio of the MLP layer.
        mlp_dropout (float): Dropout probability for the MLP layer.
        attention_dropout (float): Dropout probability for the attention layer.
        p_stochastic_dropout (float): Probability of stochastic depth.
        partition_size (int): Size of the partitions.
        n_layers (int): Number of layers in the block.
        p_stochastic (List[float]): List of probabilities for stochastic depth for each layer.
    """

    def __init__(
        self,
        # conv parameters
        in_channels: int,
        out_channels: int,
        squeeze_ratio: float,
        expansion_ratio: float,
        # conv + transformer parameters
        norm_layer: Callable[..., nn.Module],
        activation_layer: Callable[..., nn.Module],
        # transformer parameters
        head_dim: int,
        attention_dropout: float,
        # partitioning parameters
        partition_size: int,
        # number of layers
        n_layers: int,
        p_stochastic: List[float],
    ) -> None:
        super().__init__()
        if not len(p_stochastic) == n_layers:
            raise ValueError(
                f"p_stochastic must have length n_layers={n_layers}, got p_stochastic={p_stochastic}."
            )

        self.layers = nn.ModuleList()

        for idx, p in enumerate(p_stochastic):
            self.layers += [
                MaxVitLayer(
                    in_channels=in_channels if idx == 0 else out_channels,
                    out_channels=out_channels,
                    squeeze_ratio=squeeze_ratio,
                    expansion_ratio=expansion_ratio,
                    stride=1,
                    norm_layer=norm_layer,
                    activation_layer=activation_layer,
                    head_dim=head_dim,
                    attention_dropout=attention_dropout,
                    partition_size=partition_size,
                    p_stochastic_dropout=p,
                ),
            ]

        self.final_conv = nn.Conv2d(
            out_channels * n_layers, out_channels, kernel_size=1, bias=False
        )  # Reduce channels

    def forward(self, x: Tensor, cond: Tensor) -> Tensor:
        """
        Args:
            x (Tensor): Input tensor of shape (B, C, H, W).
        Returns:
            Tensor: Output tensor of shape (B, C, H, W).
        """
        block_outputs = []
        for layer in self.layers:
            x = layer(x, cond)
            block_outputs.append(x)  # Collect outputs of each block

        # Concatenate outputs along the channel dimension
        concatenated = torch.cat(block_outputs, dim=1)

        # Apply final pixel-wise transformation
        x = self.final_conv(concatenated)
        return x


class MaxVitBlocks(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        n_layers: int,
        norm_layer: Callable[..., nn.Module],
        stochastic_depth_prob: float,
        dropout: float,
    ):
        super().__init__()
        p_stochastic = np.linspace(0, stochastic_depth_prob, n_layers).tolist()
        self.blocks = ConditionalMaxVitBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            n_layers=n_layers,
            expansion_ratio=4,
            squeeze_ratio=0.25,
            norm_layer=norm_layer,
            activation_layer=nn.GELU,
            head_dim=32,
            attention_dropout=dropout,
            partition_size=8,
            p_stochastic=p_stochastic,
        )
        self.out_norm = norm_layer(out_channels, attention=False)

    def forward(self, x, cond):
        x = self.blocks(x, cond)
        return self.out_norm(x, cond)
