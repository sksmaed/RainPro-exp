from typing import Literal, Optional

import torch
import torch.nn as nn
from einops.layers.torch import Rearrange
from torch import Tensor

# Which implementation `LayerNorm` (the non-attention, NCHW branch) builds.
#
# "native" is `nn.GroupNorm(num_groups=1, ...)`, the upstream choice. Its CUDA
# forward launches `RowwiseMomentsCUDAKernel` with a grid of `N * num_groups`
# blocks -- with num_groups=1 that is one block per sample, i.e. 4 blocks at
# micro-batch 4 on a 132-SM H200. Profiling the Taiwan model put ~59% of all
# CUDA time in that one kernel at 5.4 ms/call, ~90x what the tensor's bandwidth
# cost should be, because the GPU is ~3% occupied while it runs.
#
# "var_mean" computes the same statistics through `torch.var_mean`, whose
# TensorIterator reduction splits across many blocks. More kernel launches and
# more round-trips through HBM, but no occupancy cliff.
#
# Both spell their affine parameters `weight`/`bias` with shape (C,), so a
# checkpoint trained under one loads into the other unchanged.
NormImpl = Literal["native", "var_mean"]
_NORM_IMPL: NormImpl = "native"


def set_norm_impl(impl: NormImpl) -> None:
    """Pick the implementation used by `LayerNorm`s built *after* this call."""
    global _NORM_IMPL
    if impl not in ("native", "var_mean"):
        raise ValueError(f"unknown norm impl {impl!r}; expected 'native' or 'var_mean'")
    _NORM_IMPL = impl


def get_norm_impl() -> NormImpl:
    return _NORM_IMPL


class GroupNorm1VarMean(nn.Module):
    """`nn.GroupNorm(num_groups=1, num_channels=C)` via `torch.var_mean`.

    With one group the normalized region is the whole (C, H, W) volume of each
    sample, so the statistics are a plain reduction over dims (1, 2, 3).

    Two details make this match `nn.GroupNorm` rather than merely resemble it:
    `correction=0` (GroupNorm uses the population variance; `var_mean` defaults
    to the unbiased `correction=1`), and the affine applied per *channel* after
    normalizing, not per element. `var_mean` accumulates in `acc_type`, i.e.
    fp32 even for a bf16 input, matching native GroupNorm's accumulation; the
    normalize itself stays in the input dtype so autocast still pays bf16
    bandwidth."""

    def __init__(self, num_channels: int, affine: bool = True, eps: float = 1e-5):
        super().__init__()
        self.num_channels = num_channels
        self.eps = eps
        if affine:
            self.weight = nn.Parameter(torch.ones(num_channels))
            self.bias = nn.Parameter(torch.zeros(num_channels))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def forward(self, x: Tensor) -> Tensor:
        var, mean = torch.var_mean(x, dim=(1, 2, 3), correction=0, keepdim=True)
        rstd = torch.rsqrt(var + self.eps)
        out = (x - mean.to(x.dtype)) * rstd.to(x.dtype)
        if self.weight is not None:
            out = out * self.weight.view(1, -1, 1, 1) + self.bias.view(1, -1, 1, 1)
        return out

    def extra_repr(self) -> str:
        return f"{self.num_channels}, eps={self.eps}, affine={self.weight is not None}"


class LeadTimeConditioning(nn.Module):
    def __init__(self, hot_enc_dim: int, cond_dim: int, dim_out: int, attention: bool):
        super().__init__()
        if not attention:
            rearrange = Rearrange("b c -> b c 1 1")
        else:
            rearrange = Rearrange("b (r d) -> b r 1 d", r=2)
        self.cond = nn.Sequential(
            nn.Linear(hot_enc_dim, cond_dim),
            nn.ReLU(),
            nn.Linear(cond_dim, dim_out * 2),
            rearrange,
        )

    def forward(self, x: Tensor, cond: Tensor):
        cond = self.cond(cond)
        scale, shift = cond.chunk(2, dim=1)
        return x * (scale + 1) + shift


class LayerNorm(nn.Module):
    def __init__(self, num_channels: int, attention: bool, affine: bool = True):
        super().__init__()
        if not attention:
            # See `_NORM_IMPL` above: these two are numerically the same
            # normalization and share parameter names, but differ by ~an order
            # of magnitude in how well they use the GPU.
            if _NORM_IMPL == "var_mean":
                self.norm = GroupNorm1VarMean(num_channels, affine=affine)
            else:
                self.norm = nn.GroupNorm(
                    num_groups=1, num_channels=num_channels, affine=affine
                )
        else:
            self.norm = nn.LayerNorm(num_channels)

    def forward(self, x: Tensor, cond: Optional[Tensor] = None):
        return self.norm(x)


class ConditionalLayerNorm(nn.Module):
    def __init__(
        self,
        num_channels: int,
        cond_dim: int,
        hot_enc_dim: int,
        attention: bool,
    ):
        super().__init__()
        self.norm = LayerNorm(num_channels, attention=attention)
        self.cond = LeadTimeConditioning(
            hot_enc_dim=hot_enc_dim,
            cond_dim=cond_dim,
            dim_out=num_channels,
            attention=attention,
        )

    def forward(self, x: Tensor, cond: Tensor):
        return self.cond(self.norm(x), cond)
