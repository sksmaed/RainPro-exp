from typing import Optional

import torch.nn as nn
from einops.layers.torch import Rearrange
from torch import Tensor


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
