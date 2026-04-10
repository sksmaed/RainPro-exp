from dataclasses import dataclass
from typing import Optional, Union

import numpy as np
import torch
import torch.nn as nn
from einops import rearrange

"""
Loss based on conditional probabilities to ensure consistent ordinal outputs.
https://web.fe.up.pt/~jsc/publications/conferences/2018KelwinIJCNNb.pdf (Figure 4)
"""


@dataclass
class Bucket:
    min: float
    size: float


def _sevir_buckets(normalize: bool = True):
    normalize_factor = 255 if normalize else 1

    thresholds = [16, 31, 59, 74, 100, 133, 160, 181, 219]
    buckets = []
    prev = 0
    for t in thresholds:
        size = (t - prev) / normalize_factor
        buckets.append(Bucket(min=t / normalize_factor, size=size))
        prev = t
    return buckets


class Bucketize(nn.Module):
    def __init__(self, buckets: list):
        super().__init__()
        bounds = torch.tensor([b.min for b in buckets])
        self.register_buffer("bounds", bounds)
        self.nan_bucket = len(buckets)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        nans = torch.isnan(x)
        out = torch.bucketize(x, self.bounds, right=True) - 1
        out[nans] = self.nan_bucket
        return out


class Threshold(nn.Module):
    def __init__(
        self,
        buckets: list,
        thresholds: Optional[Union[torch.Tensor, float]] = None,
        num_lead_times: int = 48,
    ):
        super().__init__()
        self.constant_time = False  # Same value across all lead times
        if thresholds is None:
            thresholds = torch.ones((num_lead_times, len(buckets))) * 0.5
            self.constant_time = True
        elif not isinstance(thresholds, torch.Tensor):
            thresholds = torch.ones((num_lead_times, len(buckets))) * thresholds
            self.constant_time = True
        assert thresholds.ndim == 2 and thresholds.shape[1] == len(buckets)
        thresholds = rearrange(thresholds, "t c -> 1 t c 1 1")
        self.register_buffer("thresholds", thresholds)
        self.buckets = buckets
        bucket_vals = torch.tensor([b.min for b in buckets])
        self.register_buffer("bucket_vals", bucket_vals)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.size(1) == self.thresholds.size(1) or self.constant_time

        x = x > self.thresholds[:, : x.size(1)]

        # select max bucket index satisfying P(val > x) > threshold
        x_idx = torch.argmax(x.int().flip(2), dim=2, keepdim=True)
        # output maximum val
        output = self.bucket_vals[-(x_idx + 1)]
        # If nothing is activated, set to 0
        output[x.int().sum(dim=2, keepdim=True) == 0] = 0
        return output


class OrdinalConsistentLoss(torch.nn.Module):
    def __init__(
        self,
        no_data_value: int,
        t_size: int,
        ratio: int = 2,
    ) -> None:
        super().__init__()
        self.no_data_value = no_data_value
        self.criterion = torch.nn.BCEWithLogitsLoss(reduction="none")
        range_tensor = torch.arange(no_data_value).view(1, 1, no_data_value, 1, 1)

        self.bucketize = Bucketize(_sevir_buckets())
        self.register_buffer("range_tensor", range_tensor)

        decay_rate = np.log(ratio) / (t_size - 1)
        # pdf of exponential distribution
        weights = np.exp(-decay_rate * np.arange(t_size))
        weights /= weights.sum()
        # normalize weights to keep loss scale the same as the unweighed loss
        weights = (weights * t_size) / weights.sum()
        weights = torch.tensor(weights, dtype=torch.get_default_dtype())
        self.register_buffer("weights", weights)

    def forward(
        self,
        preds: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        nan_mask = targets == self.no_data_value  # B, T, 1, H, W
        targets = targets.contiguous()
        targets = self.bucketize(targets)
        targets_encoded = (self.range_tensor <= targets).to(dtype=preds.dtype)

        sets_mask = (targets >= self.range_tensor - 1) & (~nan_mask)
        loss_mask = ~sets_mask

        # Compute BCE loss
        loss = self.criterion(preds, targets_encoded)

        # Lead time weights
        weights = self.weights
        weights = rearrange(weights, "t -> 1 t 1 1 1")
        loss = loss * weights

        loss = loss.masked_fill(loss_mask, torch.nan)
        loss = torch.nanmean(torch.nanmean(loss, dim=(1, 3, 4)), dim=1).mean()
        return loss
