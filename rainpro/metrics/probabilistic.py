"""CRPS and Brier score (+ reliability) from the model's per-bucket ordinal
CDF (`EvalOutputs.probs`, requires `EvalRequest(need_probs=True)`).

`rainpro.network.rainpro8.RainPro.predict()` computes
`preds = cumprod(sigmoid(outputs), dim=2)` = P(value > bucket_c.min) per
bucket channel c, non-increasing in c by construction (the cumprod *is* the
ordinal-consistency mechanism `rainpro.loss.ordinal_consistent` is named
for), then sets `EvalOutputs.probs = 1 - preds` = F(bucket_c) =
P(value <= bucket_c.min): a 16-point discretized CDF over
`rainpro.loss.ordinal_consistent.taiwan_dbz_buckets()`'s edges (5 .. 60 dBZ,
see `docs/rainpro_tw_implementation_notes.md` for why GT/classification is
dBZ, not mm/h), non-decreasing in c.

This module treats `probs` as exactly that CDF and compares `target`
directly against each bucket edge, rather than re-deriving indicators via
`Bucketize` -- simpler, and avoids any off-by-one risk in reverse-engineering
`Bucketize`'s index convention.

Tail assumption (not stated anywhere else in the codebase, made explicit
here): F = 0 for values below the first bucket edge (5 dBZ), F = 1 above the
last (60 dBZ) -- an approximation worth flagging alongside any reported
CRPS/Brier numbers.

`CRPS` reports its integrated error in mm/h, not dBZ, even though
classification happens in dBZ space: dBZ is a log scale, so integrating the
squared-CDF-error directly in dBZ would arbitrarily compress the weight given
to errors at high intensity. `dbz_to_mmh` (`rainpro.data.marshall_palmer`,
kept in the codebase specifically for this kind of post-hoc relabeling) maps
the dBZ bucket edges to mm/h once at construction time, and those *mm/h* gaps
are what weight the trapezoidal sum -- the classification indicator itself
still compares `target` (raw dBZ) against the dBZ edges. `BrierScore` and
`ReliabilityAccumulator` don't integrate over buckets, so they stay in dBZ
throughout (their bucket labels are dBZ values).
"""

import torch
import torchmetrics

from rainpro.data.marshall_palmer import dbz_to_mmh
from rainpro.loss.ordinal_consistent import taiwan_dbz_buckets
from rainpro.modules.utils import EvalOutputs


def _bucket_edges_dbz() -> torch.Tensor:
    buckets = taiwan_dbz_buckets()
    return torch.tensor([b.min for b in buckets], dtype=torch.get_default_dtype())


def _bucket_gaps_mmh(edges_dbz: torch.Tensor) -> torch.Tensor:
    """CRPS integration weights: dBZ bucket edges converted to mm/h, then
    gapped from a 0 mm/h floor (mirroring how `taiwan_dbz_buckets()` gaps its
    own `Bucket.size` from a `prev=0.0` floor)."""
    edges_mmh = dbz_to_mmh(edges_dbz.numpy())
    gaps, prev = [], 0.0
    for e in edges_mmh:
        gaps.append(float(e) - prev)
        prev = float(e)
    return torch.tensor(gaps, dtype=torch.get_default_dtype())


class CRPS(torchmetrics.Metric):
    is_differentiable = False
    full_state_update = False
    higher_is_better = False

    sum_sq: torch.Tensor
    count: torch.Tensor

    def __init__(self, num_lead_times: int):
        super().__init__()
        edges_dbz = _bucket_edges_dbz()
        self.register_buffer("edges", edges_dbz)
        self.register_buffer("gaps", _bucket_gaps_mmh(edges_dbz))

        self.add_state("sum_sq", default=torch.zeros(num_lead_times), dist_reduce_fx="sum")
        self.add_state("count", default=torch.zeros(num_lead_times), dist_reduce_fx="sum")

    def update(self, eval_outputs: EvalOutputs):
        probs = eval_outputs.probs  # (B, T, n_buckets, H, W) = F(bucket_c), dBZ classification
        target = eval_outputs.target  # (B, T, 1, H, W), raw dBZ
        assert probs is not None and target is not None

        valid = ~torch.isnan(target)
        edges = self.edges.view(1, 1, -1, 1, 1)  # dBZ, for the classification indicator
        gaps = self.gaps.view(1, 1, -1, 1, 1)  # mm/h, for the integration weight

        indicator = (target <= edges).float()  # broadcasts to (B, T, n_buckets, H, W)
        # Trapezoidal-ish sum over buckets of the squared CDF error, weighted
        # by each bucket's gap to the previous one (in mm/h -- see module
        # docstring for why the weighting unit differs from the indicator's).
        sq_err = ((probs - indicator) ** 2 * gaps).sum(dim=2, keepdim=True)  # (B,T,1,H,W)

        sq_err = torch.where(valid, sq_err, torch.zeros_like(sq_err))

        self.sum_sq += sq_err.sum(dim=(0, 2, 3, 4))
        self.count += valid.float().sum(dim=(0, 2, 3, 4))

    def compute(self) -> torch.Tensor:
        return (self.sum_sq / self.count.clamp(min=1)).mean()

    def full(self) -> dict[str, torch.Tensor]:
        return {"CRPS": self.sum_sq / self.count.clamp(min=1)}


class BrierScore(torchmetrics.Metric):
    is_differentiable = False
    full_state_update = False
    higher_is_better = False

    sum_sq: torch.Tensor
    count: torch.Tensor

    def __init__(self, num_lead_times: int):
        super().__init__()
        edges = _bucket_edges_dbz()
        self.register_buffer("edges", edges)
        self.labels = [f"{e:g}dBZ" for e in edges.tolist()]

        zeros_shape = (len(edges), num_lead_times)
        self.add_state("sum_sq", default=torch.zeros(zeros_shape), dist_reduce_fx="sum")
        self.add_state("count", default=torch.zeros(zeros_shape), dist_reduce_fx="sum")

    def update(self, eval_outputs: EvalOutputs):
        probs = eval_outputs.probs
        target = eval_outputs.target
        assert probs is not None and target is not None

        valid = ~torch.isnan(target)
        edges = self.edges.view(1, 1, -1, 1, 1)

        indicator = (target <= edges).float()
        sq_err = (probs - indicator) ** 2
        sq_err = torch.where(valid, sq_err, torch.zeros_like(sq_err))
        n_valid = valid.float().expand_as(sq_err)

        # (B, T, n_buckets, H, W) -> sum over B, H, W -> (T, n_buckets) -> (n_buckets, T)
        self.sum_sq += sq_err.sum(dim=(0, 3, 4)).transpose(0, 1)
        self.count += n_valid.sum(dim=(0, 3, 4)).transpose(0, 1)

    def compute(self) -> torch.Tensor:
        return (self.sum_sq / self.count.clamp(min=1)).mean()

    def full(self) -> dict[str, torch.Tensor]:
        return {"Brier": self.sum_sq / self.count.clamp(min=1)}


class ReliabilityAccumulator(torchmetrics.Metric):
    """Bins predicted probability into `num_bins` equal-width bins per
    (bucket, lead_time), for a reliability-diagram export. Exposed via
    `full_table()` (rows for a `wandb.Table`) rather than `full()`, since a
    bucket x lead_time x bin grid is too large to auto-plot as line charts --
    see `rainpro.callbacks.log_plots.LogPlots`.
    """

    is_differentiable = False
    full_state_update = False

    count: torch.Tensor
    sum_pred: torch.Tensor
    sum_obs: torch.Tensor

    def __init__(self, num_lead_times: int, num_bins: int = 10):
        super().__init__()
        edges = _bucket_edges_dbz()
        self.register_buffer("edges", edges)
        self.num_bins = num_bins
        self.register_buffer("bin_edges", torch.linspace(0, 1, num_bins + 1))

        zeros_shape = (len(edges), num_lead_times, num_bins)
        self.add_state("count", default=torch.zeros(zeros_shape), dist_reduce_fx="sum")
        self.add_state("sum_pred", default=torch.zeros(zeros_shape), dist_reduce_fx="sum")
        self.add_state("sum_obs", default=torch.zeros(zeros_shape), dist_reduce_fx="sum")

    def update(self, eval_outputs: EvalOutputs):
        probs = eval_outputs.probs  # (B, T, n_buckets, H, W) = F(bucket_c)
        target = eval_outputs.target
        assert probs is not None and target is not None

        b, t, c, h, w = probs.shape
        valid = (~torch.isnan(target)).expand(b, t, c, h, w)
        edges = self.edges.view(1, 1, -1, 1, 1)
        indicator = (target <= edges).float().expand(b, t, c, h, w)

        probs_clamped = probs.clamp(0, 1).contiguous()
        bin_idx = torch.bucketize(probs_clamped, self.bin_edges[1:-1])  # in [0, num_bins-1]

        lt_idx = torch.arange(t, device=probs.device).view(1, t, 1, 1, 1).expand(b, t, c, h, w)
        bucket_idx = torch.arange(c, device=probs.device).view(1, 1, c, 1, 1).expand(
            b, t, c, h, w
        )
        flat_idx = ((bucket_idx * t + lt_idx) * self.num_bins + bin_idx).long()

        flat_idx = flat_idx[valid]
        p = probs_clamped[valid]
        o = indicator[valid]

        n_cells = c * t * self.num_bins
        count = torch.bincount(flat_idx, minlength=n_cells).float()
        sum_pred = torch.bincount(flat_idx, weights=p, minlength=n_cells)
        sum_obs = torch.bincount(flat_idx, weights=o, minlength=n_cells)

        self.count += count.view(c, t, self.num_bins)
        self.sum_pred += sum_pred.view(c, t, self.num_bins)
        self.sum_obs += sum_obs.view(c, t, self.num_bins)

    def compute(self) -> torch.Tensor:
        # Weighted mean |predicted - observed| frequency (ECE-style scalar).
        pred_freq = self.sum_pred / self.count.clamp(min=1)
        obs_freq = self.sum_obs / self.count.clamp(min=1)
        weights = self.count / self.count.sum().clamp(min=1)
        return (weights * (pred_freq - obs_freq).abs()).sum()

    def full_table(self) -> list[tuple[float, int, int, float, float, float]]:
        """Rows: (bucket_dbz, lead_time, bin, mean_pred, obs_freq, count)."""
        rows = []
        edges = self.edges.tolist()
        for ci, edge in enumerate(edges):
            for lt in range(self.count.shape[1]):
                for bi in range(self.num_bins):
                    n = self.count[ci, lt, bi].item()
                    if n == 0:
                        continue
                    mean_pred = (self.sum_pred[ci, lt, bi] / n).item()
                    obs_freq = (self.sum_obs[ci, lt, bi] / n).item()
                    rows.append((edge, lt + 1, bi, mean_pred, obs_freq, n))
        return rows
