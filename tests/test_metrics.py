"""Hand-computed correctness tests for the new pooled eval metrics
(`rainpro/metrics/contingency.py`, `fss.py`, `probabilistic.py`, `regression.py`).

GT is raw QPESUMS max dBZ (see `docs/rainpro_tw_implementation_notes.md`), not
mm/h -- all test values below are dBZ.

NOT executed by the author of this file -- this sandbox has no populated
Python env (`torch`/`torchmetrics` not installed). Run with `pytest
tests/test_metrics.py` after `uv sync`.
"""

import pytest
import torch

from rainpro.metrics.contingency import ContingencyMetrics
from rainpro.metrics.fss import FractionsSkillScore
from rainpro.metrics.probabilistic import CRPS, BrierScore
from rainpro.metrics.regression import LeadTimeMAEMSE
from rainpro.modules.utils import EvalOutputs


def _tensor(values):
    # (B=1, T=1, C=1, H, W)
    t = torch.tensor(values, dtype=torch.get_default_dtype())
    return t.view(1, 1, 1, *t.shape)


def test_contingency_fbi_pod_far_hand_computed():
    # threshold = 30 dBZ
    # (0,0): target=35 (rain, >=30), forecast=35 (rain)      -> hit
    # (0,1): target=20 (no rain), forecast=35 (rain)         -> false alarm
    # (1,0): target=NaN                                       -> excluded
    # (1,1): target=20 (no rain), forecast=20 (no rain)      -> correct negative
    target = _tensor([[35.0, 20.0], [float("nan"), 20.0]])
    forecast = _tensor([[35.0, 35.0], [35.0, 20.0]])

    metric = ContingencyMetrics(num_lead_times=1, thresholds=[30.0])
    metric.update(EvalOutputs(forecast=forecast, target=target))

    full = metric.full()
    # hits=1, misses=0, false_alarms=1, correct_negatives=1 (NaN pixel excluded)
    assert full["FBI"].item() == pytest.approx((1 + 1) / (1 + 0))  # 2.0
    assert full["POD"].item() == pytest.approx(1 / (1 + 0))  # 1.0
    assert full["FAR"].item() == pytest.approx(1 / (1 + 1))  # 0.5


def test_fss_window1_is_not_csi_but_matches_its_own_closed_form():
    # FSS at window=1 (no neighborhood smoothing) is a *different* score from
    # CSI -- FSS = 1 - MSE/MSE_ref reduces, per-pixel, to a Brier-style
    # formula 2*hits / (2*hits + misses + false_alarms), NOT CSI's
    # hits / (hits + misses + false_alarms). Verify against that closed form
    # directly rather than against CSI (an earlier draft of this plan
    # incorrectly assumed the two coincide at window=1 -- they don't, except
    # in the degenerate all-hits/no-errors case).
    target = _tensor([[35.0, 20.0], [20.0, 20.0]])
    forecast = _tensor([[35.0, 35.0], [20.0, 20.0]])
    # hits=1 (0,0), false_alarms=1 (0,1), correct_negatives=2 -> misses=0

    metric = FractionsSkillScore(num_lead_times=1, thresholds=[30.0], windows=(1,))
    metric.update(EvalOutputs(forecast=forecast, target=target))

    hits, misses, false_alarms = 1, 0, 1
    expected_fss_w1 = 2 * hits / (2 * hits + misses + false_alarms)

    full = metric.full()
    assert full["FSS_w1"].item() == pytest.approx(expected_fss_w1)


def test_fss_perfect_forecast_is_one():
    target = _tensor([[35.0, 20.0], [20.0, 35.0]])
    forecast = target.clone()

    metric = FractionsSkillScore(num_lead_times=1, thresholds=[30.0], windows=(1, 2))
    metric.update(EvalOutputs(forecast=forecast, target=target))

    full = metric.full()
    assert full["FSS_w1"].item() == pytest.approx(1.0)
    assert full["FSS_w2"].item() == pytest.approx(1.0)


def test_crps_perfect_cdf_is_zero():
    # 16 bucket edges (rainpro.loss.ordinal_consistent.taiwan_dbz_buckets()); a
    # target of 30 dBZ has true CDF F(edge) = 1[30 <= edge], i.e. 0 for edges
    # < 30 (5..28) and 1 for edges >= 30 (31..60). Feed `probs` that exact
    # step function -> CRPS should be exactly 0 regardless of the mm/h gap
    # weighting (a perfect CDF has zero error at every bucket, so every
    # weight multiplies zero).
    from rainpro.loss.ordinal_consistent import taiwan_dbz_buckets

    edges = [b.min for b in taiwan_dbz_buckets()]
    target_value = 30.0
    indicator = [1.0 if target_value <= e else 0.0 for e in edges]

    target = torch.tensor([[[target_value]]]).view(1, 1, 1, 1, 1)
    probs = torch.tensor(indicator).view(1, 1, len(edges), 1, 1)

    metric = CRPS(num_lead_times=1)
    metric.update(EvalOutputs(forecast=target, target=target, probs=probs))

    assert metric.full()["CRPS"].item() == pytest.approx(0.0, abs=1e-6)


def test_brier_score_known_case():
    from rainpro.loss.ordinal_consistent import taiwan_dbz_buckets

    n = len(taiwan_dbz_buckets())
    target_value = 30.0
    # Predicted probs = 0.5 everywhere -> squared error (0.5 - indicator)^2 =
    # 0.25 at every bucket, regardless of whether indicator is 0 or 1.
    probs = torch.full((1, 1, n, 1, 1), 0.5)
    target = torch.tensor([[[target_value]]]).view(1, 1, 1, 1, 1)

    metric = BrierScore(num_lead_times=1)
    metric.update(EvalOutputs(forecast=target, target=target, probs=probs))

    brier = metric.full()["Brier"]  # [n_buckets, 1]
    assert torch.allclose(brier, torch.full_like(brier, 0.25))


def test_mae_mse_nan_masking():
    target = _tensor([[20.0, float("nan")], [35.0, 45.0]])
    forecast = _tensor([[22.0, 999.0], [35.0, 42.0]])
    # valid errors: |22-20|=2, |35-35|=0, |42-45|=3 -> excludes the NaN/999.0 pixel entirely

    metric = LeadTimeMAEMSE(num_lead_times=1)
    metric.update(EvalOutputs(forecast=forecast, target=target))

    full = metric.full()
    expected_mae = (2 + 0 + 3) / 3
    expected_mse = (2**2 + 0**2 + 3**2) / 3
    assert full["MAE"].item() == pytest.approx(expected_mae)
    assert full["MSE"].item() == pytest.approx(expected_mse)


def test_contingency_pooled_accumulation_across_batches():
    # Two `update()` calls should accumulate, not overwrite -- pooled
    # contingency table semantics (matches `CriticalSuccessIndex`'s
    # documented behavior).
    target1 = _tensor([[35.0]])
    forecast1 = _tensor([[35.0]])  # hit
    target2 = _tensor([[20.0]])
    forecast2 = _tensor([[35.0]])  # false alarm

    metric = ContingencyMetrics(num_lead_times=1, thresholds=[30.0])
    metric.update(EvalOutputs(forecast=forecast1, target=target1))
    metric.update(EvalOutputs(forecast=forecast2, target=target2))

    full = metric.full()
    assert full["POD"].item() == pytest.approx(1.0)  # 1 hit, 0 misses
    assert full["FAR"].item() == pytest.approx(0.5)  # 1 hit, 1 false alarm


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
