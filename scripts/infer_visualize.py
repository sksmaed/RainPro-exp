"""Run trained checkpoints on one init_time and plot reflectivity vs. ground truth.

Produces a lead-time x source grid: one row per forecast lead time, one column
per checkpoint plus a ground-truth column. With the default two checkpoints
(best/last) and 7 lead times that's the 21 panels you'd expect.

Probability -> dBZ
------------------
`RainPro` is ordinal-classification, not regression: `RainPro.predict` returns
`probs = 1 - cumprod(sigmoid(logits))`, i.e. the CDF at each bucket boundary,
so the exceedance curve is `S_k = 1 - probs = P(Y > t_k)` over the 16
`taiwan_dbz_buckets` boundaries [5, 10, 15, 20, 25, 28, ..., 55, 60] dBZ.
(`EvalOutputs.forecast` is NOT that expectation -- it's `Threshold`'s
deterministic pick, the largest boundary whose exceedance beats 0.5, so it can
only ever emit one of those 16 values. Hence computing the expectation here.)

Turning the exceedance curve into one continuous number is the standard
bin-probability x representative-value sum, since `cumprod` guarantees `S` is
monotonically decreasing and therefore every bin probability is >= 0:

    P(Y <= t_0)          = 1 - S_0            -> value LOW_BIN_DBZ
    P(t_{k-1} < Y <= t_k) = S_{k-1} - S_k     -> value (t_{k-1} + t_k) / 2
    P(Y > t_last)         = S_last            -> value TAIL_DBZ

Two of those representative values are judgement calls, both deliberately
conservative and both easy to change:
  * LOW_BIN_DBZ = 0, not the bin's 2.5 dBZ midpoint -- QPESUMS clear sky reads
    ~0, and a 2.5 dBZ floor would tint the entire clear domain in a way the
    ground-truth panel doesn't have.
  * TAIL_DBZ = 60, the top boundary rather than an extrapolation past it, so
    the open-ended top bin can't inflate peaks beyond what the model can
    actually resolve.

Usage:
    python scripts/infer_visualize.py \\
        --init-time 2023-06-01T06:00 \\
        --data-root '{"qpesums": "...", "sta_h8": "/work/kilin1203/datasets/STA_H8/2023/06/01"}' \\
        --variable-aliases '{"max_dbz": "MaxDBZ"}' \\
        --out figures/inference_20230601_0600.png

Note `sta_h8` may point at a raw `.btp` directory (no zarr conversion needed --
`RainPro8Dataset._get_store` falls back to the on-demand reader). Point it at
the narrowest directory that covers the init_time; the fallback walks whatever
tree it's given, so a whole-year root costs a long scan for nothing.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import matplotlib as mpl
import numpy as np
import torch

mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from rainpro.data.rainpro8_dataset import RainPro8Dataset  # noqa: E402
from rainpro.data.rainpro8_datamodule import RainPro8DataModule  # noqa: E402
from rainpro.loss.ordinal_consistent import taiwan_dbz_buckets  # noqa: E402
from rainpro.modules.rainpro8 import RainPro8Module  # noqa: E402
from rainpro.modules.utils import EvalRequest  # noqa: E402

LOW_BIN_DBZ = 0.0  # representative value for "at or below the lowest boundary"
TAIL_DBZ = 60.0  # representative value for the open-ended top bin
PLOT_BOUNDS = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60]


def expected_dbz(probs: torch.Tensor) -> torch.Tensor:
    """(B, T, K, H, W) CDF -> (B, T, H, W) expected dBZ. See module docstring."""
    exceedance = 1.0 - probs  # S_k = P(Y > t_k), decreasing in k
    boundaries = torch.tensor(
        [b.min for b in taiwan_dbz_buckets()], dtype=exceedance.dtype, device=exceedance.device
    )

    p_low = 1.0 - exceedance[:, :, 0]
    p_mid = exceedance[:, :, :-1] - exceedance[:, :, 1:]
    p_tail = exceedance[:, :, -1]

    v_mid = ((boundaries[:-1] + boundaries[1:]) / 2).view(1, 1, -1, 1, 1)
    return p_low * LOW_BIN_DBZ + (p_mid * v_mid).sum(dim=2) + p_tail * TAIL_DBZ


def load_module(ckpt_path: str, datamodule: RainPro8DataModule, device: str) -> RainPro8Module:
    # `data` is excluded from the checkpoint's hparams (`save_hyperparameters(
    # ignore="data")`), so it has to be supplied here; everything else
    # (max_epochs, dims, dropout, ...) comes back from the checkpoint.
    module = RainPro8Module.load_from_checkpoint(ckpt_path, data=datamodule, map_location=device)
    return module.eval().to(device)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--init-time", required=True, help="e.g. 2023-06-01T06:00")
    ap.add_argument("--data-root", required=True, help="same JSON dict as --data.data_root")
    ap.add_argument("--variable-aliases", default="{}")
    ap.add_argument("--ckpt", action="append", default=None,
                     help="repeatable; defaults to best.ckpt + last.ckpt under --ckpt-dir")
    ap.add_argument("--ckpt-dir", default="runs/rainpro8_2021_obs_only/checkpoints")
    ap.add_argument("--n-leadtimes", type=int, default=7,
                     help="rows in the figure; lead times are +10 min .. +10*n. Default 7 gives "
                          "7 x (2 checkpoints + GT) = 21 panels, out to +70 min; use 6 for "
                          "exactly one hour")
    ap.add_argument("--out", default="inference.png")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    # These two decide `in_dims`, so they MUST match how the checkpoint was
    # trained -- otherwise the first conv of each tier has the wrong channel
    # count and load_from_checkpoint dies on a shape mismatch deep in the
    # state_dict. Defaults are the obs-only arm this run trained.
    ap.add_argument("--include-satellite", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--include-gfs", action=argparse.BooleanOptionalAction, default=False)
    args = ap.parse_args()

    ckpts = args.ckpt or [os.path.join(args.ckpt_dir, f) for f in ("best.ckpt", "last.ckpt")]
    missing = [c for c in ckpts if not os.path.isfile(c)]
    if missing:
        raise SystemExit(f"checkpoint(s) not found: {missing}")

    data_root = json.loads(args.data_root)
    init_time = np.datetime64(args.init_time)

    # Built only for its `sources` / `frames_out`, which RainPro8Module needs to
    # size the network. `setup()` is deliberately NOT called -- that scans the
    # archives to filter split times, which inference doesn't need.
    datamodule = RainPro8DataModule(
        data_root=data_root,
        start_date="2021-01-01",
        end_date="2022-01-01",
        include_satellite=args.include_satellite,
        include_gfs=args.include_gfs,
        variable_aliases=json.loads(args.variable_aliases),
    )

    n_lead = min(args.n_leadtimes, datamodule.frames_out)
    dataset = RainPro8Dataset(
        data_root=data_root,
        sources=datamodule.sources,
        init_times=[init_time],
        jitter_km=0.0,  # inference: no augmentation, canvas centred as configured
        variable_aliases=json.loads(args.variable_aliases),
    )

    print(f"building input sample for {init_time} ...", flush=True)
    batch = {k: v.unsqueeze(0).to(args.device) for k, v in dataset[0].items()}

    truth = batch["target_2km"][0, :n_lead, 0].cpu().numpy()  # (n_lead, H, W), NaN where missing
    if np.isnan(truth).all():
        print("!! ground truth is entirely NaN -- QPESUMS has no coverage for this init_time")

    columns: list[tuple[str, np.ndarray]] = []
    for ckpt in ckpts:
        print(f"running {ckpt} on {args.device} ...", flush=True)
        module = load_module(ckpt, datamodule, args.device)
        with torch.no_grad():
            out = module(batch, EvalRequest(need_forecast=True, need_probs=True))
        assert out.probs is not None
        dbz = expected_dbz(out.probs)[0, :n_lead].cpu().numpy()
        columns.append((os.path.splitext(os.path.basename(ckpt))[0], dbz))
        del module
    columns.append(("ground truth", truth))

    cmap = mpl.colormaps["turbo"].resampled(len(PLOT_BOUNDS) - 1)
    cmap.set_bad("0.85")  # NaN (no radar coverage) reads as grey, not as 0 dBZ
    norm = mpl.colors.BoundaryNorm(PLOT_BOUNDS, cmap.N)

    fig, axes = plt.subplots(
        n_lead, len(columns), figsize=(3.1 * len(columns), 3.0 * n_lead),
        squeeze=False, layout="constrained",
    )
    for row in range(n_lead):
        for col, (label, field) in enumerate(columns):
            ax = axes[row][col]
            ax.imshow(np.ma.masked_invalid(field[row]), cmap=cmap, norm=norm, origin="lower")
            ax.set_xticks([])
            ax.set_yticks([])
            if row == 0:
                ax.set_title(label, fontsize=12)
            if col == 0:
                ax.set_ylabel(f"+{(row + 1) * 10} min", fontsize=11)

    fig.suptitle(f"RainPro-8 TW  |  init {np.datetime_as_string(init_time, unit='m')}  |  "
                 f"expected dBZ from ordinal probabilities", fontsize=13)
    fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax=axes.ravel().tolist(),
                 orientation="horizontal", fraction=0.02, shrink=0.6, aspect=45,
                 label="reflectivity (dBZ)", ticks=PLOT_BOUNDS)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    # constrained_layout already handles spacing; bbox_inches="tight" would fight it
    fig.savefig(args.out, dpi=130)
    print(f"\nwrote {args.out}  ({n_lead} lead times x {len(columns)} columns "
          f"= {n_lead * len(columns)} panels)")

    for label, field in columns:
        finite = field[np.isfinite(field)]
        if finite.size:
            print(f"  {label:<14} mean {finite.mean():6.2f} dBZ   max {finite.max():6.2f} dBZ")


if __name__ == "__main__":
    main()
