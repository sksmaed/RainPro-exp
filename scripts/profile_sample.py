"""Where does `RainPro8Dataset.__getitem__` actually spend its time?

The number that matters is CPU-seconds per sample against the budget set by
the allocation: one GPU consuming ~4 samples/s off 12 allocated CPUs leaves
~3 core-seconds per sample. Over that and the DataLoader can't keep the GPU
fed no matter how `num_workers` is tuned, because the ceiling is total core
throughput, not process count. This script says which part to attack, instead
of guessing:

    store open / regridder build   <- one-off per worker, amortized
    frame reads (I/O + decompress) <- HFS latency + zstd
    regrid prepare (cKDTree)       <- the O(N log M) nearest-neighbor query
    regrid apply (gather)          <- fancy-indexing the source frames
    mask + normalize               <- numpy elementwise over full frames

WHICH SPLIT YOU ARE MEASURING MATTERS, a lot. Sequential init_times (the
`--cadence-min` default) are the *val/test* access pattern: consecutive
samples 10 min apart share 35 of their 36 `target_2km` frames and all of
their hourly `satellite_8km` frames, so `frame_cache` absorbs most of the
reads and the per-sample number comes out several times too low to describe
training. `--random-times` (the default) draws init_times at random across
the range instead, which is what `shuffle=True` actually does to the train
split -- essentially zero cache reuse, every sample paying for all 62 frames.
Use `--sequential` to measure the val/test path deliberately.

Runs on CPU only (no GPU, no Trainer), so it's fine on a login node -- note
the login-node CPU quota (~4 cores for the whole user on this cluster) will
inflate the absolute numbers, but the *relative* breakdown is what matters.

Usage (same --data-root / --variable-aliases you pass to training):
    python scripts/profile_sample.py \\
        --data-root '{"qpesums": "...", "sta_h8": "...,..."}' \\
        --variable-aliases '{"max_dbz": "MaxDBZ"}' \\
        --n-samples 5
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from rainpro.data import rainpro8_dataset as ds_mod
from rainpro.data.rainpro8_dataset import RainPro8Dataset
from rainpro.data.rainpro8_sources import build_taiwan_sources
from rainpro.data.regrid import NearestNeighborRegridder

TIMINGS: dict[str, float] = defaultdict(float)
COUNTS: dict[str, int] = defaultdict(int)


def timed(bucket: str, fn):
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        try:
            return fn(*args, **kwargs)
        finally:
            TIMINGS[bucket] += time.perf_counter() - start
            COUNTS[bucket] += 1

    return wrapper


def instrument() -> None:
    """Wrap the hot-path pieces. Note these buckets NEST (`_load_frames` sits
    inside `__getitem__`), so they're reported against the total separately
    rather than summed."""
    RainPro8Dataset._load_frames = timed("frame reads (I/O + decompress)", RainPro8Dataset._load_frames)
    NearestNeighborRegridder.prepare = timed("regrid prepare (cKDTree query)", NearestNeighborRegridder.prepare)
    NearestNeighborRegridder.apply = timed("regrid apply (gather)", NearestNeighborRegridder.apply)
    ds_mod._mask_missing = timed("mask missing values", ds_mod._mask_missing)
    ds_mod._fill_no_echo = timed("fill no-echo sentinels", ds_mod._fill_no_echo)
    ds_mod.minmax_normalize = timed("normalize", ds_mod.minmax_normalize)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", required=True, help="same JSON dict as --data.data_root")
    ap.add_argument("--variable-aliases", default="{}", help="same JSON dict as --data.variable_aliases")
    ap.add_argument("--n-samples", type=int, default=5)
    ap.add_argument("--start-date", default="2021-03-01T00:00:00",
                     help="first init_time; --sequential steps forward from here by "
                          "--cadence-min, --random-times draws within --start-date..--end-date")
    ap.add_argument("--end-date", default="2021-12-01T00:00:00",
                     help="--random-times only: upper bound for the drawn init_times")
    ap.add_argument("--sequential", dest="random_times", action="store_false",
                     help="consecutive init_times --cadence-min apart == the val/test access "
                          "pattern, where frame_cache absorbs most reads. The default "
                          "(--random-times) mimics the shuffled train split instead; see the "
                          "module docstring, the two differ by several x")
    ap.add_argument("--random-times", dest="random_times", action="store_true", default=True)
    ap.add_argument("--seed", type=int, default=0, help="--random-times draw seed")
    ap.add_argument("--cadence-min", type=int, default=10, help="--sequential only")
    ap.add_argument("--jitter-km", type=float, default=256.0, help="0 to profile the val/test path")
    ap.add_argument("--include-satellite", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--frame-cache-size", type=int, default=128)
    ap.add_argument("--only-sources", default=None,
                     help="comma-separated subset to profile (e.g. 'satellite_8km' or "
                          "'target_2km,radar_4km') -- isolates which tier costs what, and "
                          "lets you profile one store without needing the others' paths")
    args = ap.parse_args()

    sources = build_taiwan_sources(include_satellite=args.include_satellite, include_gfs=False)
    if args.only_sources:
        wanted = {s.strip() for s in args.only_sources.split(",") if s.strip()}
        unknown = wanted - set(sources)
        if unknown:
            raise SystemExit(f"--only-sources has unknown name(s) {sorted(unknown)}; "
                             f"available: {sorted(sources)}")
        sources = {k: v for k, v in sources.items() if k in wanted}
    # One extra leading init_time: sample 0 is consumed by the warm-up below
    # (which pays the one-off store-open / cKDTree-build costs a persistent
    # worker only pays once), so profiling must start at index 1 -- otherwise
    # the first profiled sample is a pure cache hit on the warm-up's own frames.
    n_draw = args.n_samples + 1
    start = np.datetime64(args.start_date)
    if args.random_times:
        span_min = int((np.datetime64(args.end_date) - start) / np.timedelta64(1, "m"))
        rng = np.random.default_rng(args.seed)
        # snapped to the 10-min target cadence, like the real init_time grid
        init_times = [
            start + np.timedelta64(int(m) // 10 * 10, "m")
            for m in rng.integers(0, span_min, size=n_draw)
        ]
    else:
        init_times = [start + np.timedelta64(i * args.cadence_min, "m") for i in range(n_draw)]

    dataset = RainPro8Dataset(
        data_root=json.loads(args.data_root),
        sources=sources,
        init_times=init_times,
        jitter_km=args.jitter_km,
        variable_aliases=json.loads(args.variable_aliases),
        frame_cache_size=args.frame_cache_size,
    )

    instrument()

    mode = ("random init_times == shuffled TRAIN access pattern (no cache reuse)"
            if args.random_times
            else f"sequential init_times {args.cadence_min} min apart == VAL/TEST "
                 f"access pattern (heavy frame_cache reuse)")
    print(f"mode: {mode}")
    print("warming up (opening stores, building regridders)...", flush=True)
    warm_start = time.perf_counter()
    dataset[0]
    warm = time.perf_counter() - warm_start
    TIMINGS.clear()
    COUNTS.clear()
    print(f"  warm-up sample (incl. one-off setup, excluded below): {warm:6.2f} s\n")

    per_sample = []
    for i in range(1, len(init_times)):
        sample_start = time.perf_counter()
        dataset[i]
        per_sample.append(time.perf_counter() - sample_start)
        print(f"  sample {i}: {per_sample[-1]:6.2f} s", flush=True)

    total = sum(per_sample)
    n = len(per_sample)
    print(f"\n{'=' * 62}\nmean {total / n:.2f} s/sample over {n} samples "
          f"(budget is ~3 core-seconds to keep one GPU fed at 12 CPUs)\n{'=' * 62}")
    print(f"{'component':<34}{'total s':>10}{'s/sample':>10}{'% ':>8}  calls/sample")
    for bucket, seconds in sorted(TIMINGS.items(), key=lambda kv: -kv[1]):
        print(f"{bucket:<34}{seconds:>10.2f}{seconds / n:>10.2f}"
              f"{100 * seconds / total:>7.1f}%  {COUNTS[bucket] / n:>6.1f}")
    accounted = sum(TIMINGS.values())
    print(f"{'(unaccounted: stack/alloc/xarray overhead)':<34}"
          f"{total - accounted:>10.2f}{(total - accounted) / n:>10.2f}"
          f"{100 * (total - accounted) / total:>7.1f}%")


if __name__ == "__main__":
    main()
