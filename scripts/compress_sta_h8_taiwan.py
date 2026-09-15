"""One-time preprocessing: crop STA_H8 raw `.btp` files to the Taiwan region
RainPro8 actually trains on, and write the result as a compressed Zarr v3
store (one store per `--freq` period, default a whole year; `--freq quarter`
splits a year into 4 stores -- run this 4x with different --start/--end/--out
to spread output across filesystems/quotas, e.g. some quarters on /work and
some on /home; RainPro8Dataset._get_store accepts a comma-separated list of
store paths for data_root["sta_h8"] to read them back as one time series).

Why this exists
----------------
`rainpro.data.sta_h8_raw.open_sta_h8_raw()` reads full 2750x2750 (~30 MB)
`.btp` frames directly off a (typically networked, shared) filesystem, on
every access, with no caching -- fine for a quick pipeline, but during real
training this is the dominant cost: `satellite_8km` alone needs 2 timesteps x
9 bands = 18 such reads *per sample*, and nothing dedupes repeat reads of the
same (time, band) across samples. This script does the equivalent read once,
crops each frame down to the small region the model's `satellite_8km` tier
(1536 km square, see `rainpro.data.rainpro8_sources`) plus training jitter
can ever need, and writes a real chunked/compressed Zarr v3 store that
`RainPro8Dataset._get_store` will use directly (`xr.open_zarr`) instead of
the raw per-file reader -- see that module's docstring and
`rainpro.data.sta_h8_raw`'s.

Crop box: a fixed lat/lon rectangle (`--lat-min/max`, `--lon-min/max`),
default 14-33N, 110-132E (~2048x2048 km, per project instructions) rather
than a radius around a center point -- simpler to reason about and to check
against the raw archive's actual coverage. This keeps ~1160x1220 of the
native 2750x2750 grid (~19% of the area). The script errors out (via
`sta_h8_raw.crop_bounds`) if the box doesn't intersect the archive's pixels
at all, and separately WARNS (not fatal) if any edge's margin to
`--center-lat/lon` is less than `satellite_8km`'s half-width (768 km) +
`--jitter-km` (default 256, matching `rainpro8.yml`'s `train_jitter_km`) +
the regridder's own 15 km nearest-neighbor tolerance
(`NearestNeighborRegridder`'s default `max_dist_km`) -- a sample whose random
jitter draw pushes past that margin would see NaN/fill-value padding near
that edge of its canvas instead of real data. With the defaults above, the
north edge (33N, only ~1035 km from 23.7N) is the tight one: ~11 km of slack
past the bare minimum, i.e. less than that 15 km NN tolerance -- flagged as a
warning, not adjusted automatically, since the box was given explicitly.

Output schema matches `sta_h8_raw.open_sta_h8_raw()`'s in-memory `xr.Dataset`
exactly (one (time, y, x) float32 variable per band B08..B16, plus static
(y, x) `lat`/`lon` coords) so `RainPro8Dataset._read_source`'s time lookup /
`ds[var].isel(time=...).values` code path needs no changes to consume it.

Missing / bad data handling (STA_H8 has no documented missing-value sentinel,
unlike QPESUMS' -999/-99 -- see `rainpro8_sources.QPESUMS_MISSING_VALUES` --
but the real archive has several distinct ways a frame can be unusable):
  - (time, band) with no file at all               -> left at the array's
    NaN fill_value (never written).
  - file exists but is the wrong size (truncated /
    corrupt write)                                  -> left at NaN, same as
    above (same failure `rainpro.data.sta_h8_raw._load_or_nan` treats as
    missing at training time today).
  - file parses but contains non-finite values
    (NaN/Inf already in the raw record)              -> those pixels masked
    to NaN individually (rest of the frame kept).
  - file parses and is all-finite but outside a
    generously wide physical brightness-temperature
    range (`--valid-min`/`--valid-max`, default
    100-400 K)                                        -> whole frame masked
    to NaN, since a frame that far off is more likely a decode/scaling bug
    than real data and silently keeping it would poison training.
Every case is *counted*, not printed per-file (the raw reader's per-file
`warnings.warn` floods logs at scale -- 800+ lines for one training run, see
job-375131.err) -- each period prints one summary line.

Performance: a full year is ~63.6k files at ~30 MB each (~1.9 TB) if read in
full -- reading that much sequentially, one file at a time, is what makes a
naive version of this script take many hours. Two independent things cut
that down: (1) `load_rows_and_validate` reads only the row range the crop
needs (one `seek` + one contiguous read), not the full 2750x2750 frame --
with the default box this is ~42% of each file's bytes (rows only; a column
sub-range within a row isn't a contiguous byte range in this row-major
layout, so columns are still cropped in memory after the read); (2) reads
run concurrently across a thread pool (`--num-workers`, default 16) --
`np.fromfile` releases the GIL while blocked on I/O, so this is real
parallelism against the (typically networked/shared) filesystem, not fake
GIL-bound "concurrency". Writes stay single-threaded (see `compress_period`'s
docstring comment on why that's safe/sufficient). Progress prints every
~0.5% of a period's files, with elapsed time / rate / ETA, instead of the
period only printing once at the very end -- with `--freq year` that could
otherwise be many hours of silence.

Usage (defaults --out to /work/u3843478, i.e. writes under
/work/u3843478/STA_H8/, and the crop box above):
    python scripts/compress_sta_h8_taiwan.py \\
        --src /work/kilin1203/datasets/STA_H8/2021 \\
        --start 2021 --end 2021
    # quick check of what would be processed, no writes:
    python scripts/compress_sta_h8_taiwan.py --src ... --dry-run

Requires zarr>=3.0 (Zarr v3 python API: `zarr.storage.LocalStore`,
`Group.create_array`, `zarr.codecs.{BytesCodec,BloscCodec}`) -- matches the
`weather_data_compression` project's own convention for every other
lossless-only store (zstd-5, noshuffle; see that repo's
`scripts/common/zarr_io.py` / `scripts/compress/sta_h8.py`, which this
script's codec/attrs choices deliberately mirror, cropped down instead of
writing full 2750x2750 frames).
"""

from __future__ import annotations

import argparse
import datetime as dt
import itertools
import os
import sys
import time
import warnings
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait

import numpy as np
import pandas as pd
import zarr
from zarr.codecs import BloscCodec, BytesCodec

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from rainpro.data import sta_h8_raw
from rainpro.data.rainpro8_sources import build_taiwan_sources

FREQ_CHOICES = ("day", "month", "quarter", "year")
_PERIOD_FMT = {"day": "%Y%m%d", "month": "%Y%m", "year": "%Y"}  # "quarter" is special-cased below (no strftime code for it)

# Stored as int64 minutes-since-epoch + a `units` attr (CF convention, which
# xarray decodes back to datetime64 automatically on `xr.open_zarr`), not a
# native datetime64 zarr array -- matches `weather_data_compression`'s
# `common/zarr_io.py` convention for every other store's `time` coordinate,
# and sidesteps relying on zarr v3's datetime64 dtype support. `.timestamp()`
# is NOT used (it silently depends on the executing host's local $TZ for a
# naive datetime) -- same reasoning as that module's `minutes_since_epoch`.
#
# `units` must be EXACTLY "<units> since <ISO ref-date>", nothing more --
# xarray/pandas parses it as a literal timestamp string via
# `pandas.Timestamp(ref_date)`, so any trailing free text (a first attempt
# here appended a caveat in parens) makes every `xr.open_zarr` on the
# resulting store raise `ValueError: unable to decode time units ...` instead
# of silently ignoring it. The caveat goes in `time_units_note` instead.
_EPOCH = dt.datetime(1970, 1, 1)
TIME_UNITS_ATTR = "minutes since 1970-01-01T00:00:00"
TIME_UNITS_NOTE = (
    "naive -- timezone of the source filename timestamps has not been "
    "confirmed; no offset applied, values are the source timestamp as-is"
)


def minutes_since_epoch(d: dt.datetime) -> int:
    return int((d - _EPOCH).total_seconds() // 60)


def warn_if_crop_too_tight(
    lat_min: float, lat_max: float, lon_min: float, lon_max: float,
    center_lat: float, center_lon: float, jitter_km: float, nn_tolerance_km: float = 15.0,
) -> None:
    """Non-fatal sanity check: warn (don't adjust) if any edge of the crop box
    is closer to (center_lat, center_lon) than `satellite_8km` half-width +
    jitter_km + the regridder's own nearest-neighbor tolerance -- see module
    docstring for what that means for the affected samples."""
    satellite_spec = build_taiwan_sources(include_satellite=True)["satellite_8km"]
    needed_km = satellite_spec.size_km / 2 + jitter_km
    safe_km = needed_km + nn_tolerance_km
    km_per_deg_lon = 111.32 * np.cos(np.deg2rad(center_lat))
    margins_km = {
        "north": (lat_max - center_lat) * 111.32,
        "south": (center_lat - lat_min) * 111.32,
        "east": (lon_max - center_lon) * km_per_deg_lon,
        "west": (center_lon - lon_min) * km_per_deg_lon,
    }
    for edge, have_km in margins_km.items():
        if have_km < needed_km:
            warnings.warn(
                f"crop margin to the {edge} is only {have_km:.0f} km, less than "
                f"satellite_8km half-width ({satellite_spec.size_km / 2:.0f} km) + "
                f"--jitter-km ({jitter_km:.0f} km) = {needed_km:.0f} km needed -- some "
                f"samples' satellite_8km tile will have real missing (fill-value) "
                f"coverage near that edge, not just an NN-tolerance edge effect.",
                stacklevel=2,
            )
        elif have_km < safe_km:
            warnings.warn(
                f"crop margin to the {edge} is {have_km:.0f} km, only "
                f"{have_km - needed_km:.0f} km past the {needed_km:.0f} km bare minimum "
                f"(satellite_8km half-width + --jitter-km) -- less than the regridder's "
                f"{nn_tolerance_km:.0f} km nearest-neighbor tolerance, so the rare sample "
                f"whose jitter draw lands near this edge may see some NaN padding there.",
                stacklevel=2,
            )


def period_key(d: dt.datetime, freq: str) -> str:
    if freq == "quarter":
        return f"{d.year}Q{(d.month - 1) // 3 + 1}"
    return d.strftime(_PERIOD_FMT[freq])


def period_bounds(key: str, freq: str) -> tuple[dt.datetime, dt.datetime]:
    """Returns (start, end), end EXCLUSIVE."""
    if freq == "day":
        start = dt.datetime.strptime(key, "%Y%m%d")
        return start, start + dt.timedelta(days=1)
    if freq == "month":
        year, mon = int(key[:4]), int(key[4:6])
        start = dt.datetime(year, mon, 1)
        end = dt.datetime(year + (mon == 12), (mon % 12) + 1, 1)
        return start, end
    if freq == "quarter":
        year, q = int(key[:4]), int(key[5])  # "2021Q1" -> year=2021, q=1 (key[4] is literal "Q")
        start_month = (q - 1) * 3 + 1
        start = dt.datetime(year, start_month, 1)
        end = dt.datetime(year + 1, 1, 1) if start_month == 10 else dt.datetime(year, start_month + 3, 1)
        return start, end
    if freq == "year":
        year = int(key)
        return dt.datetime(year, 1, 1), dt.datetime(year + 1, 1, 1)
    raise ValueError(f"freq must be one of {FREQ_CHOICES}, got {freq!r}")


_FREQ_PRECISION = {"year": 0, "month": 1, "quarter": 1, "day": 2}  # number of "-"-separated date parts kept


def warn_if_start_end_finer_than_freq(start: str | None, end: str | None, freq: str) -> None:
    """`--start`/`--end` only filter at `--freq`'s granularity -- `period_key()`
    truncates every candidate period down to `freq` before comparing, so e.g.
    `--freq year --start 2021-06 --end 2021-06` silently keeps the WHOLE year
    (both bounds truncate to the same "2021" period key) instead of just June,
    with no error or indication anything was ignored. Warn up front (not just
    let the resulting period list speak for itself) since this is easy to
    miss, especially on a `--dry-run` where the wrong-sized run only shows up
    as a slot count you'd have to notice looks too big."""
    needed = _FREQ_PRECISION[freq]
    for flag, value in (("--start", start), ("--end", end)):
        if value is None:
            continue
        given = value.count("-")
        if given > needed:
            finer_freq = next(f for f, p in _FREQ_PRECISION.items() if p == given) if given <= 2 else "day"
            warnings.warn(
                f"{flag} {value!r} is more precise than --freq {freq!r} allows -- period "
                f"keys are truncated to {freq} before filtering, so this will silently keep "
                f"the WHOLE {freq} {value!r} falls in, not just the narrower range {flag} "
                f"looks like it's asking for. Pass --freq {finer_freq!r} (or finer) if you "
                f"meant to restrict to just that.",
                stacklevel=2,
            )


def blosc(clevel: int = 5, shuffle: str = "noshuffle", cname: str = "zstd") -> BloscCodec:
    return BloscCodec(typesize=None, cname=cname, clevel=clevel, shuffle=shuffle)


def create_array(group, name, shape, dtype, chunks, *, fill_value=None, attrs=None,
                  dimension_names=None, data=None):
    arr = group.create_array(
        name=name,
        shape=shape,
        chunks=chunks,
        dtype=dtype,
        fill_value=fill_value,
        filters=[],
        serializer=BytesCodec(),
        compressors=[blosc()],
        dimension_names=dimension_names,
        overwrite=True,
    )
    if attrs:
        arr.attrs.update(attrs)
    if data is not None:
        arr[...] = data
    return arr


def load_rows_and_validate(
    path: str, row_slice: slice, col_slice: slice, valid_min: float, valid_max: float
) -> tuple[np.ndarray | None, str | None]:
    """Returns (cropped (ny, nx) float32 frame or None, failure reason or None).

    Reads only the rows `row_slice` needs (one `seek` + one contiguous read),
    not the full 2750x2750 frame -- `.btp`'s layout is row-major with i (x,
    columns) fastest, so a row range is one contiguous byte range but a
    column range within it isn't; skipping unneeded ROWS is still a real win
    (`row_slice` here keeps ~42% of rows vs. the full frame -- see
    `default_margin`/module docstring for the box's actual pixel fractions).
    The column crop (`col_slice`) still happens in memory after the read.
    Validates the read came back exactly `n_rows * IX` values -- same
    "wrong size -> treat as corrupt/unreadable" contract as
    `sta_h8_raw.load_band_frame`, just scoped to the byte range actually
    read (corruption outside the rows we need is irrelevant and ignored).
    """
    n_rows = row_slice.stop - row_slice.start
    count = n_rows * sta_h8_raw.IX
    try:
        with open(path, "rb") as f:
            f.seek(row_slice.start * sta_h8_raw.IX * 4)  # 4 bytes/float32
            raw = np.fromfile(f, dtype="<f4", count=count)
    except OSError as e:
        return None, f"unreadable ({e})"
    if raw.size != count:
        return None, f"unreadable (expected {count} float32 values for rows[{row_slice.start}:{row_slice.stop}), got {raw.size})"
    raw = raw.reshape(n_rows, sta_h8_raw.IX)
    cropped = raw[:, col_slice].astype(np.float32, copy=True)
    nonfinite = ~np.isfinite(cropped)
    if nonfinite.any():
        cropped[nonfinite] = np.nan
    finite = cropped[~nonfinite]
    if finite.size and ((finite < valid_min) | (finite > valid_max)).all():
        return None, "out-of-range"
    return cropped, ("nonfinite-pixels" if nonfinite.any() else None)


def compress_period(
    file_index: dict[tuple[pd.Timestamp, str], str],
    out_root: str,
    period: str,
    freq: str,
    row_slice: slice,
    col_slice: slice,
    lat_crop: np.ndarray,
    lon_crop: np.ndarray,
    lat_min: float,
    lat_max: float,
    lon_min: float,
    lon_max: float,
    valid_min: float,
    valid_max: float,
    dry_run: bool,
    num_workers: int,
):
    start, end = period_bounds(period, freq)
    slots: list[pd.Timestamp] = []
    t = start
    while t < end:
        slots.append(pd.Timestamp(t))
        t += dt.timedelta(hours=1)
    n_time = len(slots)
    slot_index = {s: i for i, s in enumerate(slots)}

    present = {
        (slot_index[t], band): path
        for (t, band), path in file_index.items()
        if t in slot_index
    }
    print(
        f"[STA_H8] {period}: {len(present)}/{n_time * len(sta_h8_raw.BANDS)} "
        f"(time,band) slots have a source file"
    )
    if dry_run:
        return

    ny, nx = row_slice.stop - row_slice.start, col_slice.stop - col_slice.start
    out_path = os.path.join(out_root, "STA_H8", f"STA_H8_Taiwan_{period}.zarr")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    store = zarr.storage.LocalStore(out_path)
    group = zarr.open_group(store=store, mode="a", zarr_format=3)

    time_vals = np.array([minutes_since_epoch(s.to_pydatetime()) for s in slots], dtype="int64")
    create_array(
        group, "time", time_vals.shape, "int64", time_vals.shape,
        data=time_vals,
        attrs={"units": TIME_UNITS_ATTR, "time_units_note": TIME_UNITS_NOTE},
        dimension_names=("time",),
    )
    create_array(
        group, "lat", lat_crop.shape, "float32", lat_crop.shape,
        data=lat_crop, dimension_names=("y", "x"),
    )
    create_array(
        group, "lon", lon_crop.shape, "float32", lon_crop.shape,
        data=lon_crop, dimension_names=("y", "x"),
    )

    arrays = {
        band: create_array(
            group, band, (n_time, ny, nx), "float32", (1, ny, nx),
            fill_value=np.nan, dimension_names=("time", "y", "x"),
            attrs={"desc": f"STA_H8 {band} IR brightness temperature, K"},
        )
        for band in sta_h8_raw.BANDS
    }

    # Reads are the bottleneck (a full year is O(1-2 TB) even after the
    # row-only read in load_rows_and_validate), and each `.btp` read blocks on
    # network filesystem I/O -- np.fromfile releases the GIL while blocked, so
    # a thread pool gets real parallelism here despite the GIL, without the
    # complexity/risk of multiprocessing + shared zarr array writes. Writes
    # (`arrays[band][t_idx] = frame`) stay single-threaded in the main thread
    # as each result lands, so there's no concurrent-write question to worry
    # about; only the reads run in parallel. Tune `--num-workers` down if this
    # saturates/contends with other jobs on the shared filesystem (same
    # failure mode as the num_workers=128 DataLoader oversubscription in
    # job-375131 -- here it's one process, not 8x replicated, so there's more
    # headroom, but the storage itself is still shared with everyone else).
    #
    # Bounded in-flight window, NOT `[executor.submit(...) for item in
    # present.items()]` + `as_completed(futures)` on that full list: a
    # completed Future keeps its result (here, a ~5.9 MB cropped frame)
    # alive internally until the Future object itself is collected, and
    # `as_completed()` doesn't shrink/drain the list you pass it -- it only
    # gives back completed ones in order, the list still holds every Future
    # (done or not) for the whole loop. Submitting all `total` (up to ~78840
    # for a full year) at once means memory grows roughly linearly with how
    # many have been *read*, not how many have been *written*, capped only
    # by `total * ~5.9 MB` (~375 GB/year worst case) -- guaranteed to OOM
    # long before that on any node, login or compute. Keeping only
    # `max_inflight` Futures alive at a time (submitting a replacement each
    # time one finishes) bounds peak memory to `max_inflight * ~5.9 MB`
    # instead, independent of `total`.
    counts: Counter[str] = Counter()
    counts["missing"] = n_time * len(sta_h8_raw.BANDS) - len(present)
    total = len(present)
    done = 0
    start_t = time.time()
    progress_every = max(1, total // 200)  # ~200 progress lines across the whole period
    max_inflight = num_workers * 4

    def _read(item: tuple[tuple[int, str], str]):
        (t_idx, band), path = item
        frame, reason = load_rows_and_validate(path, row_slice, col_slice, valid_min, valid_max)
        return t_idx, band, frame, reason

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        pending_items = iter(present.items())
        futures = {executor.submit(_read, item) for item in itertools.islice(pending_items, max_inflight)}
        while futures:
            finished, futures = wait(futures, return_when=FIRST_COMPLETED)
            for future in finished:
                t_idx, band, frame, reason = future.result()
                if frame is None:
                    counts[reason] += 1
                else:
                    if reason is not None:
                        counts[reason] += 1
                    arrays[band][t_idx] = frame
                done += 1
                if done % progress_every == 0 or done == total:
                    elapsed = time.time() - start_t
                    rate = done / elapsed if elapsed > 0 else 0.0
                    eta_s = (total - done) / rate if rate > 0 else float("nan")
                    print(
                        f"[STA_H8] {period}: {done}/{total} files read ({100 * done / total:.1f}%), "
                        f"{elapsed:.0f}s elapsed, {rate:.1f} files/s, ~{eta_s:.0f}s remaining",
                        flush=True,
                    )
            futures |= {
                executor.submit(_read, item)
                for item in itertools.islice(pending_items, len(finished))
            }

    group.attrs["dataset"] = "STA_H8_Taiwan"
    group.attrs["period"] = period
    group.attrs["freq"] = freq
    group.attrs["bands"] = sta_h8_raw.BANDS
    group.attrs["crop"] = {
        "lat_min": lat_min,
        "lat_max": lat_max,
        "lon_min": lon_min,
        "lon_max": lon_max,
        "row_start": row_slice.start,
        "row_stop": row_slice.stop,
        "col_start": col_slice.start,
        "col_stop": col_slice.stop,
        "source_shape": [sta_h8_raw.IY, sta_h8_raw.IX],
    }
    group.attrs["valid_range_k"] = [valid_min, valid_max]
    try:
        zarr.consolidate_metadata(store)
    except Exception as e:  # pragma: no cover - best-effort, RainPro8Dataset opens with consolidated=False anyway
        warnings.warn(f"consolidate_metadata failed for {out_path}: {e}", stacklevel=2)

    summary = ", ".join(f"{k}={v}" for k, v in sorted(counts.items())) or "no gaps/bad frames"
    print(f"  -> wrote {out_path} ({ny}x{nx} px, {n_time} timesteps; {summary})")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, help="raw STA_H8 root dir (walked recursively for .btp files)")
    ap.add_argument("--out", default="/work/u3843478",
                     help="output root; writes {out}/STA_H8/STA_H8_Taiwan_<period>.zarr "
                          "(default: /work/u3843478)")
    ap.add_argument("--freq", default="year", choices=FREQ_CHOICES,
                     help="one zarr store per day/month/quarter/year (default: year, to match "
                          "how QPESUMS is opened as a single store by RainPro8Dataset). "
                          "'quarter' buckets by calendar quarter (Q1 Jan-Mar, ... Q4 Oct-Dec) "
                          "-- e.g. run this 4x with --freq quarter and a different --start/--end/"
                          "--out each time to split a year across filesystems/quotas; "
                          "RainPro8Dataset._get_store supports data_root['sta_h8'] being a "
                          "comma-separated list of such stores")
    ap.add_argument("--start", default=None, help="restrict to periods >= this date (YYYY[-MM[-DD]])")
    ap.add_argument("--end", default=None, help="restrict to periods <= this date (YYYY[-MM[-DD]]), inclusive")
    ap.add_argument("--lat-min", type=float, default=14.0)
    ap.add_argument("--lat-max", type=float, default=33.0)
    ap.add_argument("--lon-min", type=float, default=110.0)
    ap.add_argument("--lon-max", type=float, default=132.0)
    ap.add_argument("--center-lat", type=float, default=23.7,
                     help="rainpro8.yml's data.center_lat -- used only for the crop-margin "
                          "sanity check against --lat/lon-min/max, not for cropping itself")
    ap.add_argument("--center-lon", type=float, default=121.0,
                     help="rainpro8.yml's data.center_lon -- see --center-lat")
    ap.add_argument("--jitter-km", type=float, default=256.0,
                     help="rainpro8.yml's data.train_jitter_km -- used only for the crop-margin "
                          "sanity check against --lat/lon-min/max, not for cropping itself")
    ap.add_argument("--latlon-path", default=sta_h8_raw.DEFAULT_LATLON_PATH)
    ap.add_argument("--valid-min", type=float, default=100.0, help="K; frames entirely below this are dropped")
    ap.add_argument("--valid-max", type=float, default=400.0, help="K; frames entirely above this are dropped")
    ap.add_argument("--num-workers", type=int, default=16,
                     help="concurrent .btp reads (thread pool -- np.fromfile releases the GIL "
                          "while blocked on I/O, see compress_period's docstring comment). "
                          "Reads dominate runtime; raise this if the filesystem has headroom, "
                          "lower it if throughput degrades/stalls under load (default: 16)")
    ap.add_argument("--dry-run", action="store_true", help="report coverage per period without writing")
    args = ap.parse_args()

    if not os.path.isdir(args.src):
        print(f"! {args.src} does not exist", file=sys.stderr)
        sys.exit(2)

    warn_if_start_end_finer_than_freq(args.start, args.end, args.freq)

    warn_if_crop_too_tight(
        args.lat_min, args.lat_max, args.lon_min, args.lon_max,
        args.center_lat, args.center_lon, args.jitter_km,
    )

    lat, lon = sta_h8_raw.load_sta_h8_latlon(args.latlon_path)
    row_slice, col_slice = sta_h8_raw.crop_bounds(
        lat, lon, args.lat_min, args.lat_max, args.lon_min, args.lon_max
    )
    lat_crop = lat[row_slice, col_slice]
    lon_crop = lon[row_slice, col_slice]
    print(
        f"[STA_H8] crop: lat[{args.lat_min},{args.lat_max}] lon[{args.lon_min},{args.lon_max}] "
        f"-> rows[{row_slice.start}:{row_slice.stop}) cols[{col_slice.start}:{col_slice.stop}) "
        f"= {row_slice.stop - row_slice.start}x{col_slice.stop - col_slice.start} "
        f"({100 * (row_slice.stop - row_slice.start) * (col_slice.stop - col_slice.start) / (sta_h8_raw.IY * sta_h8_raw.IX):.1f}% of native grid)"
    )

    print(f"[STA_H8] scanning {args.src} for .btp files...")
    file_index, times = sta_h8_raw.scan_files(args.src)
    if not times:
        print(f"! no .btp files found under {args.src}", file=sys.stderr)
        sys.exit(2)
    print(f"[STA_H8] found {len(file_index)} files, {len(times)} unique timestamps "
          f"({times[0]} .. {times[-1]})")

    by_period: dict[str, None] = {}
    for t in times:
        by_period[period_key(t.to_pydatetime(), args.freq)] = None
    periods = sorted(by_period)
    if args.start:
        lo = period_key(pd.Timestamp(args.start).to_pydatetime(), args.freq)
        periods = [p for p in periods if p >= lo]
    if args.end:
        hi = period_key(pd.Timestamp(args.end).to_pydatetime(), args.freq)
        periods = [p for p in periods if p <= hi]

    for period in periods:
        compress_period(
            file_index, args.out, period, args.freq, row_slice, col_slice,
            lat_crop, lon_crop, args.lat_min, args.lat_max, args.lon_min, args.lon_max,
            args.valid_min, args.valid_max, args.dry_run, args.num_workers,
        )


if __name__ == "__main__":
    main()
