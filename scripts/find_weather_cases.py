"""Find init times for five weather regimes, to feed scripts/infer_visualize.py.

Nothing in the data labels the weather, so each regime is a heuristic over
calendar rules plus the radar field on the target_2km canvas (read exactly as
`RainPro8Dataset` builds the target):

  typhoon       inside a typhoon window (`--typhoon`, default: the 2021
                storms below), ranked by echo coverage >= 20 dBZ.
  meiyu         May 15 - Jun 30, outside typhoon windows, echo organised in an
                elongated SW-NE band, ranked by coverage >= 20 dBZ.
  summer_conv   Jun 15 - Sep 15 local afternoon (13-17 h), outside typhoon
                windows, many separate >= 35 dBZ cells rather than a band,
                and NOT widespread (coverage >= 20 dBZ at most
                `--conv-max-f20`, default 15%): without that cap the ranking
                picks large organised systems that merely happen to be
                there in the afternoon. Ranked by coverage >= 40 dBZ.
  winter_front  Nov - Mar, an elongated band that is mostly stratiform
                (little of the >= 20 dBZ area reaches 40 dBZ), ranked by
                coverage >= 20 dBZ.
  no_rain       at most `--no-rain-f20` (default 0.1%) of the canvas >= 20
                dBZ at t0 AND through the whole 6 h target window, ranked by
                the wettest of those frames' coverage >= 5 dBZ. A 512 km canvas
                around Taiwan almost always has a shower somewhere, so truly
                echo-free 6 h windows essentially don't exist.

Frames whose maximum exceeds `--max-valid-dbz` (default 75) contain values
no precipitation produces -- a data error, not weather. Init times with such
a frame in t0 .. +6 h are excluded from every regime, and the script reports
how many of the frames it read have them (they also enter training as
top-bucket targets).

Band shape comes from the >= 20 dBZ pixels' coordinate covariance:
elongation = sqrt(major / minor eigenvalue), orientation = major axis angle
from east, counter-clockwise (SW-NE ~ 20-70 deg). Cells are 8-connected
>= 35 dBZ components.

The default typhoon windows are from memory, NOT from an authoritative source
-- check them against CWA's typhoon database and override with `--typhoon`:
    In-fa (烟花)      2021-07-21 .. 2021-07-25
    Lupit (盧碧)      2021-08-04 .. 2021-08-09  (incl. the SW-flow rain after it)
    Chanthu (璨樹)    2021-09-10 .. 2021-09-13
    Kompasu (圓規)    2021-10-10 .. 2021-10-13

Local time: the store's timestamps are assumed to be UTC and shifted by
`--tz-offset-hours` (default 8) for the afternoon rule. The script prints
July-August strong-echo coverage by timestamp hour: afternoon convection
peaks around 14-17 local, so a peak near 06-09 confirms UTC; a peak near
14-17 means the timestamps are already local (rerun with --tz-offset-hours 0).

Candidates come from one split (default test -- the model never trained on
those; `--split all` to search the whole year), after
`RainPro8DataModule.setup()`'s STA_H8 filtering, on the hour. Within a regime,
picks are at least `--min-gap-hours` apart so they are different events.
Prints a table per regime and a ready-to-run infer_visualize.py command per
pick.

Usage (CPU; --split test reads ~4k frames, --split all ~41k):
    python scripts/find_weather_cases.py --config rainpro8.yml \\
        --data-root "$DATA_ROOT" --variable-aliases '{"max_dbz": "MaxDBZ"}' \\
        --split test --csv figures/weather_cases.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.ndimage import label

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from inspect_echo_stats import load_datamodule  # noqa: E402  (same scripts/ directory)

from rainpro.data.rainpro8_dataset import (  # noqa: E402
    RainPro8Dataset,
    _fill_no_echo,
    _mask_missing,
    _tolerance,
)
from rainpro.data.regrid import target_grid  # noqa: E402

DEFAULT_TYPHOONS = [
    "In-fa:2021-07-21:2021-07-25",
    "Lupit:2021-08-04:2021-08-09",
    "Chanthu:2021-09-10:2021-09-13",
    "Kompasu:2021-10-10:2021-10-13",
]
EIGHT_CONNECTED = np.ones((3, 3), dtype=bool)
FEATURES = ("f5", "f20", "f35", "f40", "max_dbz", "n_cells35", "elongation", "orientation")


def frame_features(field: np.ndarray) -> dict:
    finite = np.isfinite(field)
    n = finite.sum()
    if n == 0:
        return {k: np.nan for k in FEATURES}
    with np.errstate(invalid="ignore"):
        ge = {t: field >= t for t in (5.0, 20.0, 35.0, 40.0)}
    feats = {f"f{t:g}": ge[t].sum() / n for t in ge}
    feats["max_dbz"] = float(np.nanmax(field))
    feats["n_cells35"] = int(label(ge[35.0], structure=EIGHT_CONNECTED)[1])

    rr, cc = np.nonzero(ge[20.0])
    if len(rr) >= 20:
        # rows run south -> north (target_grid is lat-ascending), cols west -> east
        cov = np.cov(np.stack([cc, rr]).astype(float))
        evals, evecs = np.linalg.eigh(cov)
        major = evecs[:, 1]
        feats["elongation"] = float(np.sqrt(evals[1] / max(evals[0], 1e-6)))
        feats["orientation"] = float(np.degrees(np.arctan2(major[1], major[0])) % 180)
    else:
        feats["elongation"], feats["orientation"] = np.nan, np.nan
    return feats


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--variable-aliases", default=None)
    ap.add_argument("--split", default="test", choices=("train", "val", "test", "all"))
    ap.add_argument("--top-k", type=int, default=5, help="picks per regime")
    ap.add_argument("--min-gap-hours", type=float, default=24.0)
    ap.add_argument("--tz-offset-hours", type=float, default=8.0,
                     help="added to store timestamps to get Taiwan local time (0 if already local)")
    ap.add_argument("--typhoon", action="append", default=None,
                     help=f"repeatable NAME:START:END (dates inclusive); default {DEFAULT_TYPHOONS}")
    ap.add_argument("--conv-max-f20", type=float, default=0.15,
                     help="summer_conv: max coverage >= 20 dBZ (keeps out widespread systems)")
    ap.add_argument("--no-rain-f20", type=float, default=1e-3,
                     help="no_rain: max coverage >= 20 dBZ allowed in any frame t0 .. +6 h")
    ap.add_argument("--max-valid-dbz", type=float, default=75.0,
                     help="frames with a pixel above this are treated as data errors")
    ap.add_argument("--ckpt-dir", default="runs/rainpro8_2021_obs_only/checkpoints",
                     help="only used to print the infer_visualize.py commands")
    ap.add_argument("--csv", default=None, help="optional: features of every candidate init time")
    args = ap.parse_args()

    dm = load_datamodule(args)
    dm.setup()
    spec = dm.sources["target_2km"]
    splits = ("train", "val", "test") if args.split == "all" else (args.split,)
    inits = pd.DatetimeIndex(sorted(t for s in splits for t in dm.split_times[s]))
    inits = inits[inits.minute == 0]  # on the hour: enough resolution to pick cases
    print(f"{len(inits)} candidate init times (split={args.split}, on the hour)", flush=True)

    ds = RainPro8Dataset(dm.data_root, {"target_2km": spec}, [], center_lat=dm.center_lat,
                         center_lon=dm.center_lon, variable_aliases=dm.variable_aliases,
                         latlon_names=dm.latlon_names)
    handle = ds._get_store("qpesums")
    regridder = ds._get_regridder("qpesums")
    mapping = regridder.prepare(*target_grid(dm.center_lat, dm.center_lon, spec.size_km, spec.resolution_km))
    raw_name = (dm.variable_aliases or {}).get(spec.variables[0], spec.variables[0])

    # t0 plus the 36 target frames of every candidate, each frame read once.
    offsets = np.concatenate([[0], np.asarray(spec.offsets_min)])
    query = (inits.values[:, None] + offsets[None, :].astype("timedelta64[m]")).ravel()
    pos = handle.time_index.get_indexer(pd.DatetimeIndex(query), method="nearest",
                                        tolerance=_tolerance(spec)).reshape(len(inits), len(offsets))
    unique = np.unique(pos[pos >= 0])
    print(f"reading {len(unique)} QPESUMS frames ...", flush=True)
    feats = {k: np.full(len(unique), np.nan) for k in FEATURES}
    for i in range(0, len(unique), 64):
        for j, frame in enumerate(handle.read(raw_name, unique[i:i + 64])):
            frame = _mask_missing(_fill_no_echo(frame, spec.no_echo_values, spec.no_echo_fill),
                                  spec.missing_values)
            for k, v in frame_features(regridder.apply(frame, mapping, fill_value=np.nan)).items():
                feats[k][i + j] = v
        print(f"  frames {min(i + 64, len(unique))}/{len(unique)}", flush=True)

    def at(k, p):  # per-frame feature, NaN where the frame is missing
        return np.where(p >= 0, feats[k][np.clip(np.searchsorted(unique, p), 0, len(unique) - 1)], np.nan)

    bad = feats["max_dbz"] > args.max_valid_dbz
    bad_times = handle.time_index[unique[bad]]
    print(f"\nframes with max > {args.max_valid_dbz:g} dBZ (data errors): {int(bad.sum())}/{len(unique)}"
          + (f", e.g. {', '.join(f'{t:%Y-%m-%d %H:%M}' for t in bad_times[:5])}" if bad.any() else ""))
    if bad.any():
        print(f"   their max values: {np.round(np.sort(feats['max_dbz'][bad])[::-1][:10], 1).tolist()}")

    t0 = pos[:, 0]
    table = pd.DataFrame({k: at(k, t0) for k in FEATURES}, index=inits)
    table["max_f20_6h"] = np.nanmax(at("f20", pos), axis=1)  # t0 .. +6 h
    table["max_f5_6h"] = np.nanmax(at("f5", pos), axis=1)
    table["bad_frame_6h"] = (at("max_dbz", pos) > args.max_valid_dbz).any(axis=1)
    local = inits + pd.Timedelta(hours=args.tz_offset_hours)
    table["local_time"] = local

    typhoons = []
    for item in args.typhoon or DEFAULT_TYPHOONS:
        name, start, end = item.split(":")
        typhoons.append((name, pd.Timestamp(start), pd.Timestamp(end) + pd.Timedelta(days=1)))
    table["typhoon"] = ""
    for name, start, end in typhoons:
        table.loc[(local >= start) & (local < end), "typhoon"] = name
    near_typhoon = np.zeros(len(table), dtype=bool)
    for _, start, end in typhoons:  # +/- 2 days: keep typhoon rain out of the other regimes
        near_typhoon |= (local >= start - pd.Timedelta(days=2)) & (local < end + pd.Timedelta(days=2))

    # Diurnal check: which timestamp hour does Jul-Aug strong echo peak at?
    summer = table[(inits.month >= 7) & (inits.month <= 8)]
    if len(summer):
        by_hour = summer.groupby(summer.index.hour)["f35"].mean()
        peak = int(by_hour.idxmax())
        print(f"\nJul-Aug mean coverage >= 35 dBZ peaks at timestamp hour {peak:02d}"
              f" -> local {int((peak + args.tz_offset_hours) % 24):02d} with --tz-offset-hours {args.tz_offset_hours:g}"
              "  (afternoon convection should peak ~14-17 local)")

    md = local.month * 100 + local.day
    hour = local.hour
    band = (table["elongation"] >= 2.0) & table["orientation"].between(10, 80)
    clean = ~table["bad_frame_6h"]
    with np.errstate(invalid="ignore", divide="ignore"):
        convective_share = table["f40"] / table["f20"]
    regimes = {
        "typhoon": (table["typhoon"] != "", "f20", False),
        "meiyu": ((md >= 515) & (md <= 630) & ~near_typhoon & band & (table["f20"] >= 0.02), "f20", False),
        "summer_conv": ((md >= 615) & (md <= 915) & (hour >= 13) & (hour <= 17) & ~near_typhoon
                        & (table["n_cells35"] >= 10) & (table["elongation"] < 3.0)
                        & (table["f20"] <= args.conv_max_f20), "f40", False),
        "winter_front": (((local.month >= 11) | (local.month <= 3)) & band & (table["f20"] >= 0.01)
                         & (convective_share < 0.1), "f20", False),
        "no_rain": (table["max_f20_6h"] <= args.no_rain_f20, "max_f5_6h", True),
    }

    data_root = json.dumps(dm.data_root)
    aliases = json.dumps(dm.variable_aliases or {})
    for name, (mask, score, ascending) in regimes.items():
        mask = np.asarray(mask) & clean.to_numpy()
        cand = table[mask].sort_values(score, ascending=ascending)
        picks: list[pd.Timestamp] = []
        for t in cand.index:
            if len(picks) == args.top_k:
                break
            if all(abs(t - p) >= pd.Timedelta(hours=args.min_gap_hours) for p in picks):
                picks.append(t)
        print(f"\n==================== {name}: {int(mask.sum())} candidates ====================")
        if not picks:
            print("   none in this split -- try --split all, or loosen the rule")
            continue
        print(f"   {'init (store)':<17} {'local':<17} {'f20':>7} {'f40':>7} {'max':>5} {'cells35':>7}"
              f" {'elong':>6} {'orient':>6} {'typhoon':<8}")
        for t in picks:
            r = table.loc[t]
            print(f"   {t:%Y-%m-%d %H:%M} {r['local_time']:%Y-%m-%d %H:%M} {r['f20']:7.3f} {r['f40']:7.4f}"
                  f" {r['max_dbz']:5.1f} {r['n_cells35']:7.0f} {r['elongation']:6.1f} {r['orientation']:6.0f}"
                  f" {r['typhoon']:<8}")
        for t in picks:
            print(f"   python scripts/infer_visualize.py --init-time {t:%Y-%m-%dT%H:%M} \\\n"
                  f"       --data-root '{data_root}' --variable-aliases '{aliases}' \\\n"
                  f"       --ckpt-dir {args.ckpt_dir} --out figures/cases/{name}_{t:%Y%m%d_%H%M}.png")

    if args.csv:
        os.makedirs(os.path.dirname(os.path.abspath(args.csv)), exist_ok=True)
        table.to_csv(args.csv, index_label="init_time_store", quoting=csv.QUOTE_MINIMAL)
        print(f"\nwrote {args.csv}")


if __name__ == "__main__":
    main()
