"""Assign every sample of the pool (scripts/build_sample_pool.py) to a
cross-validation fold or the held-out test partition, balanced by season and
echo amount, and write the manifest `RainPro8DataModule(sample_manifest=...)`
trains from.

Why blocks, not samples. Neighbouring 10-minute samples share 35 of their 36
target frames, so splitting individual samples would put near-copies on both
sides. Samples are grouped into blocks of `--block-days` consecutive local days
(default 3), and every typhoon window (the pool's `typhoon` column) becomes a
single block so one storm never spans train and val. Whole blocks go to one
partition. The data module additionally keeps `cv_gap_hours` between roles
(train samples near val/test, and val samples near test, are dropped -- see
`manifest_splits`), enforced per fold.

Balancing. Blocks are stratified by season (typhoon blocks are their own
stratum). Within a stratum, blocks are taken in order of decreasing echo
(mean coverage >= 20 dBZ), a round of len(partitions) at a time, and in each
round every partition receives one block -- the partition with the fewest
samples so far picks first. So every partition gets a similar share of each
season, of wet and dry blocks, and of samples.

Partitions: fold0 .. fold{K-1}, plus "test" unless `--no-test`. The test
partition is NOT part of cross-validation: with K folds and a test partition,
each fold and the test set get ~1 / (K + 1) of the blocks. Train fold k with
`--data.sample_manifest <out> --data.cv_fold k`; report final numbers on test
once, at the end.

Writes `--out`: the pool's columns plus `block` and `partition`, and prints
per-partition counts, season shares and echo statistics.

Usage:
    python scripts/make_cv_folds.py --pool figures/sample_pool_2021.csv \\
        --k 5 --out splits/cv_2021_k5.csv
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

SEASONS = ("winter", "spring", "meiyu", "summer", "autumn")


def assign_blocks(pool: pd.DataFrame, block_days: int) -> pd.Series:
    dates = pd.to_datetime(pool["date_local"])
    day_index = (dates - dates.min()).dt.days
    block = "d" + (day_index // block_days).astype(str)
    typhoon = pool["typhoon"].fillna("").astype(str)
    return block.where(typhoon == "", "ty_" + typhoon)


def assign_partitions(blocks: pd.DataFrame, partitions: list[str], seed: int) -> dict[str, str]:
    """blocks: one row per block with columns n, stratum, echo. Returns
    block -> partition. See module docstring for the rule."""
    rng = np.random.default_rng(seed)
    size = dict.fromkeys(partitions, 0)
    out: dict[str, str] = {}
    for _, group in blocks.groupby("stratum", sort=True):
        # Shuffle first so equal-echo blocks don't always fall in date order.
        group = group.sample(frac=1.0, random_state=int(rng.integers(2**31))).sort_values(
            "echo", ascending=False, kind="stable"
        )
        names = list(group.index)
        for start in range(0, len(names), len(partitions)):
            free = list(partitions)
            rng.shuffle(free)  # tie-break among equally sized partitions
            for name in names[start:start + len(partitions)]:
                pick = min(free, key=lambda p: size[p])
                free.remove(pick)
                out[name] = pick
                size[pick] += int(group.loc[name, "n"])
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pool", required=True, help="CSV from scripts/build_sample_pool.py")
    ap.add_argument("--k", type=int, default=5, help="number of CV folds")
    ap.add_argument("--block-days", type=int, default=3)
    ap.add_argument("--no-test", action="store_true", help="no held-out test partition (CV folds only)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    if args.k < 2:
        raise SystemExit("--k must be at least 2")

    pool = pd.read_csv(args.pool, parse_dates=["init_time", "local_time"])
    pool["typhoon"] = pool["typhoon"].fillna("")
    pool["block"] = assign_blocks(pool, args.block_days)

    blocks = pool.groupby("block").agg(
        n=("init_time", "size"),
        season=("season", lambda s: s.mode().iloc[0]),
        typhoon=("typhoon", "first"),
        echo=("cov20_target", "mean"),
        start=("local_time", "min"),
    )
    blocks["stratum"] = np.where(blocks["typhoon"] != "", "typhoon", blocks["season"])

    partitions = [f"fold{i}" for i in range(args.k)] + ([] if args.no_test else ["test"])
    if len(blocks) < len(partitions):
        raise SystemExit(f"only {len(blocks)} blocks for {len(partitions)} partitions -- "
                         "lower --block-days or --k")
    mapping = assign_partitions(blocks, partitions, args.seed)
    pool["partition"] = pool["block"].map(mapping)
    blocks["partition"] = blocks.index.map(mapping)

    # --- report ------------------------------------------------------------
    print(f"{len(pool):,} samples in {len(blocks)} blocks ({args.block_days}-day blocks, "
          f"{(blocks['typhoon'] != '').sum()} typhoon blocks) -> {len(partitions)} partitions")
    print(f"\n{'partition':<9} {'blocks':>6} {'samples':>8} {'train-eligible':>14} {'cov20':>7} {'cov35':>7}"
          f" {'typhoon':>7}  " + "  ".join(f"{s:>6}" for s in SEASONS))
    for part in partitions:
        sel = pool["partition"] == part
        sub = pool[sel]
        shares = [(sub["season"] == s).mean() for s in SEASONS]
        print(f"{part:<9} {(blocks['partition'] == part).sum():>6} {len(sub):>8,} {sub['keep'].sum():>14,}"
              f" {sub['cov20_target'].mean():>7.3f} {sub['cov35_target'].mean():>7.4f}"
              f" {(sub['typhoon'] != '').sum():>7,}  " + "  ".join(f"{x:>6.1%}" for x in shares))
    overall = [(pool["season"] == s).mean() for s in SEASONS]
    print(f"{'all':<9} {len(blocks):>6} {len(pool):>8,} {pool['keep'].sum():>14,}"
          f" {pool['cov20_target'].mean():>7.3f} {pool['cov35_target'].mean():>7.4f}"
          f" {(pool['typhoon'] != '').sum():>7,}  " + "  ".join(f"{x:>6.1%}" for x in overall))
    print("\ntyphoon blocks:", ", ".join(f"{b} -> {blocks.loc[b, 'partition']}"
                                         for b in blocks.index[blocks["typhoon"] != ""]) or "none")
    print("\n'train-eligible' = rows the low-echo filter keeps; only train folds are filtered, "
          "val/test always use every sample (minus the gap at role boundaries).")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    pool.to_csv(args.out, index=False)
    print(f"\nwrote {args.out}")
    print(f"train fold k with: --data.sample_manifest {args.out} --data.cv_fold k   (k = 0..{args.k - 1})")


if __name__ == "__main__":
    main()
