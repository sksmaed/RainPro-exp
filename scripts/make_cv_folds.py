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
stratum); strata are dealt heaviest-echo first so later ones can even out what
the heavy ones leave. Within a stratum, blocks are taken in order of
decreasing echo (mean coverage >= 20 dBZ), a round of len(partitions) at a
time, and in each round every partition receives one block -- the partition
with the least accumulated echo mass (coverage x samples) picks first. One
block per partition per round keeps sample counts and season shares close;
picking by echo mass keeps a few very wet blocks (a typhoon, a south-westerly
flow event) from piling up in one fold.

That greedy deal alone still leaves the partitions that drew a typhoon
noticeably wetter (one block per partition per round can't compensate), so it
is refined by local search: repeatedly swap two blocks of the SAME stratum
between partitions and keep the swap if the balance score drops. Same-stratum
swaps never change how many blocks of each season a partition holds. Done
from `--tries` random starts, `--swaps` attempts each; the best split wins.

Balance score (lower = more even): coefficient of variation across
partitions of mean coverage >= 20 dBZ, of mean coverage >= 35 dBZ and of
sample count, plus 5 x the mean absolute deviation of season shares from the
pool's. It uses only the pool's own statistics -- never model results -- so
searching over it leaks nothing.

`--typhoon-in-test` reserves one typhoon block (chosen per seed) for the
test partition, so the final test score includes a typhoon; the other typhoon
blocks then go to CV folds only. Off by default: with few typhoons, each one
given to test is one fewer for training and validation.

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


def assign_partitions(
    blocks: pd.DataFrame, partitions: list[str], seed: int, typhoon_in_test: bool
) -> dict[str, str]:
    """blocks: one row per block with columns n, stratum, echo. Returns
    block -> partition. See module docstring for the rule."""
    rng = np.random.default_rng(seed)
    mass = dict.fromkeys(partitions, 0.0)  # accumulated sum of coverage x samples
    out: dict[str, str] = {}

    def give(name: str, part: str) -> None:
        out[name] = part
        mass[part] += float(blocks.loc[name, "echo"] * blocks.loc[name, "n"])

    typhoons = list(blocks.index[blocks["stratum"] == "typhoon"])
    if typhoon_in_test and "test" in partitions and typhoons:
        give(typhoons[int(rng.integers(len(typhoons)))], "test")

    blocks = blocks.assign(mass=blocks["echo"] * blocks["n"])
    order = blocks.groupby("stratum")["mass"].sum().sort_values(ascending=False).index
    for stratum in order:
        group = blocks[(blocks["stratum"] == stratum) & ~blocks.index.isin(list(out))]
        # Shuffle first so equal-echo blocks don't always fall in date order.
        group = group.sample(frac=1.0, random_state=int(rng.integers(2**31))).sort_values(
            "echo", ascending=False, kind="stable"
        )
        eligible = [p for p in partitions if not (typhoon_in_test and stratum == "typhoon" and p == "test")]
        names = list(group.index)
        for start in range(0, len(names), len(eligible)):
            free = list(eligible)
            rng.shuffle(free)  # tie-break among equal masses
            for name in names[start:start + len(eligible)]:
                pick = min(free, key=lambda p: mass[p])
                free.remove(pick)
                give(name, pick)
    return out


class Balance:
    """Fast balance score for a block -> partition-index assignment (see
    module docstring)."""

    def __init__(self, blocks: pd.DataFrame, season_counts: pd.DataFrame, n_parts: int):
        self.k = n_parts
        self.n = blocks["n"].to_numpy(float)
        self.m20 = (blocks["echo"] * blocks["n"]).to_numpy(float)
        self.m35 = (blocks["echo35"] * blocks["n"]).to_numpy(float)
        self.seasons = season_counts.to_numpy(float)
        self.overall = self.seasons.sum(axis=0) / self.seasons.sum()

    def stats(self, a: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        n = np.bincount(a, weights=self.n, minlength=self.k)
        c20 = np.bincount(a, weights=self.m20, minlength=self.k) / n
        c35 = np.bincount(a, weights=self.m35, minlength=self.k) / n
        shares = np.stack([np.bincount(a, weights=col, minlength=self.k) for col in self.seasons.T], axis=1)
        return n, c20, c35, shares / shares.sum(axis=1, keepdims=True)

    def __call__(self, a: np.ndarray) -> float:
        n, c20, c35, shares = self.stats(a)

        def cv(x: np.ndarray) -> float:
            return float(x.std() / x.mean()) if x.mean() else 0.0

        return cv(c20) + cv(c35) + cv(n) + 5 * float(np.abs(shares - self.overall).mean())


def local_search(a: np.ndarray, strata: np.ndarray, score: Balance, swaps: int,
                 rng: np.random.Generator) -> tuple[np.ndarray, float]:
    """Swap same-stratum blocks between partitions while the score improves."""
    a = a.copy()
    best = score(a)
    groups = [np.flatnonzero(strata == s) for s in np.unique(strata)]
    groups = [g for g in groups if len(g) >= 2]
    for _ in range(swaps):
        g = groups[rng.integers(len(groups))]
        i, j = rng.choice(g, size=2, replace=False)
        if a[i] == a[j]:
            continue
        a[i], a[j] = a[j], a[i]
        trial = score(a)
        if trial < best:
            best = trial
        else:
            a[i], a[j] = a[j], a[i]
    return a, best


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pool", required=True, help="CSV from scripts/build_sample_pool.py")
    ap.add_argument("--k", type=int, default=5, help="number of CV folds")
    ap.add_argument("--block-days", type=int, default=3)
    ap.add_argument("--no-test", action="store_true", help="no held-out test partition (CV folds only)")
    ap.add_argument("--seed", type=int, default=0, help="first seed tried")
    ap.add_argument("--tries", type=int, default=20, help="random starts; the best-balanced split is kept")
    ap.add_argument("--swaps", type=int, default=20000, help="local-search swap attempts per start")
    ap.add_argument("--typhoon-in-test", action="store_true",
                     help="reserve one typhoon block for the test partition")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    if args.k < 2:
        raise SystemExit("--k must be at least 2")

    pool = pd.read_csv(args.pool, parse_dates=["init_time", "local_time"], dtype={"typhoon": str})
    pool["typhoon"] = pool["typhoon"].fillna("")
    pool["block"] = assign_blocks(pool, args.block_days)

    blocks = pool.groupby("block").agg(
        n=("init_time", "size"),
        season=("season", lambda s: s.mode().iloc[0]),
        typhoon=("typhoon", "first"),
        echo=("cov20_target", "mean"),
        echo35=("cov35_target", "mean"),
        start=("local_time", "min"),
    )
    season_counts = pd.crosstab(pool["block"], pool["season"]).reindex(blocks.index)
    blocks["stratum"] = np.where(blocks["typhoon"] != "", "typhoon", blocks["season"])

    partitions = [f"fold{i}" for i in range(args.k)] + ([] if args.no_test else ["test"])
    if len(blocks) < len(partitions):
        raise SystemExit(f"only {len(blocks)} blocks for {len(partitions)} partitions -- "
                         "lower --block-days or --k")
    score = Balance(blocks, season_counts, len(partitions))
    index = {p: i for i, p in enumerate(partitions)}
    strata = blocks["stratum"].to_numpy()
    if args.typhoon_in_test:
        # Keep exactly one typhoon in test: typhoon swaps then only trade which one.
        strata = np.where(strata == "typhoon", "typhoon", strata)
    best = None
    for seed in range(args.seed, args.seed + args.tries):
        greedy = assign_partitions(blocks, partitions, seed, args.typhoon_in_test)
        start_a = np.array([index[greedy[b]] for b in blocks.index])
        greedy_score = score(start_a)
        a, s_ = local_search(start_a, strata, score, args.swaps, np.random.default_rng(seed))
        if best is None or s_ < best[0]:
            best = (s_, seed, a, greedy_score)
    final, seed, a, greedy_score = best
    mapping = {b: partitions[i] for b, i in zip(blocks.index, a)}
    n, c20, c35, _ = score.stats(a)
    print(f"best of {args.tries} starts: seed {seed}, balance score {greedy_score:.3f} (greedy) -> {final:.3f}"
          f"  | mean cov20 {c20.min():.3f}-{c20.max():.3f}, cov35 {c35.min():.4f}-{c35.max():.4f},"
          f" samples {int(n.min()):,}-{int(n.max()):,}")
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
