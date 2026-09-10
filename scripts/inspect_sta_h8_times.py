"""Inventory which timestamps/bands actually exist in a raw STA_H8 archive.

STA_H8 isn't a zarr store yet (see docs/rainpro_dataset.md) -- it's a
directory tree of per-band `.btp` files. This script doesn't assume anything
about the directory layout (e.g. `YYYY/MM/DD/YYYYMMDDHH/`); it just walks the
tree recursively and parses the timestamp + band straight out of each
filename (e.g. `2021-06-01_0200.B08.LCC.btp`), so it's robust to any nesting.

Usage:
    python scripts/inspect_sta_h8_times.py /work/kilin1203/datasets/STA_H8/2021
    # narrower scope (faster) for a quick check:
    python scripts/inspect_sta_h8_times.py /work/kilin1203/datasets/STA_H8/2021/06
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime

FNAME_RE = re.compile(r"(\d{4}-\d{2}-\d{2})_(\d{4})\.(B\d{2})\.")
EXPECTED_BANDS = {f"B{i:02d}" for i in range(8, 17)}  # B08..B16


def scan(root: str, progress_every: int = 100_000) -> tuple[dict[datetime, set[str]], int, int]:
    times_bands: dict[datetime, set[str]] = defaultdict(set)
    n_files = 0
    n_unmatched = 0
    for _, _, filenames in os.walk(root):
        for fname in filenames:
            n_files += 1
            if progress_every and n_files % progress_every == 0:
                print(f"...scanned {n_files} files", file=sys.stderr)
            m = FNAME_RE.search(fname)
            if not m:
                n_unmatched += 1
                continue
            date_str, hm_str, band = m.groups()
            ts = datetime.strptime(f"{date_str} {hm_str}", "%Y-%m-%d %H%M")
            times_bands[ts].add(band)
    return times_bands, n_files, n_unmatched


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", help="Directory to scan recursively for .btp files")
    parser.add_argument(
        "--max-list", type=int, default=20, help="Max gap/incomplete rows to print (default 20)"
    )
    args = parser.parse_args()

    times_bands, n_files, n_unmatched = scan(args.root)
    print(f"\nRoot: {args.root}")
    print(f"Total files scanned: {n_files} ({n_unmatched} unmatched filename pattern)")

    if not times_bands:
        print("No timestamped .btp files found.")
        return

    times = sorted(times_bands)
    print(f"Unique timestamps: {len(times)}")
    print(f"Range: {times[0]} .. {times[-1]}")

    deltas = [b - a for a, b in zip(times, times[1:])]
    modal_delta, modal_count = Counter(deltas).most_common(1)[0]
    print(f"\nModal cadence: {modal_delta} ({modal_count}/{len(deltas)} intervals)")

    gaps = [(a, b, d) for a, b, d in zip(times, times[1:], deltas) if d != modal_delta]
    print(f"Irregular gaps (cadence != modal): {len(gaps)}")
    for a, b, d in gaps[: args.max_list]:
        print(f"  {a} -> {b}  (gap {d})")
    if len(gaps) > args.max_list:
        print(f"  ... and {len(gaps) - args.max_list} more")

    incomplete = {t: bands for t, bands in times_bands.items() if bands != EXPECTED_BANDS}
    print(f"\nTimestamps missing >=1 of the 9 expected bands: {len(incomplete)}")
    for t, bands in list(incomplete.items())[: args.max_list]:
        print(f"  {t}: missing {sorted(EXPECTED_BANDS - bands)}")
    if len(incomplete) > args.max_list:
        print(f"  ... and {len(incomplete) - args.max_list} more")

    per_day = Counter(t.date() for t in times)
    print(f"\nDays with >=1 timestamp: {len(per_day)}")
    print(f"Avg timestamps/day: {sum(per_day.values()) / len(per_day):.1f}")


if __name__ == "__main__":
    main()
