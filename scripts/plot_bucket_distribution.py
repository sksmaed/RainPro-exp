"""Presentation version of `infer_visualize.py`'s `<stem>_hist.png`: the
model's predicted probability per dBZ bucket at two grid points, drawn for an
audience rather than for debugging.

One checkpoint, one lead time, two panels side by side (the strongest and the
weakest echo point of the ground truth at that lead, picked exactly as
`infer_visualize.pick_points` does). Compared with the diagnostic version:

  * x axis: 17 horizontal labels, each bucket's lower bound ("<5", 5, ...,
    55, ">=60"), instead of 17 rotated "a-b" ranges.
  * the observed value's bucket is a shaded band explained by the legend, not
    by a note in the axis title.
  * no in-plot statistics box; one large line under each panel instead:
    observed dBZ | model expected dBZ | probability the model gave the
    observed bucket.
  * both panels share the y axis, so bar heights compare across panels.

Writes `<stem>.png` (both panels, 16:9-slide width) and, with `--separate`,
`<stem>_max.png` / `<stem>_min.png` (one panel each, same y scale) for
laying them out on a slide yourself.

Labels are Traditional Chinese by default (`--lang zh`). That needs a CJK
font: a few common ones are tried, `--font /path/to/font.ttf` adds one, and if
none is found the figure falls back to English labels with a warning rather
than rendering empty boxes.

Usage:
    python scripts/plot_bucket_distribution.py \\
        --init-time 2021-06-04T06:00 \\
        --data-root '{"qpesums": "...", "sta_h8": "..."}' \\
        --variable-aliases '{"max_dbz": "MaxDBZ"}' \\
        --ckpt runs/rainpro8_2021_obs_only/checkpoints/best_crps.ckpt \\
        --lead 10 --out figures/dist_20210604_0600.png --separate
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
import matplotlib.font_manager as fm  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from infer_visualize import (  # noqa: E402
    BUCKET_EDGES,
    bin_probabilities,
    bucket_of,
    expected_dbz,
    load_module,
    pick_points,
)
from rainpro.data.rainpro8_dataset import RainPro8Dataset  # noqa: E402
from rainpro.data.rainpro8_datamodule import RainPro8DataModule  # noqa: E402
from rainpro.modules.utils import EvalRequest  # noqa: E402

MODEL_BLUE = "#1878A8"
OBSERVED_PINK = "#F2CBC0"
CJK_FONTS = [
    "Noto Sans CJK TC", "Noto Sans TC", "Source Han Sans TW", "Microsoft JhengHei",
    "PingFang TC", "Heiti TC", "Noto Sans CJK JP", "WenQuanYi Zen Hei", "Arial Unicode MS",
]
# Bucket lower bounds: "<5" (below the first edge), every edge up to the last,
# then the open-ended top bucket.
TICK_LABELS = {
    "zh": ["<5"] + [f"{e:g}" for e in BUCKET_EDGES[:-1]] + [f"≥{BUCKET_EDGES[-1]:g}"],
    "en": ["<5"] + [f"{e:g}" for e in BUCKET_EDGES[:-1]] + [f">={BUCKET_EDGES[-1]:g}"],
}
TEXT = {
    "zh": {
        "observed": "實際值所在區間",
        "model": "模型輸出機率",
        "xlabel": "dBZ 區間（下界）",
        "ylabel": "機率",
        "GT max": "最強回波點",
        "GT min": {"echo": "最弱回波點", "any": "最低值點"},
        "title": "{lead} 預報的機率分布｜{init} 起報",
        "footer": "實際 {gt:.1f} dBZ｜模型期望 {expected:.1f} dBZ｜實際區間機率 {p_gt:.2f}",
        "min": "分鐘",
        "hour": "小時",
    },
    "en": {
        "observed": "observed bucket",
        "model": "model probability",
        "xlabel": "dBZ bucket (lower bound)",
        "ylabel": "probability",
        "GT max": "strongest echo point",
        "GT min": {"echo": "weakest echo point", "any": "lowest-value point"},
        "title": "predicted distribution at {lead} | init {init}",
        "footer": "observed {gt:.1f} dBZ | model expected {expected:.1f} dBZ | P(observed bucket) {p_gt:.2f}",
        "min": "min",
        "hour": "h",
    },
}


def setup_font(lang: str, font_path: str | None) -> str:
    """Pick a font that can draw `lang`; returns the language actually usable."""
    plt.rcParams["axes.unicode_minus"] = False
    if lang == "en":
        return "en"
    if font_path:
        fm.fontManager.addfont(font_path)
        name = fm.FontProperties(fname=font_path).get_name()
        plt.rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
        return "zh"
    available = {f.name for f in fm.fontManager.ttflist}
    for name in CJK_FONTS:
        if name in available:
            plt.rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
            return "zh"
    print("!! no CJK font found (tried: " + ", ".join(CJK_FONTS) + "); falling back to English "
          "labels. Pass --font /path/to/NotoSansCJK.ttc for Chinese.")
    return "en"


def lead_text(lead: int, text: dict) -> str:
    if lead >= 60 and lead % 60 == 0:
        return f"+{lead // 60} {text['hour']}"
    return f"+{lead} {text['min']}"


def draw_panel(subfig, ax, probs: np.ndarray, gt_bin: int, title: str, footer: str, y_top: float,
               lang: str, show_ylabel: bool) -> None:
    text = TEXT[lang]
    x = np.arange(len(probs))
    # Full-height band behind the bars: visible even when the model gives the
    # observed bucket ~0 probability (the case worth showing).
    ax.bar(gt_bin, y_top, width=0.8, color=OBSERVED_PINK, zorder=0, label=text["observed"])
    ax.bar(x, probs, width=0.5, color=MODEL_BLUE, zorder=2, label=text["model"])

    ax.set_xlim(-0.6, len(probs) - 0.4)
    ax.set_ylim(0, y_top)
    ax.set_xticks(x, TICK_LABELS[lang], fontsize=12)
    ax.tick_params(axis="y", labelsize=12)
    ax.yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter("%.1f"))
    ax.grid(axis="y", color="0.88", linewidth=0.8, zorder=1)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_xlabel(text["xlabel"], fontsize=13)
    if show_ylabel:
        ax.set_ylabel(text["ylabel"], fontsize=13)
    ax.set_title(title, fontsize=15, pad=8)
    # Below the x-axis title, in the panel's own subfigure so constrained
    # layout reserves room for it.
    subfig.supxlabel(footer, fontsize=14)


def render(out_path: str, panels: list[dict], suptitle: str, y_top: float, lang: str) -> None:
    text = TEXT[lang]
    width = 13.33 if len(panels) > 1 else 6.8  # 16:9 slide width, or half of it
    fig = plt.figure(figsize=(width, 5.9), layout="constrained")
    fig.suptitle(suptitle, fontsize=16)
    # Legend in its own thin row under the title: a figure-level "outside"
    # legend is not stacked below the suptitle and overlaps it.
    legend_row, body = fig.subfigures(2, 1, height_ratios=[0.07, 1])
    legend_row.legend(handles=[Patch(color=OBSERVED_PINK, label=text["observed"]),
                               Patch(color=MODEL_BLUE, label=text["model"])],
                      loc="center", ncol=2, frameon=False, fontsize=13)
    subfigs = np.atleast_1d(body.subfigures(1, len(panels), wspace=0.04))
    for i, (subfig, panel) in enumerate(zip(subfigs, panels)):
        ax = subfig.subplots()
        draw_panel(subfig, ax, panel["probs"], panel["gt_bin"], panel["title"], panel["footer"], y_top,
                   lang, show_ylabel=i == 0)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"wrote {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--init-time", required=True, help="e.g. 2021-06-04T06:00")
    ap.add_argument("--data-root", required=True, help="same JSON dict as --data.data_root")
    ap.add_argument("--variable-aliases", default="{}")
    ap.add_argument("--ckpt", default=None,
                     help="checkpoint to show; default best_crps.ckpt (or best.ckpt) under --ckpt-dir")
    ap.add_argument("--ckpt-dir", default="runs/rainpro8_2021_obs_only/checkpoints")
    ap.add_argument("--lead", type=int, default=10, help="lead time (min), multiple of 10 in 10..360")
    ap.add_argument("--min-point", default="echo", choices=("echo", "any"),
                     help="weakest point over echo pixels (GT > 0) or over every covered pixel")
    ap.add_argument("--out", default="bucket_distribution.png")
    ap.add_argument("--separate", action="store_true",
                     help="also write <stem>_max.png / <stem>_min.png, one panel each")
    ap.add_argument("--lang", default="zh", choices=("zh", "en"))
    ap.add_argument("--font", default=None, help="font file with CJK glyphs, if none is installed")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    # Must match the checkpoint's training arm, see infer_visualize.py.
    ap.add_argument("--include-satellite", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--include-gfs", action=argparse.BooleanOptionalAction, default=False)
    args = ap.parse_args()

    lang = setup_font(args.lang, args.font)
    text = TEXT[lang]

    ckpt = args.ckpt
    if ckpt is None:
        ckpt = os.path.join(args.ckpt_dir, "best_crps.ckpt")
        if not os.path.isfile(ckpt):
            ckpt = os.path.join(args.ckpt_dir, "best.ckpt")
    if not os.path.isfile(ckpt):
        raise SystemExit(f"checkpoint not found: {ckpt}")

    data_root = json.loads(args.data_root)
    variable_aliases = json.loads(args.variable_aliases)
    init_time = np.datetime64(args.init_time)

    # Only for `sources` / `frames_out`; no setup(), as in infer_visualize.py.
    datamodule = RainPro8DataModule(
        data_root=data_root,
        start_date="2021-01-01",
        end_date="2022-01-01",
        include_satellite=args.include_satellite,
        include_gfs=args.include_gfs,
        variable_aliases=variable_aliases,
    )
    leads = list(datamodule.sources["target_2km"].offsets_min)
    if args.lead not in leads:
        raise SystemExit(f"--lead {args.lead} not in the target's offsets {leads}")
    li = leads.index(args.lead)

    dataset = RainPro8Dataset(
        data_root=data_root,
        sources=datamodule.sources,
        init_times=[init_time],
        jitter_km=0.0,
        variable_aliases=variable_aliases,
    )
    print(f"building input sample for {init_time} ...", flush=True)
    batch = {k: v.unsqueeze(0).to(args.device) for k, v in dataset[0].items()}
    truth = batch["target_2km"][0, li, 0].cpu().numpy()
    points = pick_points(truth, args.min_point)

    print(f"running {ckpt} on {args.device} ...", flush=True)
    module = load_module(ckpt, datamodule, args.device)
    with torch.no_grad():
        out = module(batch, EvalRequest(need_forecast=True, need_probs=True))
    exceed = (1.0 - out.probs[0, li]).cpu().numpy()  # (K, H, W), P(Y >= edge)
    expected = expected_dbz(out.probs)[0, li].cpu().numpy()

    panels = []
    for name, (r, c) in points.items():
        gt = float(truth[r, c])
        probs = bin_probabilities(exceed[:, r, c])
        gt_bin = bucket_of(gt)
        title = text[name] if name == "GT max" else text[name][args.min_point]
        footer = text["footer"].format(gt=gt, expected=float(expected[r, c]), p_gt=float(probs[gt_bin]))
        panels.append({"key": "max" if name == "GT max" else "min", "probs": probs, "gt_bin": gt_bin,
                       "title": title, "footer": footer})
        print(f"{name} (row {r}, col {c}): {footer}")

    # Shared scale across panels (and across --separate files), rounded up to
    # the next 0.1 with a little headroom.
    peak = max(float(p["probs"].max()) for p in panels)
    y_top = min(1.0, max(0.1, np.ceil(peak * 1.1 * 10) / 10))

    suptitle = text["title"].format(lead=lead_text(args.lead, text),
                                    init=np.datetime_as_string(init_time, unit="m").replace("T", " "))
    stem, ext = os.path.splitext(args.out)
    ext = ext or ".png"
    render(f"{stem}{ext}", panels, suptitle, y_top, lang)
    if args.separate:
        for panel in panels:
            render(f"{stem}_{panel['key']}{ext}", [panel], suptitle, y_top, lang)


if __name__ == "__main__":
    main()
