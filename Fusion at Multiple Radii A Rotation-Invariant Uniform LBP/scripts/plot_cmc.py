"""Plot Cumulative Match Characteristic curves from saved experiment results.

Figures 3-14 of the paper are CMC curves for Strategy 2 under each dataset,
evaluation regime and protocol.  Run the methods you want on the curve with
``--output`` and pass the JSON files here.

Usage
-----
    python scripts/plot_cmc.py results/*.json \
        --title "MMCBNU-6000 (one-session, Strategy 2, Protocol 1)" \
        --max-rank 20 --output figures/mmcbnu_s2_p1.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Sequence

import matplotlib

matplotlib.use("Agg")  # render without a display, e.g. on a headless machine
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot CMC curves from run_experiment.py result files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("results", type=Path, nargs="+",
                        help="JSON files written by run_experiment.py --output")
    parser.add_argument("--output", type=Path, default=Path("cmc.png"),
                        help="image file to write")
    parser.add_argument("--title", default=None, help="plot title")
    parser.add_argument("--max-rank", type=int, default=20,
                        help="highest rank shown on the x axis")
    parser.add_argument("--ylim", type=float, nargs=2, default=None,
                        metavar=("LOW", "HIGH"), help="y-axis limits in percent")
    parser.add_argument("--dpi", type=int, default=300, help="output resolution")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    figure, axes = plt.subplots(figsize=(7.0, 4.5))
    for path in args.results:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        curve = np.asarray(data["cmc"], dtype=float)[: args.max_rank]
        ranks = np.arange(1, len(curve) + 1)
        axes.plot(ranks, curve, marker="o", markersize=3,
                  label=str(data.get("method", path.stem)))

    axes.set_xlabel("Rank")
    axes.set_ylabel("Identification accuracy (%)")
    axes.set_xlim(1, args.max_rank)
    if args.ylim is not None:
        axes.set_ylim(*args.ylim)
    if args.title:
        axes.set_title(args.title)
    axes.grid(True, linewidth=0.4, alpha=0.6)
    axes.legend(loc="lower right", fontsize="small")
    figure.tight_layout()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=args.dpi)
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
