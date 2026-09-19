"""Command-line entry point: run one cell of the paper's experiment grid.

Examples
--------
The proposed operator with (2D)^2PCA on MMCBNU-6000, Strategy 2, Protocol 1::

    python -m mrflbp.cli --dataset mmcbnu_6000 --descriptor fused \
        --reducer 2d2pca --strategy S2 --protocol P1

The single-radius baseline on FV-USM under the session-sensitive regime::

    python -m mrflbp.cli --dataset fv_usm --descriptor lbp_8_1 \
        --strategy S2 --protocol P3 --evaluation session-sensitive
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from mrflbp.config import DESCRIPTORS, build_config
from mrflbp.datasets import DATASETS
from mrflbp.pipeline import run_experiment
from mrflbp.protocols import identity_fields


def paper_identity(config) -> tuple:
    """Identity Section 3.1 defines for this regime and strategy."""
    return identity_fields(config.evaluation, config.strategy)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mrflbp",
        description="Multi-Radius Fused LBP finger-vein identification experiments.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset", required=True, choices=sorted(DATASETS),
        help="dataset to evaluate on",
    )
    parser.add_argument(
        "--data-root", type=Path, default=None,
        help="ROI directory; overrides 'data_root' in the dataset's YAML config",
    )
    parser.add_argument(
        "--descriptor", default="fused", choices=sorted(DESCRIPTORS),
        help="feature extractor ('raw' uses pixel intensities directly)",
    )
    parser.add_argument(
        "--reducer", default="none", choices=["none", "pca", "2dpca", "2d2pca"],
        help="dimensionality reduction applied to the descriptor",
    )
    parser.add_argument(
        "--strategy", default="S2", choices=["S1", "S2"],
        help="S1 fuses a subject's fingers into one template, S2 keeps them separate",
    )
    parser.add_argument(
        "--protocol", default="P1", choices=["P1", "P2", "P3"],
        help="train/test split: P1 70/30, P2 50/50, P3 90/10",
    )
    parser.add_argument(
        "--evaluation", default=None,
        choices=["session-sensitive", "session-independent", "one-session"],
        help="how genuine pairs are defined; defaults to the dataset's YAML setting",
    )
    parser.add_argument(
        "--components", type=int, default=None,
        help="number of components for the reducer; defaults to the YAML setting",
    )
    parser.add_argument(
        "--identity", nargs="+", default=None,
        choices=["subject", "finger", "session", "hand"],
        help="override which sample attributes define an identity",
    )
    parser.add_argument(
        "--identity-convention", default="archive", choices=["archive", "paper"],
        help=(
            "'archive' scores the way the experiments behind the published "
            "tables did; 'paper' uses the definitions in Section 3.1 of the "
            "article. See docs/identity_conventions.md"
        ),
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="optional JSON file to write the scores and CMC curve to",
    )
    parser.add_argument(
        "--config-dir", type=Path, default=None,
        help="directory holding the per-dataset YAML files",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    kwargs = {}
    if args.config_dir is not None:
        kwargs["config_dir"] = args.config_dir

    config = build_config(
        dataset=args.dataset,
        descriptor=args.descriptor,
        reducer=args.reducer,
        strategy=args.strategy,
        protocol=args.protocol,
        evaluation=args.evaluation,
        data_root=args.data_root,
        n_components=args.components,
        identity=args.identity,
        identity_convention=args.identity_convention,
        **kwargs,
    )

    if args.identity and args.identity_convention != "archive":
        print("note: --identity overrides --identity-convention")

    result = run_experiment(config)
    print(result.summary())
    if not config.identity_matches_paper:
        expected = "+".join(paper_identity(config))
        active = "+".join(config.identity)
        print(
            f"\nnote: scored on {active}, which is what the published run used; "
            f"the article text describes {expected}"
        )

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "dataset": config.dataset,
            "method": config.label,
            "strategy": config.strategy.value,
            "protocol": config.protocol.value,
            "evaluation": config.evaluation.value,
            "identity": list(config.identity),
            "identity_convention": config.identity_convention,
            "identity_matches_paper": config.identity_matches_paper,
            "gallery_size": result.gallery_size,
            "probe_count": result.probe_count,
            "rank1_accuracy": result.rank1,
            "cmc": np.asarray(result.cmc).tolist(),
            "misclassified_probe_indices": result.errors,
        }
        with args.output.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        print(f"\nwrote {args.output}")

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
