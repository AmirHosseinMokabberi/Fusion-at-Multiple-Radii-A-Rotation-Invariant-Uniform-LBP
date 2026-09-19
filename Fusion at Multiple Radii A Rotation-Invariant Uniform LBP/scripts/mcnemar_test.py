"""Compare two methods with McNemar's test on matched-pair rank-1 outcomes.

Section 4 of the paper reports McNemar's test with Yates's continuity correction
at ``alpha = 0.05`` for each pairwise comparison.  The test needs both methods
evaluated on exactly the same probe set, so run the two experiments with the
same ``--dataset``, ``--strategy``, ``--protocol`` and ``--evaluation``, save
each with ``--output``, and pass the two JSON files here.

Usage
-----
    python scripts/run_experiment.py --dataset mmcbnu_6000 --descriptor fused \
        --reducer 2d2pca --strategy S2 --protocol P1 --output results/fused_2d2pca.json
    python scripts/run_experiment.py --dataset mmcbnu_6000 --descriptor lbp_8_1 \
        --strategy S2 --protocol P1 --output results/lbp_8_1.json

    python scripts/mcnemar_test.py results/fused_2d2pca.json results/lbp_8_1.json

Index lists can also be passed directly with ``--wrong-a`` / ``--wrong-b``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mrflbp.evaluation import mcnemar_test  # noqa: E402

COMPARABLE_KEYS = ("dataset", "strategy", "protocol", "evaluation", "probe_count")


def load_result(path: Path) -> Dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if "misclassified_probe_indices" not in data:
        raise ValueError(f"{path} has no 'misclassified_probe_indices' field")
    return data


def check_comparable(a: Dict[str, object], b: Dict[str, object]) -> List[str]:
    """Return the settings on which the two runs disagree."""
    return [key for key in COMPARABLE_KEYS if a.get(key) != b.get(key)]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="McNemar's test between two identification results.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("results", type=Path, nargs="*",
                        help="two JSON files written by run_experiment.py --output")
    parser.add_argument("--wrong-a", type=int, nargs="+", default=None,
                        help="misclassified probe indices of method A")
    parser.add_argument("--wrong-b", type=int, nargs="+", default=None,
                        help="misclassified probe indices of method B")
    parser.add_argument("--alpha", type=float, default=0.05,
                        help="significance level")
    parser.add_argument("--no-continuity-correction", dest="correction",
                        action="store_false",
                        help="use the uncorrected statistic instead of Yates's")
    parser.add_argument("--force", action="store_true",
                        help="run even if the two results come from different settings")
    parser.set_defaults(correction=True)
    return parser


def resolve_inputs(args: argparse.Namespace) -> Tuple[Sequence[int], Sequence[int], str, str]:
    if args.wrong_a is not None and args.wrong_b is not None:
        return args.wrong_a, args.wrong_b, "method A", "method B"

    if len(args.results) != 2:
        raise SystemExit(
            "pass exactly two result files, or both --wrong-a and --wrong-b"
        )

    first, second = (load_result(path) for path in args.results)
    differences = check_comparable(first, second)
    if differences and not args.force:
        raise SystemExit(
            "the two runs are not matched pairs; they differ in: "
            + ", ".join(differences)
            + "\nre-run them with identical settings, or pass --force"
        )
    return (
        first["misclassified_probe_indices"],  # type: ignore[return-value]
        second["misclassified_probe_indices"],  # type: ignore[return-value]
        str(first.get("method", args.results[0].name)),
        str(second.get("method", args.results[1].name)),
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    wrong_a, wrong_b, name_a, name_b = resolve_inputs(args)

    result = mcnemar_test(
        wrong_a, wrong_b, alpha=args.alpha, continuity_correction=args.correction
    )
    print(f"A: {name_a}")
    print(f"B: {name_b}")
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
