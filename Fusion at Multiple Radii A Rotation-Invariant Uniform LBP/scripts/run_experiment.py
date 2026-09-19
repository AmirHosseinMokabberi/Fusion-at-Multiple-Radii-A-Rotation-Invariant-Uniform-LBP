"""Thin wrapper so the CLI can be run without installing the package.

    python scripts/run_experiment.py --dataset utfvp --descriptor fused \
        --reducer 2d2pca --strategy S2 --protocol P1

See ``python scripts/run_experiment.py --help`` for the full argument list, and
``docs/experiment_grid.md`` for the command behind every row of Tables 1-4.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mrflbp.cli import main  # noqa: E402  (import after sys.path fix)

if __name__ == "__main__":
    raise SystemExit(main())
