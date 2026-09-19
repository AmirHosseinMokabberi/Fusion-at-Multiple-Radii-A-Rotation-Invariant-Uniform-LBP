"""Experiment configuration.

A single :class:`ExperimentConfig` fully determines one cell of the experiment
grid reported in Tables 1-4 of the paper: dataset, descriptor, reducer,
strategy, protocol and evaluation regime, plus the hyper-parameters that stay
fixed across the grid (patch size, stride, component counts, ...).

Per-dataset defaults live in the YAML files under ``configs/`` so that paths and
component counts can be changed without touching the code.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import yaml

from mrflbp.datasets import DATASETS
from mrflbp.descriptors import FUSED_CONFIGS, SINGLE_RADIUS_CONFIGS
from mrflbp.protocols import (
    Evaluation,
    Protocol,
    Strategy,
    archive_identity_fields,
    identity_fields,
)

CONFIG_DIR = Path(__file__).resolve().parent.parent / "configs"

#: Descriptor name -> (P, R) configurations; ``raw`` bypasses the LBP stage.
DESCRIPTORS: Dict[str, Optional[Tuple[Tuple[int, int], ...]]] = {
    "raw": None,
    "lbp_8_1": SINGLE_RADIUS_CONFIGS["lbp_8_1"],
    "lbp_16_1": SINGLE_RADIUS_CONFIGS["lbp_16_1"],
    "lbp_8_2": SINGLE_RADIUS_CONFIGS["lbp_8_2"],
    "fused": FUSED_CONFIGS,
}

#: Human-readable names matching the row labels of Tables 1-4.
DESCRIPTOR_LABELS: Dict[str, str] = {
    "raw": "raw pixels",
    "lbp_8_1": "LBP^riu2_(8,1)",
    "lbp_16_1": "LBP^riu2_(16,1)",
    "lbp_8_2": "LBP^riu2_(8,2)",
    "fused": "LBP^riu2_(8,1),(16,1),(8,2)",
}


@dataclass
class ExperimentConfig:
    """Everything needed to run one experiment."""

    dataset: str
    data_root: Path
    descriptor: str = "fused"
    reducer: str = "none"
    strategy: Strategy = Strategy.INDEPENDENT
    protocol: Protocol = Protocol.P1
    evaluation: Evaluation = Evaluation.ONE_SESSION

    image_size: Tuple[int, int] = (0, 0)
    patch_size: int = 10
    stride: int = 10
    denoise_h: float = 10.0
    raw_denoise: bool = False
    n_components: Optional[int] = None

    normalise_descriptor: bool = True
    normalise_features: bool = True
    max_rank: int = 100
    chunk_size: int = 64

    identity: Tuple[str, ...] = field(default_factory=tuple)
    #: "archive" reproduces what the experiments ran; "paper" uses Section 3.1.
    identity_convention: str = "archive"

    def __post_init__(self) -> None:
        self.dataset = self.dataset.strip().lower().replace("-", "_")
        if self.dataset not in DATASETS:
            raise ValueError(
                f"unknown dataset {self.dataset!r}; expected one of "
                f"{', '.join(sorted(DATASETS))}"
            )
        if self.descriptor not in DESCRIPTORS:
            raise ValueError(
                f"unknown descriptor {self.descriptor!r}; expected one of "
                f"{', '.join(DESCRIPTORS)}"
            )
        self.data_root = Path(self.data_root)
        self.image_size = tuple(self.image_size)  # type: ignore[assignment]
        if self.identity_convention not in {"archive", "paper"}:
            raise ValueError(
                f"unknown identity convention {self.identity_convention!r}; "
                "expected 'archive' or 'paper'"
            )
        if self.identity:
            self.identity = tuple(self.identity)
        elif self.identity_convention == "paper":
            self.identity = identity_fields(self.evaluation, self.strategy)
        else:
            self.identity = archive_identity_fields(
                self.dataset, self.evaluation, self.strategy,
                self.descriptor, self.reducer,
            )

    @property
    def identity_matches_paper(self) -> bool:
        """Whether the active identity is the one the article text describes."""
        return self.identity == identity_fields(self.evaluation, self.strategy)

    @property
    def feature_family(self) -> str:
        """Which of the three notebook families this configuration belongs to.

        The families resized the ROI differently, so the image size is looked up
        per family rather than per dataset.
        """
        if self.descriptor == "raw":
            return "raw"
        return "lbp" if self.reducer.lower() in {"none", "raw"} else "lbp_reduced"

    @property
    def lbp_configs(self) -> Optional[Tuple[Tuple[int, int], ...]]:
        """``(P, R)`` pairs of the descriptor, or ``None`` for raw pixels."""
        return DESCRIPTORS[self.descriptor]

    @property
    def uses_lbp(self) -> bool:
        return self.lbp_configs is not None

    @property
    def label(self) -> str:
        """Row label of this configuration in Tables 1-4."""
        base = DESCRIPTOR_LABELS[self.descriptor]
        if self.reducer.lower() in {"none", "raw"}:
            return base
        pretty = {"pca": "PCA", "2dpca": "2DPCA", "2d2pca": "(2D)^2PCA"}
        reducer = pretty.get(self.reducer.lower(), self.reducer)
        return reducer if self.descriptor == "raw" else f"{base} + {reducer}"

    def describe(self) -> str:
        return (
            f"{self.dataset} | {self.label} | {self.strategy.value} "
            f"{self.protocol.value} | {self.evaluation.value} | "
            f"identity={'+'.join(self.identity)}"
        )


def load_dataset_defaults(dataset: str, config_dir: Path | str = CONFIG_DIR) -> Dict[str, Any]:
    """Read ``configs/<dataset>.yaml`` and return it as a plain dictionary."""
    key = dataset.strip().lower().replace("-", "_")
    path = Path(config_dir) / f"{key}.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"no configuration file for dataset {dataset!r}: {path}")
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a mapping at the top level")
    return data


def _resolve_image_size(
    defaults: Dict[str, Any], descriptor: str, reducer: str
) -> Tuple[int, int]:
    """Read the ROI size for this feature family out of the YAML defaults.

    ``image_size`` may be a single ``[height, width]`` pair, or a mapping keyed
    by feature family (``raw``, ``lbp``, ``lbp_reduced``) for the datasets whose
    notebook families resized differently.
    """
    configured = defaults.get("image_size", (0, 0))
    if isinstance(configured, dict):
        if descriptor == "raw":
            family = "raw"
        else:
            family = "lbp" if reducer.lower() in {"none", "raw"} else "lbp_reduced"
        if family not in configured:
            raise ValueError(
                f"configs entry 'image_size' has no '{family}' key; "
                f"it defines {', '.join(sorted(configured))}"
            )
        configured = configured[family]
    return tuple(configured)  # type: ignore[return-value]


def build_config(
    dataset: str,
    descriptor: str,
    reducer: str,
    strategy: str,
    protocol: str,
    evaluation: Optional[str] = None,
    data_root: Optional[Path | str] = None,
    n_components: Optional[int] = None,
    identity: Optional[Sequence[str]] = None,
    identity_convention: str = "archive",
    config_dir: Path | str = CONFIG_DIR,
) -> ExperimentConfig:
    """Merge command-line arguments with the per-dataset YAML defaults."""
    defaults = load_dataset_defaults(dataset, config_dir)

    resolved_root = data_root if data_root is not None else defaults.get("data_root")
    if not resolved_root:
        raise ValueError(
            f"no data root given for {dataset!r}; pass --data-root or set "
            f"'data_root' in configs/{dataset}.yaml"
        )

    resolved_evaluation = Evaluation.parse(
        evaluation if evaluation is not None else defaults.get("evaluation", "one-session")
    )

    components = defaults.get("n_components", {}) or {}
    resolved_components = (
        n_components if n_components is not None else components.get(reducer.lower())
    )

    return ExperimentConfig(
        dataset=dataset,
        data_root=resolved_root,
        descriptor=descriptor,
        reducer=reducer,
        strategy=Strategy.parse(strategy),
        protocol=Protocol.parse(protocol),
        evaluation=resolved_evaluation,
        image_size=_resolve_image_size(defaults, descriptor, reducer),
        patch_size=int(defaults.get("patch_size", 10)),
        stride=int(defaults.get("stride", 10)),
        denoise_h=float(defaults.get("denoise_h", 10.0)),
        raw_denoise=bool(defaults.get("raw_denoise", False)),
        n_components=resolved_components,
        normalise_descriptor=bool(defaults.get("normalise_descriptor", True)),
        normalise_features=bool(defaults.get("normalise_features", True)),
        max_rank=int(defaults.get("max_rank", 100)),
        chunk_size=int(defaults.get("chunk_size", 64)),
        identity=tuple(identity) if identity else (),
        identity_convention=identity_convention,
    )
