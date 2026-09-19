"""End-to-end experiment runner.

``run_experiment`` ties the modules together in the order used throughout the
paper::

    ROI -> photometric normalisation -> descriptor -> (optional) fusion
        -> dimensionality reduction fitted on the gallery
        -> Manhattan nearest-neighbour matching -> rank-1 / CMC

The reducer is fitted on the training templates only and then applied unchanged
to the probes, so no test information leaks into the projection.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Hashable, List, Sequence, Tuple

import numpy as np

from mrflbp.config import ExperimentConfig
from mrflbp.datasets import FingerVeinDataset, Sample, build_dataset, group_templates
from mrflbp.descriptors import MultiRadiusLBP, fuse_matrices
from mrflbp.evaluation import cmc_curve, misclassified_indices, rank1_accuracy
from mrflbp.matching import NearestNeighbourMatcher
from mrflbp.preprocessing import load_grayscale, preprocess_lbp, preprocess_raw
from mrflbp.protocols import Strategy
from mrflbp.reduction import build_reducer

_EPS = 1e-12


@dataclass
class ExperimentResult:
    """Scores of one experiment, plus the material McNemar's test needs."""

    config: ExperimentConfig
    rank1: float
    cmc: np.ndarray
    gallery_size: int
    probe_count: int
    probe_identities: List[Hashable]
    errors: List[int]

    def summary(self) -> str:
        return (
            f"{self.config.describe()}\n"
            f"  gallery templates : {self.gallery_size}\n"
            f"  probe templates   : {self.probe_count}\n"
            f"  rank-1 accuracy   : {self.rank1:.2f}%\n"
            f"  rank-5 accuracy   : {self.cmc[4]:.2f}%"
        )


def _l2_scale(matrix: np.ndarray) -> np.ndarray:
    """Scale a template so that its flattened form has unit L2 norm."""
    norm = float(np.linalg.norm(matrix))
    return matrix / norm if norm > _EPS else matrix


def _template_for(sample: Sample, config: ExperimentConfig,
                  descriptor: MultiRadiusLBP | None) -> np.ndarray:
    """Descriptor matrix of one ROI image."""
    image = load_grayscale(sample.path)
    if descriptor is None:
        return preprocess_raw(
            image,
            config.image_size,
            config.denoise_h if config.raw_denoise else None,
        )
    normalised = preprocess_lbp(image, config.image_size, config.denoise_h)
    return descriptor.matrix(normalised)


def build_templates(
    samples: Sequence[Sample],
    config: ExperimentConfig,
    descriptor: MultiRadiusLBP | None,
    dataset: FingerVeinDataset,
) -> Tuple[List[np.ndarray], List[Hashable]]:
    """Turn samples into templates and identity keys according to the strategy.

    Strategy 2 yields one template per sample.  Strategy 1 concatenates all
    fingers acquired in the same capture into one composite template and skips
    any group with a missing finger, so every composite has the same shape.
    """
    if config.strategy is Strategy.INDEPENDENT:
        templates = [_template_for(s, config, descriptor) for s in samples]
        identities = [s.identity(config.identity) for s in samples]
    else:
        templates, identities = [], []
        expected = len(dataset.fingers)
        groups = group_templates(samples)
        for key in sorted(groups):
            members = groups[key]
            if len(members) != expected:
                continue  # incomplete capture: cannot form a composite template
            parts = [_template_for(s, config, descriptor) for s in members]
            templates.append(fuse_matrices(parts))
            identities.append(members[0].identity(config.identity))

    if not templates:
        raise RuntimeError(
            "no templates were built; check the dataset root and directory layout"
        )
    if config.normalise_descriptor:
        templates = [_l2_scale(t) for t in templates]
    return templates, identities


def _normalise_rows(features: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    norms[norms < _EPS] = 1.0
    return features / norms


def run_experiment(config: ExperimentConfig) -> ExperimentResult:
    """Run one cell of the experiment grid and return its scores."""
    dataset = build_dataset(config.dataset, config.data_root)
    if config.image_size == (0, 0):
        config.image_size = dataset.image_size

    descriptor = (
        MultiRadiusLBP(
            configs=config.lbp_configs,
            patch_size=config.patch_size,
            stride=config.stride,
        )
        if config.uses_lbp
        else None
    )

    train_samples, test_samples = dataset.split(config.protocol)
    gallery_templates, gallery_identities = build_templates(
        train_samples, config, descriptor, dataset
    )
    probe_templates, probe_identities = build_templates(
        test_samples, config, descriptor, dataset
    )

    reducer = build_reducer(config.reducer, config.n_components)
    gallery = reducer.fit_transform(gallery_templates)
    probes = reducer.transform(probe_templates)

    if config.normalise_features:
        gallery = _normalise_rows(gallery)
        probes = _normalise_rows(probes)

    matcher = NearestNeighbourMatcher(gallery, chunk_size=config.chunk_size)
    ranking = matcher.rank(probes, max_rank=config.max_rank)

    return ExperimentResult(
        config=config,
        rank1=rank1_accuracy(ranking, probe_identities, gallery_identities),
        cmc=cmc_curve(ranking, probe_identities, gallery_identities, config.max_rank),
        gallery_size=len(gallery_identities),
        probe_count=len(probe_identities),
        probe_identities=list(probe_identities),
        errors=misclassified_indices(ranking, probe_identities, gallery_identities),
    )
