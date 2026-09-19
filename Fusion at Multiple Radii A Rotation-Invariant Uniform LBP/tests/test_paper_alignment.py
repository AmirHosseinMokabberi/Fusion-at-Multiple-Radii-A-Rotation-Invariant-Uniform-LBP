"""Check the implementation against the numbers stated in the article.

Every value the article gives explicitly is asserted here, so a later refactor
cannot drift away from the published description without a test failing.

The three settings where the experiments and the article text disagree are
asserted as *differences*, with the article's value recorded alongside. They are
pinned deliberately: this repository reproduces what was run, and
``docs/discrepancies.md`` explains each one.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

from mrflbp.config import build_config
from mrflbp.datasets import FVUSM, MMCBNU6000, UTFVP
from mrflbp.descriptors import FUSED_CONFIGS, MultiRadiusLBP
from mrflbp.protocols import Protocol, split_indices

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"


def _load(name: str) -> ModuleType:
    """Import a file from ``scripts/`` without making it an installable package."""
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def roi_script() -> ModuleType:
    return _load("prepare_utfvp_roi")


@pytest.fixture(scope="module")
def augment_script() -> ModuleType:
    return _load("augment_utfvp")


# --- Section 2: the proposed descriptor -------------------------------------

def test_operator_fuses_the_three_stated_configurations() -> None:
    assert FUSED_CONFIGS == ((8, 1), (16, 1), (8, 2))


def test_patch_descriptor_is_38_dimensional() -> None:
    # "dim(h_(8,1)) = 10, dim(h_(16,1)) = 18, dim(h_(8,2)) = 10 ... F_patch in R^38"
    assert [p + 2 for p, _ in FUSED_CONFIGS] == [10, 18, 10]
    assert MultiRadiusLBP().patch_dimension == 38


def test_patches_are_10x10_with_zero_overlap() -> None:
    descriptor = MultiRadiusLBP()
    assert descriptor.patch_size == 10
    assert descriptor.stride == descriptor.patch_size


# --- Section 3: datasets ----------------------------------------------------

def test_dataset_sizes() -> None:
    assert FVUSM.SUBJECTS * len(FVUSM.fingers) * len(FVUSM.CAPTURES) * len(
        FVUSM.SESSIONS
    ) == 5904
    assert 100 * len(MMCBNU6000.fingers) * len(MMCBNU6000.CAPTURES) == 6000
    assert 60 * len(UTFVP.fingers) * len(UTFVP.CAPTURES) == 1440


def test_augmentation_expands_utfvp_to_5760_images() -> None:
    # "generates three variants per original image, expanding each finger to 16"
    per_capture = 1 + len(UTFVP.AUGMENTATIONS)
    assert len(UTFVP.CAPTURES) * per_capture == 16
    assert 60 * len(UTFVP.fingers) * len(UTFVP.CAPTURES) * per_capture == 5760


# --- Section 3.1: protocols -------------------------------------------------

def test_protocol_fractions() -> None:
    assert Protocol.P1.train_fraction == 0.7
    assert Protocol.P2.train_fraction == 0.5
    assert Protocol.P3.train_fraction == 0.9


def test_training_size_is_rounded_down() -> None:
    # "the training set size is rounded down to the nearest integer to guarantee
    #  at least one sample is always available for testing"
    assert split_indices(FVUSM.CAPTURES, Protocol.P3) == ((1, 2, 3, 4, 5), (6,))
    assert split_indices(MMCBNU6000.CAPTURES, Protocol.P1) == (
        (1, 2, 3, 4, 5, 6, 7), (8, 9, 10),
    )
    # 90% of 3 is 2.7, which floors to 2 and leaves one sample for testing.
    assert split_indices((1, 2, 3), Protocol.P3) == ((1, 2), (3,))


def test_pca_retains_all_non_zero_eigenvectors() -> None:
    # "In our experiments, PCA retains all nonzero eigenvectors"
    config = build_config(
        dataset="fv_usm", descriptor="fused", reducer="pca",
        strategy="S2", protocol="P1", data_root="unused",
    )
    assert config.n_components is None


# --- Section 3: UTFVP ROI pipeline ------------------------------------------

def test_roi_pipeline_parameters(roi_script: ModuleType) -> None:
    assert roi_script.CLAHE_TILE == (8, 8)
    assert roi_script.VARIANCE_THRESHOLD == 38.0
    assert roi_script.DBSCAN_EPS == 10.0
    assert roi_script.DBSCAN_MIN_SAMPLES == 10
    assert roi_script.ROTATION_THRESHOLD == 15.0


def test_augmentation_parameter_ranges(augment_script: ModuleType) -> None:
    assert augment_script.VARIANTS_PER_IMAGE == 3
    assert augment_script.BRIGHTNESS_RANGE == (0.6, 1.5)
    assert augment_script.ROTATION_RANGE == (-10.0, 10.0)
    assert augment_script.TRANSLATION_RANGE == (-4, 4)
    assert augment_script.NOISE_STD_RANGE == (10.0, 30.0)


# --- the three deliberate departures ----------------------------------------

def test_clip_limit_follows_the_experiments_not_the_text(roi_script: ModuleType) -> None:
    article_value = 0.75
    assert roi_script.CLAHE_CLIP_LIMIT == 2.0
    assert roi_script.CLAHE_CLIP_LIMIT != article_value


@pytest.mark.parametrize(
    "dataset,components",
    [("fv_usm", 137), ("mmcbnu_6000", 47), ("utfvp", 47)],
)
def test_component_counts_follow_the_experiments(dataset: str, components: int) -> None:
    # The article reports "a fixed set of 137 components across descriptors";
    # only FV-USM used 137.
    for reducer in ("2dpca", "2d2pca"):
        config = build_config(
            dataset=dataset, descriptor="fused", reducer=reducer,
            strategy="S2", protocol="P1", data_root="unused",
        )
        assert config.n_components == components


def test_augmentation_has_two_stages_the_article_does_not_mention(
    augment_script: ModuleType,
) -> None:
    parser = augment_script.build_parser()
    defaults = parser.parse_args(["--input", ".", "--output", "."])
    assert defaults.blur is True
    assert defaults.occlusion is True
