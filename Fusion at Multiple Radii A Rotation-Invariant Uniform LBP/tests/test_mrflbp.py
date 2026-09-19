"""Unit tests that need no dataset: run them with ``pytest`` from the repo root.

They cover the parts of the pipeline whose behaviour is fixed by the paper --
the riu2 mapping, the 38-dimensional fused patch descriptor, the protocol
splits, the identity definitions and McNemar's test -- so a broken refactor is
caught without touching FV-USM, MMCBNU-6000 or UTFVP.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from mrflbp.datasets import UTFVP, Sample
from mrflbp.descriptors import (
    FUSED_CONFIGS,
    MultiRadiusLBP,
    fuse_matrices,
    riu2_mapping,
)
from mrflbp.evaluation import (
    cmc_curve,
    match_ranks,
    mcnemar_test,
    misclassified_indices,
    rank1_accuracy,
)
from mrflbp.matching import NearestNeighbourMatcher
from mrflbp.protocols import (
    Evaluation,
    Protocol,
    Strategy,
    archive_identity_fields,
    identity_fields,
    split_indices,
)
from mrflbp.reduction import (
    PCA,
    Identity,
    TwoDimensionalPCA,
    TwoDirectionalTwoDimensionalPCA,
    build_reducer,
)


@pytest.fixture(scope="module")
def roi() -> np.ndarray:
    """A 60 x 128 pseudo-ROI; content is irrelevant to these tests."""
    return np.random.default_rng(0).normal(size=(60, 128))


# --- descriptor -------------------------------------------------------------

@pytest.mark.parametrize("points", [8, 16])
def test_riu2_mapping_has_p_plus_two_bins(points: int) -> None:
    table = riu2_mapping(points)
    assert table.shape == (2**points,)
    assert set(np.unique(table)) <= set(range(points + 2))
    assert table[0] == 0                    # all zeros: uniform, no set bits
    assert table[2**points - 1] == points   # all ones: uniform, all bits set


def test_fused_patch_dimension_is_38() -> None:
    assert MultiRadiusLBP().patch_dimension == 38
    assert FUSED_CONFIGS == ((8, 1), (16, 1), (8, 2))


@pytest.mark.parametrize(
    "configs,dimension",
    [(((8, 1),), 10), (((16, 1),), 18), (((8, 2),), 10), (FUSED_CONFIGS, 38)],
)
def test_descriptor_shapes(roi: np.ndarray, configs, dimension: int) -> None:
    descriptor = MultiRadiusLBP(configs=configs)
    assert descriptor.patch_dimension == dimension
    assert descriptor.grid_shape(roi.shape) == (6, 12)
    assert descriptor.matrix(roi).shape == (6, 12 * dimension)
    assert descriptor.vector(roi).shape == (6 * 12 * dimension,)


def test_each_configuration_histogram_is_a_density(roi: np.ndarray) -> None:
    histogram = MultiRadiusLBP().patch_histogram(roi[:10, :10])
    parts = [histogram[:10], histogram[10:28], histogram[28:]]
    assert np.allclose([part.sum() for part in parts], 1.0)


def test_patch_smaller_than_grid_is_rejected() -> None:
    with pytest.raises(ValueError):
        MultiRadiusLBP().grid_shape((8, 8))


def test_strategy_one_fusion_widens_the_matrix(roi: np.ndarray) -> None:
    matrix = MultiRadiusLBP().matrix(roi)
    fused = fuse_matrices([matrix] * 6)
    assert fused.shape == (matrix.shape[0], 6 * matrix.shape[1])
    with pytest.raises(ValueError):
        fuse_matrices([matrix, matrix[:-1]])


# --- protocols --------------------------------------------------------------

@pytest.mark.parametrize(
    "captures,protocol,train,test",
    [
        (6, Protocol.P1, (1, 2, 3, 4), (5, 6)),
        (6, Protocol.P2, (1, 2, 3), (4, 5, 6)),
        (6, Protocol.P3, (1, 2, 3, 4, 5), (6,)),
        (10, Protocol.P1, tuple(range(1, 8)), (8, 9, 10)),
        (10, Protocol.P2, tuple(range(1, 6)), tuple(range(6, 11))),
        (10, Protocol.P3, tuple(range(1, 10)), (10,)),
    ],
)
def test_split_indices(captures: int, protocol: Protocol, train, test) -> None:
    assert split_indices(range(1, captures + 1), protocol) == (train, test)


def test_split_always_leaves_a_test_sample() -> None:
    train, test = split_indices((1, 2), Protocol.P3)
    assert train == (1,) and test == (2,)


@pytest.mark.parametrize(
    "evaluation,strategy,expected",
    [
        (Evaluation.SESSION_SENSITIVE, Strategy.INDEPENDENT, ("subject", "session")),
        (Evaluation.SESSION_SENSITIVE, Strategy.FUSED, ("subject", "session")),
        (Evaluation.SESSION_INDEPENDENT, Strategy.INDEPENDENT, ("subject",)),
        (Evaluation.ONE_SESSION, Strategy.INDEPENDENT, ("subject", "finger")),
        (Evaluation.ONE_SESSION, Strategy.FUSED, ("subject",)),
    ],
)
def test_identity_fields(evaluation, strategy, expected) -> None:
    assert identity_fields(evaluation, strategy) == expected


@pytest.mark.parametrize(
    "protocol,n_train,n_test",
    [(Protocol.P1, 11, 5), (Protocol.P2, 8, 8), (Protocol.P3, 14, 2)],
)
def test_utfvp_splits_cover_16_images_without_overlap(protocol, n_train, n_test) -> None:
    split = UTFVP.SPLITS[protocol]
    assert (len(split["train"]), len(split["test"])) == (n_train, n_test)
    assert not set(split["train"]) & set(split["test"])


# --- identity conventions ---------------------------------------------------
# `identity_fields` is what the article text describes; `archive_identity_fields`
# is what the experiments compared, and is the repository default.

@pytest.mark.parametrize(
    "finger,hand",
    [("L_Fore", "L"), ("L_Middle", "L"), ("R_Ring", "R"), ("F3", "F3"), ("2", "2")],
)
def test_hand_is_the_leading_token_of_a_compound_finger(finger: str, hand: str) -> None:
    assert Sample(Path("x"), "001", finger, "s1", "01").hand == hand


@pytest.mark.parametrize(
    "dataset,evaluation,strategy,descriptor,reducer,expected",
    [
        # MMCBNU-6000 Strategy 2 collapsed the finger to the hand.
        ("mmcbnu_6000", Evaluation.ONE_SESSION, Strategy.INDEPENDENT,
         "fused", "2d2pca", ("subject", "hand")),
        ("mmcbnu_6000", Evaluation.ONE_SESSION, Strategy.FUSED,
         "fused", "none", ("subject",)),
        # UTFVP matched the paper.
        ("utfvp", Evaluation.ONE_SESSION, Strategy.INDEPENDENT,
         "fused", "none", ("subject", "finger")),
        # FV-USM varied by notebook family.
        ("fv_usm", Evaluation.SESSION_SENSITIVE, Strategy.INDEPENDENT,
         "lbp_8_1", "none", ("subject", "session")),
        ("fv_usm", Evaluation.SESSION_SENSITIVE, Strategy.INDEPENDENT,
         "fused", "2d2pca", ("subject", "finger", "session")),
        ("fv_usm", Evaluation.SESSION_SENSITIVE, Strategy.INDEPENDENT,
         "raw", "2dpca", ("session", "finger")),
        ("fv_usm", Evaluation.SESSION_INDEPENDENT, Strategy.INDEPENDENT,
         "fused", "none", ("subject", "finger")),
        ("fv_usm", Evaluation.SESSION_SENSITIVE, Strategy.FUSED,
         "fused", "none", ("subject", "session")),
    ],
)
def test_archive_identity_matches_what_the_experiments_compared(
    dataset, evaluation, strategy, descriptor, reducer, expected
) -> None:
    assert archive_identity_fields(
        dataset, evaluation, strategy, descriptor, reducer
    ) == expected


def test_archive_and_article_identities_agree_on_utfvp() -> None:
    for strategy in (Strategy.FUSED, Strategy.INDEPENDENT):
        assert archive_identity_fields(
            "utfvp", Evaluation.ONE_SESSION, strategy, "fused", "none"
        ) == identity_fields(Evaluation.ONE_SESSION, strategy)


# --- per-family ROI size ----------------------------------------------------

@pytest.mark.parametrize(
    "dataset,descriptor,reducer,expected",
    [
        # FV-USM: only the LBP-plus-reducer family used the other orientation.
        ("fv_usm", "raw", "pca", (300, 100)),
        ("fv_usm", "lbp_8_1", "none", (300, 100)),
        ("fv_usm", "fused", "none", (300, 100)),
        ("fv_usm", "fused", "2d2pca", (100, 300)),
        # MMCBNU-6000: the LBP families trim to 120 columns, the baselines do not.
        ("mmcbnu_6000", "raw", "2dpca", (60, 128)),
        ("mmcbnu_6000", "fused", "none", (60, 120)),
        ("mmcbnu_6000", "fused", "pca", (60, 120)),
        # UTFVP is uniform.
        ("utfvp", "raw", "pca", (60, 128)),
        ("utfvp", "fused", "2d2pca", (60, 128)),
    ],
)
def test_image_size_follows_the_feature_family(
    dataset, descriptor, reducer, expected
) -> None:
    from mrflbp.config import build_config

    config = build_config(
        dataset=dataset, descriptor=descriptor, reducer=reducer,
        strategy="S2", protocol="P1", data_root="unused",
    )
    assert config.image_size == expected


# --- reduction --------------------------------------------------------------

@pytest.fixture(scope="module")
def templates():
    rng = np.random.default_rng(1)
    return [rng.normal(size=(6, 40)) for _ in range(30)]


def test_identity_reducer_only_flattens(templates) -> None:
    assert Identity().fit_transform(templates).shape == (30, 240)


def test_pca_keeps_all_non_zero_components_by_default(templates) -> None:
    assert PCA().fit_transform(templates).shape == (30, 29)
    assert PCA(5).fit_transform(templates).shape == (30, 5)


def test_two_dimensional_reducers(templates) -> None:
    assert TwoDimensionalPCA(7).fit_transform(templates).shape == (30, 6 * 7)
    assert TwoDirectionalTwoDimensionalPCA(4, 9).fit_transform(templates).shape == (30, 36)


def test_over_requested_components_are_clipped(templates) -> None:
    # The patch grid of MMCBNU-6000/UTFVP has fewer rows than the 47 components
    # requested in the configs, so the request is clipped to what exists.
    assert build_reducer("2d2pca", 100).fit_transform(templates).shape == (30, 6 * 40)


def test_transform_before_fit_raises(templates) -> None:
    with pytest.raises(RuntimeError):
        PCA().transform(templates)


def test_unknown_reducer_is_rejected() -> None:
    with pytest.raises(ValueError):
        build_reducer("lda")


# --- matching and metrics ---------------------------------------------------

@pytest.fixture
def toy_problem():
    gallery = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [5.0, 5.0]])
    probes = np.array([[0.1, 0.0], [0.0, 1.1], [4.9, 5.2]])
    return gallery, probes, ["a", "b", "c", "d"], ["a", "c", "d"]


def test_nearest_neighbour_is_manhattan(toy_problem) -> None:
    gallery, probes, _, _ = toy_problem
    matcher = NearestNeighbourMatcher(gallery, chunk_size=2)
    assert list(matcher.predict(probes)) == [0, 2, 3]
    assert matcher.rank(probes, max_rank=2).shape == (3, 2)


def test_perfect_run_scores_100(toy_problem) -> None:
    gallery, probes, gallery_ids, probe_ids = toy_problem
    ranking = NearestNeighbourMatcher(gallery).rank(probes)
    assert rank1_accuracy(ranking, probe_ids, gallery_ids) == 100.0
    assert misclassified_indices(ranking, probe_ids, gallery_ids) == []
    curve = cmc_curve(ranking, probe_ids, gallery_ids, max_rank=10)
    assert curve.shape == (10,)
    assert np.all(np.diff(curve) >= 0)


def test_wrong_probe_is_reported(toy_problem) -> None:
    gallery, _, gallery_ids, _ = toy_problem
    ranking = NearestNeighbourMatcher(gallery).rank(np.array([[5.0, 5.0]]))
    assert misclassified_indices(ranking, ["a"], gallery_ids) == [0]
    assert match_ranks(ranking, ["a"], gallery_ids) == [3]


def test_mcnemar_counts_only_discordant_pairs() -> None:
    result = mcnemar_test([1, 2, 3], [3, 4])
    assert (result.b, result.c, result.discordant) == (2, 1, 3)
    assert result.chi2 == pytest.approx((abs(2 - 1) - 1) ** 2 / 3)


def test_mcnemar_without_discordant_pairs_is_not_significant() -> None:
    result = mcnemar_test([1, 2], [1, 2])
    assert result.discordant == 0
    assert result.p_value == 1.0
    assert not result.significant


def test_mcnemar_continuity_correction_lowers_the_statistic() -> None:
    corrected = mcnemar_test(range(20), range(15, 25))
    uncorrected = mcnemar_test(range(20), range(15, 25), continuity_correction=False)
    assert corrected.chi2 < uncorrected.chi2
    assert corrected.p_value > uncorrected.p_value
