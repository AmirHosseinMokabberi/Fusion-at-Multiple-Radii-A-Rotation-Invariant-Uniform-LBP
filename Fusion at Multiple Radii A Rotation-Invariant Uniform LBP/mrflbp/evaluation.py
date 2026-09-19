"""Identification metrics: rank-1 accuracy, CMC curves and McNemar's test.

Identities are compared as opaque hashable keys (the tuples produced by
:meth:`mrflbp.datasets.Sample.identity`), so nothing here depends on how a
dataset spells a subject, finger or session.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Hashable, Iterable, List, Sequence

import numpy as np

#: chi-square critical value with one degree of freedom at alpha = 0.05.
CHI2_CRITICAL_1DF_005 = 3.841458820694124


def _check_shapes(
    ranked_indices: np.ndarray,
    probe_identities: Sequence[Hashable],
    gallery_identities: Sequence[Hashable],
) -> None:
    if len(probe_identities) == 0:
        raise ValueError("no probes to score")
    if ranked_indices.shape[0] != len(probe_identities):
        raise ValueError(
            f"{ranked_indices.shape[0]} ranking rows for "
            f"{len(probe_identities)} probes"
        )
    if ranked_indices.size and ranked_indices.max() >= len(gallery_identities):
        raise ValueError("ranking refers to a gallery entry that does not exist")


def match_ranks(
    ranked_indices: np.ndarray,
    probe_identities: Sequence[Hashable],
    gallery_identities: Sequence[Hashable],
) -> List[int]:
    """Zero-based rank at which each probe first meets its own identity.

    A probe whose identity is absent from the inspected part of the ranking gets
    ``-1``.
    """
    _check_shapes(ranked_indices, probe_identities, gallery_identities)
    gallery = list(gallery_identities)
    hits: List[int] = []
    for row, identity in zip(ranked_indices, probe_identities):
        hit = -1
        for rank, index in enumerate(row):
            if gallery[int(index)] == identity:
                hit = rank
                break
        hits.append(hit)
    return hits


def rank1_accuracy(
    ranked_indices: np.ndarray,
    probe_identities: Sequence[Hashable],
    gallery_identities: Sequence[Hashable],
) -> float:
    """Rank-1 identification rate in percent."""
    _check_shapes(ranked_indices, probe_identities, gallery_identities)
    gallery = list(gallery_identities)
    correct = sum(
        1
        for row, identity in zip(ranked_indices, probe_identities)
        if gallery[int(row[0])] == identity
    )
    return 100.0 * correct / len(probe_identities)


def cmc_curve(
    ranked_indices: np.ndarray,
    probe_identities: Sequence[Hashable],
    gallery_identities: Sequence[Hashable],
    max_rank: int = 100,
) -> np.ndarray:
    """Cumulative Match Characteristic curve in percent, for ranks 1..``max_rank``."""
    if max_rank <= 0:
        raise ValueError("max_rank must be positive")
    limit = min(max_rank, ranked_indices.shape[1])
    hits = match_ranks(ranked_indices[:, :limit], probe_identities, gallery_identities)

    counts = np.zeros(limit, dtype=float)
    for rank in hits:
        if rank >= 0:
            counts[rank:] += 1
    curve = 100.0 * counts / len(probe_identities)
    if limit < max_rank:  # a gallery shorter than max_rank keeps its final value
        curve = np.concatenate([curve, np.full(max_rank - limit, curve[-1])])
    return curve


def misclassified_indices(
    ranked_indices: np.ndarray,
    probe_identities: Sequence[Hashable],
    gallery_identities: Sequence[Hashable],
) -> List[int]:
    """Positions of the probes a method gets wrong at rank 1.

    These index lists are the input of :func:`mcnemar_test`: two methods compared
    that way must be evaluated on the same, identically ordered probe set, so a
    probe index means the same sample for both.
    """
    _check_shapes(ranked_indices, probe_identities, gallery_identities)
    gallery = list(gallery_identities)
    return [
        i
        for i, (row, identity) in enumerate(zip(ranked_indices, probe_identities))
        if gallery[int(row[0])] != identity
    ]


@dataclass
class McNemarResult:
    """Outcome of a matched-pair comparison between two methods."""

    b: int
    c: int
    chi2: float
    p_value: float
    alpha: float
    continuity_correction: bool
    significant: bool = field(init=False)

    def __post_init__(self) -> None:
        self.significant = self.p_value < self.alpha

    @property
    def discordant(self) -> int:
        """Number of probes the two methods disagree on."""
        return self.b + self.c

    def __str__(self) -> str:
        verdict = "significant" if self.significant else "not significant"
        return (
            f"b={self.b} c={self.c} n={self.discordant} "
            f"chi2={self.chi2:.4f} p={self.p_value:.4g} "
            f"({verdict} at alpha={self.alpha})"
        )


def mcnemar_test(
    wrong_a: Iterable[int],
    wrong_b: Iterable[int],
    alpha: float = 0.05,
    continuity_correction: bool = True,
) -> McNemarResult:
    """McNemar's test for matched-pair binary outcomes.

    Parameters
    ----------
    wrong_a, wrong_b:
        Indices of the probes each method misclassifies, as returned by
        :func:`misclassified_indices`.
    alpha:
        Significance level; the paper uses 0.05.
    continuity_correction:
        Apply Yates's correction, as reported in Section 4 of the paper.

    Returns
    -------
    McNemarResult
        ``b`` counts probes only method A gets wrong and ``c`` probes only
        method B gets wrong; concordant pairs do not contribute to the statistic.
    """
    set_a, set_b = set(wrong_a), set(wrong_b)
    b = len(set_a - set_b)
    c = len(set_b - set_a)
    n = b + c

    if n == 0:
        return McNemarResult(
            b=b, c=c, chi2=0.0, p_value=1.0, alpha=alpha,
            continuity_correction=continuity_correction,
        )

    if continuity_correction:
        chi2 = (abs(b - c) - 1) ** 2 / n
    else:
        chi2 = (b - c) ** 2 / n
    # Upper tail of the chi-square distribution with one degree of freedom.
    p_value = math.erfc(math.sqrt(chi2) / math.sqrt(2.0))
    return McNemarResult(
        b=b, c=c, chi2=chi2, p_value=p_value, alpha=alpha,
        continuity_correction=continuity_correction,
    )
