"""Nearest-neighbour identification with the Manhattan (L1) distance.

Every experiment in the paper enrols the training templates in a gallery and
classifies a probe by its closest gallery entry under the city-block distance,
so this module is deliberately small: it computes probe-to-gallery distances in
memory-bounded chunks and returns the candidate ranking each metric needs.
"""

from __future__ import annotations

from typing import Iterator, Optional

import numpy as np


def manhattan_distances(probes: np.ndarray, gallery: np.ndarray) -> np.ndarray:
    """Pairwise L1 distances between every probe and every gallery template."""
    probes = np.atleast_2d(np.asarray(probes, dtype=np.float64))
    gallery = np.asarray(gallery, dtype=np.float64)
    if probes.shape[1] != gallery.shape[1]:
        raise ValueError(
            f"probe dimension {probes.shape[1]} does not match gallery "
            f"dimension {gallery.shape[1]}"
        )
    return np.abs(probes[:, None, :] - gallery[None, :, :]).sum(axis=2)


class NearestNeighbourMatcher:
    """Gallery of enrolled templates queried with the Manhattan distance.

    Parameters
    ----------
    gallery:
        ``(n_gallery, n_features)`` array of enrolled templates.
    chunk_size:
        Number of probes handled per distance computation.  The default keeps
        the temporary distance tensor small enough for the CPU-only setting the
        paper reports runtimes for.
    """

    def __init__(self, gallery: np.ndarray, chunk_size: int = 64) -> None:
        self.gallery = np.asarray(gallery, dtype=np.float64)
        if self.gallery.ndim != 2 or self.gallery.size == 0:
            raise ValueError("gallery must be a non-empty 2-D array")
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        self.chunk_size = chunk_size

    def __len__(self) -> int:
        return self.gallery.shape[0]

    def distances(self, probes: np.ndarray) -> Iterator[np.ndarray]:
        """Yield the distance rows of each probe chunk in order."""
        probes = np.atleast_2d(np.asarray(probes, dtype=np.float64))
        for start in range(0, probes.shape[0], self.chunk_size):
            yield manhattan_distances(probes[start : start + self.chunk_size], self.gallery)

    def rank(self, probes: np.ndarray, max_rank: Optional[int] = None) -> np.ndarray:
        """Gallery indices ordered by increasing distance.

        Returns
        -------
        numpy.ndarray
            ``(n_probes, max_rank)`` array of gallery indices; ``max_rank``
            defaults to the full gallery size.
        """
        limit = len(self) if max_rank is None else min(max_rank, len(self))
        ranked = []
        for block in self.distances(probes):
            order = np.argsort(block, axis=1, kind="stable")
            ranked.append(order[:, :limit])
        return np.vstack(ranked)

    def predict(self, probes: np.ndarray) -> np.ndarray:
        """Index of the closest gallery template for each probe."""
        return self.rank(probes, max_rank=1).ravel()
