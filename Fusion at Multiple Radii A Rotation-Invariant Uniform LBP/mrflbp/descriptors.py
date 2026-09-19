"""Rotation-invariant uniform LBP and the proposed multi-radius fused operator.

The paper's operator is

    LBP^{riu2}_{(8,1),(16,1),(8,2)}

i.e. three ``(P, R)`` neighbourhood configurations evaluated on the *same*
non-overlapping patch grid and concatenated into a single patch descriptor
(Eq. 5 of the paper)::

    F_patch = h_(8,1) || h_(16,1) || h_(8,2)  in R^38

with ``dim(h_(P,R)) = P + 2`` (``P + 1`` uniform bins plus one bin collecting
every non-uniform pattern).  The image-level descriptor concatenates the patch
descriptors over the whole ROI (Eq. 6).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence, Tuple

import numpy as np
from skimage.feature import local_binary_pattern

#: ``(P, R)`` pairs of the proposed fused operator, in the order used in the paper.
FUSED_CONFIGS: Tuple[Tuple[int, int], ...] = ((8, 1), (16, 1), (8, 2))

#: Single-radius baselines reported alongside the fused operator.
SINGLE_RADIUS_CONFIGS = {
    "lbp_8_1": ((8, 1),),
    "lbp_16_1": ((16, 1),),
    "lbp_8_2": ((8, 2),),
}


def riu2_mapping(P: int) -> np.ndarray:
    """Build the ``riu2`` look-up table for a ``P``-point neighbourhood.

    Every one of the ``2**P`` raw patterns is reduced to its canonical
    (lexicographically smallest) rotation.  Canonical patterns with at most two
    0/1 transitions are *uniform* and are mapped to their number of set bits
    (``0 .. P``); every other pattern is mapped to the single "non-uniform"
    bin ``P + 1``.  The table therefore has ``P + 2`` distinct output values.

    Parameters
    ----------
    P:
        Number of sampling points on the circle.

    Returns
    -------
    numpy.ndarray
        Look-up table of shape ``(2**P,)`` and dtype ``uint8``.
    """
    table = np.zeros(2**P, dtype=np.uint8)
    for value in range(2**P):
        bits = [(value >> shift) & 1 for shift in range(P)]
        canonical = min(bits[n:] + bits[:n] for n in range(P))
        closed = canonical + [canonical[0]]
        transitions = sum(closed[j] != closed[j + 1] for j in range(P))
        table[value] = sum(canonical) if transitions <= 2 else P + 1
    return table


@dataclass(frozen=True)
class MultiRadiusLBP:
    """Multi-radius fused ``LBP^{riu2}`` descriptor.

    Parameters
    ----------
    configs:
        ``(P, R)`` neighbourhood configurations to fuse.  Defaults to the
        proposed ``((8, 1), (16, 1), (8, 2))``; pass a single pair to obtain one
        of the single-radius baselines.
    patch_size:
        Side of the square patch the ROI is tiled with (10 px in the paper).
    stride:
        Step between consecutive patches.  Equal to ``patch_size`` gives the
        zero-overlap tiling used in the paper.
    """

    configs: Sequence[Tuple[int, int]] = FUSED_CONFIGS
    patch_size: int = 10
    stride: int = 10

    def __post_init__(self) -> None:
        if not self.configs:
            raise ValueError("at least one (P, R) configuration is required")
        if self.patch_size <= 0 or self.stride <= 0:
            raise ValueError("patch_size and stride must be positive")
        # ``object.__setattr__`` because the dataclass is frozen.
        object.__setattr__(self, "_tables", {P: riu2_mapping(P) for P, _ in self.configs})

    @property
    def patch_dimension(self) -> int:
        """Length of one fused patch descriptor (38 for the proposed operator)."""
        return sum(P + 2 for P, _ in self.configs)

    def grid_shape(self, image_shape: Tuple[int, int]) -> Tuple[int, int]:
        """Number of patch rows and columns for an image of ``image_shape``."""
        height, width = image_shape
        rows = (height - self.patch_size) // self.stride + 1
        cols = (width - self.patch_size) // self.stride + 1
        if rows <= 0 or cols <= 0:
            raise ValueError(
                f"image of shape {image_shape} is smaller than the "
                f"{self.patch_size}x{self.patch_size} patch"
            )
        return rows, cols

    def patch_histogram(self, patch: np.ndarray) -> np.ndarray:
        """Fused ``riu2`` histogram of a single patch."""
        parts = []
        for P, R in self.configs:
            codes = local_binary_pattern(patch, P, R, method="ror").astype(np.uint32)
            mapped = self._tables[P][codes]
            hist, _ = np.histogram(
                mapped.ravel(), bins=np.arange(0, P + 3), density=True
            )
            parts.append(hist)
        return np.concatenate(parts)

    def matrix(self, image: np.ndarray) -> np.ndarray:
        """Patch-grid descriptor of an ROI.

        Returns
        -------
        numpy.ndarray
            Array of shape ``(patch_rows, patch_cols * patch_dimension)``.  Each
            row holds the fused histograms of one row of patches, which keeps the
            vertical layout of the ROI intact -- this is the matrix consumed by
            2DPCA and (2D)^2PCA.
        """
        rows, cols = self.grid_shape(image.shape)
        out = np.zeros((rows, cols * self.patch_dimension), dtype=np.float64)
        for i in range(rows):
            y = i * self.stride
            row = []
            for j in range(cols):
                x = j * self.stride
                patch = image[y : y + self.patch_size, x : x + self.patch_size]
                row.append(self.patch_histogram(patch))
            out[i, :] = np.concatenate(row)
        return out

    def vector(self, image: np.ndarray) -> np.ndarray:
        """Flattened image-level descriptor (Eq. 6), for 1-D PCA and raw matching."""
        return self.matrix(image).ravel()


def fuse_matrices(matrices: Iterable[np.ndarray]) -> np.ndarray:
    """Horizontally concatenate per-finger descriptor matrices (Strategy 1).

    Strategy 1 builds one composite template per subject by joining the
    descriptor matrices of all of that subject's fingers side by side, which
    preserves the patch-row structure needed by the 2-D reducers.
    """
    stack = list(matrices)
    if not stack:
        raise ValueError("no matrices to fuse")
    rows = {m.shape[0] for m in stack}
    if len(rows) != 1:
        raise ValueError(f"cannot fuse matrices with differing row counts: {sorted(rows)}")
    return np.hstack(stack)
