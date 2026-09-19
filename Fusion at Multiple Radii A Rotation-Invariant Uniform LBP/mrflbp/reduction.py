"""Dimensionality reduction: PCA, 2DPCA and (2D)^2PCA.

All three reducers follow the same contract: ``fit`` learns the projection from
the training templates only, and ``transform`` maps any template (train or test)
into a flat feature vector that the nearest-neighbour matcher consumes.

* :class:`PCA` operates on flattened templates.  Following the paper it keeps
  every eigenvector with a non-negligible eigenvalue unless ``n_components`` is
  given.  The eigen-decomposition is done on the Gram matrix, which is the
  cheap direction when there are far fewer samples than features.
* :class:`TwoDimensionalPCA` keeps the template as a matrix and projects its
  columns with the leading eigenvectors of the image column-scatter matrix.
* :class:`TwoDirectionalTwoDimensionalPCA` projects rows *and* columns, giving
  the strongest compression of the three.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np

_EIGENVALUE_TOLERANCE = 1e-10


def _as_stack(templates: Sequence[np.ndarray]) -> List[np.ndarray]:
    stack = list(templates)
    if not stack:
        raise ValueError("no templates supplied")
    shapes = {t.shape for t in stack}
    if len(shapes) != 1:
        raise ValueError(f"templates must share one shape, got {sorted(shapes)}")
    return stack


class Reducer:
    """Common interface of the three reducers."""

    def fit(self, templates: Sequence[np.ndarray]) -> "Reducer":
        raise NotImplementedError

    def transform(self, templates: Sequence[np.ndarray]) -> np.ndarray:
        raise NotImplementedError

    def fit_transform(self, templates: Sequence[np.ndarray]) -> np.ndarray:
        return self.fit(templates).transform(templates)


class Identity(Reducer):
    """No reduction: templates are simply flattened."""

    name = "none"

    def fit(self, templates: Sequence[np.ndarray]) -> "Identity":
        _as_stack(templates)
        return self

    def transform(self, templates: Sequence[np.ndarray]) -> np.ndarray:
        return np.asarray([np.asarray(t, dtype=np.float64).ravel() for t in templates])


class PCA(Reducer):
    """Classical (1-D) PCA on flattened templates.

    Parameters
    ----------
    n_components:
        Number of principal components to keep.  ``None`` -- the setting used in
        the paper -- keeps every component whose eigenvalue exceeds
        ``1e-10``.
    """

    name = "pca"

    def __init__(self, n_components: Optional[int] = None) -> None:
        self.n_components = n_components
        self.mean_: Optional[np.ndarray] = None
        self.components_: Optional[np.ndarray] = None

    def fit(self, templates: Sequence[np.ndarray]) -> "PCA":
        data = np.asarray([np.asarray(t, dtype=np.float64).ravel() for t in _as_stack(templates)])
        self.mean_ = data.mean(axis=0)
        centered = data - self.mean_

        gram = centered @ centered.T
        eigenvalues, eigenvectors = np.linalg.eigh(gram)
        order = np.argsort(-eigenvalues)
        eigenvalues = eigenvalues[order]
        eigenvectors = eigenvectors[:, order]

        keep = eigenvalues > _EIGENVALUE_TOLERANCE
        eigenvalues = eigenvalues[keep]
        eigenvectors = eigenvectors[:, keep]
        if eigenvalues.size == 0:
            raise ValueError("training templates carry no variance")

        # Map the Gram-matrix eigenvectors back to feature space and normalise.
        components = (centered.T @ eigenvectors) / np.sqrt(eigenvalues)
        if self.n_components is not None:
            components = components[:, : self.n_components]
        self.components_ = components
        return self

    def transform(self, templates: Sequence[np.ndarray]) -> np.ndarray:
        if self.components_ is None or self.mean_ is None:
            raise RuntimeError("PCA.transform called before fit")
        data = np.asarray([np.asarray(t, dtype=np.float64).ravel() for t in _as_stack(templates)])
        return (data - self.mean_) @ self.components_


class TwoDimensionalPCA(Reducer):
    """2DPCA: column projection learned from the image column-scatter matrix."""

    name = "2dpca"

    def __init__(self, n_components: int) -> None:
        if n_components <= 0:
            raise ValueError("n_components must be positive")
        self.n_components = n_components
        self.mean_: Optional[np.ndarray] = None
        self.projection_: Optional[np.ndarray] = None

    def fit(self, templates: Sequence[np.ndarray]) -> "TwoDimensionalPCA":
        stack = _as_stack(templates)
        self.mean_ = sum(np.asarray(t, dtype=np.float64) for t in stack) / len(stack)
        width = self.mean_.shape[1]

        scatter = np.zeros((width, width), dtype=np.float64)
        for template in stack:
            centered = np.asarray(template, dtype=np.float64) - self.mean_
            scatter += centered.T @ centered
        scatter /= len(stack)

        eigenvalues, eigenvectors = np.linalg.eigh(scatter)
        order = np.argsort(-eigenvalues)[: self.n_components]
        self.projection_ = eigenvectors[:, order]
        return self

    def transform(self, templates: Sequence[np.ndarray]) -> np.ndarray:
        if self.projection_ is None or self.mean_ is None:
            raise RuntimeError("TwoDimensionalPCA.transform called before fit")
        # The mean enters the scatter matrix only; following Yang et al. the
        # projection is applied to the template itself, not to a centred copy.
        return np.asarray(
            [
                (np.asarray(t, dtype=np.float64) @ self.projection_).ravel()
                for t in _as_stack(templates)
            ]
        )


class TwoDirectionalTwoDimensionalPCA(Reducer):
    """(2D)^2PCA: simultaneous row and column projection, ``Y = U^T A V``."""

    name = "2d2pca"

    def __init__(self, n_row_components: int, n_col_components: int) -> None:
        if n_row_components <= 0 or n_col_components <= 0:
            raise ValueError("component counts must be positive")
        self.n_row_components = n_row_components
        self.n_col_components = n_col_components
        self.mean_: Optional[np.ndarray] = None
        self.row_projection_: Optional[np.ndarray] = None
        self.col_projection_: Optional[np.ndarray] = None

    def fit(self, templates: Sequence[np.ndarray]) -> "TwoDirectionalTwoDimensionalPCA":
        stack = _as_stack(templates)
        self.mean_ = sum(np.asarray(t, dtype=np.float64) for t in stack) / len(stack)
        height, width = self.mean_.shape

        row_scatter = np.zeros((height, height), dtype=np.float64)
        col_scatter = np.zeros((width, width), dtype=np.float64)
        for template in stack:
            centered = np.asarray(template, dtype=np.float64) - self.mean_
            row_scatter += centered @ centered.T
            col_scatter += centered.T @ centered
        row_scatter /= len(stack)
        col_scatter /= len(stack)

        row_values, row_vectors = np.linalg.eigh(row_scatter)
        col_values, col_vectors = np.linalg.eigh(col_scatter)
        # A request for more components than the matrix has is silently clipped,
        # which is what happens for the small patch grids of MMCBNU-6000/UTFVP.
        self.row_projection_ = row_vectors[:, np.argsort(-row_values)[: self.n_row_components]]
        self.col_projection_ = col_vectors[:, np.argsort(-col_values)[: self.n_col_components]]
        return self

    def transform(self, templates: Sequence[np.ndarray]) -> np.ndarray:
        if self.row_projection_ is None or self.col_projection_ is None or self.mean_ is None:
            raise RuntimeError("TwoDirectionalTwoDimensionalPCA.transform called before fit")
        # As for 2DPCA, the mean is used to build the scatter matrices only.
        out = []
        for template in _as_stack(templates):
            matrix = np.asarray(template, dtype=np.float64)
            out.append((self.row_projection_.T @ matrix @ self.col_projection_).ravel())
        return np.asarray(out)


def build_reducer(name: str, n_components: Optional[int] = None) -> Reducer:
    """Instantiate a reducer by name (``none``, ``pca``, ``2dpca`` or ``2d2pca``)."""
    key = name.strip().lower().replace("(", "").replace(")", "").replace("^", "")
    aliases = {
        "none": "none",
        "raw": "none",
        "pca": "pca",
        "2dpca": "2dpca",
        "2d2pca": "2d2pca",
        "2d22pca": "2d2pca",
    }
    if key not in aliases:
        raise ValueError(
            f"unknown reducer {name!r}; expected one of none, pca, 2dpca, 2d2pca"
        )
    kind = aliases[key]
    if kind == "none":
        return Identity()
    if kind == "pca":
        return PCA(n_components)
    if n_components is None:
        raise ValueError(f"{kind} requires an explicit number of components")
    if kind == "2dpca":
        return TwoDimensionalPCA(n_components)
    return TwoDirectionalTwoDimensionalPCA(n_components, n_components)
