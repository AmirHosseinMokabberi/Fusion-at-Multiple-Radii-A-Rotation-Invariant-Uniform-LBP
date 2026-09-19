"""ROI loading and photometric normalisation.

The experiments use two normalisation chains:

``lbp``
    resize -> non-local-means denoising (``h = 10``) -> histogram equalisation
    -> scaling to ``[0, 1]`` -> per-image z-score.  This is the input of every
    ``LBP^{riu2}`` experiment.

``raw``
    resize -> (UTFVP only) non-local-means denoising -> exact histogram
    equalisation -> per-image z-score.  This is the input of the raw-pixel PCA /
    2DPCA / (2D)^2PCA baselines.

The two chains equalise differently on purpose: the LBP chain uses OpenCV's
8-bit ``equalizeHist`` while the raw chain uses ``skimage``'s continuous
``equalize_hist``, matching the archived notebooks in each case.  For the LBP
chain the choice barely matters -- the descriptor only compares a pixel with
its neighbours, so any strictly increasing intensity mapping leaves the codes
untouched -- but the raw baselines feed intensities straight into PCA, where it
does.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
from skimage import exposure

_EPS = 1e-8


def load_grayscale(path: Path | str) -> np.ndarray:
    """Read an image as 8-bit grayscale, raising if the file cannot be decoded."""
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"could not read image: {path}")
    return image


def resize(image: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    """Resize to ``size`` given as ``(height, width)``.

    ``cv2.resize`` expects ``(width, height)``; this wrapper exists so that every
    size in the code base and in the configuration files is written in the same
    ``(height, width)`` order as ``numpy`` shapes.
    """
    height, width = size
    return cv2.resize(image, (width, height))


def zscore(image: np.ndarray) -> np.ndarray:
    """Per-image zero-mean / unit-variance normalisation."""
    image = image.astype(np.float64)
    return (image - image.mean()) / (image.std() + _EPS)


def preprocess_lbp(
    image: np.ndarray,
    size: Tuple[int, int],
    denoise_h: float = 10.0,
) -> np.ndarray:
    """Normalisation chain used for every ``LBP^{riu2}`` experiment.

    The result is a float image, so ``skimage`` emits a warning about applying
    ``local_binary_pattern`` to floating-point data.  It is expected: the LBP
    code only compares neighbours to the centre pixel, and exact ties are
    vanishingly unlikely after the z-score, so the codes are well defined.
    """
    image = resize(image, size)
    image = cv2.fastNlMeansDenoising(image, h=denoise_h)
    image = cv2.equalizeHist(image)
    return zscore(image.astype(np.float64) / 255.0)


def preprocess_raw(
    image: np.ndarray,
    size: Tuple[int, int],
    denoise_h: float | None = None,
) -> np.ndarray:
    """Normalisation chain used for the raw-pixel PCA / 2DPCA / (2D)^2PCA baselines.

    ``denoise_h`` applies non-local-means denoising first.  Only the UTFVP
    baselines did so; the FV-USM and MMCBNU-6000 ones ran without it.  The
    ``raw_denoise`` key in each dataset config selects the right behaviour.
    """
    image = resize(image, size)
    if denoise_h is not None:
        image = cv2.fastNlMeansDenoising(image, h=denoise_h)
    return zscore(exposure.equalize_hist(image))
