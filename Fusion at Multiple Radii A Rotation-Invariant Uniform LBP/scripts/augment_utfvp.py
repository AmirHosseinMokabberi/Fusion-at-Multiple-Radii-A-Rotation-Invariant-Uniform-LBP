"""Expand the UTFVP ROIs with the stochastic augmentation pipeline of the paper.

Each ROI is copied unchanged and three randomised variants are written beside
it, so every finger ends up with 16 images (4 originals + 12 augmentations) and
the dataset grows from 1,440 to 5,760 images.

Section 3 of the paper describes four transformations, all applied per variant:

* luminance adjustment with brightness factor ``U(0.6, 1.5)``
* random rotation ``U(-10, 10)`` degrees
* affine translation with offsets ``U(-4, 4)`` pixels
* additive Gaussian noise with standard deviation ``U(10, 30)``

The run that produced the published augmented set also applied a light Gaussian
blur and a small square occlusion after those four, which the article does not
mention.  Both are on by default here, because they are what ran; pass
``--no-blur`` / ``--no-occlusion`` for only the four transformations named in
the text.

Usage
-----
    python scripts/augment_utfvp.py \
        --input  data/UTFVP_ROI \
        --output data/UTFVP_ROI_augmented \
        --seed 0
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Optional, Sequence

import cv2
import numpy as np

VARIANTS_PER_IMAGE = 3
BRIGHTNESS_RANGE = (0.6, 1.5)
ROTATION_RANGE = (-10.0, 10.0)
TRANSLATION_RANGE = (-4, 4)
NOISE_STD_RANGE = (10.0, 30.0)
BLUR_KERNELS = (3, 5)
OCCLUSION_SIZE = (10, 10)


def adjust_luminance(image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Scale brightness by a random factor, clipped to the 8-bit range."""
    factor = rng.uniform(*BRIGHTNESS_RANGE)
    return np.clip(image.astype(np.float32) * factor, 0, 255).astype(np.uint8)


def random_rotation(image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Rotate about the image centre, reflecting at the borders."""
    angle = rng.uniform(*ROTATION_RANGE)
    rows, cols = image.shape
    matrix = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
    return cv2.warpAffine(image, matrix, (cols, rows), borderMode=cv2.BORDER_REFLECT)


def random_translation(image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Shift by a random integer offset, reflecting at the borders."""
    low, high = TRANSLATION_RANGE
    dx = int(rng.integers(low, high + 1))
    dy = int(rng.integers(low, high + 1))
    rows, cols = image.shape
    matrix = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(image, matrix, (cols, rows), borderMode=cv2.BORDER_REFLECT)


def add_gaussian_noise(image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Add zero-mean Gaussian noise of random intensity."""
    std = rng.uniform(*NOISE_STD_RANGE)
    noise = rng.normal(0.0, std, image.shape)
    return np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def random_blur(image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Apply a small Gaussian blur."""
    kernel = int(rng.choice(BLUR_KERNELS))
    return cv2.GaussianBlur(image, (kernel, kernel), 0)


def add_occlusion(image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Black out a small square at a random position."""
    out = image.copy()
    height, width = out.shape
    box_h, box_w = OCCLUSION_SIZE
    if height <= box_h or width <= box_w:
        return out
    y = int(rng.integers(0, height - box_h))
    x = int(rng.integers(0, width - box_w))
    out[y : y + box_h, x : x + box_w] = 0
    return out


def augment(image: np.ndarray, rng: np.random.Generator, blur: bool, occlusion: bool) -> np.ndarray:
    """Apply the augmentation chain to one ROI."""
    out = adjust_luminance(image, rng)
    out = random_rotation(out, rng)
    out = random_translation(out, rng)
    out = add_gaussian_noise(out, rng)
    if blur:
        out = random_blur(out, rng)
    if occlusion:
        out = add_occlusion(out, rng)
    return out


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Augment UTFVP ROIs (3 variants per image).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", type=Path, required=True,
                        help="directory of extracted ROIs (one folder per subject)")
    parser.add_argument("--output", type=Path, required=True,
                        help="directory the originals and augmentations are written to")
    parser.add_argument("--variants", type=int, default=VARIANTS_PER_IMAGE,
                        help="augmented variants generated per original ROI")
    parser.add_argument("--seed", type=int, default=0,
                        help="seed of the random number generator (for reproducibility)")
    parser.add_argument("--no-blur", dest="blur", action="store_false",
                        help="skip the Gaussian blur stage")
    parser.add_argument("--no-occlusion", dest="occlusion", action="store_false",
                        help="skip the occlusion stage")
    parser.set_defaults(blur=True, occlusion=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.input.is_dir():
        raise SystemExit(f"input directory not found: {args.input}")

    rng = np.random.default_rng(args.seed)
    random.seed(args.seed)

    originals = augmented = 0
    for subject_dir in sorted(p for p in args.input.iterdir() if p.is_dir()):
        target_dir = args.output / subject_dir.name
        target_dir.mkdir(parents=True, exist_ok=True)

        for image_path in sorted(subject_dir.glob("*.png")):
            image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
            if image is None:
                print(f"skipped unreadable image: {image_path}")
                continue

            cv2.imwrite(str(target_dir / image_path.name), image)
            originals += 1

            for variant in range(1, args.variants + 1):
                variant_image = augment(image, rng, args.blur, args.occlusion)
                name = f"{image_path.stem}_{variant}_Augmented.png"
                cv2.imwrite(str(target_dir / name), variant_image)
                augmented += 1

    print(f"wrote {originals} originals and {augmented} augmentations to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
