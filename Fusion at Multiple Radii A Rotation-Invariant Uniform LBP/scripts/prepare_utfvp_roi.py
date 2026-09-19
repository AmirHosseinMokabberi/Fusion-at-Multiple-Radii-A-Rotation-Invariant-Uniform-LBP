"""Extract finger-vein ROIs from the raw UTFVP images.

UTFVP is the only dataset of the three that ships without pre-extracted ROIs.
The six-stage pipeline below is the one described in Section 3 of the paper,
adapted from Lu et al., "Robust finger vein ROI localization based on flexible
segmentation" (Sensors, 2013):

1. CLAHE enhancement with 8x8 tiles and clip limit 2.0.
2. Prewitt-based edge detection of the upper and lower finger boundaries.
3. Abnormal-placement check on the boundary variance, with a small padding retry.
4. DBSCAN cleaning of the finger's middle line.
5. Tilt correction when the estimated orientation exceeds the rotation threshold.
6. ROI extraction centred on the brightest column of the central half of the
   image, followed by histogram equalisation.

Usage
-----
    python scripts/prepare_utfvp_roi.py \
        --input  /path/to/UTFVP/dataset/data \
        --output data/UTFVP_ROI

The input directory is expected to hold one folder per subject, each containing
PNG captures named ``<subject>_<finger>_<capture>_<timestamp>.png``; the output
keeps the same folder structure with the timestamp dropped.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Sequence, Tuple

import cv2
import numpy as np
from scipy.ndimage import rotate as ndimage_rotate
from skimage import exposure
from sklearn.cluster import DBSCAN

# --- the settings the published ROIs were produced with ----------------------
RESIZED_SHAPE = (128, 256)   # (height, width) the capture is resized to
ROI_SIZE = (60, 128)         # (height, width) of the extracted ROI
CLAHE_TILE = (8, 8)
CLAHE_CLIP_LIMIT = 2.0       # value used for the published ROIs
#                              (Section 3 of the article states 0.75)
PADDING_ROWS = 3
VARIANCE_THRESHOLD = 38.0
DBSCAN_EPS = 10.0
DBSCAN_MIN_SAMPLES = 10
ROTATION_THRESHOLD = 15.0


def apply_clahe(image: np.ndarray, clip_limit: float, tile: Tuple[int, int]) -> np.ndarray:
    """Stage 1: contrast-limited adaptive histogram equalisation."""
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile)
    return clahe.apply(image)


def _prewitt_masks() -> Tuple[np.ndarray, np.ndarray]:
    upper = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], dtype=np.float32)
    return upper, -upper


def _edge_response(image: np.ndarray) -> np.ndarray:
    """Stage 2: Prewitt response, oriented outwards in each half of the image."""
    upper_mask, lower_mask = _prewitt_masks()
    middle = image.shape[0] // 2
    upper = cv2.filter2D(image[:middle, :], -1, upper_mask)
    lower = cv2.filter2D(image[middle:, :], -1, lower_mask)
    stacked = np.vstack([upper, lower])
    return cv2.normalize(stacked, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def find_edges(response: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Per-column indices of the strongest upper and lower boundary responses."""
    top = np.argmax(response, axis=0)
    bottom = response.shape[0] - 1 - np.argmax(response[::-1], axis=0)
    return top, bottom


def coarse_binarization(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fill the region between the detected boundaries of every column."""
    response = _edge_response(image)
    top, bottom = find_edges(response)
    mask = np.zeros_like(image, dtype=np.uint8)
    for column in range(image.shape[1]):
        if top[column] < bottom[column]:
            mask[top[column] : bottom[column], column] = 255
    return mask, top, bottom


def abnormal_case(top: np.ndarray, bottom: np.ndarray, threshold: float) -> Optional[str]:
    """Stage 3: flag a boundary that is unstable across both halves of the image."""
    half = len(top) // 2
    if np.var(top[:half]) > threshold and np.var(top[half:]) > threshold:
        return "upper"
    if np.var(bottom[:half]) > threshold and np.var(bottom[half:]) > threshold:
        return "lower"
    return None


def elaborate_binarization(
    image: np.ndarray, threshold: float, padding: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Re-run the boundary search on a padded image when a boundary looks unstable."""
    mask, top, bottom = coarse_binarization(image)
    case = abnormal_case(top, bottom, threshold)
    if case is None:
        return top, bottom, image

    pad = ((padding, 0), (0, 0)) if case == "upper" else ((0, padding), (0, 0))
    padded = np.pad(image, pad, mode="constant")
    _, top, bottom = coarse_binarization(padded)
    return top, bottom, padded


def clean_middle_line(
    middle_line: np.ndarray, eps: float, min_samples: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Stage 4: drop DBSCAN outliers from the finger's middle line."""
    x = np.arange(len(middle_line))
    points = np.column_stack([x, middle_line])
    labels = DBSCAN(eps=eps, min_samples=min_samples).fit(points).labels_
    keep = labels != -1
    return x[keep], middle_line[keep]


def orientation_angle(x: np.ndarray, y: np.ndarray) -> float:
    """Least-squares tilt of the cleaned middle line, in degrees."""
    if len(x) < 2:
        return 0.0
    x_mean, y_mean = np.mean(x), np.mean(y)
    denominator = np.sum((x - x_mean) ** 2)
    if denominator == 0:
        return 0.0
    slope = np.sum((x - x_mean) * (y - y_mean)) / denominator
    return float(np.degrees(np.arctan(slope)))


def rotate_image(image: np.ndarray, angle: float) -> np.ndarray:
    """Stage 5: de-rotate the finger without changing the image size."""
    return ndimage_rotate(image, -angle, reshape=False, order=3, mode="constant", cval=0)


def extract_roi(
    image: np.ndarray, top: np.ndarray, bottom: np.ndarray, roi_size: Tuple[int, int]
) -> np.ndarray:
    """Stage 6: crop around the finger centre line and the brightest column."""
    height, width = image.shape
    roi_height, roi_width = roi_size

    y_center = int(np.median((top + bottom) // 2))
    y1 = max(0, y_center - roi_height // 2)
    y2 = y1 + roi_height

    start, end = width // 4, 3 * width // 4
    column = start + int(np.argmax(np.sum(image[:, start:end], axis=0)))
    x1 = max(0, column - roi_width // 2)
    x2 = x1 + roi_width

    crop = image[y1 : min(y2, height), x1 : min(x2, width)]
    resized = cv2.resize(crop, (roi_width, roi_height), interpolation=cv2.INTER_CUBIC)
    return (exposure.equalize_hist(resized) * 255).astype(np.uint8)


def process_image(path: Path, args: argparse.Namespace) -> np.ndarray:
    """Run the full six-stage pipeline on one capture."""
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"could not read {path}")

    enhanced = apply_clahe(image, args.clip_limit, tuple(args.clahe_tile))
    resized = cv2.resize(
        enhanced, (args.resize[1], args.resize[0]), interpolation=cv2.INTER_CUBIC
    )
    top, bottom, working = elaborate_binarization(
        resized, args.variance_threshold, args.padding_rows
    )

    x, y = clean_middle_line((top + bottom) // 2, args.dbscan_eps, args.dbscan_min_samples)
    angle = orientation_angle(x, y)
    if abs(angle) > args.rotation_threshold:
        working = rotate_image(working, angle)

    return extract_roi(working, top, bottom, tuple(args.roi_size))


def output_name(filename: str) -> str:
    """``0001_1_1_120509-135315.png`` -> ``0001_1_1.png``."""
    return "_".join(Path(filename).stem.split("_")[:3]) + ".png"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract UTFVP finger-vein ROIs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", type=Path, required=True,
                        help="UTFVP dataset/data directory (one folder per subject)")
    parser.add_argument("--output", type=Path, required=True,
                        help="directory the extracted ROIs are written to")
    parser.add_argument("--clip-limit", type=float, default=CLAHE_CLIP_LIMIT,
                        help="CLAHE clip limit (2.0 produced the published ROIs)")
    parser.add_argument("--clahe-tile", type=int, nargs=2, default=list(CLAHE_TILE),
                        metavar=("ROWS", "COLS"), help="CLAHE tile grid")
    parser.add_argument("--resize", type=int, nargs=2, default=list(RESIZED_SHAPE),
                        metavar=("H", "W"), help="size the capture is resized to")
    parser.add_argument("--roi-size", type=int, nargs=2, default=list(ROI_SIZE),
                        metavar=("H", "W"), help="size of the extracted ROI")
    parser.add_argument("--variance-threshold", type=float, default=VARIANCE_THRESHOLD,
                        help="boundary-variance threshold of the abnormal-placement check")
    parser.add_argument("--padding-rows", type=int, default=PADDING_ROWS,
                        help="rows added before retrying an abnormal placement")
    parser.add_argument("--dbscan-eps", type=float, default=DBSCAN_EPS,
                        help="DBSCAN radius in pixels")
    parser.add_argument("--dbscan-min-samples", type=int, default=DBSCAN_MIN_SAMPLES,
                        help="DBSCAN minimum samples")
    parser.add_argument("--rotation-threshold", type=float, default=ROTATION_THRESHOLD,
                        help="tilt in degrees above which the finger is de-rotated")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.input.is_dir():
        raise SystemExit(f"input directory not found: {args.input}")

    written = failed = 0
    for subject_dir in sorted(p for p in args.input.iterdir() if p.is_dir()):
        target_dir = args.output / subject_dir.name
        target_dir.mkdir(parents=True, exist_ok=True)
        for image_path in sorted(subject_dir.glob("*.png")):
            try:
                roi = process_image(image_path, args)
            except Exception as error:  # noqa: BLE001 - report and continue
                print(f"failed: {image_path.name}: {error}", file=sys.stderr)
                failed += 1
                continue
            cv2.imwrite(str(target_dir / output_name(image_path.name)), roi)
            written += 1

    print(f"wrote {written} ROIs to {args.output} ({failed} failed)")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
