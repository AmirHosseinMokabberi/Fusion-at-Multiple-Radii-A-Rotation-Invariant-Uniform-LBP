# Cross-check against the article

Every value the article states explicitly, checked against the code in this
repository. `tests/test_paper_alignment.py` asserts all of them, so the table
below cannot silently go stale.

**24 of the 27 checkable claims match.** The three that differ are the ones
listed in [`discrepancies.md`](discrepancies.md); the code keeps what the
experiments used.

## Section 2 — the proposed descriptor

| Claim | Article | Code |
| --- | --- | --- |
| Fused configurations `(P, R)` | (8,1), (16,1), (8,2) | (8,1), (16,1), (8,2) | 
| Per-configuration bins `P + 2` | 10, 18, 10 | 10, 18, 10 |
| Fused patch descriptor `F_patch` | ℝ³⁸ | 38 |
| Patch size, zero overlap | 10 × 10, stride 10 | 10 × 10, stride 10 |

## Section 3 — datasets

| Claim | Article | Code |
| --- | --- | --- |
| FV-USM images (123 × 4 × 6 × 2) | 5,904 | 5,904 |
| MMCBNU-6000 images (100 × 6 × 10) | 6,000 | 6,000 |
| UTFVP images (60 × 6 × 4) | 1,440 | 1,440 |
| UTFVP after augmentation | 5,760 | 5,760 |
| UTFVP images per finger | 16 | 16 |

## Section 3 — UTFVP ROI pipeline

| Claim | Article | Code |
| --- | --- | --- |
| CLAHE tile grid | 8 × 8 px | 8 × 8 |
| CLAHE clip limit | 0.75 | **2.0** |
| Abnormal-placement variance threshold | 38 | 38 |
| DBSCAN radius / minimum samples | 10 px / 10 | 10 / 10 |
| Tilt-correction rotation threshold | 15° | 15 |
| Extracted ROI size | 60 × 128 | 60 × 128 |

## Section 3 — augmentation

| Claim | Article | Code |
| --- | --- | --- |
| Variants per original image | 3 | 3 |
| Brightness factor | U(0.6, 1.5) | (0.6, 1.5) |
| Rotation | U(−10, 10)° | (−10, 10) |
| Affine translation | U(−4, 4) px | (−4, 4) |
| Gaussian noise intensity | U(10, 30) | (10, 30) |
| Any further stages | none stated | **blur and occlusion also applied** |

## Section 3.1 — protocols and reduction

| Claim | Article | Code |
| --- | --- | --- |
| Protocol splits | 70/30, 50/50, 90/10 | 0.7, 0.5, 0.9 |
| Rounding when not integral | round training set down | `floor`, never below 1 |
| PCA components | all non-zero eigenvectors | `None` (all above 1e-10) |
| 2DPCA / (2D)²PCA components | 137 across descriptors | 137 FV-USM, **47** MMCBNU-6000 and UTFVP |

## Not stated in the article

The article does not specify the classifier or the distance metric. Every
experiment enrols the training templates in a gallery and assigns a probe the
identity of its nearest neighbour under the **Manhattan (L1)** distance, which
is what all 142 notebooks do; `mrflbp/matching.py` implements that.

Nor does it state which sample attributes constitute an identity beyond the
regime definitions in Section 3.1. That turns out to matter, and is treated
separately in [`identity_conventions.md`](identity_conventions.md).
