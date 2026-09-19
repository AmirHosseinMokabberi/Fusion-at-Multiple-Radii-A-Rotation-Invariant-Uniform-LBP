# Archived experiment notebooks

These are the Google Colab notebooks that produced Tables 1–4 of the paper. They are kept as the
record of what was actually run; `mrflbp/` in the repository root is the consolidated,
documented version of the same pipeline.

## What was changed

Only packaging, never logic:

- **All cell outputs removed**, together with execution counts. No results are distributed here.
- **Colab execution metadata removed** (`executionInfo`, `outputId`, `mount_file_id`,
  `authorship_tag`), which carried account information and drive identifiers.
- **Renamed and regrouped** as
  `<dataset>/<method>/<dataset>_<method>[_regime]_S<strategy>_P<protocol>.ipynb`.

Cell sources are untouched, including the hard-coded `/content/drive/MyDrive/Datasets/...`
paths — change these to your own before running anything.

## What was left out

This directory holds 142 notebooks: 141 of the 174 archived experiment notebooks, plus the
McNemar's-test notebook. The 33 that were dropped do not correspond to anything reported in the
article:

- **AlexNet** experiments (24 notebooks) — no CNN was trained for the paper; the CNN entries in
  Table 5 are quoted from the literature.
- **Genetic-algorithm feature selection** (5 notebooks) — exploratory, not reported.
- **UTFVP augmentation variants 1–3** — superseded by variant 4, whose parameter ranges are the
  ones quoted in Section 3.
- **The first UTFVP ROI notebook** — superseded by the version that adds tilt correction and ROI
  histogram equalisation, kept here as `preprocessing/utfvp_roi_extraction.ipynb`.

## Layout

```
notebooks/
├── fv_usm/          pca, 2dpca, 2d2pca, lbp_riu2, lbp_riu2_pca, lbp_riu2_2dpca, lbp_riu2_2d2pca
├── mmcbnu_6000/     same seven method folders
├── utfvp/           same seven method folders
├── preprocessing/   UTFVP ROI extraction and augmentation
└── analysis/        McNemar's test
```

Naming:

- `S1` / `S2` — Strategy 1 (fingers fused into one composite template) / Strategy 2 (each finger
  independent).
- `P1` / `P2` / `P3` — 70/30, 50/50 and 90/10 train/test splits.
- `session-sensitive` / `session-independent` — FV-USM only. Notebooks whose original filename
  carried the `2_` prefix are the session-independent runs.
- `lbp_riu2` covers **all four descriptors**: the `LBP_CONFIGS` list near the top of each
  notebook selects the neighbourhood, written as `(radius, points)`. `[(1, 8)]`, `[(1, 16)]` and
  `[(2, 8)]` give the single-radius baselines `LBP^riu2_(8,1)`, `LBP^riu2_(16,1)` and
  `LBP^riu2_(8,2)`; `[(1, 8), (1, 16), (2, 8)]` gives the proposed fused operator. The inactive
  variants are present as commented-out lines.

`fv_usm/2dpca/fv_usm_2dpca_single_session_comparative.ipynb` is the single-session FV-USM run
behind the “Our study” rows of Table 5, where the comparison methods use one session only.

## Differences between the notebooks and `mrflbp/`

The notebooks grew one experiment at a time and are not internally uniform. `mrflbp/`
reproduces each family's behaviour rather than normalising it, so its defaults match what was
run. Where a setting varies, the library makes it explicit:

| Aspect | Notebooks | `mrflbp/` |
| --- | --- | --- |
| **Identity definition** | inlined per notebook as string splitting, and **not uniform across them** | reproduced per family; `--identity-convention paper` for the article's definitions. See [`docs/identity_conventions.md`](../docs/identity_conventions.md) |
| ROI size, FV-USM | 300 × 100 for the raw and LBP-only notebooks, 100 × 300 for LBP + reducer | same, via the per-family `image_size` in `configs/fv_usm.yaml` |
| ROI size, MMCBNU-6000 | 60 × 128 for the raw baselines, 60 × 120 for the LBP families | same, per family |
| CLAHE clip limit (UTFVP ROI) | 2.0 | 2.0 (`--clip-limit` to change it) |
| L2 normalisation | applied before reduction in some notebooks, after it in others | applied to the descriptor and again to the final feature vector |

Two apparent inconsistencies turn out **not** to matter, and the library follows
the notebooks rather than "fixing" them:

- **Test-set normalisation.** Some notebooks z-score the training images but
  feed the test images to the LBP stage as `uint8`. This changes nothing: the
  descriptor only compares a pixel with its neighbours, and the z-score is a
  strictly increasing affine map, so every comparison — and therefore every
  code — is identical either way.
- **2DPCA / (2D)²PCA projection.** The notebooks use the mean only to build the
  scatter matrices and project the template itself (`Y = UᵀAV`), which is the
  formulation of Yang et al. `mrflbp/reduction.py` does the same.

The raw-pixel baselines equalise with `skimage.exposure.equalize_hist`
(continuous) rather than OpenCV's 8-bit `equalizeHist`, matching what those
notebooks did, and only the UTFVP baselines denoise first (`raw_denoise` in the
configs).

[`docs/discrepancies.md`](../docs/discrepancies.md) lists every point where the
code and the article text disagree.
