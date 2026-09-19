# Where the code and the article text differ

The code in this repository is the reference: it is what produced the published
results, so every default here follows what was run rather than what the article
says. Seven settings differ between the two. None is hidden, and each is
reachable from the command line if you want the other value.

| # | Article text | What ran, and what this repo does | Where |
| --- | --- | --- | --- |
| 1 | CLAHE clip limit **0.75** | **2.0** | `scripts/prepare_utfvp_roi.py`, `--clip-limit` |
| 2 | "2DPCA and 2D²PCA use a fixed set of **137** components across descriptors" | 137 on FV-USM, **47** on MMCBNU-6000 and UTFVP | `configs/*.yaml`, `--components` |
| 3 | Augmentation = brightness, rotation, translation, Gaussian noise | those four **plus** a Gaussian blur and a small occlusion | `scripts/augment_utfvp.py`, `--no-blur` / `--no-occlusion` |
| 4 | "ROIs from original images are preserved for testing, while augmented images are used only for training" | P1 and P2 test sets also contain one augmented image per capture; P3 tests on originals only | `UTFVP.SPLITS` in `mrflbp/datasets.py` |
| 5 | Session-independent: "any same-subject pair is genuine" | Strategy 2 compared subject + finger | `--identity-convention` |
| 6 | Session-sensitive: genuine when "both subject and session match" | the reducer families also compared the finger; raw 2DPCA compared session + finger and never the subject | `--identity-convention` |
| 7 | MMCBNU-6000 identity (implied subject + finger) | subject + **hand**, because the finger directory names contain an underscore | `--identity-convention` |

Items 5–7 are set out case by case in
[`identity_conventions.md`](identity_conventions.md), including which of them
look deliberate and which look like defects in the label parsing.

Three further inconsistencies exist *within* the experiments rather than against
the article, and the repository reproduces each faithfully rather than
normalising it:

- **ROI orientation and size vary by feature family.** FV-USM used 300 × 100 for
  the raw-pixel baselines and for LBP without a reducer, but 100 × 300 for LBP
  with one; MMCBNU-6000 used 60 × 128 for the raw baselines and 60 × 120 for the
  LBP families. `image_size` in each config file is keyed by family for exactly
  this reason.
- **Histogram equalisation differs by chain.** The LBP chain uses OpenCV's 8-bit
  `equalizeHist`; the raw-pixel chain uses `skimage`'s continuous
  `equalize_hist`. Only the UTFVP raw baselines denoise first (`raw_denoise`).
- **L2 normalisation** was applied before reduction in some notebooks and after
  it in others; the library applies it to the descriptor and again to the final
  feature vector.

Two apparent inconsistencies turn out not to matter, and are noted here so they
are not mistaken for defects:

- Some notebooks z-score the training images but pass the test images to the LBP
  stage as `uint8`. The descriptor only compares a pixel with its neighbours and
  a z-score is strictly increasing, so the codes are identical either way.
- 2DPCA and (2D)²PCA use the mean only to build the scatter matrices and project
  the template itself (`Y = UᵀAV`). That is the Yang et al. formulation, and
  `mrflbp/reduction.py` does the same.
