# Fusion at Multiple Radii — a rotation-invariant uniform LBP for finger-vein identification

Reference implementation and archived experiment notebooks for:

> A. Mokabberi and Ö. Toygar, **“Fusion at multiple radii: a rotation-invariant uniform LBP for
> finger-vein identification”**, *Signal, Image and Video Processing*, 19:1419, 2025.
> DOI: [10.1007/s11760-025-05009-3](https://doi.org/10.1007/s11760-025-05009-3)

The paper introduces a **Multi-Radius Fused** rotation-invariant uniform Local Binary Pattern
operator, `LBP^riu2_(8,1),(16,1),(8,2)`, designed for finger-vein identification on
resource-constrained hardware, and evaluates it — on its own and combined with PCA, 2DPCA and
(2D)²PCA — on **FV-USM**, **MMCBNU-6000** and **UTFVP** under two fusion strategies and three
train/test protocols.

This repository contains the code, the first page of the article, and the MSc thesis that was
built on the same code (see [The thesis](#the-thesis)). No datasets, no trained models and no
experiment outputs are distributed here; see [Data availability](#data-availability).

**The code is the reference.** Where the article text and the experiments disagree on a
setting, the defaults here follow what was run, and the difference is documented rather than
quietly resolved. Those cases are listed in
[`docs/discrepancies.md`](docs/discrepancies.md).

---

## Contents

| Path | What it holds |
| --- | --- |
| `mrflbp/` | Reference implementation: descriptor, reducers, protocols, matching, metrics |
| `configs/` | Per-dataset hyper-parameters (paths, ROI size, component counts) |
| `scripts/` | Command-line entry points: experiments, UTFVP ROI extraction and augmentation, McNemar's test, CMC plots |
| `notebooks/` | The original Colab notebooks that produced Tables 1–4, cleaned and reorganised |
| `tests/` | Unit tests that need no dataset |
| `docs/experiment_grid.md` | The exact command behind every row of Tables 1–4 |
| `docs/paper_crosscheck.md` | Every value the article states, checked against the code |
| `docs/identity_conventions.md` | What counts as "the same identity", per grid cell |
| `docs/discrepancies.md` | The seven places the code and the article text differ |
| `Fusion_at_multiple_radii_a_rotation_inva.pdf` | First page of the published article |
| `Eastern_Mediterranean_University__EMU__Thesis___By_AmirHossein_Mokabberi.pdf` | The MSc thesis built on this code |
| `Eastern_Mediterranean_University__EMU__Thesis___By_AmirHossein_Mokabberi.zip` | LaTeX source of the thesis |

```
mrflbp/
├── descriptors.py    riu2 mapping, multi-radius fused LBP, Strategy 1 fusion
├── preprocessing.py  ROI loading and the two photometric normalisation chains
├── datasets.py       FV-USM / MMCBNU-6000 / UTFVP readers and their splits
├── protocols.py      strategies, protocols, identity definitions
├── reduction.py      PCA, 2DPCA, (2D)²PCA
├── matching.py       Manhattan nearest-neighbour gallery
├── evaluation.py     rank-1, CMC, McNemar's test
├── pipeline.py       end-to-end experiment runner
├── config.py         experiment configuration and YAML merging
└── cli.py            command-line interface
```

---

## The method in brief

**Descriptor.** For every non-overlapping 10 × 10 patch of the ROI, three `riu2` histograms are
computed at `(P, R) = (8, 1)`, `(16, 1)` and `(8, 2)` and concatenated (Eq. 5 of the paper):

```
F_patch = h_(8,1) ‖ h_(16,1) ‖ h_(8,2)  ∈ R^38
```

with `dim(h_(P,R)) = P + 2` — one bin per uniform pattern plus a single bin collecting every
non-uniform pattern. Patch descriptors are then concatenated over the whole ROI (Eq. 6). The
implementation keeps the patch grid as a matrix (`patch_rows × patch_cols · 38`) so that 2DPCA
and (2D)²PCA can exploit its spatial structure; 1-D PCA and raw matching simply flatten it.

**Strategies.** *Strategy 1* joins all fingers of a subject captured at the same acquisition
index into one composite template. *Strategy 2* enrols and probes every finger sample on its own.

**Protocols.** *P1* 70 % train / 30 % test, *P2* 50/50, *P3* 90/10. When a percentage does not
divide evenly, the training set is rounded down so at least one sample is always available for
testing.

**Evaluation regimes.** FV-USM has two sessions and is evaluated both *session-sensitive* and
*session-independent*; MMCBNU-6000 and UTFVP are treated as single-session collections. What
each regime counts as a genuine pair is set out in the note below.

**Matching.** Nearest neighbour under the Manhattan (L1) distance, scored with rank-1 accuracy
and CMC curves. Pairwise comparisons use McNemar's test with Yates's continuity correction at
α = 0.05.

> **Identity conventions.** A score depends on what counts as the same identity, and the
> experiments were not uniform about it — most notably on MMCBNU-6000, where Strategy 2
> compares subject + *hand* rather than subject + finger. This repository reproduces the
> experiments by default; `--identity-convention paper` switches to the definitions in
> Section 3.1 of the article.
> [`docs/identity_conventions.md`](docs/identity_conventions.md) documents every case.

---

## Installation

Clone or download this repository, then from its root folder:

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Or install the package itself, which also provides an `mrflbp` command:

```bash
pip install -e .
```

Python 3.9 or newer. Everything runs on CPU; the paper reports the fused descriptor training in
about 15 minutes with a minimum of 2.5 GB RAM on a Google Colab CPU-only runtime.

The unit tests need no dataset. They cover the descriptor, the protocol splits, the identity
conventions, the reducers and McNemar's test, and assert every value the article states
explicitly against the code:

```bash
pip install -e ".[dev]"
pytest tests -q
```

---

## Datasets

The three datasets are third-party and are **not** redistributed here. Request them from their
owners and point `configs/<dataset>.yaml` (or `--data-root`) at your local copy.

| Dataset | Content | ROIs |
| --- | --- | --- |
| [FV-USM](http://drfendi.com/fv_usm_database/) | 123 subjects × 4 fingers × 6 captures × 2 sessions = 5,904 images | provided, 300 × 100 px |
| [MMCBNU-6000](http://multilab.jbnu.ac.kr/) | 100 subjects × 6 fingers × 10 captures = 6,000 images | provided, 128 × 60 px |
| [UTFVP](https://www.utwente.nl/en/eemcs/dmb/downloads/utfvp/) | 60 subjects × 6 fingers × 4 captures = 1,440 images | **not** provided — extract them first |

Expected layouts:

```
FV-USM        <root>/1st_session/extractedvein/vein001_1/01.jpg
MMCBNU-6000   <root>/001/L_Fore/01.bmp
UTFVP         <root>/0001/0001_1_1.png                  (original ROI)
              <root>/0001/0001_1_1_1_Augmented.png      (augmentation 1)
```

### Preparing UTFVP

UTFVP ships raw captures, so build the ROIs and the augmented set first:

```bash
python scripts/prepare_utfvp_roi.py \
    --input  /path/to/UTFVP/dataset/data \
    --output data/UTFVP_ROI

python scripts/augment_utfvp.py \
    --input  data/UTFVP_ROI \
    --output data/UTFVP_ROI_augmented \
    --seed 0
```

The first command implements the six-stage ROI pipeline: CLAHE with 8 × 8 tiles and clip limit
2.0 → Prewitt boundary detection → abnormal-placement check with variance threshold 38 → DBSCAN
cleaning with radius 10 and minimum 10 samples → tilt correction above 15° → ROI extraction
centred on the brightest column of the central half of the image. The second expands each ROI
into four images (the original plus three stochastic variants), giving 5,760 in total.

Because the augmentation is stochastic, `--seed` fixes the random state; a different seed
produces a different augmented set and therefore slightly different scores.

---

## Running an experiment

Every cell of Tables 1–4 is one command. For example, the proposed operator with (2D)²PCA on
MMCBNU-6000 under Strategy 2, Protocol 1:

```bash
python scripts/run_experiment.py \
    --dataset mmcbnu_6000 --descriptor fused --reducer 2d2pca \
    --strategy S2 --protocol P1
```

The single-radius `LBP^riu2_(8,1)` baseline on FV-USM, session-sensitive, Protocol 3:

```bash
python scripts/run_experiment.py \
    --dataset fv_usm --descriptor lbp_8_1 \
    --strategy S2 --protocol P3 --evaluation session-sensitive
```

Options:

| Flag | Values |
| --- | --- |
| `--dataset` | `fv_usm`, `mmcbnu_6000`, `utfvp` |
| `--descriptor` | `raw`, `lbp_8_1`, `lbp_16_1`, `lbp_8_2`, `fused` |
| `--reducer` | `none`, `pca`, `2dpca`, `2d2pca` |
| `--strategy` | `S1` (fused fingers), `S2` (independent fingers) |
| `--protocol` | `P1` (70/30), `P2` (50/50), `P3` (90/10) |
| `--evaluation` | `session-sensitive`, `session-independent`, `one-session` |
| `--components` | override the reducer's component count |
| `--identity` | override which attributes define an identity: `subject`, `finger`, `session`, `hand` |
| `--identity-convention` | `archive` (default, reproduces the published runs) or `paper` (Section 3.1's definitions) |
| `--output` | write scores, the CMC curve and the misclassified probe indices to JSON |

`docs/experiment_grid.md` lists the command behind every row of Tables 1–4.

### Statistical comparison

McNemar's test needs both methods evaluated on the same probe set, so run them with identical
dataset, strategy, protocol and evaluation settings and save each with `--output`:

```bash
python scripts/run_experiment.py --dataset mmcbnu_6000 --descriptor fused \
    --reducer 2d2pca --strategy S2 --protocol P1 --output results/fused_2d2pca.json
python scripts/run_experiment.py --dataset mmcbnu_6000 --descriptor lbp_8_1 \
    --strategy S2 --protocol P1 --output results/lbp_8_1.json

python scripts/mcnemar_test.py results/fused_2d2pca.json results/lbp_8_1.json
```

### CMC curves

```bash
python scripts/plot_cmc.py results/*.json \
    --title "MMCBNU-6000 (one-session, Strategy 2, Protocol 1)" \
    --max-rank 20 --output figures/mmcbnu_s2_p1.png
```

---

## Library use

```python
from mrflbp import MultiRadiusLBP
from mrflbp.preprocessing import load_grayscale, preprocess_lbp

descriptor = MultiRadiusLBP()                       # (8,1), (16,1), (8,2)
roi = preprocess_lbp(load_grayscale("roi.png"), size=(60, 128))

matrix = descriptor.matrix(roi)                     # patch grid, for 2DPCA / (2D)²PCA
vector = descriptor.vector(roi)                     # flattened, for PCA / L1 matching
print(descriptor.patch_dimension)                   # 38
```

---

## The archived notebooks

`notebooks/` holds the Colab notebooks that produced the published tables, reorganised as
`<dataset>/<method>/<dataset>_<method>[_regime]_S<strategy>_P<protocol>.ipynb`, with all cell
outputs and Colab execution metadata removed. They are kept verbatim otherwise, as the record of
what was run; `mrflbp/` is the consolidated, documented version of the same pipeline. See
[`notebooks/README.md`](notebooks/README.md) for the differences between the two and for the
notebooks' own conventions.

---

## The thesis

> A. Mokabberi, **“Advancing Finger Vein Recognition: A Comparative Analysis of Feature
> Extraction and Matching”**, MSc thesis, Department of Computer Engineering, Eastern
> Mediterranean University, February 2026. Supervisor: Prof. Dr. Önsen Toygar.

The thesis was built on the code in this repository and covers the same descriptor, datasets,
strategies and protocols as the article, in more detail. The compiled thesis is
[`Eastern_Mediterranean_University__EMU__Thesis___By_AmirHossein_Mokabberi.pdf`](Eastern_Mediterranean_University__EMU__Thesis___By_AmirHossein_Mokabberi.pdf);
its LaTeX source (`EMU_THESIS_FINAL_VERSION.tex`, the EMU thesis class, the bibliography and
every figure) is in the `.zip` of the same name.

The thesis also compares the descriptor with a modified AlexNet. Those CNN experiments are not
part of the article, and their notebooks are not included here; see
[`notebooks/README.md`](notebooks/README.md).

---

## Data availability

The datasets analysed in this study are publicly available from their respective owners under
their own licences and are not redistributed in this repository:

- FV-USM — <http://drfendi.com/fv_usm_database/>
- MMCBNU-6000 — <http://multilab.jbnu.ac.kr/>
- UTFVP — <https://www.utwente.nl/en/eemcs/dmb/downloads/utfvp/>

## Code availability

All code needed to reproduce the experiments reported in the article is available in this
repository under the MIT licence.

---

## Citing

If you use this code, please cite the article:

```bibtex
@article{Mokabberi2025FusionMultipleRadii,
  author  = {Mokabberi, AmirHossein and Toygar, {\"O}nsen},
  title   = {Fusion at multiple radii: a rotation-invariant uniform {LBP}
             for finger-vein identification},
  journal = {Signal, Image and Video Processing},
  volume  = {19},
  number  = {1419},
  year    = {2025},
  doi     = {10.1007/s11760-025-05009-3}
}
```

`CITATION.cff` carries the same metadata in a machine-readable form.

To refer to the thesis:

```bibtex
@mastersthesis{Mokabberi2026Thesis,
  author  = {Mokabberi, AmirHossein},
  title   = {Advancing Finger Vein Recognition: A Comparative Analysis of Feature
             Extraction and Matching},
  school  = {Eastern Mediterranean University},
  address = {Famagusta, North Cyprus},
  month   = feb,
  year    = {2026}
}
```

## License

The MIT licence ([`LICENSE`](LICENSE)) covers the code. The thesis and the article's first page
are included for reference and are not covered by it: the article is © The Author(s), under
exclusive licence to Springer-Verlag London Ltd., part of Springer Nature 2025. The datasets
are covered by their own licences.
