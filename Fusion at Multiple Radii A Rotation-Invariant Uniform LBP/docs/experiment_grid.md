# Experiment grid

Tables 1–4 of the paper are one grid evaluated four times. Each table has the same ten method
rows and six columns (Strategy 1 and Strategy 2, each under Protocols 1–3); the tables differ
only in dataset and evaluation regime.

| Table | Dataset | `--dataset` | `--evaluation` |
| --- | --- | --- | --- |
| 1 | FV-USM | `fv_usm` | `session-sensitive` |
| 2 | FV-USM | `fv_usm` | `session-independent` |
| 3 | UTFVP | `utfvp` | `one-session` |
| 4 | MMCBNU-6000 | `mmcbnu_6000` | `one-session` |

Scores also depend on what counts as the same identity. The commands below score the way the
published runs did; add `--identity-convention paper` for the definitions in Section 3.1 of
the article. See [`identity_conventions.md`](identity_conventions.md).

## Method rows

| Row label in the paper | `--descriptor` | `--reducer` |
| --- | --- | --- |
| PCA | `raw` | `pca` |
| 2DPCA | `raw` | `2dpca` |
| (2D)²PCA | `raw` | `2d2pca` |
| LBP^riu2_(8,1) | `lbp_8_1` | `none` |
| LBP^riu2_(16,1) | `lbp_16_1` | `none` |
| LBP^riu2_(8,2) | `lbp_8_2` | `none` |
| LBP^riu2_(8,1),(16,1),(8,2) | `fused` | `none` |
| LBP^riu2_(8,1),(16,1),(8,2) + PCA | `fused` | `pca` |
| LBP^riu2_(8,1),(16,1),(8,2) + 2DPCA | `fused` | `2dpca` |
| LBP^riu2_(8,1),(16,1),(8,2) + (2D)²PCA | `fused` | `2d2pca` |

## Columns

| Column | `--strategy` | `--protocol` | Split |
| --- | --- | --- | --- |
| S1 P1 | `S1` | `P1` | 70 % train / 30 % test |
| S1 P2 | `S1` | `P2` | 50 / 50 |
| S1 P3 | `S1` | `P3` | 90 / 10 |
| S2 P1 | `S2` | `P1` | 70 / 30 |
| S2 P2 | `S2` | `P2` | 50 / 50 |
| S2 P3 | `S2` | `P3` | 90 / 10 |

Which acquisitions land in which set:

| Dataset | Captures per finger | P1 train / test | P2 train / test | P3 train / test |
| --- | --- | --- | --- | --- |
| FV-USM | 6 per session | 1–4 / 5–6 | 1–3 / 4–6 | 1–5 / 6 |
| MMCBNU-6000 | 10 | 1–7 / 8–10 | 1–5 / 6–10 | 1–9 / 10 |
| UTFVP | 16 after augmentation | 11 / 5 | 8 / 8 | 14 / 2 |

UTFVP is the irregular case: its 16 images per finger are 4 originals plus 12 augmentations, and
the split is defined over `(capture, variant)` pairs so that original ROIs are kept for testing
wherever the protocol allows. `UTFVP.SPLITS` in `mrflbp/datasets.py` lists the exact pairs.

## Running one cell

```bash
python scripts/run_experiment.py \
    --dataset fv_usm --evaluation session-sensitive \
    --descriptor fused --reducer 2d2pca \
    --strategy S2 --protocol P3 \
    --output results/table1_fused_2d2pca_S2_P3.json
```

## Running a whole table

```bash
#!/usr/bin/env bash
# Table 4: MMCBNU-6000, one-session.
set -euo pipefail

DATASET=mmcbnu_6000
EVAL=one-session

run () {  # descriptor reducer
    for strategy in S1 S2; do
        for protocol in P1 P2 P3; do
            python scripts/run_experiment.py \
                --dataset "$DATASET" --evaluation "$EVAL" \
                --descriptor "$1" --reducer "$2" \
                --strategy "$strategy" --protocol "$protocol" \
                --output "results/${DATASET}_$1_$2_${strategy}_${protocol}.json"
        done
    done
}

run raw      pca
run raw      2dpca
run raw      2d2pca
run lbp_8_1  none
run lbp_16_1 none
run lbp_8_2  none
run fused    none
run fused    pca
run fused    2dpca
run fused    2d2pca
```

On Windows, the same loop in PowerShell:

```powershell
$dataset = "mmcbnu_6000"
$methods = @(
    @("raw", "pca"), @("raw", "2dpca"), @("raw", "2d2pca"),
    @("lbp_8_1", "none"), @("lbp_16_1", "none"), @("lbp_8_2", "none"),
    @("fused", "none"), @("fused", "pca"), @("fused", "2dpca"), @("fused", "2d2pca")
)
foreach ($m in $methods) {
    foreach ($s in @("S1", "S2")) {
        foreach ($p in @("P1", "P2", "P3")) {
            python scripts/run_experiment.py --dataset $dataset --evaluation one-session `
                --descriptor $m[0] --reducer $m[1] --strategy $s --protocol $p `
                --output "results/$dataset`_$($m[0])_$($m[1])_$s`_$p.json"
        }
    }
}
```

## Figures 3–14

The published CMC curves are all Strategy 2. Run the methods you want on one plot with
`--output`, then:

```bash
python scripts/plot_cmc.py results/mmcbnu_6000_*_S2_P1.json \
    --title "MMCBNU-6000 (one-session, Strategy 2, Protocol 1)" \
    --max-rank 20 --output figures/fig09.png
```

| Figure | Dataset | Regime | Column |
| --- | --- | --- | --- |
| 3–5 | FV-USM | session-sensitive | S2 P1, P2, P3 |
| 6–8 | FV-USM | session-independent | S2 P1, P2, P3 |
| 9–11 | MMCBNU-6000 | one-session | S2 P1, P2, P3 |
| 12–14 | UTFVP | one-session | S2 P1, P2, P3 |

## Table 5

Table 5 compares against published results obtained under each source's own preprocessing, so
only the “Our study” and “Proposed methods” rows come from this repository. Those rows use a
single FV-USM session; `notebooks/fv_usm/2dpca/fv_usm_2dpca_single_session_comparative.ipynb` is
the archived run for that setting.
