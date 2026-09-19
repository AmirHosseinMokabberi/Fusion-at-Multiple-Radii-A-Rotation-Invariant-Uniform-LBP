# Identity conventions

An identification score depends on two things: the ranking, and what counts as
"the same identity". This page documents the second, because the experiments and
the article text do not always agree on it — and the experiments were not
internally uniform either.

**This repository reproduces the experiments.** `--identity-convention archive`
is the default, so a run scores the way the run behind the published table
scored. `--identity-convention paper` switches to the definitions given in
Section 3.1 of the article. The CLI prints a note whenever the two differ, and
every result JSON records which was used.

## What the experiments compared

Reconstructed by reading the scoring cell of all 142 notebooks; each derived its
identity by splitting a label string on `_`.

| Table | Rows | Strategy | Identity compared |
| --- | --- | --- | --- |
| 1 — FV-USM, session-sensitive | all | S1 | subject + session |
| 1 | `LBP^riu2` with no reducer | S2 | subject + session |
| 1 | LBP + PCA / 2DPCA / (2D)²PCA | S2 | subject + finger + session |
| 1 | raw PCA, raw (2D)²PCA | S2 | subject + finger + session |
| 1 | raw 2DPCA | S2 | session + finger |
| 2 — FV-USM, session-independent | all | S1 | subject |
| 2 | all | S2 | subject + finger |
| 3 — UTFVP, one-session | all | S1 | subject |
| 3 | all | S2 | subject + finger |
| 4 — MMCBNU-6000, one-session | all | S1 | subject |
| 4 | all | S2 | subject + hand |

## Where the article text says something different

Section 3.1 defines the regimes as: session-sensitive is genuine when subject
*and* session match; session-independent is genuine for any same-subject pair.
Reading that literally gives subject + session and subject, with the finger
never part of a FV-USM identity, and subject + finger under the one-session
regimes. Four groups of cells depart from it:

| Cells | Ran with | Article implies |
| --- | --- | --- |
| Table 1, S2, LBP or raw with a reducer | subject + finger + session | subject + session |
| Table 1, S2, raw 2DPCA | session + finger | subject + session |
| Table 2, S2, all rows | subject + finger | subject |
| Table 4, S2, all rows | subject + hand | subject + finger |

Two of those look unintended rather than deliberate:

**MMCBNU-6000, Strategy 2 (all 21 notebooks).** MMCBNU names its finger
directories `L_Fore` … `R_Ring`, so a label reads `001_L_Fore_img01` and
`label.split("_")[1]` returns `L`, not `L_Fore`. The comparison is therefore
subject + hand: 200 classes rather than 600, and a probe from `L_Fore` matched
against `L_Ring` of the same subject counts as correct. The scoring cells
themselves print "Match (Subject + Finger correct)", so subject + finger was the
intent. FV-USM and UTFVP are unaffected — their finger tokens (`f2`, `F3`,
`finger2`) contain no underscore.

**FV-USM, session-sensitive Strategy 2, raw 2DPCA (3 notebooks).** The label is
`session1_subject005_fingervein2`, and the scoring cell compares `parts[0]`
against `parts[0]` and `parts[-1]` against `parts[-1]` — session against session
and finger against finger. The subject token sits at `parts[1]` and is never
read.

Every method row within an affected table was scored the same way, so the
comparisons those tables support are unaffected; the absolute percentages are
what a stricter identity would change.

## Choosing a convention

```bash
# Default: score as the published run did (200 MMCBNU classes, subject + hand)
python scripts/run_experiment.py --dataset mmcbnu_6000 --descriptor fused \
    --reducer 2d2pca --strategy S2 --protocol P1

# The article's definition instead (600 classes, subject + finger)
python scripts/run_experiment.py --dataset mmcbnu_6000 --descriptor fused \
    --reducer 2d2pca --strategy S2 --protocol P1 --identity-convention paper

# Or state the fields outright; this is what the default resolves to above
python scripts/run_experiment.py --dataset mmcbnu_6000 --descriptor fused \
    --reducer 2d2pca --strategy S2 --protocol P1 --identity subject hand
```

`--identity` accepts `subject`, `finger`, `session` and `hand` in any
combination and takes precedence over `--identity-convention`. Result JSON
records `identity`, `identity_convention` and an `identity_matches_paper` flag,
so a saved run is never ambiguous about how it was scored.

For UTFVP and for every Strategy 1 cell the two conventions resolve to the same
fields, and the choice changes nothing.
