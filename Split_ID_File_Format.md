# 22AIE304 — Split-ID file format

**For the Week-2 deliverable (Handout §2: "2-page proposal + baseline numbers + your split-ID files"), due 23:59 IST, Monday 12 October 2026** — extended once from Friday 9 October. Issued 9 October 2026. Earlier issues of the pack named split-ID files as a deliverable without defining them. This sheet defines them.

**Upload your split-ID files and your summary file through the Week 2 upload link by 23:59 IST on Monday 12 October 2026.**

## What the files are for

A split-ID file records, for one protocol on your card, which samples you put in train, validation and test, and which you excluded. We use it to check three things without running your code:

1. no independent unit appears in more than one of train, validation and test;
2. where your card came with a frozen split, your membership is the issued membership;
3. your counts match the card.

**Upload the membership you actually train and evaluate on.** If your card issued a frozen file, that is the issued membership. Your independent Week-1 reproduction belongs in the readiness note. In the proposal, state in one line whether it matched, and how many samples differed if it did not.

**Generate the files with your pipeline code, not by hand.** The script that writes them belongs in your repository.

## The split file

**One file per protocol.** The file names are listed per card in the table below.

**File name:** `T<team>_<card>_<protocol>.csv`. The team number has two digits, for example `T08_E2_matched.csv` or `T16_I4-S_F1_within.csv`.

**Encoding:** UTF-8 without a byte-order mark, comma-separated, with one header row. Quote fields with standard CSV quoting; several PlantVillage paths contain spaces. Row order does not matter.

**Columns, exactly these five, in this order:**

| Column | Content |
|---|---|
| `sample_id` | The sample, written as the table below defines it for your card. Unique within the file. Where an issued file exists, write it exactly as that file writes it. |
| `unit_id` | The independent unit the split is made on, such as a leaf group, lot, patient, cell, recording, scene group, bearing or region. |
| `partition` | One of `train`, `val`, `test`, `excluded`. Nothing else. |
| `label` | The class name; for multi-label samples, the positive classes joined by `;` in the order your card lists them; for regression, the target value. Leave it empty where the table says so. Do not put test-set predictions here. |
| `reason` | Required when `partition` is `excluded`, empty otherwise. Use a short fixed word, for example `cap`, `guard`, `degenerate`, `no_target`, `no_superclass`, `not_in_fold`, `unmapped_class`, `targets_span_partitions`. |

**Every sample in your card's approved pool appears exactly once, in one of the four partitions.** Samples you drop, whether by a cap, a guard band or an exclusion rule, are listed as `excluded` with their reason, not left out. That way the totals in the table below can be checked by adding up the four partitions.

**The no-shared-unit rule, and its exceptions.** No `unit_id` may appear in more than one of `train`, `val` and `test`. Excluded rows are not counted. The rule does not apply to these protocols, which split within a unit by design (Handout §4, §9):

- the Track A `cwru` files (guarded within-recording blocks);
- `T16_I4-S_F1_within` (guarded within-scene reference);
- A3's `audit` file and I2's `audit101`–`audit105` (audit controls).

For Tracks C and D the unit is the issue time. There, we check the boundary rule (every target inside its own partition) from the `sample_id` dates, not from unit overlap.

## The summary file

Upload one per team: `T<team>_<card>_split_summary.csv`, same encoding, columns `file,partition,n_samples,n_units,sha256`.

- Give one row for each split file and each of the four partitions, including zero-count partitions.
- `n_units` is the number of distinct `unit_id` values in that partition.
- `sha256` is the SHA-256 of the split file, repeated on each of its rows.

We recompute every number from your files, and any mismatch is queried. This catches the commonest parser error: a count that does not add up to its stated total.

## What each card uploads

Expected counts come from the card and annexes. Where the pack gives none, you report your own.

| Card | Files (`<protocol>`) | `sample_id` | `unit_id` | `label` | Expected train / val / test (excluded) |
|---|---|---|---|---|---|
| **A1** | `cwru`, `paderborn` | `cwru`: `<file>:<start>`, e.g. `105.mat:0`, where `start` is the window's first sample at 12 kHz after resampling. `paderborn`: `<bearing>/<record>.mat:<start>`, with `start` at 16 kHz | `cwru`: recording code, e.g. `IR007_0`, `Normal_2`. `paderborn`: bearing code, e.g. `KA04` | healthy / inner / outer / ball (CWRU); healthy / inner / outer (Paderborn) | `cwru` 527 / 139 / 155 windows; `paderborn` 3,116 / 1,863 / 3,724 |
| **A2** | `cwru` | as A1 | as A1 | as A1 | 527 / 139 / 155 |
| **A3** | `cwru`, `audit` | as A1; the audit file's windows start at stride 512 | recording code | as A1 | `cwru` 527 / 139 / 155; `audit`: yours |
| **B1** | `source`, `target` | `source`: `path` exactly as in `PlantVillage_color_split.csv`. `target`: path from the PlantDoc repository root, e.g. `test/<class folder>/<file>` | `source`: `leaf_id` as issued. `target`: MD5 of the image file's bytes, so identical photographs share a unit | PlantVillage class (`source`); PlantDoc class folder (`target`) | `source` 7,972 / 2,521 / 2,645, with the other 41,167 rows `excluded` (`cap` or `unmapped_class`); 54,305 rows in all. `target`: 236 `test`; the 2,342 PlantDoc training images `excluded` (`plantdoc_train_unused`) |
| **B2** | `colour` | `path` as issued | `leaf_id` as issued | PlantVillage class | 10,766 / 3,429 / 3,530, the other 36,580 `excluded` (`cap`); 54,305 rows. Segmented and greyscale follow `PlantVillage_B2_alignment.csv` and need no file of their own |
| **B3** | `cap10`, `cap25`, `cap50` | `path` as issued | `leaf_id` as issued | PlantVillage class | train 380 / 949 / 1,899 images in the three files; val 746 and test 750 in every file; the rest `excluded` (`cap` or `ineligible_class`); 54,305 rows in each |
| **C1–C3** | `forecast` | issue month `YYYY-MM` | same as `sample_id` | empty | 505 / 109 / 229, plus 22 `excluded` (`targets_span_partitions`); 865 rows |
| **D1, D2** | `metr-la` | 0-based row index of the issue time in `metr-la.h5`; list only the issue times Annex D4 selects | same as `sample_id` | empty | 3,995 / 285 / 571 |
| **D3** | `metr-la`, `pems-bay` | as D1, row index in each city's file | same as `sample_id` | empty | METR-LA 3,995 / 285 / 571; PEMS-BAY 6,077 / 434 / 868 |
| **E1** | `builtin` | 0-based row index in the pickle (the wafer ID) | `lotName` | normalised `failureType` | 43,308 / 11,047 / 118,595; 172,950 labelled wafers, with unlabelled wafers not listed |
| **E2** | `builtin`, `matched` | as E1 | `lotName` | as E1 | `builtin` as E1; `matched` 103,853 / 34,440 / 34,657 |
| **E3** | `builtin`, `pool` | as E1 | `lotName` | as E1; empty in `pool` | `builtin` as E1. `pool`: 10,000 rows, all `train`, `reason` empty, in the issued hash order; the first 2,000 rows are the 2,000 cap |
| **I1** | `opssat`, `smap_msl` | `opssat`: `segment`. `smap_msl`: `<stream>:<file>:<start>-<end>`, a half-open native index range, e.g. `P-1:train:0-<end>` | `opssat`: `segment`. `smap_msl`: stream, `P-1` or `C-1` | `opssat`: `anomaly` (0/1). `smap_msl`: empty | `opssat`: 1,019 `train`, 254 `val` (nominal development), the provided test fragments `test`, and anomalous training fragments `excluded` (`anomalous_train_unused`); 2,123 rows. `smap_msl`: one row per range (fit, guard, calibration, test) per stream, the guard `excluded` (`guard`) |
| **I2** | `official`, `audit101`…`audit105` | `ecg_id` | `patient_id` | superclasses joined by `;` in NORM;MI;STTC;CD;HYP order | `official` 17,073 / 2,145 / 2,157, plus 424 `excluded` (`no_superclass`); 21,799 rows. Each audit file: 17,100 / 2,137 / 2,138 plus the same 424 |
| **I3** | `cells` | cell ID in lower case | same as `sample_id` | `cycle_life_target` | 75 / 24 / 29, plus 7 `excluded` (`no_target`); 135 physical cells |
| **I4-S** | `F1_within`, `F1_transfer`, `F2`, `F3`, `F4` | `patch_id` | `scene_group` as issued | `is_positive` (0/1) | every file has 3,799 rows. `F1_within` 1,702 / 507 / 570, with 1,020 `excluded` (`guard` 141, `degenerate` 114, `not_in_fold` 765). `F1_transfer` 644 / 121 / 570. `F2` 3,414 / 121 / 150. `F3` 3,070 / 494 / 121. `F4` 3,041 / 150 / 494. The rest of each file is `excluded` |
| **I5** | `focal`, `powdermill` | `focal`: `XC<number>:<w>`, with window index `w` from 0 to 5. `powdermill`: `<segment mp3 file name>:<w>`, `w` from 0 to 59 | `focal`: `XC<number>`. `powdermill`: `Recording_1` … `Recording_4` | `focal`: the recording's species code. `powdermill`: positive codes among your 15, joined by `;`, empty if none | `focal`: windows from 540 / 180 / 180 recordings, at most 5,400; drop invalid windows under the RMS rule as `excluded` (`silent`). `powdermill`: 4,620 rows, all `test` |
| **I6** | `parcels` | parcel `id` as in `I6_parcel_selection.csv` | region (`frh01`–`frh04`) | class name | 13,317 / 4,500 / 4,500 |

**Not needed this week:** I5's noise-clip pairing, I2's audit-overlap statistics and B2's 100-image attribution set. They are experiment configuration, not split membership, and belong in the report and repository.

**If a count does not match the table,** upload what your pipeline produced and explain the difference in the proposal. A documented mismatch is a finding and costs nothing. A silent one is a leakage risk.
