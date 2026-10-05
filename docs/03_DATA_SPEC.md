# 03. Data specification

## 1. Dataset choice

Use NASA C-MAPSS **FD001**. The verified archive README describes 100 training engines, 100 test engines, one sea-level operating condition and high-pressure-compressor degradation. Training trajectories end at failure; test trajectories stop before failure, with a separate endpoint RUL vector.

This is simulated data. It contains no maintenance interventions, costs or real factory downtime. Source details and observed hashes are in [sources](10_SOURCES.md).

SECOM is deferred because process pass/fail modelling would change the target and cannot support the same run-to-failure RUL contract.

## 2. Acquisition and provenance

1. Follow the NASA repository's dataset 6 download, not the similarly named simulator-software catalog entry.
2. The outer ZIP contains another CMAPSSData.zip. Extract only the required text files and preserve its README/citation. Validate member paths remain inside the chosen data directory.
3. Save train_FD001.txt, test_FD001.txt and RUL_FD001.txt under data/raw/FD001/.
4. Record download URL/time, archive hash, individual file hashes, citation and terms in data/source_manifest.json.
5. Verify expected hashes and dimensions. If upstream bytes change, investigate and record provenance before proceeding.
6. Do not commit raw data. Review redistribution terms before any future public data release; do not treat a code license as a dataset license.

## 3. Raw schema

Whitespace-separated rows, no header. Ignore empty trailing whitespace; never remove a populated column to force the schema.

| Positions | Internal names | Type and meaning |
|---|---|---|
| 1 | unit_id | Positive integer, unique within source partition |
| 2 | cycle | Positive integer operating cycle |
| 3-5 | op_1, op_2, op_3 | Three finite numeric operating settings |
| 6-26 | s01 through s21 | Twenty-one finite numeric sensor measurements |

The source README's last sensor line has a numbering inconsistency. The verified 26-field rows imply 21 sensor fields after the five identifier/settings fields. Preserve generic sensor IDs; physical descriptions require a verified mapping.

Observed raw sizes: 20,631 training rows, 13,096 test rows and 100 endpoint RUL values. Structural QA may inspect test features, but training and tuning must never consume test rows or labels.

Validation rules:

- Exactly 26 fields per measurement row; RUL vector has one finite nonnegative value per test unit.
- No duplicate (source_partition, unit_id, cycle) key.
- Per-engine cycles start at 1, increase by 1 and contain no gaps. Detect errors before sorting; documented raw-file normalisation may sort by unit/cycle.
- Numeric values must be finite. MVP rejects missing readings; imputation requires a separately documented change.
- Train and test both use IDs such as 1: globally identify an engine by (partition, unit_id), never join train and test by the integer alone.
- Join endpoint RUL to test IDs 1-100 in ascending order and test this mapping explicitly.

## 4. Labels and predictor exclusions

For a training engine with failure endpoint T, label at cycle t:

`rul_true = T - t`

For a test endpoint, use the corresponding official RUL value directly. If a retrospective intermediate-cycle diagnostic is needed:

`rul_true(t) = official_endpoint_rul + last_observed_cycle - t`

True RUL may use future outcomes for offline evaluation. Feature generation cannot use T, remaining life, endpoint labels, total trajectory length, failure flags or any observation after t.

Default training target: min(rul_true, 125). This cap is a modelling assumption. Primary metrics always compare predictions with **uncapped** true endpoint RUL. Report capped metrics only as separately labelled diagnostics. Predictions are clipped at zero, without an upper clipping bound.

## 5. Frozen engine splits

1. Sort training engine IDs and permute with numpy.random.default_rng(42).
2. First 80 IDs are development; remaining 20 are validation. Save exact IDs, RNG/library versions and split hash.
3. Use five GroupKFold splits within development, grouping by engine ID. Persist fold assignments.
4. All feature selection and fitted preprocessing occur inside each fold. Never split overlapping windows across training and validation.
5. Official NASA test engines remain the final holdout. No test-driven feature selection, tuning or cutoff adjustment.

For validation, assign each engine one reproducible endpoint: permute its 20 IDs with the same declared seed, then assign remaining-life offsets 10, 30, 60, 90 cyclically. Cut at `T - offset`; retain only the prefix. Require cut cycle >=20; if impossible, use the largest eligible offset from that list and log the exception. Freeze this snapshot manifest before experiments. These engineered endpoint proportions are artificial and must be stated in the report.

Inner-fold evaluation uses eligible offsets 10, 30, 60, 90 for each held-out development engine; aggregate with equal engine weight. Full validation trajectories may support separately labelled anomaly diagnostics, using causal features at each time. Their results are not the primary endpoint benchmark.

## 6. Shared feature contract

- Require at least 20 consecutive cycles and use the final 20 rows ending at prediction cycle t.
- For each nonconstant sensor: value at t, trailing mean, population std (ddof=0), and least-squares slope versus relative cycle positions 0-19.
- RUL predictors may include current cycle and the three current operating settings. Drop constant columns based only on the fitting split.
- Anomaly predictors exclude cycle, IDs, RUL and policy flags. Fit their own scaler/feature selector on the proxy training subset.
- No centred windows, backward fill from future rows, full-lifetime averages or future normalisation.
- Never calculate a window across engine boundaries.
- For model fitting, eligible rows are cycles >=20. Give each engine total sample weight 1 so long trajectories do not dominate RUL fitting.
- API and CLI use the exact same feature builder and persisted column order.

## 7. Anomaly proxy labels

- High-RUL proxy: eligible training rows with true RUL >=100 cycles.
- Near-failure proxy: rows with true RUL <=30 cycles.
- Intermediate rows are neither proxy category.
- These are retrospective evaluation/training choices; high RUL does not certify mechanically healthy operation and near failure is not an observed anomaly label.
- Isolation Forest fitting samples at most 50 evenly spaced eligible proxy rows per development engine. At least 30 contributing engines are required; otherwise record insufficient support.

## 8. Derived files

| Path | Purpose |
|---|---|
| data/source_manifest.json | Acquisition, citation and checksums |
| data/processed/split_manifest.json | Engine membership and fold assignments |
| data/processed/snapshot_manifest.json | Frozen prefix cutoffs and endpoint labels |
| data/processed/features_*.parquet | Features, partition metadata and separate label fields |
| reports/data_quality.json | Schema, counts, missingness, chronology and constant columns |

Processed tables must retain provenance keys for auditing; the model input allowlist excludes metadata and labels. prepare may calculate deterministic causal feature values, but it cannot fit selectors/scalers on the full dataset. Those fitted operations belong inside training folds. Official-test evaluation joins labels only in the evaluation module.
