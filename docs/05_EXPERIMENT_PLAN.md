# 05. Experiment and evaluation protocol

## 1. Predeclared protocol

Before experiments, freeze data hashes, engine membership, snapshot cutoffs, target cap, feature definitions, search budget and metrics. Record each run in local MLflow and a JSON run manifest. Seed 42 is the default.

The data specification owns the 80/20 engine split, five inner folds, 20-cycle causal history and uncapped evaluation labels. Never use official-test metrics to select a model.

## 2. RUL experiment sequence

| ID | Experiment | Purpose |
|---|---|---|
| E01 | DummyRegressor median | Constant prediction baseline |
| E02 | Ridge using cycle only | Determine what engine age explains without sensors |
| E03 | Ridge using current sensors/settings/cycle | Simple sensor baseline |
| E04 | XGBoost using current values | Nonlinear model before rolling features |
| E05 | XGBoost using current values plus 20-cycle mean/std/slope | Test whether recent history adds value |
| E06 | Winner without cycle | Measure dependence on age |
| E07 | Winner trained on uncapped RUL | Target-cap sensitivity, exploratory; cannot displace champion automatically |

E01-E05 are the selection pool. E06-E07 are declared diagnostics. StandardScaler applies to Ridge; XGBoost does not require scaling. Fit sensor selection and preprocessing on each training fold only. Apply equal total training weight per engine.

Use the same frozen held-out snapshots for paired comparisons. Within inner CV, evaluate each engine's eligible four cutoffs, then give each engine equal weight in aggregated squared error. Report fold RMSE and between-fold variation.

## 3. Search budget and champion selection

- Ridge alpha candidates: 0.1, 1, 10, 100.
- XGBoost search covers depth 2-5, learning rate 0.02-0.10, estimator count 100-500, subsample 0.7-1.0, colsample_bytree 0.7-1.0 and bounded regularisation.
- At most 30 XGBoost configurations total across E04/E05, each evaluated on the same five folds. Declare the sampling list and seed before execution.
- Fixed estimator counts avoid early-stopping ambiguity. If early stopping is later introduced, its evaluation data must be separately grouped inside each training fold.
- Choose lowest mean inner-CV endpoint RMSE. Define the best model's standard error as sample std of its five fold RMSE values divided by sqrt(5). If another eligible model is within that standard error of the best mean, prefer Ridge, then shallower/smaller XGBoost. The constant baseline can win the comparison but cannot meet the nonconstant promotion requirement.
- Select feature set and configuration using inner folds. Fit the chosen RUL model on the 80 development engines and evaluate the 20 validation endpoints.
- Validation is a promotion check, not permission to switch to whichever candidate looks best on those 20 engines.
- Keep the final 80-engine fit for this release. Do not refit on validation before official evaluation; this preserves the selected policy and validation evidence.

If validation gates fail, keep the result as unpromoted and investigate on development data. A material protocol change receives a new version and states that validation has been reused. Never describe repeated validation as an untouched holdout.

## 4. RUL metrics

For endpoint error `e = predicted_rul - true_rul`:

- MAE = mean(abs(e)); RMSE = sqrt(mean(e squared)).
- Mean signed error measures bias. Positive error overestimates remaining life.
- Overestimation rate = mean(e >0); also report mean positive error and the fraction with e >10 cycles.
- NASA asymmetric score: sum(exp(-e/13)-1) for e <0, plus sum(exp(e/10)-1) for e >=0. Lower is better; report both total and mean per engine.
- Report errors in true-RUL bands <=30, 31-100 and >100, with group sizes.
- Use 1,000 seeded engine-level bootstrap draws for 95% percentile intervals and paired RMSE differences. Never bootstrap correlated rows as independent engines.

The NASA score is supported by [source S3](10_SOURCES.md). It is an error penalty, not money or a certified operational cost.

## 5. Maintenance policy diagnostics

- Actual near-failure proxy: true RUL <=30 cycles.
- Predicted horizon flag: estimated RUL <=30 cycles.
- Report TP/FP/FN/TN, precision, recall and counts. Use null with a reason for undefined ratios.
- Rank one endpoint per engine using the serving policy. At k=max(1,ceil(0.20*N)), report precision@k, recall@k and lift over prevalence.
- Compare constant and age-only baselines using the same sorting and tie rules.
- Report the artificial validation snapshot prevalence alongside the official-test prevalence. Do not equate either with a real maintenance fleet.
- No invented financial benefit, maintenance intervention or downtime reduction.

## 6. Isolation Forest protocol

1. Use eligible development rows with true RUL >=100 as a retrospective reference proxy. Sample at most 50 evenly spaced rows per engine, with >=30 contributing engines.
2. Exclude cycle, unit IDs, targets and horizon flags. Fit sensor-feature selector and StandardScaler on this subset only.
3. Fixed initial model: 200 trees, max_samples='auto', contamination='auto', seed 42. Bound parallelism and record it.
4. Define anomaly_score = -score_samples. Larger is more unusual. Do not use predict's default cutoff or reinterpret the score as a probability.
5. Freeze cutoff at the 99th percentile of scores on that sampled development proxy, quantile method 'linear'; anomaly_flag is score >cutoff.
6. Evaluate on validation full-trajectory high-RUL and near-failure proxy rows, with causal features. Report per-engine rates and their macro averages. Gate: high-RUL macro false-flag rate <=5%; require >=10 validation engines contributing proxy rows, otherwise mark inconclusive.
7. Report near-failure proxy flag rate and synthetic perturbation sensitivity as diagnostics. There are no true anomaly labels to establish anomaly precision or fault diagnosis accuracy.

The fitted training cutoff may be optimistic. Validation measures this limitation; do not retune the cutoff on official test. If it fails, keep the anomaly component unpromoted and record the reason.

## 7. Explainability

- Use SHAP TreeExplainer for XGBoost and the compatible linear explainer for Ridge. Background data, when needed, are development-only.
- Explain raw RUL output in cycles and verify full contribution sum plus expected value matches it within numerical tolerance.
- Report a global mean absolute contribution view and individual top-three signed contributions. Group related rolling features by sensor in an additional summary without hiding the original features.
- Include good predictions, large overestimates and large underestimates. SHAP explains model behaviour, not causal mechanical failure.
- A constant baseline has no meaningful sensor explanations and cannot masquerade as the promoted explanation-capable model.

## 8. Drift experiment

- Freeze reference features from one development snapshot per engine using the documented cutoff assignment procedure. Store 80 rows, schema, feature names, sampling policy and bundle version.
- Current batches use one latest endpoint per engine, the same feature code and the same input scope. Do not include IDs or labels in drift tests.
- Compare at least 30 independent engines per batch; smaller batches produce insufficient_data rather than a reassuring pass.
- Explicit starting method: numeric Wasserstein drift on selected varying sensor features with threshold 0.1; dataset warning when at least 50% of these features drift. Verify the pinned Evidently version supports the intended normalisation and record the actual method/threshold in the report.
- Controls: an exact copy of reference should show no change; shift one varying raw sensor by +2 reference standard deviations across its recent histories, rebuild features, and verify that sensor's features are flagged. This is a synthetic check, not production validation.
- Report cycle/stage mix and warn when cohort composition differs. Progressive degradation can cause expected drift. No automatic retraining or model failure conclusion follows from a warning.
- Export HTML and JSON with sample sizes, reference/current definitions, quality failures and warnings. Diagnostic alarms for live fleets need separate calibration.

## 9. Final holdout procedure

1. Freeze RUL/anomaly models, policy, features, dependency lock, config and validation gate outcomes into a versioned bundle.
2. Score only each official test engine's last observed eligible cycle. Write and hash predictions before joining RUL_FD001.txt.
3. Run the official evaluation for this frozen release and record all metrics, targets and limitations, including failures.
4. Do not tune and rescore to improve the same held-out claim. A bug-fixed rerun must preserve the original report, explain the bug and disclose that the holdout has been observed.
5. Keep one row per engine. If a test engine has <20 cycles, report a protocol incompatibility rather than silently dropping it or padding with future information.
