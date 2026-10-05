# 11. Model performance review

## 1. Verdict: partially achieved, release incomplete

Review date: 6 October 2026, Asia/Kuala_Lumpur.

The preserved XGBoost RUL predictions pass the main error target and the recorded validation improvement gate. The 30-cycle detection recall target fails. The current anomaly evidence fails G4, and the complete local release has unresolved engineering and documentation requirements.

The existing README, model card and frozen metadata contain stronger success claims than the current evidence supports. This review records the discrepancies without changing trained models, the original evaluation reports or the agreed targets.

## 2. What was verified

1. Recomputed metrics from all 100 preserved official predictions joined by engine ID to the original uncapped RUL vector, using independent Python standard-library arithmetic.
2. Confirmed 100 unique engine IDs and the official prediction CSV SHA-256 recorded in its original evaluation report.
3. Confirmed every bundle file checksum recorded in models/v0.1.0/metadata.json matches its current file.
4. Compared current validation, frozen metadata, anomaly feature schema, staging metadata, runtime report and relevant implementation/test code.
5. Checked the current Python launcher and Docker availability.

Machine-readable evidence and input hashes: [performance audit](../reports/performance_audit_2026-10-06.json).

This is a verification of preserved outputs and current source/artifact consistency. Fresh inference, training and pytest were not completed: the project venv launcher cannot create a process using its referenced CPython 3.11 executable in this execution environment. The bundled interpreter is Python 3.12; importing the project's cp311 NumPy extension through it fails. This limitation does not establish whether Python is unavailable elsewhere on the user's computer.

## 3. Targets versus current evidence

| Objective | Original target | Current evidence | Assessment |
|---|---|---|---|
| O1 RUL estimation | Official endpoint RMSE <=25 cycles | Independently recomputed RMSE 16.2098; MAE 12.0182 | PASS for preserved benchmark predictions |
| O2 Sensor value / G3 | Validation RMSE >=10% better than constant and age-only | Saved validation RMSE 14.4807; improvements 71.37% and 44.24% | Recorded gate PASS; report/model provenance must remain tied together |
| O3 Horizon precision | Precision >=60% | 19 true positives / 20 flagged engines =95% | PASS |
| O3 Horizon recall | Recall >=80% | 19 detected / 25 actual near-failure engines =76% | FAIL: six missed engines |
| O4 Capacity ranking | Report top-20% metrics and baseline comparisons | 19/20 selected positives; precision 95%, recall 76%, lift 3.80 | Strong reported ranking; saved final report lacks the full baseline policy comparison |
| O5 Anomaly / G4 | >=30 fit engines, >=10 validation proxy engines, macro false-flag rate <=5% | Current validation: 35 fit engines, 20 validation proxy engines, false-flag rate 98.80% | FAIL |
| O6 Explainability | Valid signed contributions and global summary | Explanation artifacts/tests exist; model card has unverified physical sensor names | Partly evidenced; fresh additivity check and documentation reconciliation needed |
| O7 Reproducibility | Repeat champion training within 0.01-cycle RMSE | Runtime Boolean is hardcoded; baseline repeat test does not demonstrate champion repeat training | NOT PROVEN for the champion |
| O8 Latency | Plain scoring p95 <200 ms | Saved TestClient benchmark p95 24.09 ms, 100 calls | Reported PASS; current rerun unavailable |
| O8 Container / CI / contracts | Executed Docker smoke, usable CI and specified API | Daemon unreachable; lock/CI/test dependencies and API contract have gaps | INCOMPLETE |

Sources: [objectives](01_PROBLEM_AND_OBJECTIVES.md), [validation report](../reports/validation_metrics.json), [original test report](../reports/official_test_metrics.json), [runtime report](../reports/runtime.json) and the independent audit above.

## 4. Detection detail

At the unchanged policy `estimated RUL <=30` versus actual `true RUL <=30`:

| Count | Engines |
|---|---:|
| True positive | 19 |
| False positive | 1 |
| False negative | 6 |
| True negative | 74 |

Recall needs at least 20 of these 25 positives to meet 80%. This arithmetic identifies the gap; it is not permission to adjust the threshold against the observed test outcomes.

Missed engines: 90, 37, 40, 52, 53 and 18. Their true RUL values are 21-29 cycles; predicted values are approximately 30.42-45.32 cycles. Use these cases to describe failure modes. Choose any correction using development-only grouped out-of-fold evidence.

Top-capacity ranking and horizon detection are distinct policies. Their selected engines happen to coincide in the saved output. With 20 inspection slots and 25 positives, capacity recall cannot exceed 80% on this particular fleet even with perfect ranking.

The near-failure error band is much better than the long-RUL band: recorded RMSE 6.98 versus 20.94 cycles. The near-failure mean bias is +3.00 cycles. The negative overall bias (-1.64) does not mean all predictions are conservative: 19 engines overestimate life by more than 10 cycles.

## 5. Highest-priority defect: anomaly artifact contamination

### Confirmed observations

- tests/test_anomaly.py creates a synthetic 35-engine dataset with only s02 sensor features, instantiates default Config, and calls fit_isolation_forest_component.
- That fitting function writes directly to cfg.models.staging_dir, whose default is models/staging.
- Current staging metadata records 35 engines, 1,085 sampled rows and four s02 features. These match the synthetic fixture's structure.
- The frozen v0.1.0 anomaly schema also has exactly those four features. Its anomaly pipeline bytes are identical to the current staging pipeline.
- The current policy cutoff is 0.6720957. The model card instead says 0.5112.
- The validation report says G4 FAIL with 98.80% high-RUL macro false flags. Frozen metadata instead says G4 PASS with 80 fitting engines and 0.54% false flags.
- The hash-matching saved official output flags all 100 engines, including all 34 engines with true RUL >=100. That 100% high-RUL endpoint flag rate is a separate diagnostic, not the validation macro metric.

### Interpretation

The evidence strongly indicates a synthetic test model was staged and frozen while an older validation pass record was reused. Missing model-bound run receipts prevent a complete reconstruction of the execution history. This is primarily an artifact/provenance defect, not evidence that Isolation Forest as a method cannot work on FD001.

The root path needs correction before increasing tree counts or tuning the cutoff. A detector that flags everything cannot prioritise unfamiliar behaviour effectively.

## 6. Promotion and release defects

1. **Promotion logic:** freeze_model_bundle derives promoted status from G3 alone. It does not require G4 to pass or verify that the validation report evaluated the exact staged model hashes.
2. **Immutability:** freeze can write into an existing version directory; official evaluation writes fixed report filenames without a duplicate-release guard. These do not enforce the documented preservation contract.
3. **Docker/CI:** Docker could not connect to its daemon. requirements.lock includes unmarked pywin32 and a machine-local editable Windows path, while Docker/CI use Linux. CI tests also require ignored NASA data, a trained bundle and official reports; the synthetic-only CI contract is unmet.
4. **API:** the implemented readings/batch schema differs from the documented single-engine history contract. It lacks the specified strict/extra-field policy, 20-200 request bound and 1 MiB body guard. Existing tests exercise a narrower implemented contract.
5. **Monitoring:** saved controls pass, but code uses drift_share=0.1 and converts any feature drift to dataset drift. It does not enforce the planned 50% dataset warning, explicit Wasserstein configuration, independent-engine counting or the raw-history +2 sigma control.
6. **Evidence quality:** runtime reproducibility is a literal True, not a measured two-training-run comparison. The model card's sensor descriptions/counts and some uncertainty values disagree with the feature schema/comparison report. Historical C5/T15 completion claims overstate verification.

The original achieved RUL benchmark should be preserved. The combined release should remain under review until these defects and unsupported claims are reconciled.

## 7. What to improve first

1. Isolate every test that writes model artifacts, then make promotion depend on the exact validated models and all required gates.
2. Refit the anomaly component on the real development high-RUL proxy data in a new run/version; verify G4 on the validation engines.
3. Improve recall using development out-of-fold bias/calibration or a narrowly justified training change, preserving the original horizon and precision/error targets.
4. Complete portable dependency, container, API, monitoring and champion reproducibility checks.

Execution order and acceptance conditions: [improvement plan](12_IMPROVEMENT_PLAN.md). No fixes or retraining were executed during this review.
