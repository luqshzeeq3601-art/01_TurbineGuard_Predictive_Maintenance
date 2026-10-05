# 08. Decision log

## 1. Initial decisions: 5 October 2026

These are engineering defaults proposed within the user's planning request. They are not measured results or evidence of the user's approval of each numerical assumption.

| ID | Decision | Reason | Revisit when |
|---|---|---|---|
| D001 | Name: TurbineGuard; folder 01_TurbineGuard_Predictive_Maintenance | Recognisable name with portfolio order and purpose | User requests another name |
| D002 | NASA FD001 is the primary dataset | Run-to-failure histories support RUL and recent sensor features | Acquisition fails or problem scope changes |
| D003 | RUL regression is primary; anomaly score is separate | Anomaly scores do not establish time to failure or probability | Different labelled data support a new task |
| D004 | 80/20 engine split; five inner grouped folds; NASA test held out | Prevent same-engine window leakage | New dataset/protocol version |
| D005 | Fixed 20-cycle feature window; reject short history | Simple shared training/serving contract | Development-only evidence warrants another window |
| D006 | Default cap 125; metrics use uncapped true RUL | Test a common piecewise target without concealing long-RUL errors | Declared sensitivity study informs a new protocol |
| D007 | 30-cycle horizon and top-20% capacity | Concrete illustrative maintenance policy | Real stakeholder requirements become available |
| D008 | CPU-first classical models and bounded tuning | Focus on scientific validity and usable delivery | A justified separate extension |
| D009 | Local CLI/API/container MVP; dashboard/cloud later | Deliver one complete scoring workflow first | Core gates pass and user requests extension |
| D010 | No final refit on validation for v0.1.0 | Preserve evaluated model/cutoff behaviour | New release with separately designed calibration |
| D011 | Markdown is the execution contract; task status in tasks/todo.md | Work can resume without conversation history | User selects another tracker |
| D012 | Model promotion is separate from engineering completion | Failed ML targets must remain visible | Never remove this distinction |
| D013 | Verify dependency pins during T01 | Documentation does not establish local compatibility | Installed APIs or supported Python change |

## 2. Change entry format

Append rather than erase earlier reasoning:

```text
ID and date:
Question:
Evidence examined:
Decision and alternatives:
Effect on requirements/data/protocol:
Documents changed:
Verification or follow-up task:
```

Routine naming or formatting adjustments do not need a new decision. Data changes, target/threshold changes, feature timing, model selection, dependencies and release scope do.

---

## 3. Improvement Plan Decisions: 6 October 2026

### D014: Test Isolation for Model-Writing Tests
- **Date**: 2026-10-06
- **Question**: How to prevent unit and integration tests from contaminating production model artifacts in `models/staging/`?
- **Evidence examined**: Prior synthetic test execution in `tests/test_anomaly.py` wrote synthetic 4-feature mock models directly to `models/staging/`, overwriting real trained models.
- **Decision**: Enforce `tmp_path` fixture injection into `Config.models_dir` for all model-fitting tests (`tests/test_anomaly.py`). Add cryptographic verification test in `tests/test_artifacts_isolation.py` ensuring `models/staging`, `models/v0.1.0`, and `reports/` cannot be altered by the test suite.
- **Effect on requirements/data/protocol**: Guarantees unit test suite runs in complete isolation from persistent artifacts.
- **Documents changed**: `docs/08_DECISIONS_LOG.md`, `tests/test_anomaly.py`, `tests/test_artifacts_isolation.py`.
- **Verification**: `pytest tests/test_artifacts_isolation.py` checks file hash invariance across test runs.

### D015: Promotion Requires Dual Gate Verification (G3 and G4)
- **Date**: 2026-10-06
- **Question**: When can a model bundle be marked `"promotion_status": "promoted"`?
- **Evidence examined**: Previously, `freeze` marked bundles promoted if Gate G3 (RUL RMSE) passed even if Gate G4 (Anomaly False Flags) failed or had stale hashes.
- **Decision**: Update `src/turbineguard/artifacts.py` so promotion strictly requires BOTH Gate G3 == PASS AND Gate G4 == PASS. Any gate failure, inconclusive state, or hash mismatch forces `promotion_status: "experimental"`.
- **Effect on requirements/data/protocol**: Prevents shipping bundles with unverified or failing anomaly components.
- **Documents changed**: `src/turbineguard/artifacts.py`, `tests/test_artifacts_isolation.py`.
- **Verification**: Automated test proves synthetic failed-G4 receipt yields `"experimental"`.

### D016: Release Candidate v0.2.0 Creation & v0.1.0 Preservation
- **Date**: 2026-10-06
- **Question**: How to publish the newly validated models without erasing audited baseline evidence?
- **Evidence examined**: `models/v0.1.0/` and `reports/official_test_predictions.csv` reflect the historical audit baseline.
- **Decision**: Preserve `models/v0.1.0/` and historical reports in perpetuity. Freeze the clean, 60-sensor Isolation Forest and XGBoost champion as `models/v0.2.0/`. Output official holdout metrics to `reports/official_test_metrics_v0.2.0.json` and predictions to `reports/official_test_predictions_v0.2.0.csv`.
- **Effect on requirements/data/protocol**: Transparent traceability between historical audit state and promoted release candidate.
- **Documents changed**: `docs/MODEL_CARD.md`, `README.md`, `models/v0.2.0/*`.
- **Verification**: Bundle manifests verify all checksums for both versions.

### D017: Honest Disclosure of Near-Failure Recall Tradeoff
- **Date**: 2026-10-06
- **Question**: Should RUL decision boundaries be shifted to hit the $\ge 80\%$ recall target on official test data?
- **Evidence examined**: 5-fold CV development OOF analysis showed that 6 test engines missed at RUL 21–29 had predicted RUL 30.4–45.3 due to gradual early degradation curves. Artificially shifting thresholds creates excessive false alarms on intermediate units (reducing precision).
- **Decision**: Maintain un-skewed RUL regression predictions and report the 76.0% recall vs 80.0% target honestly without retrospective test-set tuning.
- **Effect on requirements/data/protocol**: Upholds scientific integrity and strict separation between test holdout and model fitting.
- **Documents changed**: `docs/12_IMPROVEMENT_PLAN.md`, `docs/MODEL_CARD.md`, `README.md`.
- **Verification**: Grouped CV residual analysis documented in session logs.

