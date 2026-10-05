# 02. Product requirements document

## 1. Product definition

- **Name:** TurbineGuard.
- **Version:** planning v0.1, 5 October 2026.
- **Owner:** user; implementation combines ML engineering and project management.
- **Problem:** recent sensor histories are difficult to turn into a defensible inspection priority.
- **Value:** one validated pipeline produces an RUL estimate, anomaly flag and interpretable worklist.
- **Delivery environment:** local CPU, Windows development, Linux Docker serving.

## 2. Primary user journey

1. A planner supplies a CSV of engine histories or an API request containing one engine's recent history.
2. The system validates fields and chronology, then uses the last 20 consecutive cycles.
3. The system returns estimated RUL in cycles, a horizon flag and a separate anomaly score/flag.
4. Batch output contains one row per engine, ordered by ascending estimated RUL, then anomalous first for ties, then engine ID.
5. The reliability engineer optionally requests the top three model contributions.
6. The ML engineer compares a new batch with a frozen reference and reviews a drift report.

## 3. User stories

| ID | Story | Acceptance |
|---|---|---|
| US1 | As a planner, I want to inspect the most urgent engines first | Batch output contains each valid engine exactly once and follows the declared sort |
| US2 | As a reliability engineer, I want to understand a low estimate | Explanation includes signed contributions, baseline and model version |
| US3 | As an ML engineer, I want to reproduce a reported result | Bundle links to hashes, split manifest, configuration, dependency lock and run |
| US4 | As an operator, I want invalid data identified | Errors name the invalid field or engine without producing a misleading prediction |
| US5 | As an ML engineer, I want to notice changed data | Monitoring reports reference/current definitions, drift method, sample size and limitations |

## 4. Functional requirements

| ID | Priority | Requirement | Verification |
|---|---|---|---|
| FR1 | Must | Acquire FD001 from the NASA-linked archive; preserve provenance and hashes | Source manifest and schema report |
| FR2 | Must | Split training data by engine; freeze official test | Disjoint engine manifest and leakage tests |
| FR3 | Must | Generate last-value, rolling mean/std and slope features using only current/past cycles | Future-perturbation and cross-engine isolation tests |
| FR4 | Must | Compare constant, age-only, Ridge and bounded XGBoost experiments | Reproducible comparison report with fold scores |
| FR5 | Must | Predict continuous RUL; expose threshold flag when estimate <=30 | Shared batch/API policy tests |
| FR6 | Must | Fit Isolation Forest on a high-RUL proxy training subset; keep its score separate | Frozen cutoff and score-direction tests |
| FR7 | Must | Produce one latest prediction per engine and a ranked batch CSV | Duplicate, order and cardinality checks |
| FR8 | Must | Explain promoted RUL model with appropriate SHAP method | Contribution additivity and feature mapping checks |
| FR9 | Must | Serve /health, /ready, /model-info and /predict | FastAPI contract and readiness tests |
| FR10 | Must | Package a versioned model bundle, run manifest and Docker image | Clean-environment inference smoke test |
| FR11 | Must | Produce Evidently drift HTML plus machine-readable summary | No-change and synthetic-change checks |
| FR12 | Must | Track experiments and verify changes in CI | Local MLflow evidence and CI workflow checks |
| FR13 | Later | Add a browser dashboard for histories and rankings | Separate follow-up specification |
| FR14 | Later | Deploy externally or extend FD002-FD004 | Separate approval and dataset-specific validation |

## 5. Nonfunctional requirements

| ID | Requirement | Acceptance |
|---|---|---|
| NFR1 | Reproducibility | Seed 42 by default; artifact/config/data hashes; same-environment RMSE tolerance 0.01 cycles |
| NFR2 | Responsiveness | Warm local /predict p95 <200 ms without explanations; measure 100 requests, 20-cycle input, single concurrent client |
| NFR3 | Explanation latency | Record explanation p95 separately; aspiration <1 second, never include it in the plain-scoring claim |
| NFR4 | Reliability | Missing or incompatible bundle makes /ready return 503; /health still reports process health |
| NFR5 | Validation | Reject missing/unknown fields, nonfinite values, short histories and nonconsecutive cycles; request limits in technical design |
| NFR6 | Maintainability | Typed reusable package, tested contracts, no duplicated notebook-only inference |
| NFR7 | Resource use | CPU only; <=30 XGBoost configurations across five inner folds; record hardware and run time |
| NFR8 | Input containment | No file path or URL supplied to prediction endpoints; no public model uploads or automatic artifact deserialization |

## 6. Output definitions

- **estimated_rul_cycles:** nonnegative continuous estimate; no conversion to wall-clock time.
- **within_horizon:** estimate <=30 cycles; a thresholded estimate, not a calibrated probability.
- **anomaly_score:** negative Isolation Forest score_samples; higher means more unusual relative to the fitted proxy reference.
- **anomaly_flag:** score exceeds the frozen training-derived cutoff.
- **inspection_priority:** review_soon when within_horizon; otherwise investigate when anomalous; otherwise routine_review. These labels do not certify equipment condition.
- **top_contributions:** three largest absolute signed contributions to the raw RUL prediction, when requested.
- Include engine ID, latest cycle, history length, dataset ID, bundle version and policy version.

## 7. MVP boundaries

Included: FD001, offline training, batch CSV, stateless prediction, explanations, container, CI, local experiment tracking, monitoring and evaluation reports.

Excluded: maintenance actuation, aviation certification, real factory claims, failure probability, causal diagnosis, live PLC/IoT ingestion, Kafka, multi-tenant accounts, paid cloud resources, automated retraining, deep learning and financial ROI estimation.

## 8. Release decision

- Every Must requirement must have evidence or an explicit unresolved status.
- The engineering delivery and model promotion decisions are recorded separately.
- Target failures remain visible. No unsupported result is inserted into README or a resume.
- Public release is a later user-directed action. Local Docker and reproducible evidence complete the MVP delivery environment.
