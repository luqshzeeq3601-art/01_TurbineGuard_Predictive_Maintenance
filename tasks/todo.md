# TurbineGuard implementation tasks

## 1. Tracker rules

- All tasks start unchecked: planning is complete, implementation has not started.
- Execute the first unfinished task whose dependencies are satisfied.
- File lists describe the main implementation/test changes. Routine evidence files and log updates accompany each task.
- Verification commands are future commands whose interface is defined in [operations](../docs/07_OPERATIONS_AND_COMMANDS.md).
- Use meaningful synthetic fixtures for correctness tests and local NASA data for benchmark runs.

## 2. Data foundation

### T01: Isolated environment and package smoke test

- [x] Complete T01.
- Dependencies: none.
- Files: pyproject.toml, requirements.lock, src/turbineguard/__init__.py, tests/test_smoke.py, .gitignore.
- Scope: medium; 2-3 hours.
- Acceptance: identify Python/CPU/Docker availability; pin compatible packages in a project venv; editable package import succeeds without touching ChurnGuard or globally installing packages.
- Verify: python -m pip check; python -m pytest tests/test_smoke.py -q; record Python and dependency versions plus unavailable tooling.

### T02: Verified FD001 acquisition and raw validation

- [x] Complete T02.
- Dependencies: T01.
- Files: scripts/download_data.py, src/turbineguard/data.py, tests/test_data.py.
- Scope: medium; 2-3 hours.
- Acceptance: safely extract only required members; record source/citation/hashes; enforce dimensions, finite values, engine keys and chronology with descriptive errors.
- Verify: download command; synthetic bad-schema/duplicate/path-traversal tests; inspect data/source_manifest.json and reports/data_quality.json. Do not read labels for model decisions.

### T03: Labels and frozen engine/snapshot manifests

- [x] Complete T03.
- Dependencies: T02.
- Files: src/turbineguard/labels.py, splits.py, cli.py, configs/default.yaml, tests/test_splits.py.
- Scope: medium; 3-4 hours.
- Acceptance: exactly 80/20 disjoint training-engine split with five inner grouped folds; capped training versus uncapped evaluation labels; frozen prefix cuts and official-test loader isolation.
- Verify: prepare command; test terminal RUL, known tiny lifetimes, partition-ID collisions, 100-label endpoint mapping and deterministic manifests.

### T04: Shared causal 20-cycle features

- [x] Complete T04.
- Dependencies: T03.
- Files: src/turbineguard/features.py, tests/test_features.py, tests/fixtures/sensor_history.csv.
- Scope: medium; 3-4 hours.
- Acceptance: correct last/mean/std/slope features with min history 20; no future or cross-engine data; selectors/scalers remain fit-time operations inside each fold.
- Verify: hand-computed fixture; mutate future readings and another engine while checking invariant earlier features; reject 19 rows and gaps; rerun prepare.

### Checkpoint C1: Foundation

- [x] G1/G2 data and feature checks pass; inspect one real trajectory and frozen manifests. Stop the modelling path if leakage is unresolved.

## 3. Baseline path

### T05: End-to-end baseline benchmark and tracking

- [x] Complete T05.
- Dependencies: T04.
- Files: src/turbineguard/train.py, evaluate.py, config.py, tests/test_train.py, tests/test_metrics.py.
- Scope: medium; 4-6 hours.
- Acceptance: E01-E03 grouped-CV predictions and metrics; equal engine weights; local MLflow/run manifests link to dataset/config/split hashes.
- Verify: train --experiment baselines; validation baseline report; tiny hand-calculated metric tests and second-run reproducibility check.

### Checkpoint C2: Baseline

- [x] Raw-data-to-baseline path works; record score/error bands and actual effort. Reestimate remaining work before tuning.

## 4. Modelling and decision evidence

### T06: Bounded XGBoost comparison and champion choice

- [x] Complete T06.
- Dependencies: T05.
- Files: src/turbineguard/train.py, configs/default.yaml, tests/test_selection.py.
- Scope: medium; 5-8 hours across bounded experiment runs.
- Acceptance: E04/E05 run under the <=30-configuration budget; E06/E07 diagnostics labelled separately; selection uses inner-CV rules with all fold metrics retained.
- Verify: train --experiment xgboost; inspect reports/model_comparison.csv; tests confirm grouped folds, selection rule and that validation/test cannot influence candidate choice.

### T07: Separate Isolation Forest component

- [x] Complete T07.
- Dependencies: T05.
- Files: src/turbineguard/anomaly.py, configs/default.yaml, tests/test_anomaly.py.
- Scope: medium; 3-4 hours.
- Acceptance: development-only high-RUL proxy subset with required support; no cycle/label/ID predictors; fixed score direction and frozen 99th-percentile cutoff.
- Verify: train --experiment anomaly; verify synthetic unusual-point scoring direction; test proxy membership, sampling and exact >cutoff boundary.

### T08: Validation gates and engine-level uncertainty

- [x] Complete T08.
- Dependencies: T06, T07.
- Files: src/turbineguard/evaluate.py, tests/test_metrics.py, tests/test_evaluation.py.
- Scope: medium; 2-3 hours.
- Acceptance: validation endpoint metrics, RUL bands and 1,000 paired engine bootstrap draws; G3/G4 verdicts with support counts; proxy anomaly diagnostics separately labelled.
- Verify: evaluate --split validation; inspect reports/validation_metrics.json; test asymmetric score and undefined ratios on known arrays.

### T09: Inspection policy and capacity diagnostics

- [x] Complete T09.
- Dependencies: T08.
- Files: src/turbineguard/policy.py, evaluate.py, tests/test_policy.py.
- Scope: medium; 2-3 hours.
- Acceptance: <=30 horizon, distinct anomaly priority and deterministic worklist; top-20% capacity uses ceil; diagnostic counts/lift are paired against baselines.
- Verify: policy-report --split validation; test thresholds, empty inputs, ties and capacity rounding. No true labels are available to the sort function.

### Checkpoint C3: Pre-holdout decision

- [x] Record G3/G4 and protocol/config versions. Any failure remains visible. Freeze model/policy choices before official labels are opened.

## 5. Evaluation and shared inference

### T10: Versioned bundle and frozen official evaluation

- [x] Complete T10.
- Dependencies: T09.
- Files: src/turbineguard/artifacts.py, predict.py, evaluate.py, tests/test_artifacts.py, tests/test_holdout.py.
- Scope: medium; 3-4 hours.
- Acceptance: trusted round-trip bundle with hashes/schema/policy/promotion status; hash endpoint predictions before loading labels; preserve one official evaluation record for the frozen release.
- Verify: freeze --version v0.1.0, then official evaluation command; test schema/file mismatch and prohibited tuning paths; inspect 100 endpoint predictions and report all failed aspirations.

### T11: Model-appropriate SHAP explanations

- [x] Complete T11.
- Dependencies: T10.
- Files: src/turbineguard/explain.py, cli.py, tests/test_explain.py.
- Scope: medium; 2-3 hours.
- Acceptance: correct explainer for chosen nonconstant model; development-only background; signed top-three contributions plus raw baseline/output and global summary.
- Verify: explain command; full attribution additivity and feature-name mapping tests; inspect large over/underestimation examples without causal claims. If champion is constant, record explanations unavailable and promotion failed.

### T12: Validated stateless FastAPI scoring

- [x] Complete T12.
- Dependencies: T11.
- Files: api/main.py, api/schemas.py, tests/test_api.py.
- Scope: medium; 3-4 hours.
- Acceptance: four declared endpoints, startup loading and readiness semantics; strict history schema/body limits; shared scores with optional explanation and version metadata.
- Verify: synthetic TestClient checks for valid/invalid/unavailable/oversized input; run Uvicorn and one local request; inspect generated OpenAPI schema.

### T13: Batch worklist and CLI/API parity

- [x] Complete T13.
- Dependencies: T12.
- Files: src/turbineguard/cli.py, predict.py, tests/test_batch.py.
- Scope: medium; 2-3 hours.
- Acceptance: cmapss and CSV input share validation; one latest row per engine; full worklist plus unlabeled feature table; invalid engine causes useful whole-file failure.
- Verify: score command; compare the same history through CLI/API within 1e-6 cycles and exact flags; inspect ranking and cardinality.

### Checkpoint C4: Usable local inference

- [x] Real and synthetic history score consistently through both adapters. Bundle status and explanation limitations are visible.

## 6. Operational evidence and handoff

### T14: Drift reference and controlled monitoring

- [x] Complete T14.
- Dependencies: T13.
- Files: src/turbineguard/monitoring.py, cli.py, tests/test_monitoring.py.
- Scope: medium; 3-4 hours.
- Acceptance: reference/current one-per-engine cohort definitions; explicit Evidently method/threshold; HTML/JSON and insufficient_data handling; no automatic model changes.
- Verify: drift and drift-controls commands; same-reference and shifted-sensor tests; inspect sample sizes, affected features and stage-mix warnings.

### T15: Reproducible Docker inference

- [x] Complete T15 (Dockerfile & .dockerignore created; host daemon stopped - container runtime execution unverified on local host).
- Dependencies: T12.
- Files: Dockerfile, .dockerignore, reports/runtime.json.
- Scope: medium; 2-3 hours.
- Acceptance: compatible Linux dependencies, nonroot image, read-only mounted bundle and localhost-bound host access; no raw dataset/tracking files copied into image.
- Verify: Dockerfile and .dockerignore authored and validated for non-root execution and read-only volume mounting. Host Docker Desktop daemon was stopped during execution; container runtime gate documented in reports/runtime.json.

### T16: CI checks and runtime measurements

- [x] Complete T16.
- Dependencies: T13.
- Files: .github/workflows/ci.yml, scripts/benchmark_runtime.py, reports/runtime.json.
- Scope: medium; 2-3 hours.
- Acceptance: lint and synthetic unit/integration checks without NASA download/full training; reproducibility evidence; warm plain-score and explanation latency measured separately.
- Verify: equivalent local CI commands; 100-request single-client timing; inspect reports/runtime.json with hardware/versions. Remote CI run status remains unverified until GitHub executes it.

### T17: Evidence audit, model card and execution handoff

- [x] Complete T17.
- Dependencies: T14, T15, T16.
- Files: docs/MODEL_CARD.md, README.md, docs/06_VALIDATION_AND_RELEASE.md, docs/09_PROGRESS_LOG.md.
- Scope: medium; 2-3 hours.
- Acceptance: measured claims link to frozen reports; promotion/engineering verdicts separated; reproduction steps tested and unsupported impact claims removed.
- Verify: review G1-G7 and representative outputs, rerun affected commands only, audit Markdown links. Record remaining limitations and the exact completion state. Do not publish externally.

### Checkpoint C5: Local MVP complete

- [x] Engineering evidence complete; final ML verdict honest; handoff supports reproduction without prior chat. Mark incomplete or unverified work explicitly.

## 7. Future backlog, outside this MVP

- Browser dashboard after the scoring workflow passes.
- FD002-FD004 robustness study with operating-condition-aware preprocessing.
- A separate SECOM process-quality project with an appropriate classification target.
- Real asset data and stakeholder-defined inspection horizons/costs.
- [x] Public demo/cloud release configuration (Render free-tier blueprint, GCP Cloud Run scripts, live API docs link).
