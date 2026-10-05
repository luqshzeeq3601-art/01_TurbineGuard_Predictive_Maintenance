# 06. Validation and release gates

## 1. Gate structure

| Gate | Required evidence | Outcome |
|---|---|---|
| G1 Data | Source hashes, schema, target checks and frozen manifests | Pass/fail |
| G2 Leakage | Engine-disjoint folds, causal features, train-only fitting and test isolation | Pass/fail |
| G3 RUL promotion | Nonconstant champion validation RMSE >=10% better than both constant and age-only baselines | Pass/fail; intervals also reported |
| G4 Anomaly promotion | >=30 fitting engines; >=10 validation proxy engines; high-RUL macro false-flag rate <=5% | Pass/fail/inconclusive |
| G5 Engineering | Shared inference, API errors, readiness, bundle checks, Docker and CI | Pass/fail/unverified |
| G6 Monitoring | No-change control, shifted-sensor control and insufficient-data behaviour | Pass/fail |
| G7 Evidence | Frozen official evaluation, reports, model card and honest README | Complete/incomplete |

All ML targets are unmeasured at planning time. Official RMSE <=25 and horizon recall/precision targets are aspirations reported separately. They do not override data integrity or permit test-set tuning.

A model bundle is eligible for validation-based promotion when G1-G4 pass. The usable local MVP additionally requires G5-G7. A failed G3/G4 may still be served for local experimental demonstration with explicit unpromoted metadata. G7 records the final performance verdict without rewriting earlier evidence. Final reports must state the basis of promotion and any missed official-test aspirations.

## 2. Meaningful tests

| Area | Required cases |
|---|---|
| Parsing | 26-field schema, extra populated column, bad numeric input, missing values, duplicate keys |
| Chronology | Unsorted input, duplicates, cycle gaps and engine boundary isolation |
| Labels | Tiny known lifetimes, terminal RUL zero, test endpoint vector mapping and uncapped evaluation |
| Splits | Exact manifests, disjoint engines, no overlapping windows across folds |
| Features | Known 20-cycle mean/std/slope; changing future rows leaves earlier features unchanged; short history rejected |
| Fitted preprocessing | Validation values cannot affect selector/scaler; feature order survives save/reload |
| Models | Equal engine weights; deterministic seeded run; official labels cannot enter training loaders |
| Metrics | Small hand-computed MAE/RMSE/bias; asymmetric score penalises +10 more than -10; undefined precision/recall explicit |
| Policy | <=30 inclusive; score >cutoff; deterministic capacity rounding and ties |
| Explainability | Full SHAP additivity against raw output; clipped output clearly distinguished; feature names valid |
| Artifacts | Checksum/schema mismatch, missing file, compatible round trip, no untrusted upload path |
| API | Valid history, 19 rows, >200 rows, unknown field, nonfinite numeric, 1 MiB cap, unavailable bundle |
| Batch | Latest snapshot only, one output per engine, whole-file failure with useful errors |
| Monitoring | Same reference, shifted sensor, insufficient independent engines and quality failure |

Use synthetic fixtures in CI. Full NASA training is a deliberate local evaluation job, not a requirement for every commit. Avoid tests that merely duplicate implementation lines; test observable contracts and leakage risks.

## 3. Engineering acceptance

- Ruff checks and relevant pytest checks pass; critical data/feature/policy/API behaviour has tests.
- Batch/API RUL estimates differ by <=1e-6 cycles, with identical flags and versions.
- Same-environment repeated training meets 0.01-cycle RMSE tolerance; record any nondeterministic dependency.
- Warm local scoring p95 <200 ms on 100 sequential 20-cycle requests; report CPU, RAM, OS, versions and cold-start separately. Explanations have a separate latency report.
- Docker build and inference smoke test pass with a trusted bundle and synthetic request; nonroot runtime verified.
- CI workflow is present and locally equivalent checks pass. A remote green run is unverified until the workflow actually executes on GitHub.
- No global installation, hidden test data dependence or generated secrets in the intended release set.

No arbitrary coverage percentage substitutes for contract testing. Record coverage as supporting evidence if measured.

## 4. Required final evidence

| Artifact | Content |
|---|---|
| reports/data_quality.json | Provenance, schema, counts and split QA |
| reports/model_comparison.csv | Fold metrics, selected configuration and selection rationale |
| reports/validation_metrics.json | Baselines, champion, intervals and G3/G4 decisions |
| reports/official_test_metrics.json | Frozen endpoint metrics, bands, policy diagnostics and missed aspirations |
| reports/official_test_predictions.csv | One prediction per official test engine and bundle version |
| reports/explanations/ | Global and example explanation artifacts with units |
| reports/drift/ | Evidently HTML and summary JSON with control outcomes |
| reports/runtime.json | Reproducibility, latency and container evidence |
| docs/MODEL_CARD.md | Intended use, data, limitations, performance and promotion status |
| README.md | Reproduction instructions and measured claims linked to evidence |

Paths are future deliverables, not existing artifacts. Generated evidence stays local until a curated publication is requested.

## 5. Completion decisions

1. **Task done:** its criteria pass and the progress log links to evidence.
2. **Engineering complete:** G1/G2/G5/G6 pass, final evidence exists, and blockers are resolved.
3. **Model promoted:** G3/G4 also pass. Otherwise record experimental/unpromoted status.
4. **Portfolio ready locally:** G7 complete, results defensible, commands reproducible and limitations visible.
5. **Published:** a separate user-authorized release with actual external links verified.

Do not substitute planned commands, README claims or a built image for executed verification.
