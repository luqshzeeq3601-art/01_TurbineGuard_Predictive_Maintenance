# 12. TurbineGuard improvement plan

## 1. Goal and scope

Resolve the failed anomaly gate, missed recall target and incomplete release evidence identified in [the review](11_MODEL_PERFORMANCE_REVIEW.md). Preserve the useful RUL model and all original v0.1.0 results.

This is a proposed follow-up plan, not work already completed. The user's evaluation request authorized review and planning. Execute implementation only when the user requests it. Use tasks/todo.md as the task-status authority; reconcile its overclaimed completion states when follow-up work begins rather than maintaining a second completion tracker here.

Do not start by adding a dashboard, deep learning, more datasets or a larger hyperparameter search. First repair the evidence and artifact path.

## 2. Phase A: make verification and artifacts trustworthy

Affected original tasks: T01, T07, T08, T10, T16, T17.

### A1. Restore a compatible local verification environment

- Identify an available CPython 3.11 interpreter or explicitly resolve a compatible runtime in an isolated environment.
- Pin and record the versions used for the existing/new model; do not load cp311 extensions under Python 3.12 or silently cross library versions.
- Check imports, pip check and one package smoke test before claiming fresh model verification.
- Keep machine-local interpreter paths out of portable requirements files.

### A2. Isolate model-writing tests

- Give anomaly fitting tests a tmp_path staging directory through their Config; audit all other model/report-writing tests.
- Use synthetic fixtures generated inside test directories. Separate optional NASA integration tests from default CI.
- Hash real staging, frozen bundles and official reports before and after the synthetic test suite. Assert no changes.
- Add a regression case showing a synthetic anomaly test cannot replace a real candidate's staged model.

The appropriate pytest isolation mechanism is documented in [tmp_path guidance](https://docs.pytest.org/en/stable/how-to/tmp_path.html).

### A3. Bind promotion to model-specific evidence

- Use an explicit run ID/directory instead of a shared unverified latest-model location.
- Record training input hashes, development engine IDs, actual feature names, sampled proxy keys/counts, configuration, lockfile and model hashes in the run receipt.
- Require matching receipts and G1-G4 pass status for combined model promotion. Any fail/inconclusive/missing gate means experimental.
- Reject stale validation reports whose model/config/data hashes differ from the staged candidate.
- Refuse to overwrite an existing frozen version or original official evaluation. Store reports under their version/run ID.
- Correct README/model-card/task completion claims using verified outcomes; leave historical failure evidence visible.

**Exit condition:** synthetic tests do not alter real artifacts; a candidate with failed G4 cannot be marked promoted; stale receipts and duplicate freeze/evaluation attempts are rejected.

## 3. Phase B: recover and validate the real anomaly component

Affected original tasks: T07-T10.

- Use the existing real 80 development engines and the documented true-RUL >=100 proxy; do not use official test labels for training/cutoff selection.
- Sample at most 50 eligible rows per engine. Record actual proxy engine/row IDs and counts.
- Fit the documented sensor-only pipeline with training-proxy-fitted selectors/scaling. The present four-feature synthetic schema is not an acceptable substitute for a justified real-data feature selection.
- Start with the planned 200-tree Isolation Forest and training-proxy 99th-percentile cutoff. Do not tune a cutoff merely to manufacture a validation pass.
- Recompute validation full-trajectory high-RUL macro false flags and support counts, tied to the exact new model hash.
- If it still fails, use grouped development-only holdouts to compare justified sensor subsets/reference sampling choices. Keep proxy limitations explicit; expand search only with a documented reason.
- Do not refit the good RUL component merely to repair the separate anomaly artifact.

**Exit condition:** >=30 real fitting engines, >=10 validation proxy engines, macro high-RUL false flags <=5%; correct score direction, persisted schema and matched evidence. If these fail, retain experimental status and the actual failed numbers.

## 4. Phase C: improve near-failure recall through development evidence

Affected original tasks: T06, T08, T09 and T10.

Current gap: recall 76% versus target >=80%, while precision is 95% versus target >=60%. This indicates room to investigate recall, not a guaranteed easy improvement.

1. Generate grouped out-of-fold RUL predictions on the development engines. Inspect residuals and horizon confusion counts, especially near 30 cycles.
2. Compare the unchanged model with a small number of predeclared correction candidates: a development-derived residual correction, or a modest near-horizon/asymmetric training weighting change if calibration is insufficient.
3. Keep the maintenance horizon at 30 cycles. Fit/tune any correction only using development out-of-fold data, and evaluate its precision/recall, RMSE, NASA score and overestimation risk.
4. Select a candidate with the documented complexity preference, then check validation once for that version. Disclose that the original 20-engine validation set has already been used.
5. If additional cutoffs/long-RUL snapshots are needed, version the development evaluation protocol before running them. They are a protocol change, not an invisible replacement for the original benchmark.

**Exit condition:** development/validation evidence supports recall >=80% and precision >=60%, with the original RMSE aspiration retained and no hidden deterioration of the asymmetric error diagnostics. If no candidate meets the constraints, report the tradeoff instead of lowering the goal.

The principle of selecting decision policies outside test data is supported by [scikit-learn threshold guidance](https://scikit-learn.org/stable/modules/classification_threshold.html). This project uses RUL regression, so any residual correction needs its own explicit regression/policy implementation rather than blindly applying a classifier threshold tool.

Avoid immediately removing the 125-cycle cap: existing development E07 uncapped RMSE 18.84 is worse than E05's 14.49. Long-RUL improvements can be a separate bounded experiment after the failed primary gates are resolved.

## 5. Phase D: finish engineering and monitoring evidence

Affected original tasks: T12-T17.

| Work | Concrete action | Acceptance |
|---|---|---|
| Portable dependencies | Remove the absolute editable path; mark Windows-only dependencies appropriately; install the package separately | Clean Linux dependency installation plus pip check |
| CI | Default tests use only synthetic fixtures; NASA/bundle-dependent tests have a separate explicit integration lane | Default suite passes without ignored data, trained models or prior reports |
| API contract | Implement the PRD history schema, strict input/unknown-field rules, length/body caps and unavailable-bundle behaviour | Positive/negative contract tests cover the stated requirements |
| Docker | Build, run as nonroot with read-only model mount, check readiness and send a valid scoring request | Executed smoke evidence, not Dockerfile presence |
| Champion reproducibility | Fit the selected RUL configuration twice in isolated real-development runs | Independently calculated endpoint RMSE difference <=0.01 cycles with both receipts |
| Latency | Repeat plain/explanation measurements on the corrected version and record environment/sample protocol | Plain p95 <200 ms; explanation measured separately |
| Monitoring | Set the declared method/threshold and 50% dataset rule; count unique engine IDs; rebuild features after raw-history perturbation | No-change, +2 sigma changed-sensor and insufficient independent-engine controls pass; cohort/stage mix reported |
| Documentation | Reconcile model parameters, features, cutoffs, physical labels, uncertainty and gate claims | Every final claim matches version-specific evidence; no unsupported production/safety claim |

Version compatibility for persisted models is described in [scikit-learn persistence guidance](https://scikit-learn.org/stable/model_persistence.html). Platform markers and editable requirements are described in [pip requirements guidance](https://pip.pypa.io/en/stable/reference/requirements-file-format/).

## 6. Phase E: final versioned review and honest reporting

- Create a new release candidate version after Phase A-D evidence is complete. Preserve v0.1.0 bytes and its reports.
- Keep model promotion and engineering completion separate; do not set C5 complete while a mandatory Docker/API/monitoring check is unverified.
- Compare error, horizon metrics, capacity diagnostics and anomaly false flags with the audited baseline.
- Any reevaluation on the already-observed FD001 official test is a reused benchmark diagnostic, not a fresh independent holdout. A new untouched holdout or separately predeclared outer grouped validation is required for a new independent generalisation claim.
- Keep the original result and disclose protocol/data reuse. Never choose a model or correction by trying to recover the specific six observed test misses.
- Publish externally only if the user later requests it.

**Final stop condition:** all mandatory original gates have truthful evidence, or the report explicitly states which model targets remain unmet. An explanation of a failed metric does not turn it into a pass.

## 7. Suggested execution instruction

> Read AGENTS.md, docs/11_MODEL_PERFORMANCE_REVIEW.md, docs/12_IMPROVEMENT_PLAN.md and the original specifications. Execute Phase A first. Reconcile the task tracker, isolate model-writing tests, bind promotion to exact validated artifacts, and verify that original v0.1.0 files are unchanged. Record evidence and stop at Phase A's exit condition. Do not retrain or tune against the official test during this phase.
