# 01. Problem and objectives

## 1. Problem statement

A maintenance planner receives many noisy sensor readings but needs a practical answer: **which engine should be inspected first, and how close might it be to its failure endpoint?** A single unusual reading does not establish a failure, and an average prediction error does not reveal whether estimates dangerously overstate remaining life.

TurbineGuard converts recent sensor histories into remaining-life estimates, separate anomaly scores and an ordered inspection list. Its benchmark tests whether those estimates generalise to engines excluded from training.

## 2. Evidence and assumptions

- NASA provides simulated run-to-failure engine histories and separately truncated test histories. This supports retrospective RUL evaluation. See [sources S1-S3](10_SOURCES.md).
- NASA's prognostics paper discusses remaining-life estimates and asymmetric penalties for late predictions. This motivates reporting overestimation alongside RMSE.
- The persona, inspection capacity and maintenance horizon below are project assumptions. We have no interviews, factory records, maintenance costs or downtime records.
- The spreadsheet maps this topic to sensor modelling and maintenance skills. It does not establish that any named employer uses this dataset or currently has a relevant vacancy.

## 3. Users and decisions

| User | Problem | Useful output |
|---|---|---|
| Maintenance planner | Too many assets to review at once | Ranked list with estimated RUL and anomaly flag |
| Reliability engineer | Needs to investigate suspicious trends | Top model contributions and recent sensor context |
| ML engineer | Needs trustworthy and repeatable results | Versioned bundle, evaluation report, drift diagnostics |

The planner scenario is fictional. FD001 represents turbofan engines; transferring to semiconductor equipment would require different data and a new evaluation.

## 4. Objectives

Targets below are predeclared design goals, not achieved results or published benchmark claims.

| ID | Objective | Measurement and target |
|---|---|---|
| O1 | Estimate remaining life on unseen engines | Report endpoint MAE, RMSE, bias, overestimation rate and NASA score; aspiration: official-test RMSE <=25 cycles |
| O2 | Show that sensors add value | Validation RMSE at least 10% below both the constant and age-only baselines; report paired engine bootstrap intervals |
| O3 | Identify engines near a maintenance horizon | For true RUL <=30 cycles, report precision/recall of predicted RUL <=30; aspiration: recall >=0.80 and precision >=0.60 |
| O4 | Demonstrate capacity-based prioritisation | Rank one latest snapshot per engine; report precision, recall and lift at the top 20% against declared baselines |
| O5 | Flag unfamiliar sensor behaviour honestly | Separate anomaly score from RUL; validation false-flag rate <=5% on the high-RUL proxy subset; report near-failure proxy recall without calling it true anomaly accuracy |
| O6 | Make individual predictions understandable | Provide top three signed model contributions for a promoted nonconstant model, plus a global feature summary |
| O7 | Reproduce outputs | Same data hashes, split, config, lockfile and seed reproduce endpoint RMSE within 0.01 cycles on the same environment |
| O8 | Deliver usable engineering | Shared batch/API predictions within 1e-6 cycles, valid Docker smoke test, CI checks and a drift report |

## 5. Business interpretation

1. **Inspection prioritisation:** a low estimated RUL places an engine earlier in the worklist. An anomaly flag adds investigation context.
2. **Explainability:** contributions describe what influenced the model. They do not establish mechanical root causes.
3. **Monitoring:** drift identifies changes to investigate. It does not prove model accuracy declined.

The default 30-cycle horizon and 20% inspection capacity are illustrative policies. Cycles are not days, hours or money. This project cannot demonstrate avoided downtime or financial savings without real intervention outcomes and costs.

## 6. Success and failure

- Engineering completion requires reproducibility, input validation, meaningful tests, serving and honest reports.
- Model promotion separately requires the validation gates in [validation and release](06_VALIDATION_AND_RELEASE.md).
- If targets are missed, keep the measured results and document failure analysis. The project can be an honest completed benchmark study while the model remains unpromoted.
- Final test results assess generalisation. They must not drive another tuning round disguised as the same held-out evaluation.

## 7. Portfolio value

This project adds grouped time-series validation, causal rolling features, regression, anomaly detection, asymmetric error analysis and model monitoring to the existing churn portfolio. A recruiter should be able to trace a claim from README to a report, configuration and reproducible run.
