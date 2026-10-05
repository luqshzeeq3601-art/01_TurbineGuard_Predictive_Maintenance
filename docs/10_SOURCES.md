# 10. Sources and evidence register

## 1. Evidence checked on 5 October 2026

| ID | Source | What it supports |
|---|---|---|
| S0 | [Portfolio spreadsheet](../../Classical_ML_Portfolio_Plan_Malaysia.xlsx), Project Tracker A1 and A4:F4 | Project 1 theme, classical techniques and engineering additions |
| S1 | [NASA PCoE repository, dataset 6](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/) | Simulated turbofan degradation data, citation and NASA-linked download |
| S2 | [NASA-linked archive](https://phm-datasets.s3.amazonaws.com/NASA/6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip), nested CMAPSSData.zip/readme.txt | FD001 counts, operating/fault scope, train/test meaning and schema |
| S3 | [NASA technical paper](https://ntrs.nasa.gov/api/citations/20090029214/downloads/20090029214.pdf), performance evaluation section | Simulation context and asymmetric RUL error scoring |
| S4 | [scikit-learn grouped cross-validation](https://scikit-learn.org/stable/modules/cross_validation.html) | Group-aware model evaluation |
| S5 | [Isolation Forest API](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html) | Fitting/scoring methods and score conventions |
| S6 | [XGBoost Python API](https://xgboost.readthedocs.io/en/stable/python/python_api.html) | Regressor and version-specific training interface |
| S7 | [SHAP TreeExplainer](https://shap.readthedocs.io/en/latest/generated/shap.TreeExplainer.html) | Raw-output tree explanations and additivity |
| S8 | [FastAPI request bodies](https://fastapi.tiangolo.com/tutorial/body/) | Typed request schemas |
| S9 | [MLflow tracking](https://mlflow.org/docs/latest/ml/tracking/) | Local experiment parameters, metrics and artifacts |
| S10 | [Evidently drift customisation](https://docs.evidentlyai.com/metrics/customize_data_drift) and [preset reference](https://evidentlyai.github.io/evidently/api-reference/main/evidently/presets.html) | Reference/current data, configurable drift methods and report API |
| S11 | [Docker build practices](https://docs.docker.com/build/building/best-practices/) | Container build/runtime practices |

Library documentation was checked for planning. It does not establish installed-version compatibility; T01 pins versions and each implementation task verifies its actual API.

Dataset citation: A. Saxena and K. Goebel (2008), Turbofan Engine Degradation Simulation Data Set, NASA Prognostics Data Repository, NASA Ames Research Center.

## 2. Remote archive verification

The archive was fetched and inspected in memory, without saving dataset files. The outer ZIP contains a directory and nested CMAPSSData.zip. The required files were present and decoded for structural checks.

| Item | Observed SHA-256 |
|---|---|
| Outer archive | c9c5dec12a945a82e8bb4446589d7fb3cc057b5e5d81fa1a12e25ee9912ad3b2 |
| train_FD001.txt | 963b5e22825b34d8b21c69e1aeb4af3e647050eb672ee8834ba4b5d91d2de0f8 |
| test_FD001.txt | 3cda7109ce17bafb5443f2ac926cfcf88154b941b8c4cf95eb55d1ddd6f52851 |
| RUL_FD001.txt | a19c8ec94931949d0485bdc35118206e9c81c4547b422efb9cf86f4ceddbceca |

Observed counts: train 20,631 rows/26 fields, test 13,096 rows/26 fields, RUL 100 rows/one field. Full numeric quality checks, engine IDs, chronological checks and local reproducibility are T02-T04 work, not already verified.

The README identifies 100 training and 100 test trajectories, one operating condition and one degradation mode. Its final sensor-number line is inconsistent with the 26-field layout; the project uses 21 generic sensor fields and avoids unverified physical names.

## 3. Availability and usage limitations

- NASA's dataset 6 repository link successfully provided the archive during this planning session. Recheck availability at acquisition time.
- A [similarly named NASA simulator catalog](https://data.nasa.gov/dataset/c-mapss-aircraft-engine-simulator-data) displays an unavailable-download note. It is not the acquisition URL used here.
- NASA asks users to acknowledge the repository and data contributors. Preserve attribution; record archive terms during acquisition and do not infer unrestricted redistribution from an open download.
- Source links do not support any achieved RMSE, anomaly accuracy, employer adoption, salary claim or real business savings for TurbineGuard.

## 4. Assumption register

| Assumption | Owner | Verification/change point |
|---|---|---|
| Maintenance planner persona | PRD | Real stakeholder research in a future project |
| FD001-only initial scope | Decision D002 | T02 acquisition and quality review |
| 20-cycle history, target cap 125 | Data specification | Predeclared experiments; version any change |
| 30-cycle horizon, 20% inspection capacity | PRD | Stakeholder-defined policy if available |
| RMSE/precision/recall targets | Objectives | Validation/final reports, no guaranteed outcome |
| 45-65 hours and 10 hours/week | Project plan | Reestimate after T05 with actual capacity |
| Local CPU and Python 3.11 | Technical design | T01 compatibility check |

Distinguish source facts, engineering choices and observed results in every final report.
