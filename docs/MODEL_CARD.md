# TurbineGuard Model Card: Version v0.2.0

## 1. Model Details
- **Model Name**: TurbineGuard Turbofan Remaining Useful Life (RUL) Predictor & Sensor Anomaly Detector
- **Version**: `v0.2.0` (frozen on 2026-10-06; supersedes exploratory staging and audited `v0.1.0`)
- **Model Architecture**:
  - **RUL Predictor**: Extreme Gradient Boosting Regressor (`xgb.XGBRegressor`, `max_depth=4`, `n_estimators=300`, `learning_rate=0.03`, `subsample=0.8`, `colsample_bytree=0.8`, histogram tree method)
  - **Anomaly Detector**: Isolation Forest (`sklearn.ensemble.IsolationForest`, `n_estimators=200`, contamination="auto", standard scaled) with frozen 99th percentile anomaly cutoff (`0.5112`)
- **Input Features**: 64 causal features derived from 20-cycle rolling windows across 15 active non-constant sensors (`s02`, `s03`, `s04`, `s06`, `s07`, `s08`, `s09`, `s11`, `s12`, `s13`, `s14`, `s15`, `s17`, `s20`, `s21`), 3 operational settings, and cumulative operating cycle. 6 invariant flatline sensors (`s01`, `s05`, `s10`, `s16`, `s18`, `s19`) dropped during feature preprocessing.
- **Anomaly Feature Schema**: 60 sensor-only rolling statistics (`last`, `mean`, `std`, `slope` for the 15 active sensors); strictly excludes cycle, unit ID, operational settings, and RUL labels.
- **License**: MIT
- **Release Status**: **Promoted** (Validation Gates G1–G4 passed: Gate G3 RUL promotion PASS, Gate G4 Anomaly promotion PASS with 0.54% false-flag rate on validation high-RUL baseline engines).

---

## 2. Intended Use & Domain
- **Primary Domain**: Predictive maintenance and condition-based monitoring for commercial turbofan engines under constant sea-level operating conditions (NASA C-MAPSS FD001).
- **Intended Users**: Maintenance planners, reliability engineers, and fleet operations dispatchers.
- **Intended Workflow**:
  1. Ingestion of multi-cycle sensor logs per engine via CLI or REST API.
  2. Batch/API scoring to estimate remaining operational cycles and identify unusual sensor signatures.
  3. Ranking fleet assets into a capacity-constrained inspection worklist with priority categorizations (`review_soon`, `investigate`, `routine_review`).
  4. Inspection of local SHAP feature attributions to explain degradation drivers.

---

## 3. Training & Validation Data
- **Dataset**: NASA Commercial Modular Aero-Propulsion System Simulation (C-MAPSS) Dataset 1 (FD001).
  - **Provenance**: Downloaded directly from NASA Dash repository with SHA-256 integrity verification (`b430872c64bf5ded185f3b3dad62bcdc24aed56446e5ceba356dc59dad613693`).
- **Data Partitions**:
  - **Development Set**: 80 engines (15,007 feature records), partitioned into 5 engine-disjoint cross-validation folds (16 engines per fold).
  - **Validation Set**: 20 engines held out at engine-level (evaluated via frozen snapshot cutoffs at offsets [10, 30, 60, 90] cycles before failure).
  - **Official Benchmark Test Set**: 100 truncated test engine trajectories (13,096 raw cycles, 11,196 feature rows). Evaluated under frozen protocol with predictions hashed before label access.
- **Pre-processing**: Causal 20-cycle rolling statistics (mean, std, linear slope, latest reading) strictly backward-looking. RUL target capped at 125 cycles during training to model healthy-state plateau.

---

## 4. Empirical Evaluation & Benchmarks

### 4.1 Cross-Validation and Model Selection (5-Fold Grouped CV)
| Candidate Experiment | Model Family | Features | Mean CV RMSE | Selection Outcome |
| :--- | :--- | :--- | :--- | :--- |
| **E01** | Dummy Regressor (Median) | Constant baseline | 50.92 cycles (±0.85) | Baseline |
| **E02** | Ridge Linear Regression | Cycle (Age only) | 29.27 cycles (±0.67) | Baseline |
| **E03** | Ridge Linear Regression | Instantaneous sensors | 18.29 cycles (±0.61) | Baseline |
| **E04** | XGBoost | Instantaneous sensors | 16.87 cycles (±0.58) | Candidate |
| **E05** | **XGBoost (Champion)** | **Current + 20-cycle rolling** | **14.49 cycles (±0.59)** | **Selected Champion (1-SE Rule)** |
| **E06** | XGBoost (Diagnostic) | Rolling sensors without cycle | 15.41 cycles (±0.57) | Proves sensor degradation signal |
| **E07** | XGBoost (Diagnostic) | Trained on uncapped RUL | 18.84 cycles (±0.66) | Proves benefit of piecewise cap |

### 4.2 Official Benchmark Performance (100 Test Engines)
| Evaluation Metric / Band | Official Benchmark Measured | Target Aspiration / Reference | Status |
| :--- | :--- | :--- | :--- |
| **Overall RMSE** | **16.21 cycles** | Target $\le 25.0$ cycles | **ACHIEVED** |
| **Overall MAE** | **12.02 cycles** | | Informational |
| **Overall Bias** | **-1.64 cycles** | Safe underestimate skew | Informational |
| **NASA Asymmetric Score Total** | **383.71** | Mean penalty: 3.84 | Informational |
| **Near-Failure Band ($RUL \le 30$)** | **RMSE: 6.98 cycles, MAE: 5.17 cycles** | Most critical maintenance decisions | High accuracy |
| **Intermediate Band ($31 \le RUL \le 100$)** | **RMSE: 15.88 cycles, MAE: 12.42 cycles** | Scheduled overhaul planning | Good calibration |
| **Long-RUL Band ($RUL > 100$)** | **RMSE: 20.94 cycles, MAE: 16.69 cycles** | Plateau effect from 125-cycle cap | Expected cap effect |
| **Horizon Precision ($\le 30$ cycles)** | **95.0%** (19 TP / 20 flagged) | Target $\ge 60\%$ | **ACHIEVED** |
| **Horizon Recall ($\le 30$ cycles)** | **76.0%** (19 TP / 25 near-failure) | Target $\ge 80\%$ | **6 Missed Units (RUL 21-29)** |
| **Worklist Precision@k (top 20% = 20)** | **95.0%** (19 of 20 top engines near failure) | **3.80x lift** over 25.0% prevalence | Strong ranking |

*Note on Horizon Recall*: 6 test engines with true RUL between 21 and 29 cycles were predicted between 30.4 and 45.3 cycles due to gradual initial degradation curves. Investigation on development OOF data confirms a precision-recall tradeoff where shifting decision boundaries improves recall to ~76% but increases false positive inspections.

---

## 5. Explainability & Diagnostics
- **Explainability Engine**: TreeSHAP feature attributions computed natively via XGBoost booster.
- **Top Predictive Features**: High-pressure compressor outlet temperature (`s09_mean`), physical fan speed (`s14_mean`), bleed enthalpy (`s11_last`), and core speed (`s04_mean`).
- **Explanation Artifacts**: Inference responses can return top-3 signed feature contributions, raw baseline ($E[f(x)] = 80.70$), and impact direction.

---

## 6. Data Drift & Monitoring
- **Monitoring Tool**: Evidently (`evidently.legacy.report.Report` with `DataDriftPreset(drift_share=0.50)`).
- **Reference Cohort**: 80 frozen development engine snapshots (`monitoring_reference.parquet`).
- **Statistical Minimum**: Requires $\ge 30$ independent engine records; cohorts $< 30$ return `insufficient_data` status to avoid false alarms.
- **Drift Controls**: Verified with 100% pass on no-change control (0 drift), shifted-sensor control ($+2\sigma$ perturbation detected), and insufficient-data control.

---

## 7. Known Limitations & Operational Boundaries
1. **Piecewise Target Cap Plateau**: Healthy engines with $RUL > 125$ cycles will predict near $\approx 125$ cycles by design.
2. **Single Operating Regime (FD001)**: The model is validated solely for sea-level static flight profiles. It must NOT be applied to multi-regime datasets (FD002, FD004) without retraining.
3. **No Direct Actuator Control**: Outputs are decision-support recommendations for human maintenance scheduling; never interface directly with automated engine controls.
4. **Minimum History Requirement**: Requires at least 20 continuous operational cycles for causal feature window extraction.
