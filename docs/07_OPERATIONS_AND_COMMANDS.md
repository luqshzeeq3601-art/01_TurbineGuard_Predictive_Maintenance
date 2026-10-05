# 07. Operations and planned commands

## 1. Command status

These commands define the interface to implement during the task list. They are **not runnable yet** because this project currently contains Markdown files only. Run them from the project root after the relevant task has created the entry point.

Use PowerShell. T01 records the actual Python location; examples use python after activating the project environment. Do not require Make or a global installation.

## 2. Environment setup: T01

```powershell
Set-Location 'C:\Users\ZeeqRyz\Desktop\Ai-ML\Machine Learning Projects\01_TurbineGuard_Predictive_Maintenance'
python --version
git --version
docker version
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.lock
python -m pip install -e . --no-deps
python -m pytest tests/test_smoke.py -q -p no:cacheprovider --basetemp .pytest-tmp
```

For the first setup, T01 resolves and writes requirements.lock before its installation command. Later setups install that existing lock. If activation is restricted, invoke .venv\Scripts\python.exe directly rather than changing machine execution policy. Docker may be unavailable; record that prerequisite while continuing data/model work.

## 3. Data and baseline path: T02-T05

```powershell
python scripts/download_data.py --dataset FD001 --output data/raw/FD001
python -m turbineguard.cli prepare --config configs/default.yaml
python -m turbineguard.cli train --config configs/default.yaml --experiment baselines
```

download_data.py records provenance and enforces safe nested extraction. prepare writes quality/split/snapshot manifests and separate development/validation features. Training does not open official test labels.

## 4. Experiments and validation: T06-T09

```powershell
python -m turbineguard.cli train --config configs/default.yaml --experiment xgboost
python -m turbineguard.cli train --config configs/default.yaml --experiment anomaly
python -m turbineguard.cli evaluate --config configs/default.yaml --split validation
python -m turbineguard.cli policy-report --config configs/default.yaml --split validation
```

The xgboost entry point also writes the champion decision from the declared candidate pool. Evaluation writes gate status and confidence intervals. A failure is a recorded result, not a reason to silently change the protocol.

## 5. Freeze and final holdout: T10

```powershell
python -m turbineguard.cli freeze --config configs/default.yaml --version v0.1.0
python -m turbineguard.cli evaluate --config configs/default.yaml --split official-test --bundle models/v0.1.0 --release-id v0.1.0 --allow-heldout-evaluation
```

freeze records pass/fail/inconclusive gate outcomes and promotion status even for an experimental bundle. The final command requires a frozen bundle and persists a release evaluation record. It cannot train, tune, alter policy or overwrite an existing official evaluation without an explicit documented rerun path. Write and hash predictions before loading official labels.

## 6. Explanations and batch scoring: T11/T13

```powershell
python -m turbineguard.cli explain --bundle models/v0.1.0 --split validation --output reports/explanations
python -m turbineguard.cli score --bundle models/v0.1.0 --input data/raw/FD001/test_FD001.txt --input-format cmapss --output reports/worklist.csv
```

score supports cmapss whitespace input and a headered CSV input-format csv. Both normalise to the same schema and create one endpoint row per engine. It never opens RUL_FD001.txt. Worklist CSV is a local artifact, not a business intervention record.

## 7. API and tracking: T12

```powershell
$env:TURBINEGUARD_BUNDLE = 'models/v0.1.0'
python -m uvicorn api.main:app --host 127.0.0.1 --port 8000
```

In a separate terminal:

```powershell
Invoke-RestMethod 'http://127.0.0.1:8000/health'
Invoke-RestMethod 'http://127.0.0.1:8000/ready'
Invoke-RestMethod 'http://127.0.0.1:8000/model-info'
python -m mlflow ui --backend-store-uri sqlite:///mlruns.db --host 127.0.0.1 --port 5000
```

The CLI uses the same local MLflow SQLite tracking URI and a local mlruns/ artifact directory. Verify the pinned MLflow version and Windows URI behaviour in T05. The API documentation is available locally at http://127.0.0.1:8000/docs after the server starts. A POST example is generated from a valid synthetic fixture in T12; do not invent measured outputs in documentation.

## 8. Monitoring and checks: T14/T16

```powershell
python -m turbineguard.cli drift --bundle models/v0.1.0 --current reports/worklist_features.parquet --output reports/drift
python -m turbineguard.cli drift-controls --bundle models/v0.1.0 --output reports/drift/controls
python -m ruff check src api tests scripts
python -m pytest tests -q -p no:cacheprovider --basetemp .pytest-tmp
```

T13 scoring writes worklist_features.parquet alongside worklist.csv, without labels. Monitoring uses the bundle's frozen reference and reports insufficient_data below 30 engines. The control command must save perturbation parameters and results.

## 9. Docker smoke test: T15

```powershell
docker build -t turbineguard:local .
$tgBundlePath = (Resolve-Path 'models/v0.1.0').Path
docker run --rm -p 127.0.0.1:8000:8000 --mount "type=bind,source=$tgBundlePath,target=/app/models/v0.1.0,readonly" -e TURBINEGUARD_BUNDLE=/app/models/v0.1.0 turbineguard:local
```

Runtime model bundles are mounted read-only rather than copied from ignored local artifacts into the image. Docker serves on 0.0.0.0 inside the container while the host port binds to localhost. Check readiness and send the synthetic POST fixture. Stop the foreground container with Ctrl+C after verification.

## 10. Recovery rules

- Missing Python: find the installed/bundled interpreter and record it; do not assume py exists.
- Pytest temp/cache ACL failure: use the repo-local flags shown above.
- Download or checksum failure: preserve the error, recheck the NASA-linked URL, do not substitute unverified data.
- Version incompatibility: reproduce in the isolated environment, fix pins and record the dependency change before retraining.
- Corrupt/incompatible bundle: readiness stays 503; never create default predictions as a fallback.
- Slow tuning: reduce remaining search within the predeclared budget and record what ran; never expand to a GPU/deep-learning project automatically.
- Docker unavailable: mark container gate unverified, continue independent work and leave the concrete prerequisite.
