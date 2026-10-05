# 00. Execution guide

## 1. What we are planning

**TurbineGuard** is a local maintenance decision-support demonstrator. It receives an engine's recent sensor history and returns a remaining-life estimate, an anomaly flag and an inspection priority. We will develop and evaluate it on NASA C-MAPSS FD001.

The owner is the user. The implementer acts as ML engineer and project manager: keep the scope small, validate the science, record evidence and finish tasks in dependency order.

## 2. Reading order

1. README.md and AGENTS.md: scope and working instructions.
2. 01_PROBLEM_AND_OBJECTIVES.md and 02_PRD.md: intended outcome.
3. tasks/plan.md and tasks/todo.md: order and acceptance criteria.
4. 03_DATA_SPEC.md and 05_EXPERIMENT_PLAN.md before touching data or models.
5. 04_TECHNICAL_DESIGN.md and 07_OPERATIONS_AND_COMMANDS.md before implementation.
6. 06_VALIDATION_AND_RELEASE.md before declaring a milestone complete.
7. 08_DECISIONS_LOG.md and 09_PROGRESS_LOG.md whenever resuming.

All files in this list are in docs/ unless a path is explicitly given. [Sources](10_SOURCES.md) provide supporting references.

## 3. Defaults already chosen

| Decision | Default |
|---|---|
| Problem | Estimate remaining life and support inspection prioritisation |
| Dataset | FD001 only; SECOM would be a separate quality-prediction project |
| Device | CPU, local Windows development |
| Primary models | Constant and age-only baselines, Ridge, XGBoost regressor |
| Anomaly model | Isolation Forest fitted on a high-RUL proxy subset |
| History | Last 20 consecutive cycles; insufficient history is rejected |
| Internal split | 80 development engines and 20 validation engines from NASA training data |
| Final holdout | NASA's separate 100-engine test set |
| Delivery | CLI, FastAPI, Docker, CI checks, local MLflow and Evidently report |
| Task tracker | tasks/todo.md only |
| Work estimate | 45-65 focused hours, roughly 5-7 weeks at 10 hours/week; assumptions only |

## 4. First execution session

1. Start **T01**: inspect available Python, Git and Docker; create the isolated environment and minimal package.
2. Do not train until T02-T04 have established provenance, labels and an engine-disjoint split.
3. Implement command entry points as needed by tasks. Commands in the operations document are the future interface, not proof that code exists today.
4. Verify each checkpoint before advancing. Finish the baseline path before tuning.

## 5. How to resume without this conversation

- Read the latest progress entry and find the next unfinished, dependency-ready task.
- Recheck the current files and evidence; logs may describe an older working state.
- Preserve completed outputs. Rerun only what changed or failed verification.
- If blocked, record the exact failure and continue independent ready work. Do not label the milestone complete.

## 6. Document authority

| Question | Owning file |
|---|---|
| Should a feature exist? | 02_PRD.md |
| Is a value or split valid? | 03_DATA_SPEC.md |
| Which model or threshold wins? | 05_EXPERIMENT_PLAN.md |
| What is the API or artifact contract? | 04_TECHNICAL_DESIGN.md |
| Does work pass? | 06_VALIDATION_AND_RELEASE.md |
| What is next? | tasks/todo.md and 09_PROGRESS_LOG.md |

Changes to these defaults require a decision entry and updates to affected documents. Routine implementation details within the contract can be decided by the engineer.
