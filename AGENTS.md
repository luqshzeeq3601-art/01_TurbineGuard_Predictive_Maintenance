# TurbineGuard implementation instructions

## 1. Communication

- Explain directly, briefly and in numbered steps with useful bullets.
- State the result, verification evidence and next ready task.

## 2. Before working

1. Read [START_HERE](docs/00_START_HERE.md), [progress](docs/09_PROGRESS_LOG.md), [decisions](docs/08_DECISIONS_LOG.md) and [tasks](tasks/todo.md).
2. Read the PRD and technical documents relevant to the selected task.
3. Inspect existing files and working changes. Preserve unrelated work and the neighbouring ChurnGuard project.
4. Select the first unfinished task whose dependencies are satisfied. A future instruction to start or continue authorizes the documented local implementation work; routine steps do not need repeated confirmation.

## 3. Execution rules

- Treat the Markdown documents as the implementation contract. Do not depend on earlier chat history.
- Requirements belong in the PRD; data rules in the data specification; model rules in the experiment plan; task status only in tasks/todo.md.
- If documents conflict, resolve and record the decision before implementing the affected behaviour. Human instructions take precedence.
- Complete one bounded task, run its checks, record evidence, then proceed to the next ready task within the authorized session.
- Put reusable logic in src/turbineguard/. Keep notebooks exploratory.
- Use Python type hints, snake_case functions and pytest tests for meaningful behaviour.
- Keep data processing and serving on the same feature implementation.
- Never use random row splits, future sensor readings, engine lifetime or labels as predictors.
- Keep official test labels unavailable to training, tuning, feature selection and threshold selection.
- Record missed targets honestly. Do not change thresholds or objectives after observing official test results to create a passing outcome.
- Check official documentation for the versions actually installed; pin compatible dependencies during T01.
- Model artifact loading accepts only project-generated trusted bundles. Never deserialize a user-uploaded model.
- Do not install dependencies globally, touch other projects, fabricate results or publish files merely because a planned command exists.

## 4. Scope control

- Proceed with documented local setup, data acquisition, training, testing and report generation after implementation is requested.
- Ask for direction when changing the primary dataset, problem definition or evaluation protocol.
- External publication, paid resources, public cloud deployment and contacting others require user authorization unless already explicitly authorized.
- Dashboards, streaming, deep learning, automatic retraining and equipment control are outside the MVP.

## 5. End each session

1. Mark a task complete only when its acceptance criteria have evidence.
2. Append a dated entry to docs/09_PROGRESS_LOG.md with commands, results, changed files and unresolved issues.
3. Record material design changes in docs/08_DECISIONS_LOG.md and update the owning specification.
4. Leave a concrete next task ID and any prerequisite.
