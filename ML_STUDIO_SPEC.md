# ML STUDIO

## Stateful Agentic AutoML — Project Specification

> **IMPORTANT:** This document is the source of truth for implementing the ML Studio project.
>
> Build the project according to this specification.
>
> **Do NOT build a web UI, frontend, dashboard, React application, Streamlit application, or graphical interface for the MVP.**
>
> The MVP must be **CLI-based**.
>
> Prioritize correctness, clean architecture, reproducibility, agentic state management, and ML workflow orchestration over visual design.

---

# 1. PROJECT OVERVIEW

## Project Name

**ML Studio**

## Project Type

University-level Advanced Machine Learning mini-project.

## Core Concept

ML Studio is a **stateful, agentic AutoML system for tabular machine learning**.

The system accepts a dataset and dynamically constructs an ML workflow.

It should NOT simply execute a predefined pipeline such as:

```text
Dataset
   ↓
Preprocessing
   ↓
Train Models
   ↓
Compare
   ↓
Best Model
```

Instead, the core workflow is:

```text
                  ┌─────────────────┐
                  │     Dataset     │
                  └────────┬────────┘
                           ↓
                     Observe data
                           ↓
                    Generate EDA
                           ↓
                    Update STATE
                           ↓
                  ┌─────────────────┐
                  │   AI Agent      │
                  │ Observe         │
                  │ Reason          │
                  │ Decide          │
                  └────────┬────────┘
                           ↓
                     Execute action
                           ↓
                  ML/EDA Tool Engine
                           ↓
                     Observe result
                           ↓
                    Update STATE
                           ↓
                  ┌─────────────────┐
                  │   AI Agent      │
                  │ Re-evaluate     │
                  │ Decide next     │
                  └────────┬────────┘
                           ↓
                         ...
                           ↓
                  Final recommendation
```

The important research/engineering concept is:

> **Observe → Update State → Reason → Decide → Execute → Observe Again**

---

# 2. MAIN PROJECT OBJECTIVE

Build a system in which an AI/LLM agent can:

1. Understand structured information about a dataset.
2. Identify potential ML problem characteristics.
3. Detect data-quality issues.
4. Recommend preprocessing.
5. Select appropriate candidate algorithms.
6. Execute ML experiments through deterministic tools.
7. Observe experiment results.
8. Update its internal workflow state.
9. Decide whether another experiment or tuning step is necessary.
10. Select a final model based on an explicit evaluation objective.
11. Explain the decisions made throughout the workflow.

The LLM is the **reasoning/orchestration layer**.

Python/ML libraries are the **execution layer**.

Do not allow the LLM to perform numerical ML computation itself.

---

# 3. IMPORTANT PROJECT PHILOSOPHY

Do NOT attempt to create an AI that magically knows the universally correct ML pipeline.

Instead:

> Build an agent that uses structured evidence and intermediate results to make justified, adaptive ML workflow decisions.

The system should acknowledge uncertainty.

For ambiguous decisions:

```text
Agent recommendation
        ↓
Human confirmation / override
        ↓
Continue workflow
```

Human-in-the-loop behavior is a feature, not a failure.

---

# 4. MVP SCOPE

The MVP should support **tabular datasets**.

## Input

Primary:

* CSV

Optional if implementation is straightforward:

* Excel

Do not add unnecessary file formats.

---

# 5. SUPPORTED ML TASKS

## Classification

Support:

* Binary classification
* Multiclass classification

## Regression

Support:

* Continuous numerical regression

## Unsupervised Learning

Do NOT implement a complete unsupervised AutoML system in the MVP.

If no target is selected, the system may report:

```text
No supervised-learning target was selected.
Clustering / unsupervised learning is outside the MVP scope.
```

Clustering may be listed as future work.

---

# 6. TARGET COLUMN DETECTION

The system should attempt to identify likely target columns using deterministic heuristics and/or agent reasoning.

Consider:

* datatype
* number of unique values
* uniqueness ratio
* column name
* identifier-like behavior
* target-like naming
* dataset structure

However:

## NEVER silently select the target.

Example CLI output:

```text
============================================================
TARGET DETECTION
============================================================

Recommended target: churn
Confidence: 0.96

Reason:
  - Binary categorical variable
  - Plausible outcome variable
  - Not identifier-like

Alternative candidates:
  1. churn
  2. customer_status

Please confirm target.

Target [churn]:
```

The user must be able to enter another column.

---

# 7. PROBLEM TYPE DETECTION

Determine:

* binary classification
* multiclass classification
* regression

Use:

* target datatype
* target cardinality
* number of unique values
* distribution
* user input

Example:

```text
============================================================
PROBLEM DETECTION
============================================================

Target: churn
Unique values: 2
Values: [Yes, No]

Detected problem:
Binary Classification

Confidence:
0.97

Continue? [Y/n]:
```

If confidence is low, explicitly ask the user.

---

# 8. DATASET VALIDATION

Before ML processing, validate the dataset.

Check:

* file exists
* file is readable
* empty dataset
* number of rows
* number of columns
* duplicate rows
* constant columns
* all-missing columns
* missing values
* mixed datatypes
* target validity
* target with only one class
* extremely small datasets
* high-dimensional datasets
* high-cardinality features
* identifier-like columns

Distinguish:

### Error

Cannot safely continue.

### Warning

Workflow can continue but user should be informed.

Example:

```text
============================================================
DATASET VALIDATION
============================================================

✓ File readable
✓ 15,432 rows
✓ 23 columns
✓ Target contains multiple classes

Warnings:
⚠ 4 columns contain missing values
⚠ 1 high-cardinality categorical feature
⚠ 1 possible identifier column
⚠ Target class imbalance detected

Validation status: PASSED WITH WARNINGS
```

---

# 9. EDA ENGINE

EDA must be implemented deterministically in Python.

Do not ask the LLM to inspect the raw dataset directly unless absolutely necessary.

Generate structured metadata.

## Dataset-level information

Calculate:

* rows
* columns
* datatypes
* memory usage

## Numerical features

Calculate:

* mean
* median
* standard deviation
* minimum
* maximum
* quantiles
* skewness where useful

## Categorical features

Calculate:

* unique count
* frequency distribution
* cardinality ratio

## Missing values

Calculate:

* missing count
* missing percentage

## Duplicate detection

Calculate:

* duplicate row count

## Outliers

Use a documented statistical method such as IQR.

Do not automatically delete outliers.

## Target analysis

Calculate:

* target distribution
* class distribution
* imbalance ratio

## Feature relationships

Where appropriate calculate:

* correlations
* basic associations

## Risk detection

Detect:

* identifier-like columns
* high-cardinality features
* potential leakage
* suspicious target relationships

---

# 10. STATEFUL AGENT ARCHITECTURE

This is the CORE of the project.

Create a persistent workflow state.

Example:

```python
state = {
    "dataset": {},
    "problem": {},
    "data_quality": {},
    "risk_flags": {},
    "user_context": {},
    "preprocessing": {},
    "candidate_models": [],
    "experiments": [],
    "best_model": {},
    "optimization_metric": {},
    "resource_budget": {},
    "agent_history": [],
    "current_stage": ""
}
```

Use a proper typed structure such as a Python dataclass or Pydantic model if appropriate.

Do not use an unstructured global dictionary throughout the application.

---

# 11. STATE CONTENT

The state should represent what the agent currently knows.

Example:

```json
{
  "dataset": {
    "rows": 15000,
    "columns": 23,
    "numerical_features": 14,
    "categorical_features": 7,
    "datetime_features": 2
  },

  "problem": {
    "task": "classification",
    "target": "churn",
    "confidence": 0.96
  },

  "data_quality": {
    "missing_values": true,
    "outliers": true,
    "duplicates": false,
    "constant_columns": false,
    "high_cardinality": true
  },

  "risk_flags": {
    "possible_leakage": ["last_login"],
    "class_imbalance": true
  },

  "preprocessing": {},

  "experiments": [],

  "best_model": null,

  "optimization_metric": "f1",

  "resource_budget": {
    "max_models": 5,
    "max_tuning_trials": 20
  },

  "agent_history": [],

  "current_stage": "EDA_COMPLETE"
}
```

The state must be updated after every major workflow action.

---

# 12. AGENT LOOP

Implement the following core loop:

```text
1. Observe current state
2. Analyze current state
3. Decide next action
4. Execute action using deterministic tool
5. Collect result
6. Update state
7. Repeat
```

Example:

```text
EDA
 ↓
State update:
class imbalance = true
 ↓
Agent decision:
use F1/PR-AUC
 ↓
Train baseline models
 ↓
Results:
LR F1 = 0.79
RF F1 = 0.87
XGBoost F1 = 0.90
 ↓
State update
 ↓
Agent decision:
tune XGBoost
 ↓
Tune
 ↓
Evaluate
 ↓
Final recommendation
```

The agent should NOT blindly execute all possible steps.

---

# 13. AGENT VS ML ENGINE

This separation is mandatory.

## Agent responsibility

The agent decides:

* what the dataset appears to contain
* what problem type is likely
* which issues matter
* what preprocessing strategy to use
* which models to test
* which metric matters
* whether tuning is worthwhile
* what to do next
* how to interpret experiment results
* when to stop

## ML engine responsibility

The ML engine performs:

* data transformations
* train/test splitting
* cross-validation
* model training
* hyperparameter search
* metric calculation
* predictions
* feature importance
* plots/statistics

The agent must not calculate metrics itself.

---

# 14. AGENT TOOLS

Implement deterministic tools/functions that the agent can invoke.

Potential tools:

```text
validate_dataset()
profile_dataset()
detect_target_candidates()
detect_problem_type()
detect_data_quality_issues()
detect_leakage_risks()
build_preprocessing_pipeline()
train_model()
evaluate_model()
compare_models()
tune_model()
generate_feature_importance()
```

The exact implementation can differ, but responsibilities must remain separated.

---

# 15. STRUCTURED AGENT DECISIONS

Avoid unrestricted natural-language actions.

Agent decisions should follow a structured schema.

Example:

```json
{
  "observation": "Target has 8% minority class",
  "decision": "Use F1 as primary metric",
  "next_action": "train_baseline_models",
  "reason": "Accuracy may be misleading",
  "confidence": 0.91
}
```

Use enums or validated schemas for:

* action
* stage
* task type
* metric
* preprocessing strategy
* model selection

This reduces hallucinated tool calls and invalid workflow states.

---

# 16. PREPROCESSING

The agent should recommend preprocessing based on dataset characteristics.

## Numerical

Support:

* mean/median imputation
* standard scaling
* robust scaling where appropriate

## Categorical

Support:

* one-hot encoding
* controlled handling of high-cardinality features

## Dates

Support basic feature extraction:

* year
* month
* day
* weekday
* weekend

## Missing values

Agent chooses an appropriate strategy based on EDA.

## Outliers

Agent can recommend:

* retain
* transform
* cap
* remove

But do not automatically delete unusual values without justification.

---

# 17. HIGH-CARDINALITY FEATURES

Detect features such as:

```text
Customer_ID
Product_ID
Transaction_ID
```

Example:

```text
Feature: Customer_ID
Unique values: 100000
Rows: 100000
Uniqueness ratio: 100%

WARNING:
Likely identifier / high-cardinality feature.

Recommendation:
Exclude unless domain context indicates otherwise.
```

The user should be able to override the recommendation.

---

# 18. DATA LEAKAGE

Implement heuristic leakage detection.

Look for:

* target-like column names
* identifier-like columns
* suspiciously strong relationships
* post-outcome-looking features
* domain-provided constraints

Example:

```text
============================================================
POTENTIAL DATA LEAKAGE
============================================================

Feature:
loan_approval_date

Risk:
HIGH

Reason:
Feature appears potentially available only after
the prediction event.

Recommendation:
Exclude from model unless user confirms otherwise.

Continue? [Y/n]:
```

Do NOT claim that the system can detect all semantic leakage.

---

# 19. CLASS IMBALANCE

Detect class imbalance.

Example:

```text
Class distribution:

0 → 91.8%
1 → 8.2%

Imbalance detected.

Agent recommendation:
- Use stratified cross-validation
- Avoid relying solely on accuracy
- Consider F1 or PR-AUC
- Consider class weighting
```

Support class weights initially.

Resampling such as SMOTE may be added if implementation remains manageable.

---

# 20. MODEL CANDIDATES

Keep the initial model set manageable.

## Classification

Implement:

* Logistic Regression
* Decision Tree
* Random Forest
* SVM
* Gradient Boosting

Optionally:

* XGBoost if dependency/environment permits

## Regression

Implement:

* Linear Regression
* Decision Tree Regressor
* Random Forest Regressor
* Gradient Boosting Regressor

Do not attempt dozens of algorithms.

The agent should select a sensible subset.

---

# 21. MODEL SELECTION LOGIC

The agent should consider dataset characteristics.

Examples:

### High dimensionality

Prefer:

* regularized linear models
* feature selection

### Nonlinear relationships

Consider:

* tree-based models
* boosting

### Small datasets

Avoid unnecessarily complex experimentation.

### Severe class imbalance

Use:

* suitable metrics
* stratification
* class weighting

The agent should provide a short reason for model selection.

---

# 22. RESOURCE-AWARE EXPERIMENTATION

The system should not blindly test:

```text
10 preprocessing methods
×
20 algorithms
×
100 hyperparameter combinations
```

Define an experiment budget.

Example:

```json
{
  "max_models": 5,
  "max_tuning_trials": 20,
  "max_experiment_time": 300
}
```

The exact limits may be configurable.

The agent should use resources intelligently.

---

# 23. MODEL EVALUATION

Use a consistent evaluation methodology.

Recommended structure:

```text
Dataset
   ↓
Train/Test Split
   ↓
Training data
   ↓
Cross-validation
   ↓
Model selection / tuning
   ↓
FINAL untouched test set
   ↓
Final evaluation
```

Never repeatedly optimize against the final test set.

---

# 24. METRICS

## Classification

Support:

* Accuracy
* Precision
* Recall
* F1
* ROC-AUC
* PR-AUC

## Regression

Support:

* MAE
* RMSE
* R²

The user should be able to specify the primary metric.

If not specified, the agent can recommend a metric.

Example:

```text
Class imbalance detected.

Recommended primary metric:
F1

Reason:
Accuracy may hide poor minority-class performance.

Accept F1? [Y/n]:
```

---

# 25. BASELINE EXPERIMENTS

Do not immediately perform expensive tuning.

First run baseline models.

Example:

```text
MODEL                  F1       ROC-AUC
------------------------------------------------
Logistic Regression    0.79      0.84
Decision Tree          0.75      0.80
Random Forest          0.87      0.91
XGBoost                0.90      0.94
```

Then feed the results back into the agent state.

---

# 26. ADAPTIVE EXPERIMENTATION

This is a major feature.

The agent should decide what to do after seeing baseline results.

Example:

```text
Baseline results:

Logistic Regression → 0.79
Random Forest       → 0.87
XGBoost             → 0.90

Agent observation:
XGBoost is currently strongest.

Agent decision:
Allocate tuning budget to XGBoost.

Reason:
Current F1 exceeds alternatives by meaningful margin.
```

This demonstrates that the workflow is adaptive rather than static.

---

# 27. HYPERPARAMETER TUNING

Tune only promising candidates.

Possible tools:

* GridSearchCV
* RandomizedSearchCV
* Optuna, if appropriate

Record:

* parameters
* metric
* score
* runtime
* model
* experiment ID

Example state entry:

```json
{
  "experiment_id": 7,
  "model": "XGBoost",
  "type": "hyperparameter_tuning",
  "parameters": {
    "max_depth": 6,
    "learning_rate": 0.05
  },
  "metric": "f1",
  "score": 0.912
}
```

---

# 28. FAIR MODEL COMPARISON

All candidate models must use:

* same target
* comparable train/test split
* appropriate preprocessing
* same primary evaluation metric
* consistent validation methodology

The execution engine must enforce this.

Do not rely on the LLM to enforce experimental fairness.

---

# 29. FINAL MODEL SELECTION

Define "best model" explicitly.

The best model is:

> The model with the strongest validated performance according to the selected optimization metric, while considering stability, computational cost, and final test performance.

Do not simply use the highest accuracy.

Example final output:

```text
============================================================
FINAL MODEL RECOMMENDATION
============================================================

Model:
XGBoost

Primary metric:
F1

Cross-validation F1:
0.902 ± 0.018

Final test F1:
0.914

ROC-AUC:
0.946

Training time:
8.3 seconds

Why selected:
- Highest validated F1
- Strong ROC-AUC
- Stable cross-validation performance
- Acceptable training cost
```

---

# 30. EXPLAINABILITY

The system should explain:

* why the problem type was selected
* why preprocessing was selected
* why models were selected
* why the metric was selected
* why additional experiments were performed
* why the final model won

Provide:

* model comparison
* confusion matrix for classification
* feature importance
* ROC/PR curve where appropriate

SHAP may be added if practical.

Do not expose hidden chain-of-thought.

Only show concise decision summaries and supporting evidence.

Example:

```text
Agent decision:
F1 selected.

Evidence:
Minority class = 8.2%

Reason:
Accuracy would not adequately represent minority-class performance.
```

---

# 31. CLI-FIRST REQUIREMENT

The MVP must be entirely CLI based.

Do NOT implement:

* React
* Next.js
* Vue
* Angular
* Streamlit
* Gradio
* Flask frontend
* FastAPI frontend
* HTML dashboard
* CSS
* browser interface

A future UI can be added later.

The CLI must be clear enough for demonstration during a university presentation/viva.

---

# 32. CLI COMMANDS

At minimum support:

```bash
python -m src.main --dataset path/to/data.csv
```

Optionally support:

```bash
python -m src.main \
    --dataset path/to/data.csv \
    --target churn \
    --metric f1
```

Useful additional commands may include:

```bash
python -m src.main --help
python -m src.main --version
```

If appropriate, use `argparse` or `typer`.

Do not introduce unnecessary CLI complexity.

---

# 33. CLI OUTPUT STYLE

The CLI should be readable and structured.

Example:

```text
╔══════════════════════════════════════════════════════════╗
║                       ML STUDIO                          ║
║              Stateful Agentic AutoML                     ║
╚══════════════════════════════════════════════════════════╝

Dataset:
  customer_churn.csv

------------------------------------------------------------
[1/8] DATASET VALIDATION
------------------------------------------------------------

✓ File loaded
✓ 15,432 rows
✓ 23 columns
✓ No duplicate rows

Warnings:
⚠ Missing values detected
⚠ Class imbalance detected
⚠ Possible leakage feature detected

Status: PASSED WITH WARNINGS


------------------------------------------------------------
[2/8] DATASET PROFILING
------------------------------------------------------------

Numerical features: 14
Categorical features: 7
Datetime features: 2

Missing values: 4.2%
Class imbalance: 91.8 / 8.2%


------------------------------------------------------------
[3/8] AGENT ANALYSIS
------------------------------------------------------------

Target recommendation:
  churn

Problem:
  Binary Classification

Optimization metric:
  F1

Reason:
  Minority class represents 8.2% of observations.


------------------------------------------------------------
[4/8] PREPROCESSING DECISION
------------------------------------------------------------

Numerical:
  Median imputation
  Standard scaling

Categorical:
  One-hot encoding

Class imbalance:
  Class weighting

Potential leakage:
  last_login → excluded


------------------------------------------------------------
[5/8] BASELINE EXPERIMENTS
------------------------------------------------------------

Model                  F1       ROC-AUC
------------------------------------------------------------
Logistic Regression    0.790      0.840
Random Forest          0.870      0.910
Gradient Boosting      0.880      0.921
XGBoost                0.900      0.940


------------------------------------------------------------
[6/8] AGENT DECISION
------------------------------------------------------------

Observation:
  XGBoost currently performs best.

Decision:
  Tune XGBoost.

Reason:
  Strongest F1 and ROC-AUC among baseline candidates.


------------------------------------------------------------
[7/8] HYPERPARAMETER TUNING
------------------------------------------------------------

Trials:
  20

Best configuration:
  max_depth = 6
  learning_rate = 0.05
  n_estimators = 300

Best CV F1:
  0.912


------------------------------------------------------------
[8/8] FINAL EVALUATION
------------------------------------------------------------

FINAL RECOMMENDATION
────────────────────────────────────────────────────────────

Model:
  XGBoost

F1:
  0.914

ROC-AUC:
  0.946

Status:
  ✓ COMPLETED

Reason:
  Highest validated performance with stable
  cross-validation results and acceptable cost.

Output:
  outputs/reports/final_report.json
  outputs/reports/final_report.txt
  outputs/models/best_model.joblib
```

The actual values above are only examples. Never hard-code them.

---

# 34. INTERACTIVE CLI CHECKPOINTS

Important decisions should support user confirmation.

Example:

```text
Recommended target: churn

Accept? [Y/n/change]:
```

For leakage:

```text
Potential leakage:
last_login

Exclude this feature? [Y/n]:
```

For metric:

```text
Recommended metric:
F1

Accept? [Y/n/change]:
```

For final model:

```text
Recommended model:
XGBoost

Accept? [Y/n]:
```

The user should also be able to run the workflow non-interactively when all configuration is supplied through CLI arguments.

---

# 35. PROJECT FILE STRUCTURE

Create the following structure.

```text
ml-studio/
│
├── README.md
├── ML_STUDIO_SPEC.md
├── requirements.txt
├── .env.example
├── .gitignore
│
├── src/
│   ├── __init__.py
│   ├── main.py
│   │
│   ├── agent/
│   │   ├── __init__.py
│   │   ├── agent.py
│   │   ├── state.py
│   │   ├── decisions.py
│   │   ├── prompts.py
│   │   └── memory.py
│   │
│   ├── data/
│   │   ├── __init__.py
│   │   ├── loader.py
│   │   ├── validator.py
│   │   └── profiler.py
│   │
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   ├── strategies.py
│   │   └── pipeline.py
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── classification.py
│   │   ├── regression.py
│   │   └── tuning.py
│   │
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── metrics.py
│   │   ├── validation.py
│   │   └── comparison.py
│   │
│   ├── tools/
│   │   ├── __init__.py
│   │   └── ml_tools.py
│   │
│   └── utils/
│       ├── __init__.py
│       ├── logging.py
│       └── config.py
│
├── tests/
│   ├── __init__.py
│   ├── test_loader.py
│   ├── test_validator.py
│   ├── test_profiler.py
│   ├── test_state.py
│   ├── test_preprocessing.py
│   ├── test_models.py
│   ├── test_evaluation.py
│   └── test_agent.py
│
├── datasets/
│   └── .gitkeep
│
├── outputs/
│   ├── reports/
│   ├── experiments/
│   ├── models/
│   └── logs/
│
└── docs/
    ├── architecture.md
    ├── agent_workflow.md
    └── experiment_design.md
```

Keep modules small and focused.

Do not create unnecessary files merely to make the architecture look complex.

---

# 36. RESPONSIBILITY OF EACH DIRECTORY

## `src/main.py`

CLI entry point.

Responsible for:

* parsing arguments
* initializing workflow
* starting agent
* handling user interaction

It should NOT contain ML implementation.

---

## `src/agent/`

Contains the agentic layer.

### `agent.py`

Responsible for:

* observe
* reason
* decide
* execute
* update state

### `state.py`

Defines the workflow state.

### `decisions.py`

Defines structured agent decisions and action schemas.

### `prompts.py`

Contains LLM system prompts and decision prompts.

### `memory.py`

Stores workflow history / experiment history where needed.

---

## `src/data/`

Contains deterministic dataset handling.

### `loader.py`

Load CSV/Excel.

### `validator.py`

Dataset validation.

### `profiler.py`

EDA and profiling.

---

## `src/preprocessing/`

Contains deterministic preprocessing.

The agent chooses strategies.

This package executes them.

---

## `src/models/`

Contains model definitions and training.

---

## `src/evaluation/`

Contains:

* metrics
* cross-validation
* comparison
* final evaluation

---

## `src/tools/`

Expose deterministic operations that the agent can call.

---

# 37. LLM INTEGRATION

Design the LLM layer so that the provider can be changed.

Do not tightly couple the entire application to one specific provider.

Create a clean abstraction such as:

```python
class LLMClient:
    def generate_decision(...):
        ...
```

The actual provider implementation can be plugged in later.

The application should also support a deterministic/mock agent mode for testing.

Example:

```bash
python -m src.main \
    --dataset data.csv \
    --agent-mode mock
```

This allows the ML pipeline to be tested without API calls.

---

# 38. ENVIRONMENT CONFIGURATION

Use environment variables for API credentials.

Example `.env.example`:

```text
LLM_API_KEY=
LLM_MODEL=
```

Never hard-code API keys.

Never commit secrets.

---

# 39. RAW DATA PRIVACY

Do not send the entire raw dataset to the LLM by default.

The agent should primarily receive:

* schema
* statistics
* EDA summaries
* warnings
* experiment results
* user-provided domain context

Example:

```json
{
  "rows": 15000,
  "columns": 23,
  "missing_values": {
    "age": 0.04
  },
  "class_distribution": {
    "0": 0.918,
    "1": 0.082
  }
}
```

This is preferable to passing all customer records to the LLM.

---

# 40. LOGGING

Log important workflow events.

Example:

```text
2026-08-31 20:00:01 | DATASET_LOADED
2026-08-31 20:00:02 | EDA_COMPLETE
2026-08-31 20:00:04 | AGENT_DECISION | SELECT_METRIC
2026-08-31 20:00:05 | PREPROCESSING_COMPLETE
2026-08-31 20:00:10 | EXPERIMENT_COMPLETE | RandomForest
```

Do not log sensitive raw records.

---

# 41. EXPERIMENT RECORD

Every experiment should have a structured record.

Example:

```json
{
  "experiment_id": "EXP-004",
  "stage": "baseline",
  "model": "RandomForest",
  "preprocessing": {
    "numeric": "median_imputation+scaling",
    "categorical": "one_hot"
  },
  "metric": "f1",
  "cv_mean": 0.87,
  "cv_std": 0.02,
  "test_score": null,
  "runtime_seconds": 4.2,
  "status": "completed"
}
```

This experiment history becomes part of the agent state.

---

# 42. STOPPING CONDITIONS

The agent should not run indefinitely.

Stop when one of the following is reached:

* sufficient model evidence
* budget exhausted
* maximum experiments reached
* tuning completed
* improvement becomes negligible
* strong candidate is identified
* fatal error occurs

The stopping reason should be recorded.

Example:

```text
Agent stopping condition:
No significant improvement after 3 additional experiments.
```

---

# 43. ERROR HANDLING

The application must fail gracefully.

Examples:

```text
ERROR:
Dataset contains only one target class.

Cannot perform classification.

Please select a valid target.
```

or:

```text
ERROR:
Unable to load dataset.

Reason:
Invalid CSV format.
```

Do not show raw Python tracebacks to normal CLI users unless debug mode is enabled.

---

# 44. TESTING REQUIREMENTS

Create unit tests for:

* dataset loading
* validation
* EDA
* state transitions
* preprocessing
* model training
* evaluation
* agent decisions

Create at least one integration test:

```text
dataset
 ↓
validation
 ↓
EDA
 ↓
state
 ↓
agent
 ↓
preprocessing
 ↓
model
 ↓
evaluation
 ↓
final recommendation
```

Use a small synthetic dataset for automated tests.

Tests should not require an external LLM API.

---

# 45. MOCK AGENT

Implement a mock/deterministic agent mode.

Example:

```bash
python -m src.main \
    --dataset datasets/test.csv \
    --agent-mode mock
```

This mode should return deterministic decisions based on state.

Purpose:

* testing
* debugging
* offline development
* reproducibility
* avoiding unnecessary API costs

The real LLM agent can then be used for demonstration.

---

# 46. REPRODUCIBILITY

Set random seeds where applicable.

Record:

* random seed
* dataset information
* preprocessing strategy
* model
* hyperparameters
* metrics
* experiment IDs

The same configuration should produce approximately reproducible results.

---

# 47. OUTPUT FILES

After a successful run, save:

```text
outputs/
├── reports/
│   ├── final_report.json
│   └── final_report.txt
│
├── experiments/
│   └── experiment_history.json
│
├── models/
│   └── best_model.joblib
│
└── logs/
    └── workflow.log
```

Do not save sensitive raw data unless explicitly required.

---

# 48. FINAL REPORT

The final report should contain:

## Dataset

* filename
* shape
* target
* problem type

## Data quality

* missing values
* duplicates
* outliers
* imbalance
* high-cardinality features
* potential leakage

## Preprocessing

* transformations performed
* reasoning

## Experiments

* models
* metrics
* runtime
* CV performance

## Tuning

* tuned model
* parameter search
* best parameters
* performance improvement

## Final model

* selected model
* final metrics
* reasoning

## Agent workflow

* major decisions
* observations
* state transitions
* stopping reason

---

# 49. DEVELOPMENT PHASES

Do NOT implement the entire project blindly in one pass.

Implement in phases.

---

## PHASE 1 — Project skeleton

Create:

* directory structure
* requirements
* configuration
* CLI
* README

Verify:

```bash
python -m src.main --help
```

works.

---

## PHASE 2 — Dataset engine

Implement:

* loader
* validator
* profiler

Test using a small dataset.

CLI should be able to display:

```text
Dataset
Shape
Datatypes
Missing values
Duplicates
Target candidates
```

Do not introduce the LLM yet.

---

## PHASE 3 — State engine

Implement:

* state model
* state updates
* state transitions
* experiment history

Demonstrate:

```text
EDA
 ↓
STATE
 ↓
STATE UPDATE
```

---

## PHASE 4 — Agent layer

Implement:

* agent interface
* structured decisions
* mock agent
* LLM abstraction

First demonstrate:

```text
EDA
 ↓
State
 ↓
Agent decision
 ↓
Action
```

---

## PHASE 5 — Preprocessing

Implement deterministic preprocessing.

Connect:

```text
Agent decision
 ↓
Preprocessing tool
 ↓
Updated state
```

---

## PHASE 6 — Model training

Implement baseline models.

Connect:

```text
Agent
 ↓
Model selection
 ↓
Training
 ↓
Results
 ↓
State
```

---

## PHASE 7 — Agentic experiment loop

Implement:

```text
Observe
 ↓
Reason
 ↓
Decide
 ↓
Execute
 ↓
Observe
 ↓
Update
 ↓
Repeat
```

This is the most important phase.

---

## PHASE 8 — Hyperparameter tuning

Add adaptive tuning.

Only tune promising models.

---

## PHASE 9 — Explainability

Add:

* feature importance
* confusion matrix
* comparison
* decision explanations

---

## PHASE 10 — Final evaluation

Run complete workflow on representative datasets.

---

## PHASE 11 — Research experiment

Compare:

### Baseline

Static ML pipeline.

### Proposed

Stateful agentic pipeline.

Measure:

* final model performance
* number of experiments
* runtime
* tuning efficiency
* computational cost
* decision quality
* explainability

---

# 50. DO NOT OVER-ENGINEER

This is a university mini-project.

Prefer:

```text
Simple
Reliable
Understandable
Testable
Demonstrable
```

over:

```text
Massive
Distributed
Production-scale
Overly abstract
```

Do not introduce:

* Kubernetes
* Docker orchestration
* microservices
* message queues
* distributed training
* databases unless actually necessary
* frontend frameworks
* cloud infrastructure

unless a later requirement explicitly demands them.

A local Python application is sufficient for the MVP.

---

# 51. RESEARCH CONTRIBUTION

The project should be positioned around:

> **State-aware adaptive ML workflow orchestration using an AI agent.**

The research question can be:

> Can a stateful LLM-based agent dynamically construct and adapt a machine-learning pipeline based on dataset characteristics and intermediate experimental results?

A secondary question can be:

> Does adaptive agentic orchestration provide advantages over a static AutoML-style workflow in model performance, computational efficiency, and explainability?

---

# 52. BASELINE VS PROPOSED SYSTEM

Implement a way to compare:

## Static baseline

```text
Dataset
 ↓
Fixed preprocessing
 ↓
Fixed model list
 ↓
Fixed evaluation
 ↓
Best model
```

## ML Studio

```text
Dataset
 ↓
EDA
 ↓
Agent
 ↓
Adaptive preprocessing
 ↓
Experiments
 ↓
State update
 ↓
Agent
 ↓
Adaptive model selection
 ↓
Tuning
 ↓
Final evaluation
```

Compare:

| Metric                 | Static | ML Studio |
| ---------------------- | -----: | --------: |
| Best primary metric    |        |           |
| Runtime                |        |           |
| Number of experiments  |        |           |
| Tuning trials          |        |           |
| Final test performance |        |           |
| Computational cost     |        |           |

---

# 53. SUCCESS CRITERIA

The project is considered successful if:

### Technical

* Dataset can be loaded.
* Dataset can be profiled.
* State is maintained.
* Agent can make structured decisions.
* Decisions can invoke deterministic tools.
* Results update state.
* Agent can adapt based on results.
* Models can be compared fairly.
* Final model can be evaluated.
* Workflow is reproducible.

### Agentic

The system must demonstrate at least one clear adaptive behavior.

Example:

```text
EDA detects imbalance
        ↓
Agent chooses F1
        ↓
Baseline results show XGBoost is strongest
        ↓
Agent chooses XGBoost for tuning
        ↓
Tuning improves performance
        ↓
Agent stops experimentation
```

This sequence should be demonstrable in the CLI.

---

# 54. WHAT NOT TO CLAIM

The project must NOT claim:

* perfect automatic target detection
* perfect leakage detection
* universal best model
* complete automatic feature engineering
* replacement of domain experts
* production-grade AutoML
* support for every ML problem
* guaranteed optimal model

Use language such as:

* recommendation
* confidence
* heuristic
* evidence
* adaptive decision
* human confirmation

---

# 55. FINAL IMPLEMENTATION PRINCIPLE

The most important implementation requirement is:

> **The agent must not merely generate a plan once.**

It must be capable of:

```text
OBSERVE
   ↓
REASON
   ↓
ACT
   ↓
OBSERVE RESULT
   ↓
UPDATE STATE
   ↓
REASON AGAIN
   ↓
ACT AGAIN
```

The agent's second decision must be capable of changing because of the result of the first action.

If the entire workflow can be predetermined before execution, then the implementation is NOT sufficiently agentic.

---

# 56. FIRST DEMONSTRATION TARGET

Before implementing every feature, get the following working end-to-end:

```text
CSV
 ↓
Dataset validation
 ↓
EDA
 ↓
State creation
 ↓
Agent reads state
 ↓
Agent recommends target/problem/metric
 ↓
User confirms
 ↓
Agent selects preprocessing
 ↓
Preprocessing executes
 ↓
Baseline models train
 ↓
Results enter state
 ↓
Agent reviews results
 ↓
Agent selects promising model
 ↓
Tuning
 ↓
Final evaluation
 ↓
CLI recommendation
```

Do not move to UI development.

Do not add unnecessary infrastructure.

Get this workflow correct first.

---

# 57. EXPECTED END STATE

A user should eventually be able to run:

```bash
python -m src.main --dataset datasets/customer_churn.csv
```

and experience a complete CLI workflow where ML Studio:

```text
1. Loads the dataset
2. Validates it
3. Profiles it
4. Detects issues
5. Maintains state
6. Makes an agent decision
7. Executes the selected action
8. Updates state
9. Runs experiments
10. Reviews results through the agent
11. Adapts the workflow
12. Tunes promising models
13. Evaluates the final model
14. Explains the recommendation
15. Saves the experiment history and model
```

The final system should demonstrate that the **AI agent is acting as an adaptive ML workflow orchestrator**, while deterministic Python components remain responsible for actual data processing and machine-learning computation.

---

# END OF ML STUDIO SPECIFICATION