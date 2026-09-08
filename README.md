# ML Studio ⚡
### Stateful Agentic AutoML for Tabular Machine Learning

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.2%2B-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Google Gemini API](https://img.shields.io/badge/Google%20Gemini-Multi--Key%20Rotation-8E75C2?style=for-the-badge&logo=google&logoColor=white)](https://ai.google.dev/)
[![Ollama](https://img.shields.io/badge/Ollama-Offline%20Failover-black?style=for-the-badge&logo=ollama&logoColor=white)](https://ollama.ai/)
[![Pytest](https://img.shields.io/badge/Tests-24%2F24%20Passing-brightgreen?style=for-the-badge&logo=pytest&logoColor=white)](https://pytest.org/)

---

## 📌 Executive Overview

**ML Studio** is an advanced, university-level **Stateful Agentic AutoML platform** engineered specifically for tabular datasets (CSV and Excel formats). 

Unlike legacy AutoML tools that rely on brute-force grid searches or rigid sequential pipelines, ML Studio implements an **Observe-Reason-Decide-Act** cognitive loop powered by Large Language Models (LLMs). The agent inspects exploratory data analysis (EDA) distributions, assesses data quality flags (outliers, leakage, cardinality, imbalance), formulates theoretical machine learning hypotheses, dynamically curates tailored candidate algorithms (**Bagging**, **Boosting**, **Decision Trees**, **Kernel Methods**, and **Stacking/Voting Ensembles**), and guides hyperparameter optimization to build production-grade models.

---

## 🚀 Key Architectural Capabilities

### 1. 🧠 Multi-Tier Resilient Agent Reasoning
* **Tier 1 (Google Gemini Rotation)**: Automated round-robin rotation across multiple Gemini API keys (`GEMINI_API_KEY1` through `GEMINI_API_KEY8`) with rate-limit (`429`) and auth-error (`401`) self-healing.
* **Tier 2 (Local Ollama Failover)**: Automatic local server fallback (supporting `gemma4:e4b` for complex reasoning and `llama3.2:3b` for minor tasks) with generous timeout tolerance.
* **Tier 3 (Deterministic Heuristic Fallback)**: Built-in deterministic AutoML reasoning engine ensuring zero pipeline downtime even when completely offline.

### 2. 📊 Context-Aware Preprocessing & Feature Transformation
* **EDA-Driven Scaling**: Automatically selects `RobustScaler` when Interquartile Range (IQR) outliers are flagged, or `StandardScaler` / `MinMaxScaler` otherwise.
* **Intelligent Categorical Encoding**: Dynamically routes low/medium cardinality features to `OneHotEncoder` and high-cardinality features to `OrdinalEncoder` to prevent sparse dimensional explosion in tree-based algorithms.
* **Robust Imputation**: Handles missing values via median, mean, most frequent, or passthrough strategies based on missingness patterns.
* **Data Leakage & Identifier Scrubbing**: Detects and drops constant columns and high-cardinality identifier fields (`customer_id`, `UUID`, etc.) to prevent memorization.

### 3. 🌲 Advanced Model Zoo & Meta-Ensemble Strategies
* **Decision Trees & Linear Baselines**: Single interpretable `DecisionTreeClassifier`/`Regressor`, `LogisticRegression` (with class weighting), and `Ridge`.
* **Bagging (Bootstrap Aggregation)**: `RandomForestClassifier` / `Regressor` with multi-threaded feature subsampling to reduce variance.
* **Boosting (Residual Loss Minimization)**: `GradientBoostingClassifier` / `Regressor` for complex non-linear boundaries.
* **Kernel Methods**: Support Vector Machines (`SVC`) with Linear and Radial Basis Function (`RBF`) kernels.
* **Voting Ensembles**: Soft-voting meta-ensembles averaging class probability distributions across diverse model families.
* **Stacking Ensembles**: Stacked generalization combining out-of-fold predictions from heterogeneous base models into a meta-learner (`LogisticRegression` / `Ridge`).

### 4. 📈 5-Fold Stratified Cross-Validation & Metric Optimization
* Multi-metric tracking: **Accuracy**, **F1-Macro**, **ROC-AUC**, **Precision**, **Recall**, **RMSE**, **MAE**, and **R²**.
* Dynamic `RandomizedSearchCV` hyperparameter optimization exploring algorithm-specific parameter grids.

### 5. 🔍 Auditable Artifact Generation
* Exports complete experiment histories (`experiment_history.json`), executive text reports (`final_report.txt`), JSON deliverables (`final_report.json`), serialized models (`best_model.joblib`), and execution traces (`workflow.log`).

---

## 🔄 The 8-Stage Agentic Workflow

```text
┌────────────────────────────────────────────────────────────────────────┐
│                        ML STUDIO WORKFLOW ENGINE                       │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
    [1/8] DATASET VALIDATION        ▼  Load file, check duplicates & validate schema
    [2/8] STATISTICAL EDA           ▼  Compute nulls, IQR outliers, types & cardinality
    [3/8] AGENT PREPROCESSING       ▼  LLM selects imputation, scaling, encoding & drop cols
    [4/8] FEATURE TRANSFORMATION    ▼  Execute ColumnTransformer & inspect feature space
    [5/8] AGENT MODEL STRATEGY      ▼  LLM reasons on data geometry -> selects algorithms
    [6/8] BASELINE TOURNAMENT       ▼  5-Fold Cross-Validation (Trees, Ensembles, Kernels)
    [7/8] AGENT TUNING & ENSEMBLING ▼  LLM selects champion -> runs Hyperparameter Tuning
    [8/8] HOLDOUT EVALUATION        ▼  Test on untouched split, extract feature importances & export
```

---

## 📂 Repository Layout

```text
MLStudio/
├── README.md                          # Project documentation and guide
├── ML_STUDIO_SPEC.md                  # Comprehensive architectural specification
├── requirements.txt                   # Production dependencies
├── .env.example                       # Multi-key environment template
├── .gitignore                         # Git exclusion rules
├── datasets/                          # Dataset storage directory
│   └── synthetic_dataset.xlsx         # Validation benchmark dataset
├── src/
│   ├── main.py                        # Unified 8-stage CLI entry point
│   ├── agent/                         # Multi-tier agent cognitive reasoning
│   │   ├── agent.py                   # LLMClient (Gemini rotation/Ollama), BaseAgent, LLMAgent, MockAgent
│   │   ├── prompts.py                 # Structured prompts for EDA, architecture, tuning & synthesis
│   │   ├── state.py                   # WorkflowState tracker & JSON persistence
│   │   ├── decisions.py               # Enums & AgentDecision data classes
│   │   └── memory.py                  # Trajectory tracker
│   ├── data/                          # Dataset loading & statistical profiling
│   │   ├── loader.py                  # CSV / Excel loader with type detection
│   │   ├── validator.py               # Quality checks, leakage warnings, identifiers
│   │   └── profiler.py                # Statistical EDA & IQR outlier calculations
│   ├── preprocessing/                 # Feature pipeline engine
│   │   ├── pipeline.py                # ColumnTransformer with scaling, encoding & passthrough
│   │   └── strategies.py              # Preprocessing strategy enums
│   ├── models/                        # Model Zoo & Ensembles
│   │   ├── classification.py          # Decision Tree, RF, GB, SVM, Voting, Stacking classifiers
│   │   ├── regression.py              # Linear, Ridge, Decision Tree, RF, GB, Voting, Stacking regressors
│   │   └── tuning.py                  # HyperparameterTuner with RandomizedSearchCV
│   ├── evaluation/                    # Cross-validation & comparison
│   │   ├── metrics.py                 # Classification & regression metric calculators
│   │   ├── validation.py              # 5-Fold Stratified K-Fold validation engine
│   │   └── comparison.py              # ModelComparator tournament ranker
│   ├── tools/                         # Unified MLTools execution layer
│   │   └── ml_tools.py                # Consolidated tool interfaces & artifact exporter
│   └── utils/                         # Config & logging
│       ├── config.py                  # AppConfig (multi-key discovery & Ollama URLs)
│       └── logging.py                 # Timestamped file & console logger
├── tests/                             # Automated test suite (24 tests)
│   ├── test_agent.py                  # Agent reasoning & JSON parsing tests
│   ├── test_evaluation.py             # Validation & metric calculation tests
│   ├── test_loader.py                 # File loader tests
│   ├── test_models.py                 # Factory, ensemble & tuning tests
│   ├── test_preprocessing.py          # Preprocessing pipeline tests
│   ├── test_profiler.py               # Profiler & outlier tests
│   ├── test_state.py                  # State persistence tests
│   └── test_validator.py              # Validation constraint tests
└── outputs/                           # Generated project deliverables
    ├── reports/                       # final_report.json, final_report.txt
    ├── experiments/                   # experiment_history.json
    ├── models/                        # best_model.joblib
    └── logs/                          # workflow.log
```

---

## 🛠️ Setup & Installation

### 1. Clone & Setup Virtual Environment
```powershell
# Navigate into the project folder
cd MLStudio

# Create and activate virtual environment
python -m venv myvenv
.\myvenv\Scripts\activate
```

### 2. Install Dependencies
```powershell
pip install -r requirements.txt
```

### 3. Configure Environment Variables
Copy `.env.example` to `.env` and insert your Gemini API keys or Ollama endpoints:
```powershell
copy .env.example .env
```

Example `.env` configuration:
```env
# Local Ollama Host (Offline Tier)
OLLAMA_BASE_URL = http://localhost:11434
OLLAMA_MAJOR_MODEL = gemma4:e4b
OLLAMA_MINOR_MODEL = llama3.2:3b

# Google Gemini Multi-Key Rotation (Online Tier)
GEMINI_MODEL = gemini-2.5-flash
GEMINI_API_KEY1 = AIzaSy...
GEMINI_API_KEY2 = AIzaSy...
GEMINI_API_KEY3 = AIzaSy...
```

---

## 💻 CLI Usage & Commands

### 1. Run Complete Autonomous AutoML Workflow
```powershell
# Run with AI Agent (Automatic Target, Preprocessing, Model Strategy & Tuning)
python -m src.main --dataset datasets/synthetic_dataset.xlsx

# Run with an external dataset (e.g. Dry Bean Dataset)
python -m src.main --dataset "path/to/your/dataset.xlsx"
```

### 2. Interactive Human-in-the-Loop Mode
Enables human approval checkpoints before executing preprocessing and model training:
```powershell
python -m src.main --dataset datasets/synthetic_dataset.xlsx --interactive
```

### 3. Explicit Target & Metric Overrides
```powershell
python -m src.main --dataset datasets/synthetic_dataset.xlsx --target churn --metric f1
```

### 4. Deterministic Mock Mode (Offline Testing)
```powershell
python -m src.main --dataset datasets/synthetic_dataset.xlsx --agent-mode mock
```

### 5. Run Automated Test Suite
```powershell
pytest -v
```

---

## 📊 Sample Execution Output

```text
╔══════════════════════════════════════════════════════════╗
║                       ML STUDIO                          ║
║              Stateful Agentic AutoML                     ║
╚══════════════════════════════════════════════════════════╝

------------------------------------------------------------
[1/8] DATASET VALIDATION
------------------------------------------------------------
✓ File loaded
✓ 100 rows
✓ 7 columns
✓ No duplicate rows
Status: PASSED WITH WARNINGS

------------------------------------------------------------
[3/8] AGENT EDA ANALYSIS & PREPROCESSING DECISION
------------------------------------------------------------
Target recommendation: churn
Problem: Binary Classification
Optimization metric: F1
Features excluded (Risk/Leakage/Constant): ['customer_id', 'status_code']
Scaling strategy: Standard
Encoding strategy: One-Hot
Observation: Dataset contains 100 observations across 7 features with 3 data quality flags.
Reason: Selecting 'churn' as target with F1-score optimization balances precision/recall under class weighting.

------------------------------------------------------------
[5/8] AGENT MODEL STRATEGY & ENSEMBLE SELECTION
------------------------------------------------------------
Agent Model Hypothesis: Given the 80 training samples and 5 engineered features, a diverse suite of models is chosen to balance bias-variance trade-offs: Random Forest (Bagging) reduces variance, Gradient Boosting (Boosting) captures non-linear residual loss, SVM with RBF kernels creates robust decision hyperplanes, and a Soft Voting Ensemble averages predictions across diverse model families.
Selected Candidate Suite: logistic_regression, random_forest, gradient_boosting, svm, voting_ensemble

------------------------------------------------------------
[6/8] BASELINE & ENSEMBLE TOURNAMENT (5-Fold Cross-Validation)
------------------------------------------------------------

Model / Architecture           F1         ROC-AUC    Runtime   
-----------------------------------------------------------------
Gradient Boosting (Boosting)   0.4511     0.6738     0.23  s
Voting Ensemble (Soft)         0.4511     0.3838     0.95  s
Random Forest (Bagging)        0.4398     0.5176     0.72  s
SVM (Kernel)                   0.4297     0.3295     0.04  s
Logistic Regression            0.3468     0.3038     0.04  s

------------------------------------------------------------
[7/8] AGENT DECISION & HYPERPARAMETER TUNING
------------------------------------------------------------
Searching parameter space for Gradient Boosting (20 trials)...
Trials completed: 20
Best configuration: {'subsample': 0.8, 'n_estimators': 50, 'max_depth': 3, 'learning_rate': 0.01}
Best CV F1: 0.4735
Tuning time: 4.74s

------------------------------------------------------------
[8/8] FINAL EVALUATION & ARTIFACT EXPORT
------------------------------------------------------------
Model: Gradient Boosting
Final Holdout Test F1: 0.4737
Final Test Accuracy: 0.9000
Top 5 Important Features:
  - monthly_charges: 0.6851
  - age: 0.1571
  - tenure: 0.1403
  - gender_Female: 0.0121
  - gender_Male: 0.0053

Outputs Saved:
  ✓ outputs/reports/final_report.json
  ✓ outputs/reports/final_report.txt
  ✓ outputs/experiments/experiment_history.json
  ✓ outputs/models/best_model.joblib
  ✓ outputs/logs/workflow.log
```

---

## 🧪 Testing & Verification

ML Studio comes with a comprehensive test suite of 24 unit tests covering every layer of the architecture:

```powershell
pytest -v
```
```text
tests/test_agent.py::test_mock_agent_decisions PASSED                    [  4%]
tests/test_agent.py::test_llm_json_extractor PASSED                      [  8%]
tests/test_evaluation.py::test_metrics_classification PASSED             [ 12%]
tests/test_evaluation.py::test_metrics_regression PASSED                 [ 16%]
tests/test_evaluation.py::test_model_comparator PASSED                   [ 20%]
tests/test_evaluation.py::test_validation_engine_cross_validate PASSED   [ 25%]
tests/test_loader.py::test_loader_invalid_file PASSED                    [ 29%]
tests/test_loader.py::test_loader_unsupported_format PASSED              [ 33%]
tests/test_models.py::test_classifier_factory PASSED                     [ 37%]
tests/test_models.py::test_classifier_factory_create_candidates PASSED   [ 41%]
tests/test_models.py::test_classifier_factory_invalid PASSED             [ 45%]
tests/test_models.py::test_regressor_factory PASSED                      [ 50%]
tests/test_models.py::test_regressor_factory_create_candidates PASSED    [ 54%]
tests/test_models.py::test_hyperparameter_tuning PASSED                  [ 58%]
tests/test_preprocessing.py::test_preprocessing_pipeline PASSED          [ 62%]
tests/test_profiler.py::test_profiler_summary PASSED                     [ 66%]
tests/test_profiler.py::test_profiler_columns PASSED                     [ 70%]
tests/test_profiler.py::test_profiler_outliers PASSED                    [ 75%]
tests/test_state.py::test_workflow_state_initialization PASSED           [ 79%]
tests/test_state.py::test_workflow_state_to_dict PASSED                  [ 83%]
tests/test_validator.py::test_validator_empty_dataframe PASSED           [ 87%]
tests/test_validator.py::test_validator_valid_dataframe PASSED           [ 91%]
tests/test_validator.py::test_validator_missing_target PASSED            [ 95%]
tests/test_validator.py::test_validator_warnings PASSED                  [100%]

============================= 24 passed in 5.13s =============================
```

---

## 📜 Academic Integrity & License

Developed as part of the **Applied Machine Learning (AML)** curriculum. Built with standard open-source Python libraries (`pandas`, `numpy`, `scikit-learn`, `requests`, `joblib`, `pytest`).