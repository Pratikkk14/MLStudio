"""Structured prompt templates for the ML Studio Agent Reasoning Layer."""

SYSTEM_PROMPT = """You are the Lead AutoML AI Scientist for ML Studio, a stateful agentic machine learning platform.
Your objective is to guide tabular machine learning workflows through an adaptive reasoning cycle:
Observe Dataset Profile/EDA -> Select Context-Aware Preprocessing -> Formulate Model Hypothesis (Bagging, Boosting, Trees, Kernels, Ensembles) -> Evaluate Baselines -> Tune & Synthesize Final Deployment Pipeline.

RULES:
1. Do NOT perform numerical computations yourself; delegate all ML training, evaluation, and calculations to deterministic tools.
2. If class imbalance (>85% majority) is present, prioritize F1-score or ROC-AUC over accuracy and recommend class weighting.
3. If high-cardinality identifier columns (e.g. customer_id, uuid) or constant columns (only 1 unique value) are detected, recommend excluding them from model training to prevent overfitting or leakage.
4. When choosing categorical encoding: choose 'one_hot' for low-to-medium cardinality categories; choose 'ordinal' if cardinality is high to prevent sparse dimensional explosion for tree-based models.
5. When choosing scaling: choose 'robust' if significant outliers are flagged in EDA; choose 'standard' otherwise.
6. Output your response strictly as a JSON object adhering to the specified schema, without extraneous text.
"""

ANALYSIS_DECISION_PROMPT = """Current Stage: DATASET_EDA_AND_PREPROCESSING_DECISION

Dataset Overview:
- Total Observations (Rows): {rows}
- Total Features (Columns): {columns}
- Numerical Features Count: {numerical_features}
- Categorical Features Count: {categorical_features}
- Total Missing Cells: {missing_pct}%
- Target Candidates: {target_candidates}

Per-Column Statistical EDA Insights:
{column_details}

Data Quality & Outlier Warnings:
{warnings}

Task:
1. Confirm the target column and problem type ({detected_problem}).
2. Select the optimal primary evaluation metric (e.g., f1, accuracy, roc_auc, rmse, r2).
3. Identify features to exclude (identifiers, constant columns, leakage risks).
4. Recommend tailored preprocessing:
   - numerical_imputation ('median' or 'mean')
   - scaling ('standard', 'robust' for outliers, or 'minmax')
   - categorical_encoding ('one_hot' or 'ordinal' for high cardinality)
   - use_class_weights (true/false)

Respond with a JSON object with this exact structure:
{{
  "observation": "Detailed EDA assessment of data distributions, cardinality, and quality flags",
  "decision": "Formal recommendation for target, metric, preprocessing choices, and dropped columns",
  "next_action": "preprocess_data",
  "reason": "Scientific rationale connecting data distributions to chosen scaling, encoding, and metric",
  "confidence": 0.95,
  "parameters": {{
    "target": "{default_target}",
    "problem_type": "{detected_problem}",
    "primary_metric": "f1",
    "exclude_columns": ["col1"],
    "numerical_imputation": "median",
    "scaling": "standard",
    "categorical_encoding": "one_hot",
    "use_class_weights": true
  }}
}}
"""

MODEL_SELECTION_STRATEGY_PROMPT = """Current Stage: MODEL_ARCHITECTURE_AND_ENSEMBLE_SELECTION

Dataset & Feature Space Context:
- Problem Type: {problem_type}
- Primary Optimization Metric: {primary_metric}
- Training Observations: {train_rows}
- Preprocessed Engineered Features: {feature_count}
- Scaling Strategy: {scaling_applied}
- Encoding Strategy: {encoding_applied}
- Outlier & Distribution Notes: {eda_notes}

Available Algorithm Families:
1. 'logistic_regression' or 'ridge' (Linear Models)
2. 'decision_tree' (Single interpretable tree)
3. 'random_forest' (Bagging ensemble of trees with feature subsampling)
4. 'gradient_boosting' (Sequential boosting ensemble minimizing residual loss)
5. 'svm' (Kernel methods with Linear/RBF hyperplanes)
6. 'voting_ensemble' (Soft/hard probability averaging of diverse algorithms)
7. 'stacking_ensemble' (Stacked generalization using meta-learner)

Task:
Reason on the dataset geometry, sample-to-feature ratio, non-linearity, and task requirements. Select a tailored suite of 3 to 5 candidate algorithms (including single models and ensemble approaches like Bagging, Boosting, Stacking, or Voting) to enter the 5-fold cross-validation tournament.

Respond with a JSON object with this exact structure:
{{
  "observation": "Analysis of feature dimensionality and dataset complexity for model suitability",
  "decision": "Selected tailored candidate model suite for cross-validation tournament",
  "next_action": "train_models",
  "reason": "Theoretical hypothesis explaining why these specific algorithms and ensemble strategies match the data structure",
  "confidence": 0.94,
  "parameters": {{
    "selected_candidates": ["random_forest", "gradient_boosting", "decision_tree", "svm", "voting_ensemble"]
  }}
}}
"""

TUNING_DECISION_PROMPT = """Current Stage: BASELINE_TOURNAMENT_EVALUATION

Problem Type: {problem_type}
Optimization Metric: {primary_metric}

Cross-Validated Tournament Results:
{baseline_results_table}

Task:
1. Evaluate the comparative validation scores, standard deviations (variance/stability), and runtimes across candidate models and ensembles.
2. Select the champion model or ensemble to advance to hyperparameter optimization.
3. Allocate a tuning trial budget (between 10 and 30 trials).

Respond with a JSON object with this exact structure:
{{
  "observation": "Comparative analysis of tournament scores, generalization margins, and variance",
  "decision": "Select <ModelName> for hyperparameter optimization",
  "next_action": "tune_model",
  "reason": "Scientific justification for why this model or ensemble architecture has the highest optimization upside",
  "confidence": 0.93,
  "parameters": {{
    "selected_model": "<ModelName>",
    "trials": 20
  }}
}}
"""

FINAL_RECOMMENDATION_PROMPT = """Current Stage: FINAL_EVALUATION

Problem Type: {problem_type}
Primary Metric: {primary_metric}
Best Model / Architecture: {best_model_name}
Validation Score: {cv_score}
Holdout Test Score: {test_score}
All Candidate Results:
{all_results_table}

Task:
Provide a final executive synthesis justifying why the selected model/ensemble is optimal for production deployment, detailing robustness, feature importance, and performance trade-offs.

Respond with a JSON object with this exact structure:
{{
  "observation": "Final validated performance summary on holdout data",
  "decision": "Approve {best_model_name} for production deployment",
  "next_action": "stop_workflow",
  "reason": "Comprehensive final justification for model selection and generalization reliability",
  "confidence": 0.98,
  "parameters": {{
    "deployable_model": "{best_model_name}",
    "status": "APPROVED"
  }}
}}
"""
