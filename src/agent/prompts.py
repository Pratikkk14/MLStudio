"""Structured prompt templates for the ML Studio Agent Reasoning Layer."""

SYSTEM_PROMPT = """You are the Lead AutoML AI Scientist for ML Studio, a stateful agentic machine learning platform.
Your objective is to guide tabular machine learning workflows through an adaptive reasoning cycle:
Observe Dataset Profile/Results -> Update State -> Reason on Constraints -> Decide Next Action -> Explain Decision.

RULES:
1. Do NOT perform numerical computations yourself; delegate all ML training, evaluation, and calculations to deterministic tools.
2. If class imbalance (>85% majority) is present, prioritize F1-score or ROC-AUC over accuracy and recommend class weighting.
3. If high-cardinality identifier columns (e.g. customer_id, uuid) or constant columns (only 1 unique value) are detected, recommend excluding them from model training to prevent overfitting or leakage.
4. Output your response strictly as a JSON object adhering to the specified schema, without extraneous text.
"""

ANALYSIS_DECISION_PROMPT = """Current Stage: DATASET_ANALYSIS

Dataset Information:
- Total Rows: {rows}
- Total Columns: {columns}
- Numerical Features: {numerical_features}
- Categorical Features: {categorical_features}
- Missing Cells Percentage: {missing_pct}%
- Target Candidates: {target_candidates}

Validation Warnings:
{warnings}

Task:
1. Select or confirm the target column and problem type (binary_classification, multiclass_classification, or regression).
2. Select the optimal primary evaluation metric (e.g., f1, roc_auc, accuracy, rmse, r2).
3. Identify features to exclude (identifiers, constant columns, leakage risks).
4. Recommend preprocessing strategies (imputation, scaling, categorical encoding, class weights).

Respond with a JSON object with this exact structure:
{{
  "observation": "Concise summary of dataset characteristics and data quality risks",
  "decision": "Formal recommendation for target, metric, preprocessing, and features to exclude",
  "next_action": "preprocess_data",
  "reason": "Clear scientific justification for metric and preprocessing choices",
  "confidence": 0.95,
  "parameters": {{
    "target": "{default_target}",
    "problem_type": "{detected_problem}",
    "primary_metric": "f1",
    "exclude_columns": ["col1", "col2"],
    "numerical_imputation": "median",
    "scaling": "standard",
    "categorical_encoding": "one_hot",
    "use_class_weights": true
  }}
}}
"""

TUNING_DECISION_PROMPT = """Current Stage: BASELINE_EXPERIMENTS_EVALUATED

Problem Type: {problem_type}
Optimization Metric: {primary_metric}

Baseline Experiment Results (Cross-Validated):
{baseline_results_table}

Task:
1. Evaluate the comparative performance, stability, and runtime of baseline models.
2. Select the single most promising candidate algorithm for hyperparameter tuning.
3. Allocate a tuning trial budget (between 10 and 30 trials).

Respond with a JSON object with this exact structure:
{{
  "observation": "Summary of baseline model rankings and performance margins",
  "decision": "Select <ModelName> for hyperparameter optimization",
  "next_action": "tune_model",
  "reason": "Justification for why this model has the highest upside for tuning",
  "confidence": 0.92,
  "parameters": {{
    "selected_model": "<ModelName>",
    "trials": 20
  }}
}}
"""

FINAL_RECOMMENDATION_PROMPT = """Current Stage: FINAL_EVALUATION

Problem Type: {problem_type}
Primary Metric: {primary_metric}
Best Model: {best_model_name}
Validation Score: {cv_score}
Holdout Test Score: {test_score}
All Candidate Results:
{all_results_table}

Task:
Provide a final executive recommendation justifying why the selected model is optimal for deployment, summarizing key trade-offs (performance vs. training time/complexity).

Respond with a JSON object with this exact structure:
{{
  "observation": "Final validated performance summary",
  "decision": "Deploy <ModelName> as the primary production pipeline",
  "next_action": "stop_workflow",
  "reason": "Comprehensive final justification for model selection",
  "confidence": 0.98,
  "parameters": {{
    "deployable_model": "{best_model_name}",
    "status": "APPROVED"
  }}
}}
"""
