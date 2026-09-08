import argparse
import sys
from pathlib import Path
from typing import Dict, Any
import numpy as np
import pandas as pd

from src.agent.state import WorkflowState
from src.agent.agent import LLMAgent, MockAgent, LLMClient
from src.tools.ml_tools import MLTools
from src.evaluation.validation import ValidationEngine
from src.evaluation.comparison import ModelComparator
from src.utils.logging import setup_logger

VERSION = "0.1.0"

def print_header():
    """Prints styled CLI header."""
    header = """
╔══════════════════════════════════════════════════════════╗
║                       ML STUDIO                          ║
║              Stateful Agentic AutoML                     ║
╚══════════════════════════════════════════════════════════╝
"""
    fallback_header = """
+----------------------------------------------------------+
|                       ML STUDIO                          |
|              Stateful Agentic AutoML                     |
+----------------------------------------------------------+
"""
    try:
        if hasattr(sys.stdout, 'reconfigure'):
            sys.stdout.reconfigure(encoding='utf-8')
        print(header.strip())
    except UnicodeEncodeError:
        print(fallback_header.strip())

def main():
    parser = argparse.ArgumentParser(
        description="ML Studio: A stateful agentic AutoML system for tabular datasets.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--dataset",
        type=str,
        help="Path to the input CSV/Excel dataset file."
    )
    parser.add_argument(
        "--target",
        type=str,
        help="The name of the target column in the dataset (optional, auto-detected by agent)."
    )
    parser.add_argument(
        "--metric",
        type=str,
        help="Primary evaluation metric (e.g. f1, accuracy, rmse)."
    )
    parser.add_argument(
        "--agent-mode",
        type=str,
        choices=["real", "mock"],
        default="real",
        help="The operation mode: 'real' (Gemini rotation with Ollama fallback) or 'mock' (deterministic testing)."
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Enable interactive human-in-the-loop checkpoints before major actions."
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"ML Studio v{VERSION}",
        help="Show application version and exit."
    )

    args = parser.parse_args()
    print_header()

    logger = setup_logger()
    logger.info(f"STARTING_ML_STUDIO | Version: {VERSION} | Agent Mode: {args.agent_mode}")

    if not args.dataset:
        print("\n[!] Please specify a dataset path via --dataset <path>.")
        print("Example: python -m src.main --dataset datasets/synthetic_dataset.xlsx")
        print("Use --help to view all available commands.")
        sys.exit(0)

    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        print(f"\n[!] Error: Dataset file not found at '{args.dataset}'")
        logger.error(f"FILE_NOT_FOUND | Path: {args.dataset}")
        sys.exit(1)

    print(f"\nDataset: {dataset_path.name}")
    print(f"Agent Mode: {args.agent_mode.upper()}")

    # Initialize State & Agent
    state = WorkflowState()
    state.user_context["dataset_path"] = str(dataset_path)

    if args.agent_mode == "real":
        agent = LLMAgent(state)
    else:
        agent = MockAgent(state)

    # ------------------------------------------------------------
    # [1/8] DATASET LOADING & VALIDATION
    # ------------------------------------------------------------
    print("\n------------------------------------------------------------")
    print("[1/8] DATASET VALIDATION")
    print("------------------------------------------------------------")
    try:
        df = MLTools.load_dataset(str(dataset_path))
        logger.info(f"DATASET_LOADED | Shape: {df.shape}")
    except Exception as e:
        print(f"[!] Error loading dataset: {e}")
        logger.error(f"LOAD_ERROR | {e}")
        sys.exit(1)

    errors, warnings = MLTools.validate_dataset(df, args.target)

    print("✓ File loaded")
    print(f"✓ {df.shape[0]:,} rows")
    print(f"✓ {df.shape[1]:,} columns")

    dup_count = df.duplicated().sum()
    if dup_count == 0:
        print("✓ No duplicate rows")
    else:
        print(f"⚠ {dup_count:,} duplicate rows detected")

    if errors:
        print("\nValidation Errors:")
        for err in errors:
            print(f" ✗ {err}")
        print("\nValidation status: FAILED")
        logger.error("DATASET_VALIDATION_FAILED")
        sys.exit(1)

    if warnings:
        print("\nWarnings:")
        for w in warnings:
            print(f" ⚠ {w}")
        print("\nStatus: PASSED WITH WARNINGS")
    else:
        print("\nStatus: PASSED")

    state.data_quality["warnings"] = warnings
    state.current_stage = "VALIDATED"

    # ------------------------------------------------------------
    # [2/8] DATASET PROFILING
    # ------------------------------------------------------------
    print("\n------------------------------------------------------------")
    print("[2/8] DATASET PROFILING")
    print("------------------------------------------------------------")
    profile = MLTools.profile_dataset(df)
    summary = profile["summary"]
    cols_profile = profile["columns"]

    total_cells = df.size
    total_missing = df.isnull().sum().sum()
    missing_pct = (total_missing / total_cells * 100) if total_cells > 0 else 0.0
    summary["missing_percentage"] = missing_pct

    print(f"Numerical features: {summary['numerical_features_count']}")
    print(f"Categorical features: {summary['categorical_features_count']}")
    print(f"Missing values: {missing_pct:.1f}%")

    target_candidates = MLTools.detect_target_candidates(df)
    if args.target and args.target in df.columns:
        selected_target = args.target
    else:
        selected_target = target_candidates[0]

    detected_problem = MLTools.detect_problem_type(df, selected_target)
    state.dataset = summary
    state.problem = {"target": selected_target, "task": detected_problem}
    state.current_stage = "PROFILED"

    # ------------------------------------------------------------
    # [3/8] AGENT EDA ANALYSIS & PREPROCESSING DECISION
    # ------------------------------------------------------------
    print("\n------------------------------------------------------------")
    print("[3/8] AGENT EDA ANALYSIS & PREPROCESSING DECISION")
    print("------------------------------------------------------------")
    
    print("Observing dataset distributions, cardinality, and data quality flags...")
    decision_1 = agent.select_analysis_decision(profile, warnings, target_candidates)
    params = decision_1.parameters

    target = args.target or params.get("target", selected_target)
    problem_type = params.get("problem_type", detected_problem)
    primary_metric = (args.metric or params.get("primary_metric", "f1")).lower()
    exclude_cols = params.get("exclude_columns", [])
    use_cw = params.get("use_class_weights", True)

    state.problem["target"] = target
    state.problem["task"] = problem_type
    state.optimization_metric = primary_metric
    state.preprocessing = {
        "exclude_columns": exclude_cols,
        "numerical_imputation": params.get("numerical_imputation", "median"),
        "scaling": params.get("scaling", "standard"),
        "categorical_encoding": params.get("categorical_encoding", "one_hot"),
        "use_class_weights": use_cw
    }

    print(f"\nTarget recommendation: {target}")
    print(f"Problem: {problem_type.replace('_', ' ').title()}")
    print(f"Optimization metric: {primary_metric.upper()}")
    print(f"Features excluded (Risk/Leakage/Constant): {exclude_cols if exclude_cols else 'None'}")
    print(f"Scaling strategy: {state.preprocessing['scaling']}")
    print(f"Encoding strategy: {state.preprocessing['categorical_encoding']}")
    print(f"Observation: {decision_1.observation}")
    print(f"Reason: {decision_1.reason}")

    if args.interactive:
        confirm = input("\nAccept Agent Recommendation? [Y/n]: ").strip().lower()
        if confirm == "n":
            target = input(f"Enter target column [{target}]: ").strip() or target
            primary_metric = input(f"Enter primary metric [{primary_metric}]: ").strip().lower() or primary_metric
            state.problem["target"] = target
            state.optimization_metric = primary_metric

    state.record_transition("ANALYZED", decision_1.next_action.value, decision_1.to_dict())

    # ------------------------------------------------------------
    # [4/8] PREPROCESSING EXECUTION & FEATURE SPACE ANALYSIS
    # ------------------------------------------------------------
    print("\n------------------------------------------------------------")
    print("[4/8] PREPROCESSING EXECUTION & FEATURE SPACE ANALYSIS")
    print("------------------------------------------------------------")

    # Clean target
    df_clean = df.dropna(subset=[target]).copy()
    y_raw = df_clean[target]
    X_raw = df_clean.drop(columns=[target])

    # Convert y to numerical if classification strings
    if "classification" in problem_type:
        classes, y_encoded = np.unique(y_raw.astype(str), return_inverse=True)
    else:
        classes = None
        y_encoded = y_raw.values.astype(float)

    # Holdout Train/Test Split (80% Train, 20% Untouched Final Test)
    validator_engine = ValidationEngine(task_type=problem_type, test_size=0.2, random_seed=42)
    X_train_df, X_test_df, y_train, y_test = validator_engine.split(X_raw, y_encoded)

    # Build and fit preprocessing pipeline on Train
    preproc_pipeline = MLTools.build_preprocessing_pipeline(state.preprocessing)
    X_train_proc = preproc_pipeline.fit_transform(X_train_df, y_train)
    X_test_proc = preproc_pipeline.transform(X_test_df)
    feature_names = preproc_pipeline.get_feature_names()
    feat_summary = preproc_pipeline.get_feature_summary()

    print(f"Train split: {X_train_proc.shape[0]} rows | Holdout test split: {X_test_proc.shape[0]} rows")
    print(f"Engineered feature dimensionality: {X_train_proc.shape[1]} features")
    print(f"Scaling applied: {feat_summary['scaling_applied'].title()} | Encoding applied: {feat_summary['encoding_applied'].replace('_', ' ').title()}")
    if use_cw and "classification" in problem_type:
        print("Class imbalance: Balanced class weighting active")
    
    state.current_stage = "PREPROCESSED"

    # ------------------------------------------------------------
    # [5/8] AGENT MODEL STRATEGY & CANDIDATE SELECTION
    # ------------------------------------------------------------
    print("\n------------------------------------------------------------")
    print("[5/8] AGENT MODEL STRATEGY & ENSEMBLE SELECTION")
    print("------------------------------------------------------------")
    print("Reasoning on dataset geometry, sample-to-feature ratio, and non-linearity...")

    decision_model_strat = agent.select_model_strategy(
        profile=profile,
        feature_summary=feat_summary,
        problem_type=problem_type,
        primary_metric=primary_metric,
        train_shape=X_train_proc.shape
    )
    selected_candidates = decision_model_strat.parameters.get("selected_candidates", [])
    
    print(f"\nAgent Model Hypothesis: {decision_model_strat.reason}")
    print(f"Selected Candidate Suite: {', '.join(selected_candidates)}")
    state.record_transition("MODEL_STRATEGY_SELECTED", decision_model_strat.next_action.value, decision_model_strat.to_dict())

    # ------------------------------------------------------------
    # [6/8] BASELINE & ENSEMBLE TOURNAMENT (5-Fold Cross-Validation)
    # ------------------------------------------------------------
    print("\n------------------------------------------------------------")
    print("[6/8] BASELINE & ENSEMBLE TOURNAMENT (5-Fold Cross-Validation)")
    print("------------------------------------------------------------")
    
    baseline_results = MLTools.run_baseline_experiments(
        task_type=problem_type,
        X_train=X_train_proc,
        y_train=y_train,
        primary_metric=primary_metric,
        use_class_weights=use_cw,
        selected_candidates=selected_candidates
    )

    comparator = ModelComparator(primary_metric=primary_metric)
    for res in baseline_results:
        comparator.add_result(res)
        state.record_experiment({
            "stage": "baseline",
            "model": res["model_name"],
            "metrics": res["metrics"],
            "cv_mean": res["cv_mean"],
            "cv_std": res["cv_std"],
            "runtime_seconds": res["runtime_seconds"]
        })

    # Print baseline results table
    header_col = f"{primary_metric.upper():<10}"
    print(f"\n{'Model / Architecture':<30} {header_col} {'ROC-AUC':<10} {'Runtime':<10}")
    print("-" * 65)
    for r in comparator.get_ranked_models():
        m_score = r["metrics"].get(primary_metric, 0.0)
        roc_score = r["metrics"].get("roc_auc", 0.0)
        print(f"{r['model_name']:<30} {m_score:<10.4f} {roc_score:<10.4f} {r['runtime_seconds']:<6.2f}s")

    state.current_stage = "BASELINES_TRAINED"

    # ------------------------------------------------------------
    # [7/8] AGENT DECISION & HYPERPARAMETER TUNING
    # ------------------------------------------------------------
    print("\n------------------------------------------------------------")
    print("[7/8] AGENT DECISION & HYPERPARAMETER TUNING")
    print("------------------------------------------------------------")
    
    decision_2 = agent.select_tuning_decision(baseline_results, primary_metric)
    selected_tune_model = decision_2.parameters.get("selected_model", comparator.get_best_model()["model_name"])
    tuning_trials = int(decision_2.parameters.get("trials", 20))

    print(f"Observation: {decision_2.observation}")
    print(f"Decision: {decision_2.decision}")
    print(f"Reason: {decision_2.reason}")

    state.record_transition("TUNING_DECISION", decision_2.next_action.value, decision_2.to_dict())

    print(f"\nSearching parameter space for {selected_tune_model} ({tuning_trials} trials)...")
    tuning_res = MLTools.tune_candidate(
        model_name=selected_tune_model,
        X_train=X_train_proc,
        y_train=y_train,
        task_type=problem_type,
        primary_metric=primary_metric,
        trials=tuning_trials,
        use_class_weights=use_cw
    )

    best_estimator = tuning_res["best_estimator"]
    best_cv_score = tuning_res["best_score"]
    best_params = tuning_res["best_params"]

    print(f"Trials completed: {tuning_res['trials_run']}")
    print(f"Best configuration: {best_params if best_params else 'Default configuration'}")
    print(f"Best CV {primary_metric.upper()}: {best_cv_score:.4f}")
    print(f"Tuning time: {tuning_res['tuning_time_seconds']:.2f}s")

    state.record_experiment({
        "stage": "hyperparameter_tuning",
        "model": selected_tune_model,
        "best_params": best_params,
        "cv_score": best_cv_score,
        "runtime_seconds": tuning_res["tuning_time_seconds"]
    })
    state.current_stage = "TUNED"

    # ------------------------------------------------------------
    # [8/8] FINAL EVALUATION & SUMMARY
    # ------------------------------------------------------------
    print("\n------------------------------------------------------------")
    print("[8/8] FINAL EVALUATION & ARTIFACT EXPORT")
    print("------------------------------------------------------------")

    # Fit best estimator on full training set and evaluate on untouched holdout test set
    best_estimator.fit(X_train_proc, y_train)
    test_metrics, conf_matrix = MLTools.evaluate_holdout_test(best_estimator, X_test_proc, y_test, problem_type)
    
    # Feature importances
    feat_importances = MLTools.extract_feature_importance(best_estimator, feature_names)

    best_model_record = {
        "model_name": selected_tune_model,
        "metrics": test_metrics,
        "cv_score": best_cv_score,
        "test_score": test_metrics.get(primary_metric, 0.0),
        "best_params": best_params,
        "estimator": best_estimator
    }
    state.best_model = {
        "model_name": selected_tune_model,
        "primary_metric": primary_metric,
        "cv_score": best_cv_score,
        "test_score": test_metrics.get(primary_metric, 0.0),
        "best_params": best_params
    }

    # Final agent synthesis
    decision_3 = agent.select_final_recommendation(best_model_record, comparator.get_ranked_models())
    state.record_transition("FINAL_EVALUATION", decision_3.next_action.value, decision_3.to_dict())

    # Build report text
    fi_text = "\n".join([f"  - {f}: {score:.4f}" for f, score in feat_importances[:5]])
    report_text = f"""
============================================================
FINAL MODEL RECOMMENDATION
============================================================

Model:
  {selected_tune_model}

Problem Type:
  {problem_type.replace('_', ' ').title()}

Target:
  {target}

Primary Metric:
  {primary_metric.upper()}

Cross-Validation {primary_metric.upper()}:
  {best_cv_score:.4f}

Final Holdout Test {primary_metric.upper()}:
  {test_metrics.get(primary_metric, 0.0):.4f}

Final Test Accuracy:
  {test_metrics.get('accuracy', 0.0):.4f}

Final Test ROC-AUC:
  {test_metrics.get('roc_auc', 0.0):.4f}

Top 5 Important Features:
{fi_text}

Why Selected:
  - {decision_3.reason}
  - Highest validated performance under {primary_metric.upper()} metric.
  - Generalizes reliably on untouched test holdout.

Status:
  [✓] COMPLETED
"""
    print(report_text.strip())

    # Save all output artifacts
    final_report_dict = {
        "dataset": {
            "filename": dataset_path.name,
            "shape": list(df.shape),
            "target": target,
            "problem_type": problem_type
        },
        "data_quality": {
            "warnings": warnings,
            "missing_pct": missing_pct,
            "duplicates": int(dup_count)
        },
        "preprocessing": state.preprocessing,
        "baseline_experiments": state.experiments,
        "best_model": {
            "name": selected_tune_model,
            "cv_score": best_cv_score,
            "test_metrics": test_metrics,
            "best_params": best_params,
            "top_features": feat_importances[:10],
            "confusion_matrix": conf_matrix
        },
        "agent_workflow": state.agent_history,
        "final_recommendation_reason": decision_3.reason
    }

    MLTools.export_artifacts(
        state_dict=state.to_dict(),
        best_model=best_estimator,
        final_report=final_report_dict,
        report_text=report_text,
        out_dir="outputs"
    )

    print("\nOutputs Saved:")
    print("  ✓ outputs/reports/final_report.json")
    print("  ✓ outputs/reports/final_report.txt")
    print("  ✓ outputs/experiments/experiment_history.json")
    print("  ✓ outputs/models/best_model.joblib")
    print("  ✓ outputs/logs/workflow.log")

    logger.info("WORKFLOW_COMPLETED_SUCCESSFULLY")

if __name__ == "__main__":
    main()
