import argparse
import sys
from pathlib import Path
from src.agent.state import WorkflowState
from src.agent.agent import MockAgent
from src.tools.ml_tools import MLTools
from src.utils.logging import setup_logger
from src.utils.config import AppConfig

VERSION = "0.1.0"

def print_header():
    """Prints the styled CLI header as specified in ML_STUDIO_SPEC.md."""
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
        help="The name of the target column in the dataset."
    )
    parser.add_argument(
        "--metric",
        type=str,
        help="Primary evaluation metric (e.g. f1, accuracy, rmse)."
    )
    parser.add_argument(
        "--agent-mode",
        type=str,
        choices=["mock", "real"],
        default="mock",
        help="The operation mode for the agent reasoning engine."
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"ML Studio v{VERSION}",
        help="Show application version and exit."
    )

    args = parser.parse_args()

    print_header()

    # Setup logger
    logger = setup_logger()
    logger.info(f"STARTING_ML_STUDIO | Version: {VERSION} | Agent Mode: {args.agent_mode}")

    if not args.dataset:
        print("\n[!] Please specify a dataset path via --dataset <path>.")
        print("Use --help to view all available commands.")
        sys.exit(0)

    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        print(f"\n[!] Error: Dataset file not found at '{args.dataset}'")
        logger.error(f"FILE_NOT_FOUND | Path: {args.dataset}")
        sys.exit(1)

    print(f"\nDataset: {dataset_path.name}\n")

    # 1. LOAD DATASET
    try:
        df = MLTools.load_dataset(args.dataset)
        logger.info(f"DATASET_LOADED | Shape: {df.shape}")
    except Exception as e:
        print(f"[!] Error loading dataset: {e}")
        logger.error(f"LOAD_ERROR | {e}")
        sys.exit(1)

    # 2. DATASET VALIDATION
    print("------------------------------------------------------------")
    print("[1/8] DATASET VALIDATION")
    print("------------------------------------------------------------")
    
    errors, warnings = MLTools.validate_dataset(df, args.target)
    
    print("✓ File loaded")
    print(f"✓ {df.shape[0]:,} rows")
    print(f"✓ {df.shape[1]:,} columns")
    
    duplicate_rows = df.duplicated().sum()
    if duplicate_rows == 0:
        print("✓ No duplicate rows")
    else:
        print(f"⚠ {duplicate_rows:,} duplicate rows detected")

    if errors:
        print("\nErrors:")
        for err in errors:
            print(f" ✗ {err}")
        print("\nValidation status: FAILED")
        logger.error("DATASET_VALIDATION_FAILED")
        sys.exit(1)

    if warnings:
        print("\nWarnings:")
        for warn in warnings:
            print(f" ⚠ {warn}")
        print("\nValidation status: PASSED WITH WARNINGS")
        logger.warning("DATASET_VALIDATION_PASSED_WITH_WARNINGS")
    else:
        print("\nValidation status: PASSED")
        logger.info("DATASET_VALIDATION_PASSED")

    # 3. DATASET PROFILING
    print("\n------------------------------------------------------------")
    print("[2/8] DATASET PROFILING")
    print("------------------------------------------------------------")
    
    profile = MLTools.profile_dataset(df)
    summary = profile["summary"]
    cols_profile = profile["columns"]

    print(f"Numerical features: {summary['numerical_features_count']}")
    print(f"Categorical features: {summary['categorical_features_count']}")
    
    # Calculate global cell missing percentage
    total_cells = df.size
    total_missing = df.isnull().sum().sum()
    missing_pct = (total_missing / total_cells * 100) if total_cells > 0 else 0.0
    print(f"Missing values: {missing_pct:.1f}% of total cells")

    # Display columns detailed profiles
    print("\nColumn Profiles:")
    for col, col_prof in cols_profile.items():
        missing_info = f"{col_prof['missing_count']} missing ({col_prof['missing_percent']*100:.1f}%)"
        outliers_str = ""
        if "outliers" in col_prof:
            outliers_str = f" | IQR Outliers: {col_prof['outliers']['count']}"
        print(f"  - {col} ({col_prof['dtype']}): {col_prof['unique_count']} unique values | {missing_info}{outliers_str}")

    # Heuristically detect target candidates
    target_candidates = MLTools.detect_target_candidates(df)
    print("\nDetected Target Candidates:")
    for idx, cand in enumerate(target_candidates, 1):
        print(f"  {idx}. {cand}")

    # Initialize workflow state
    state = WorkflowState()
    state.user_context["dataset_path"] = args.dataset
    state.dataset = summary
    state.current_stage = "PROFILED"

    if args.target:
        state.problem["target"] = args.target
        state.problem["task"] = MLTools.detect_problem_type(df, args.target)
        logger.info(f"PROBLEM_TYPE_DETECTED | Target: {args.target} | Task: {state.problem['task']}")
    else:
        # Default to first target candidate
        detected_target = target_candidates[0]
        state.problem["target"] = detected_target
        state.problem["task"] = MLTools.detect_problem_type(df, detected_target)
        logger.info(f"HEURISTIC_TARGET_DETECTED | Target: {detected_target} | Task: {state.problem['task']}")

    if args.metric:
        state.optimization_metric = args.metric

    # In Phase 2, we terminate here and don't execute any agent layers
    print("\nPhase 2 Complete: Dataset engine successfully executed.")
    logger.info("PHASE_2_COMPLETED")

if __name__ == "__main__":
    main()
