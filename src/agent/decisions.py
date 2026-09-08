from enum import Enum
from typing import Any, Dict, Optional

class WorkflowStage(str, Enum):
    INITIALIZED = "INITIALIZED"
    VALIDATED = "VALIDATED"
    PROFILED = "PROFILED"
    PREPROCESSED = "PREPROCESSED"
    BASELINES_TRAINED = "BASELINES_TRAINED"
    TUNED = "TUNED"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"

class AgentAction(str, Enum):
    VALIDATE_DATASET = "validate_dataset"
    PROFILE_DATASET = "profile_dataset"
    SELECT_TARGET_PROBLEM_METRIC = "select_target_problem_metric"
    PREPROCESS_DATA = "preprocess_data"
    RUN_BASELINES = "run_baselines"
    TUNE_MODEL = "tune_model"
    EVALUATE_FINAL_MODEL = "evaluate_final_model"
    STOP_WORKFLOW = "stop_workflow"

class TaskType(str, Enum):
    BINARY_CLASSIFICATION = "binary_classification"
    MULTICLASS_CLASSIFICATION = "multiclass_classification"
    REGRESSION = "regression"
    UNSUPERVISED = "unsupervised"

class MetricType(str, Enum):
    ACCURACY = "accuracy"
    PRECISION = "precision"
    RECALL = "recall"
    F1 = "f1"
    ROC_AUC = "roc_auc"
    PR_AUC = "pr_auc"
    MAE = "mae"
    RMSE = "rmse"
    R2 = "r2"

class AgentDecision:
    """Represents a structured decision made by the agent."""
    def __init__(
        self,
        observation: str,
        decision: str,
        next_action: AgentAction,
        reason: str,
        confidence: float,
        parameters: Optional[Dict[str, Any]] = None
    ):
        self.observation = observation
        self.decision = decision
        self.next_action = next_action
        self.reason = reason
        self.confidence = confidence
        self.parameters = parameters or {}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "observation": self.observation,
            "decision": self.decision,
            "next_action": self.next_action.value,
            "reason": self.reason,
            "confidence": self.confidence,
            "parameters": self.parameters
        }
