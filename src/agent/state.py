import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

@dataclass
class WorkflowState:
    """Represents the complete, persistent state of the AutoML workflow."""
    dataset: Dict[str, Any] = field(default_factory=dict)
    problem: Dict[str, Any] = field(default_factory=dict)
    data_quality: Dict[str, Any] = field(default_factory=dict)
    risk_flags: Dict[str, Any] = field(default_factory=dict)
    user_context: Dict[str, Any] = field(default_factory=dict)
    preprocessing: Dict[str, Any] = field(default_factory=dict)
    candidate_models: List[str] = field(default_factory=list)
    experiments: List[Dict[str, Any]] = field(default_factory=list)
    best_model: Optional[Dict[str, Any]] = None
    optimization_metric: Optional[str] = None
    resource_budget: Dict[str, Any] = field(default_factory=lambda: {
        "max_models": 5,
        "max_tuning_trials": 20,
        "max_experiment_time_seconds": 300
    })
    agent_history: List[Dict[str, Any]] = field(default_factory=list)
    current_stage: str = "INITIALIZED"

    def record_experiment(self, experiment: Dict[str, Any]) -> None:
        """Adds an executed experiment record to history."""
        self.experiments.append(experiment)

    def record_transition(self, stage: str, action: str, decision: Dict[str, Any]) -> None:
        """Records a stage transition in the agent trajectory."""
        self.current_stage = stage
        self.agent_history.append({
            "stage": stage,
            "action": action,
            "decision": decision
        })

    def to_dict(self) -> Dict[str, Any]:
        """Converts state dataclass to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WorkflowState":
        """Instantiates WorkflowState from dictionary."""
        return cls(**data)

    def save_json(self, file_path: str = "outputs/experiments/experiment_history.json") -> None:
        """Serializes state to JSON file."""
        p = Path(file_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)
