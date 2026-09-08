from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

@dataclass
class WorkflowState:
    """Represents the persistent workflow state of the AutoML agent."""
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
        "max_experiment_time": 300
    })
    agent_history: List[Dict[str, Any]] = field(default_factory=list)
    current_stage: str = "INITIALIZED"

    def to_dict(self) -> Dict[str, Any]:
        """Convert state to a dictionary representation."""
        return {
            "dataset": self.dataset,
            "problem": self.problem,
            "data_quality": self.data_quality,
            "risk_flags": self.risk_flags,
            "user_context": self.user_context,
            "preprocessing": self.preprocessing,
            "candidate_models": self.candidate_models,
            "experiments": self.experiments,
            "best_model": self.best_model,
            "optimization_metric": self.optimization_metric,
            "resource_budget": self.resource_budget,
            "agent_history": self.agent_history,
            "current_stage": self.current_stage
        }
