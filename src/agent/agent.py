from typing import Any, Dict, Protocol
from src.agent.state import WorkflowState
from src.agent.decisions import AgentDecision, AgentAction

class LLMClient:
    """Base client class for invoking external LLM providers."""
    def __init__(self, api_key: str, model: str):
        self.api_key = api_key
        self.model = model

    def generate_decision(self, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
        """Invokes LLM provider and parses structured decision dictionary."""
        # Skeleton implementation
        return {
            "observation": "LLM client skeleton observation",
            "decision": "LLM client skeleton decision",
            "next_action": "stop_workflow",
            "reason": "Not implemented in skeleton",
            "confidence": 1.0,
            "parameters": {}
        }

class BaseAgent:
    """Abstract/base agent structure to govern AutoML workflow."""
    def __init__(self, state: WorkflowState):
        self.state = state

    def select_next_action(self) -> AgentDecision:
        """Observe state and return structured next action."""
        raise NotImplementedError

class LLMAgent(BaseAgent):
    """LLM reasoning layer implementing agent loop decisions via LLM API."""
    def __init__(self, state: WorkflowState, client: LLMClient):
        super().__init__(state)
        self.client = client

    def select_next_action(self) -> AgentDecision:
        # Skeleton implementation
        decision_dict = self.client.generate_decision("system prompt", "user prompt")
        return AgentDecision(
            observation=decision_dict["observation"],
            decision=decision_dict["decision"],
            next_action=AgentAction(decision_dict["next_action"]),
            reason=decision_dict["reason"],
            confidence=decision_dict["confidence"],
            parameters=decision_dict.get("parameters", {})
        )

class MockAgent(BaseAgent):
    """Deterministic mock agent that navigates the workflow stages for testing and dry runs."""
    def select_next_action(self) -> AgentDecision:
        # Basic skeleton deterministic progression based on current_stage
        stage = self.state.current_stage
        
        if stage == "INITIALIZED":
            return AgentDecision(
                observation="Dataset loaded, workflow initialized.",
                decision="Validate dataset",
                next_action=AgentAction.VALIDATE_DATASET,
                reason="Dataset needs checks before model construction.",
                confidence=1.0
            )
        elif stage == "VALIDATED":
            return AgentDecision(
                observation="Dataset validation completed.",
                decision="Profile dataset",
                next_action=AgentAction.PROFILE_DATASET,
                reason="Profiling provides column statistics and shape.",
                confidence=1.0
            )
        elif stage == "PROFILED":
            return AgentDecision(
                observation="Dataset profiling completed.",
                decision="Select target and problem characteristics.",
                next_action=AgentAction.SELECT_TARGET_PROBLEM_METRIC,
                reason="Target and task type must be established.",
                confidence=1.0
            )
        elif stage == "METRIC_SELECTED":
            return AgentDecision(
                observation="Target column and evaluation metric established.",
                decision="Preprocess dataset.",
                next_action=AgentAction.PREPROCESS_DATA,
                reason="Encode categoricals, handle scaling/imputation.",
                confidence=1.0
            )
        elif stage == "PREPROCESSED":
            return AgentDecision(
                observation="Preprocessing complete.",
                decision="Run baseline models.",
                next_action=AgentAction.RUN_BASELINES,
                reason="Establish baseline score profiles.",
                confidence=1.0
            )
        elif stage == "BASELINES_TRAINED":
            return AgentDecision(
                observation="Baseline algorithms evaluated.",
                decision="Tune best candidate.",
                next_action=AgentAction.TUNE_MODEL,
                reason="Attempt hyperparameter search on best baseline candidate.",
                confidence=1.0
            )
        elif stage == "TUNED":
            return AgentDecision(
                observation="Tuning search completed.",
                decision="Run final validation.",
                next_action=AgentAction.EVALUATE_FINAL_MODEL,
                reason="Calculate test set performance for final model.",
                confidence=1.0
            )
        else:
            return AgentDecision(
                observation="Workflow steps completed.",
                decision="Stop workflow",
                next_action=AgentAction.STOP_WORKFLOW,
                reason="No more execution tasks remaining.",
                confidence=1.0
            )

