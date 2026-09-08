from typing import Any, Dict, List

class AgentMemory:
    """Manages the history of actions, decisions, and states traversed by the agent."""
    def __init__(self):
        self.history: List[Dict[str, Any]] = []

    def record_step(self, stage: str, action: str, decision: Dict[str, Any], state_snapshot: Dict[str, Any]) -> None:
        """Records a single step in the agent's workflow trajectory."""
        self.history.append({
            "stage": stage,
            "action": action,
            "decision": decision,
            "state_snapshot": state_snapshot
        })

    def get_history(self) -> List[Dict[str, Any]]:
        """Returns the full execution history."""
        return self.history

    def clear(self) -> None:
        """Clears all recorded memory."""
        self.history.clear()
