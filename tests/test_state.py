from src.agent.state import WorkflowState

def test_workflow_state_initialization():
    state = WorkflowState()
    assert state.current_stage == "INITIALIZED"
    assert len(state.experiments) == 0
    assert state.best_model is None
    assert state.optimization_metric is None

def test_workflow_state_to_dict():
    state = WorkflowState(current_stage="VALIDATED")
    d = state.to_dict()
    assert d["current_stage"] == "VALIDATED"
    assert "dataset" in d
    assert "experiments" in d
