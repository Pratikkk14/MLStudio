from src.agent.state import WorkflowState
from src.agent.agent import MockAgent
from src.agent.decisions import AgentAction

def test_mock_agent_progression():
    state = WorkflowState()
    agent = MockAgent(state)
    
    # Step 1: Initialized -> Validate
    decision = agent.select_next_action()
    assert decision.next_action == AgentAction.VALIDATE_DATASET
    
    state.current_stage = "VALIDATED"
    # Step 2: Validated -> Profile
    decision = agent.select_next_action()
    assert decision.next_action == AgentAction.PROFILE_DATASET
    
    state.current_stage = "COMPLETED"
    decision = agent.select_next_action()
    assert decision.next_action == AgentAction.STOP_WORKFLOW
