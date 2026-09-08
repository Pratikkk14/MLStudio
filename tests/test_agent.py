from src.agent.state import WorkflowState
from src.agent.agent import MockAgent, LLMClient
from src.agent.decisions import AgentAction

def test_mock_agent_decisions():
    state = WorkflowState()
    agent = MockAgent(state)
    
    # 1. Analysis decision
    profile = {
        "summary": {"rows": 100, "columns": 5, "missing_percentage": 2.0}
    }
    warnings = ["Column 'customer_id' has high cardinality. Likely identifier."]
    candidates = ["churn", "status"]
    
    dec_1 = agent.select_analysis_decision(profile, warnings, candidates)
    assert dec_1.next_action == AgentAction.PREPROCESS_DATA
    assert dec_1.parameters["target"] == "churn"
    assert "customer_id" in dec_1.parameters["exclude_columns"]

    # 1.5 Model strategy decision
    feat_summary = {"total_engineered_features": 8, "scaling_applied": "standard", "encoding_applied": "one_hot"}
    dec_strat = agent.select_model_strategy(profile, feat_summary, "binary_classification", "f1", (80, 8))
    assert dec_strat.next_action == AgentAction.RUN_BASELINES
    assert "random_forest" in dec_strat.parameters["selected_candidates"]
    assert "voting_ensemble" in dec_strat.parameters["selected_candidates"]

    # 2. Tuning decision
    baselines = [
        {"model_name": "Logistic Regression", "metrics": {"f1": 0.75}},
        {"model_name": "Random Forest", "metrics": {"f1": 0.88}}
    ]
    dec_2 = agent.select_tuning_decision(baselines, "f1")
    assert dec_2.next_action == AgentAction.TUNE_MODEL
    assert dec_2.parameters["selected_model"] == "Random Forest"

    # 3. Final decision
    best_model = {"model_name": "Random Forest", "metrics": {"f1": 0.90}}
    dec_3 = agent.select_final_recommendation(best_model, baselines)
    assert dec_3.next_action == AgentAction.STOP_WORKFLOW
    assert dec_3.parameters["status"] == "APPROVED"

def test_llm_json_extractor():
    client = LLMClient()
    
    # Plain json
    res1 = client._extract_json('{"key": "value"}')
    assert res1 == {"key": "value"}
    
    # Markdown fenced json
    res2 = client._extract_json('Here is the output:\n```json\n{"status": "ok"}\n```\nDone.')
    assert res2 == {"status": "ok"}
