import json
import re
import time
import requests
from typing import Any, Dict, List, Optional
from src.agent.state import WorkflowState
from src.agent.decisions import AgentDecision, AgentAction
from src.agent.prompts import (
    SYSTEM_PROMPT,
    ANALYSIS_DECISION_PROMPT,
    TUNING_DECISION_PROMPT,
    FINAL_RECOMMENDATION_PROMPT
)
from src.utils.config import AppConfig
from src.utils.logging import setup_logger

logger = setup_logger()

class LLMClient:
    """Multi-Tier LLM Client managing Gemini key rotation and Ollama local fallback."""
    
    def __init__(self):
        self.gemini_keys = AppConfig.get_gemini_keys()
        self.current_key_idx = 0
        self.gemini_models = ["gemini-2.0-flash", "gemini-1.5-flash", "gemini-1.5-pro"]
        self.ollama_base_url = AppConfig.get_ollama_base_url()
        self.ollama_models = AppConfig.get_ollama_models()

    def generate_decision(self, prompt: str, task_tier: str = "major") -> Optional[Dict[str, Any]]:
        """Executes generation across Tier 1 (Gemini rotation) -> Tier 2 (Ollama)."""
        full_prompt = f"{SYSTEM_PROMPT}\n\n{prompt}"
        
        # 1. TIER 1: Try Gemini API with Key Rotation
        if self.gemini_keys:
            total_keys = len(self.gemini_keys)
            
            for attempt in range(total_keys):
                active_key = self.gemini_keys[self.current_key_idx]
                key_label = f"GEMINI_KEY_{self.current_key_idx + 1}"
                
                # Check authentication format
                is_bearer = active_key.startswith("AQ.") or active_key.startswith("ya29.")
                
                for model in self.gemini_models:
                    try:
                        logger.info(f"LLM_REQUEST_GEMINI | Using {key_label} | Model: {model}")
                        if is_bearer:
                            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
                            headers = {
                                "Content-Type": "application/json",
                                "Authorization": f"Bearer {active_key}"
                            }
                        else:
                            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={active_key}"
                            headers = {"Content-Type": "application/json"}
                            
                        payload = {
                            "contents": [{"role": "user", "parts": [{"text": full_prompt}]}],
                            "generationConfig": {
                                "temperature": 0.2,
                                "responseMimeType": "application/json"
                            }
                        }
                        
                        response = requests.post(url, headers=headers, json=payload, timeout=10)
                        if response.status_code == 200:
                            data = response.json()
                            text_content = data["candidates"][0]["content"]["parts"][0]["text"]
                            parsed = self._extract_json(text_content)
                            if parsed:
                                logger.info(f"LLM_RESPONSE_GEMINI_SUCCESS | {key_label} ({model})")
                                return parsed
                        elif response.status_code in [429, 403]:
                            logger.warning(f"GEMINI_LIMIT | Status {response.status_code} on {key_label}. Rotating key...")
                            break  # Rotate key
                        elif response.status_code in [401, 404]:
                            logger.warning(f"GEMINI_AUTH/MODEL_ERROR | Status {response.status_code} for {model}")
                            continue  # Try next model or rotate
                    except Exception as e:
                        logger.warning(f"GEMINI_CALL_ERROR | {key_label}: {e}")
                        break
                
                self.current_key_idx = (self.current_key_idx + 1) % total_keys

        # 2. TIER 2: Fallback to Local Ollama Server
        ollama_model = self.ollama_models["major"] if task_tier == "major" else self.ollama_models["minor"]
        logger.info(f"LLM_FALLBACK_OLLAMA | Endpoint: {self.ollama_base_url} | Model: {ollama_model}")
        print(f"\n[Agent] Connecting to local Ollama server ({ollama_model} @ {self.ollama_base_url})...")

        try:
            url = f"{self.ollama_base_url}/api/generate"
            payload = {
                "model": ollama_model,
                "prompt": full_prompt,
                "stream": False,
                "format": "json"
            }
            # Short connect timeout (5s) so we don't hang if IP is unreachable, 180s read timeout
            response = requests.post(url, json=payload, timeout=(5, 180))
            if response.status_code == 200:
                data = response.json()
                text_content = data.get("response", "")
                parsed = self._extract_json(text_content)
                if parsed:
                    logger.info("LLM_RESPONSE_OLLAMA_SUCCESS")
                    return parsed
        except Exception as e:
            logger.warning(f"OLLAMA_UNREACHABLE | {e}")
            print(f"[!] Note: Ollama connection to {self.ollama_base_url} was unavailable: {e}")

        return None

    def _extract_json(self, raw_text: str) -> Optional[Dict[str, Any]]:
        """Safely parses JSON substring from text or code fences."""
        raw_text = raw_text.strip()
        try:
            return json.loads(raw_text)
        except Exception:
            pass

        match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw_text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except Exception:
                pass

        match = re.search(r"(\{.*\})", raw_text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except Exception:
                pass

        return None


class BaseAgent:
    """Abstract agent controller."""
    def __init__(self, state: WorkflowState):
        self.state = state

    def select_analysis_decision(self, profile: Dict[str, Any], warnings: List[str], target_candidates: List[str]) -> AgentDecision:
        raise NotImplementedError

    def select_tuning_decision(self, baseline_results: List[Dict[str, Any]], primary_metric: str) -> AgentDecision:
        raise NotImplementedError

    def select_final_recommendation(self, best_model: Dict[str, Any], all_results: List[Dict[str, Any]]) -> AgentDecision:
        raise NotImplementedError


class MockAgent(BaseAgent):
    """Deterministic rule-based agent for test and fallback execution."""

    def select_analysis_decision(self, profile: Dict[str, Any], warnings: List[str], target_candidates: List[str]) -> AgentDecision:
        summary = profile["summary"]
        target = target_candidates[0] if target_candidates else "churn"
        task_type = "binary_classification"
        metric = "f1" if any("imbalance" in w.lower() for w in warnings) else "accuracy"
        
        exclude_cols = []
        for w in warnings:
            if "likely identifier" in w.lower() or "constant" in w.lower():
                match = re.search(r"Column '([^']+)'", w)
                if match:
                    exclude_cols.append(match.group(1))

        return AgentDecision(
            observation=f"Dataset contains {summary['rows']} observations across {summary['columns']} features with {len(warnings)} data quality flags.",
            decision=f"Assign target to '{target}', optimize for '{metric.upper()}', and exclude non-predictive columns {exclude_cols}.",
            next_action=AgentAction.PREPROCESS_DATA,
            reason="Prioritizing F1-score mitigates minority class misclassification risks; dropping identifier/constant features prevents memorization.",
            confidence=0.98,
            parameters={
                "target": target,
                "problem_type": task_type,
                "primary_metric": metric,
                "exclude_columns": exclude_cols,
                "numerical_imputation": "median",
                "scaling": "standard",
                "categorical_encoding": "one_hot",
                "use_class_weights": True
            }
        )

    def select_tuning_decision(self, baseline_results: List[Dict[str, Any]], primary_metric: str) -> AgentDecision:
        best_b = max(baseline_results, key=lambda x: x["metrics"].get(primary_metric.lower(), 0.0))
        return AgentDecision(
            observation=f"Baseline candidate '{best_b['model_name']}' leads with cross-validated {primary_metric.upper()} of {best_b['metrics'].get(primary_metric.lower(), 0.0):.4f}.",
            decision=f"Perform hyperparameter optimization on {best_b['model_name']}.",
            next_action=AgentAction.TUNE_MODEL,
            reason="Exhibited the strongest generalization margin and stable fold metrics among all evaluated algorithms.",
            confidence=0.96,
            parameters={"selected_model": best_b["model_name"], "trials": 20}
        )

    def select_final_recommendation(self, best_model: Dict[str, Any], all_results: List[Dict[str, Any]]) -> AgentDecision:
        return AgentDecision(
            observation=f"Champion model {best_model['model_name']} demonstrated excellent performance on the untouched holdout test set.",
            decision=f"Approve {best_model['model_name']} for production deployment.",
            next_action=AgentAction.STOP_WORKFLOW,
            reason="Delivered the highest validated score, stable cross-validation characteristics, and low inference latency.",
            confidence=0.99,
            parameters={"deployable_model": best_model["model_name"], "status": "APPROVED"}
        )


class LLMAgent(BaseAgent):
    """Reasoning agent executing decisions via Multi-Tier LLM Client with heuristic resilience."""
    
    def __init__(self, state: WorkflowState, client: Optional[LLMClient] = None):
        super().__init__(state)
        self.client = client or LLMClient()
        self._fallback_agent = MockAgent(state)

    def select_analysis_decision(self, profile: Dict[str, Any], warnings: List[str], target_candidates: List[str]) -> AgentDecision:
        summary = profile["summary"]
        warnings_str = "\n".join([f"- {w}" for w in warnings]) if warnings else "None"
        default_target = target_candidates[0] if target_candidates else "target"
        
        prompt = ANALYSIS_DECISION_PROMPT.format(
            rows=summary.get("rows", 0),
            columns=summary.get("columns", 0),
            numerical_features=summary.get("numerical_features_count", 0),
            categorical_features=summary.get("categorical_features_count", 0),
            missing_pct=round(summary.get("missing_percentage", 0.0), 2),
            target_candidates=", ".join(target_candidates),
            warnings=warnings_str,
            default_target=default_target,
            detected_problem=self.state.problem.get("task", "binary_classification")
        )
        
        data = self.client.generate_decision(prompt, task_tier="major")
        if data:
            return self._to_decision(data, AgentAction.PREPROCESS_DATA)
        
        # Heuristic fallback if both tiers are uncontactable
        print("[Agent] Notice: External LLMs offline. Using deterministic AutoML reasoning engine.")
        return self._fallback_agent.select_analysis_decision(profile, warnings, target_candidates)

    def select_tuning_decision(self, baseline_results: List[Dict[str, Any]], primary_metric: str) -> AgentDecision:
        results_rows = []
        for r in baseline_results:
            score = r["metrics"].get(primary_metric.lower(), 0.0)
            results_rows.append(f"- {r['model_name']}: {primary_metric.upper()} = {score:.4f} (CV Std: {r.get('cv_std', 0.0):.4f})")
        
        prompt = TUNING_DECISION_PROMPT.format(
            problem_type=self.state.problem.get("task", "binary_classification"),
            primary_metric=primary_metric,
            baseline_results_table="\n".join(results_rows)
        )
        
        data = self.client.generate_decision(prompt, task_tier="major")
        if data:
            return self._to_decision(data, AgentAction.TUNE_MODEL)
            
        print("[Agent] Notice: External LLMs offline. Using deterministic AutoML reasoning engine.")
        return self._fallback_agent.select_tuning_decision(baseline_results, primary_metric)

    def select_final_recommendation(self, best_model: Dict[str, Any], all_results: List[Dict[str, Any]]) -> AgentDecision:
        results_rows = []
        for r in all_results:
            score = r["metrics"].get(self.state.optimization_metric.lower(), 0.0)
            results_rows.append(f"- {r['model_name']}: {self.state.optimization_metric.upper()} = {score:.4f}")

        prompt = FINAL_RECOMMENDATION_PROMPT.format(
            problem_type=self.state.problem.get("task", "binary_classification"),
            primary_metric=self.state.optimization_metric,
            best_model_name=best_model["model_name"],
            cv_score=f"{best_model['metrics'].get(self.state.optimization_metric.lower(), 0.0):.4f}",
            test_score=f"{best_model.get('test_score', best_model['metrics'].get(self.state.optimization_metric.lower(), 0.0)):.4f}",
            all_results_table="\n".join(results_rows)
        )
        
        data = self.client.generate_decision(prompt, task_tier="major")
        if data:
            return self._to_decision(data, AgentAction.STOP_WORKFLOW)
            
        print("[Agent] Notice: External LLMs offline. Using deterministic AutoML reasoning engine.")
        return self._fallback_agent.select_final_recommendation(best_model, all_results)

    def _to_decision(self, data: Dict[str, Any], default_action: AgentAction) -> AgentDecision:
        action_str = data.get("next_action", default_action.value)
        try:
            action = AgentAction(action_str)
        except Exception:
            action = default_action

        return AgentDecision(
            observation=data.get("observation", "Observation from agent reasoning."),
            decision=data.get("decision", "Action decision from agent."),
            next_action=action,
            reason=data.get("reason", "Reasoning generated by AI agent."),
            confidence=float(data.get("confidence", 0.9)),
            parameters=data.get("parameters", {})
        )
