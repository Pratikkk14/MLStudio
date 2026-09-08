import os
import re
from pathlib import Path
from typing import List
from dotenv import load_dotenv

# Load .env file from project root
root_dir = Path(__file__).resolve().parent.parent.parent
load_dotenv(dotenv_path=root_dir / ".env")

class AppConfig:
    """Manages dynamic multi-key configuration and LLM endpoints."""
    
    @classmethod
    def get_gemini_keys(cls) -> List[str]:
        """Discovers all GEMINI_API_KEY variables in environment in sorted order."""
        keys_dict = {}
        for key, value in os.environ.items():
            if key.startswith("GEMINI_API_KEY") and value.strip():
                # Extract numeric index if present (e.g. GEMINI_API_KEY1 -> 1)
                match = re.search(r"\d+", key)
                idx = int(match.group()) if match else 999
                keys_dict[idx] = value.strip()
            elif key == "LLM_API_KEY" and value.strip():
                keys_dict[1000] = value.strip()
                
        # Sort by key index (1, 2, 3...)
        sorted_keys = [keys_dict[k] for k in sorted(keys_dict.keys())]
        return sorted_keys

    @classmethod
    def get_ollama_base_url(cls) -> str:
        """Retrieves Ollama host URL from environment."""
        url = os.getenv("OLLAMA_BASE_URL") or os.getenv("OLLAMA_HOST") or "http://10.10.8.7:11434"
        return url.strip().rstrip("/")

    @classmethod
    def get_ollama_models(cls) -> dict:
        """Retrieves designated Ollama model names."""
        return {
            "major": os.getenv("OLLAMA_MAJOR_MODEL", "gemma4:e4b"),
            "minor": os.getenv("OLLAMA_MINOR_MODEL", "llama3.2:3b")
        }

    @classmethod
    def get_gemini_model(cls) -> str:
        """Default Gemini model identifier."""
        return os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
