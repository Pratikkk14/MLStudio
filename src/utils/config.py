import os
from pathlib import Path
from dotenv import load_dotenv

# Load env file from root directory
root_dir = Path(__file__).parent.parent.parent
load_dotenv(dotenv_path=root_dir / ".env")

class AppConfig:
    """Manages application configurations and environment variables."""
    LLM_API_KEY: str = os.getenv("LLM_API_KEY", "")
    LLM_MODEL: str = os.getenv("LLM_MODEL", "gpt-4")
    
    @classmethod
    def get_provider(cls) -> str:
        """Retrieves provider based on model naming convention."""
        model_name = cls.LLM_MODEL.lower()
        if "gpt" in model_name:
            return "openai"
        elif "claude" in model_name:
            return "anthropic"
        elif "gemini" in model_name:
            return "google"
        else:
            return "mock"
