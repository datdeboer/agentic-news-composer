"""Centralized LLM factory. Call configure() before each run to set the provider."""
import os

from langchain_openai import ChatOpenAI

PROVIDERS = {
    "openrouter": {
        "label": "OpenRouter",
        "base_url": "https://openrouter.ai/api/v1",
        "env_key": "OPENROUTER_API_KEY",
        "models": [
            {"id": "openai/gpt-4o-mini", "label": "GPT-4o Mini (free tier)"},
            {"id": "meta-llama/llama-3.3-70b-instruct:free", "label": "Llama 3.3 70B (free)"},
            {"id": "google/gemma-3-27b-it:free", "label": "Gemma 3 27B (free)"},
        ],
    },
    "groq": {
        "label": "Groq",
        "base_url": "https://api.groq.com/openai/v1",
        "env_key": "GROQ_API_KEY",
        "models": [
            {"id": "llama-3.3-70b-versatile", "label": "Llama 3.3 70B Versatile"},
            {"id": "openai/gpt-oss-120b", "label": "GPT OSS 120B"},
            {"id": "llama-3.1-8b-instant", "label": "Llama 3.1 8B Instant (fast)"},
            {"id": "mixtral-8x7b-32768", "label": "Mixtral 8x7B"},
        ],
    },
}

_provider: str = "openrouter"
_model: str | None = None


def configure(provider: str, model: str) -> None:
    """Set the active provider and model for the current run."""
    global _provider, _model
    _provider = provider
    _model = model


def get_llm() -> ChatOpenAI:
    """Return a ChatOpenAI instance configured for the active provider."""
    cfg = PROVIDERS.get(_provider, PROVIDERS["openrouter"])
    api_key = os.environ.get(cfg["env_key"], "")
    model = _model or cfg["models"][0]["id"]
    return ChatOpenAI(base_url=cfg["base_url"], api_key=api_key, model=model)
