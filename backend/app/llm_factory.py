import os

from . import gemini_client
from .base_llm import BaseLLMClient


class GeminiLLMClient(BaseLLMClient):
    def invoke(self, prompt: str, query: str = "", context: str = "") -> dict:
        return gemini_client.route_and_invoke(
            prompt=prompt,
            query=query,
            context=context,
        )


class GroqLLMClient(BaseLLMClient):
    def invoke(self, prompt: str, query: str = "", context: str = "") -> dict:
        from . import groq_client

        return groq_client.route_and_invoke(
            prompt=prompt,
            query=query,
            context=context,
        )


def get_llm_client() -> BaseLLMClient:
    """
    Return active LLM client based on LLM_BACKEND env var.
    Supports:
      - 'gemini' (default): Google Gemini Flash/Pro via Google GenAI SDK
      - 'groq': Free ultra-fast Llama-3.3-70B via Groq API
    """
    backend = os.getenv("LLM_BACKEND", "gemini").lower()
    if backend == "groq":
        return GroqLLMClient()
    return GeminiLLMClient()
