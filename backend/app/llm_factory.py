from . import gemini_client
from .base_llm import BaseLLMClient


class GeminiLLMClient(BaseLLMClient):
    def invoke(self, prompt: str, query: str = "", context: str = "") -> dict:
        return gemini_client.route_and_invoke(
            prompt=prompt,
            query=query,
            context=context,
        )


def get_llm_client() -> BaseLLMClient:
    return GeminiLLMClient()
