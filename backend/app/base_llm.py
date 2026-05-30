from abc import ABC, abstractmethod


class BaseLLMClient(ABC):
    @abstractmethod
    def invoke(self, prompt: str, query: str = "", context: str = "") -> dict:
        raise NotImplementedError
