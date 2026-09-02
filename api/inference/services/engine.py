from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class InferenceResult:
    output: dict[str, Any]
    confidence: float | None = None


class InferenceEngine(ABC):
    """
    Interface implemented by the AI inference engine.

    The Django application does not need to know
    how the underlying models perform inference.
    """

    @abstractmethod
    def predict(
        self,
        *,
        model,
        input_type: str,
        input_data: dict,
        files: list,
    ) -> InferenceResult:
        raise NotImplementedError
