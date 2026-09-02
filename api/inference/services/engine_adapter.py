from .engine import (
    InferenceEngine,
    InferenceResult,
)


class AIInferenceEngineAdapter(InferenceEngine):
    """
    Adapter around the actual AI inference engine.

    Replace the implementation of predict()
    with your engine integration.
    """

    def predict(
        self,
        *,
        model,
        input_type,
        input_data,
        files,
    ):
        if input_type == "TABULAR":
            return self._predict_tabular(
                model=model,
                input_data=input_data,
            )

        if input_type == "IMAGE":
            return self._predict_image(
                model=model,
                files=files,
            )

        if input_type == "MULTIMODAL":
            return self._predict_multimodal(
                model=model,
                input_data=input_data,
                files=files,
            )

        raise ValueError(f"Unsupported input type: {input_type}")

    def _predict_tabular(
        self,
        *,
        model,
        input_data,
    ):
        # TODO: Connect your tabular engine.

        return InferenceResult(
            output={
                "prediction": None,
            },
            confidence=None,
        )

    def _predict_image(
        self,
        *,
        model,
        files,
    ):
        # TODO: Connect your image engine.

        return InferenceResult(
            output={
                "prediction": None,
            },
            confidence=None,
        )

    def _predict_multimodal(
        self,
        *,
        model,
        input_data,
        files,
    ):
        # TODO: Connect your multimodal engine.

        return InferenceResult(
            output={
                "prediction": None,
            },
            confidence=None,
        )
