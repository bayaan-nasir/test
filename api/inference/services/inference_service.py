from .engine_adapter import (
    AIInferenceEngineAdapter,
)


class InferenceService:
    def __init__(self, engine=None):
        self.engine = engine or AIInferenceEngineAdapter()

    def execute(self, inference):
        files = list(inference.input_files.all())

        result = self.engine.predict(
            model=inference.model,
            input_type=inference.input_type,
            input_data=inference.input_data,
            files=files,
        )

        return result
