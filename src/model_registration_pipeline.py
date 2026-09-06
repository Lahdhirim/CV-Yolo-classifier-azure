import os

from src.azure.azure_service import AzureService
from src.base_pipeline import BasePipeline
from src.utils.logger import create_logger

logger = create_logger(
    "registration_pipeline", "logs/registration_pipeline.log", propagate=False
)


class RegistrationPipeline(BasePipeline):
    """Azure ML model registration pipeline class"""

    def __init__(self, config: dict, azure_service: AzureService):
        super().__init__(config)

        self.azure_service = azure_service

        # Load configuration parameters
        self.models = self.config.get("models", [])

    def run(self) -> None:
        logger.info(
            f"Starting model registration pipeline with {len(self.models)} models to register: {self.models}"
        )

        if not self.models:
            logger.error("No models specified for registration. Exiting the pipeline.")
            raise ValueError(
                "No models specified for registration. Please check the configuration file."
            )

        successful_registrations, failed_registrations = [], []
        for model in self.models:
            try:
                logger.info(f"Registering model: {model}")
                model_name, model_path = model.get("name"), model.get("path")
                if not model_name or not model_path:
                    logger.error(
                        f"[ERROR] Model name or path is missing for model: {model}. Skipping registration for model: {model_name}."
                    )
                    failed_registrations.append((model, "Missing name or path"))
                    continue

                if not os.path.exists(model_path):
                    logger.error(
                        f"[ERROR] Model file does not exist: {model_path}. Skipping registration for model: {model_name}."
                    )
                    failed_registrations.append((model, "Model path does not exist"))
                    continue

                if not model_path.endswith(".pt"):
                    logger.error(
                        f"[ERROR] Model file is not a .pt file: {model_path}. Skipping registration for model: {model_name}."
                    )
                    failed_registrations.append((model, "Model path is not a .pt file"))
                    continue

                self.azure_service.register_model(
                    model_name=model_name, model_path=model_path
                )
                successful_registrations.append((model, "Registered successfully"))
                logger.info(f"[SUCCESS] Successfully registered model: {model_name}")

            except Exception as e:
                failed_registrations.append((model, str(e)))
                logger.error(f"[ERROR] Failed to register model: {model}. Error: {e}")

        # Log summary of registration results
        logger.info("Model registration summary:")
        for model, status in successful_registrations:
            logger.info(f"Model registered successfully: {model} - Status: {status}")
        for model, status in failed_registrations:
            logger.error(f"Model registration failed: {model} - Status: {status}")
        logger.info("Model registration pipeline completed.")
