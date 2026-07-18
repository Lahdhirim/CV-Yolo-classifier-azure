from src.base_pipeline import BasePipeline
from src.utils.logger import azure_logger as logger


class RegistrationPipeline(BasePipeline):
    """Azure ML model registration pipeline class"""

    def __init__(self, config: dict):
        super().__init__(config)

        # Load configuration parameters

    def run(self):
        logger.info("Starting model registration pipeline...")
        pass
        logger.info("Model registration pipeline completed.")
