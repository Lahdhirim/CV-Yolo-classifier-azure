from src.base_pipeline import BasePipeline
from src.utils.logger import training_logger as logger

class TrainingPipeline(BasePipeline):
    """Training pipeline class"""

    def run(self) -> None:
        """run the training pipeline."""
        logger.info(f"Running training pipeline with config: {self.config}")