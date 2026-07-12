from src.base_pipeline import BasePipeline
from src.training.preprocess import DatasetProcessor
from src.utils.logger import training_logger as logger


class TrainingPipeline(BasePipeline):
    """Training pipeline class"""

    def __init__(self, config: dict):
        super().__init__(config)

        data_config = self.config.get("data", {})
        self.input_dir = data_config.get("input_dir", None)
        self.img_size = tuple(data_config.get("img_size", [640, 640]))
        self.train_ratio = data_config.get("train_ratio", 0.7)
        self.val_ratio = data_config.get("val_ratio", 0.15)

        self.seed = self.config.get("seed", 42)
        self.dataset_processor = DatasetProcessor(
            input_dir=self.input_dir,
            img_size=self.img_size,
            seed=self.seed,
            train_ratio=self.train_ratio,
            val_ratio=self.val_ratio,
        )

    def run(self) -> None:
        """run the training pipeline."""
        logger.info(f"Running training pipeline with config: {self.config}")

        # Load and preprocess the dataset
        self.dataset_processor.load_images()
        self.dataset_processor.split_dataset()
