import logging

import torch
from ultralytics import YOLO

from src.models.training_result import TrainingTracker, TrainStatus

logger = logging.getLogger(__name__)


class YoloTrainer:
    def __init__(
        self,
        data: str,
        epochs: int,
        imgsz: tuple[int, int],
        batch_size: int,
        learning_rate: float,
        optimizer: str,
        enable_gpu: bool,
        num_workers: int,
    ):
        self.data = data
        self.epochs = epochs
        self.imgsz = imgsz
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.optimizer = optimizer
        self.device = self._get_device(enable_gpu)
        self.num_workers = num_workers

    def _get_device(self, enable_gpu: bool) -> int | None:
        if enable_gpu:
            if torch.cuda.is_available():
                logger.info(
                    "[GPU] GPU is available. Training will be performed on GPU."
                )
                return 0

            else:
                logger.warning(
                    "[GPU] GPU is not available. Training will be performed on CPU."
                )
                return None

        else:
            logger.info(
                "[GPU] GPU usage is disabled. Training will be performed on CPU."
            )
            return None

    def train(self, model_tracker: TrainingTracker) -> TrainingTracker:
        logger.info(
            f"[TRAINING - {model_tracker.model_name}] Starting training with the following parameters: epochs={self.epochs}, imgsz={self.imgsz}, batch_size={self.batch_size}, learning_rate={self.learning_rate}, optimizer={self.optimizer}, device={'GPU' if self.device is not None else 'CPU'}, num_workers={self.num_workers}"
        )

        if self.device is not None:
            torch.cuda.empty_cache()

        try:
            model = YOLO(model_tracker.model_name)
            train_metrics = model.train(
                data=self.data,
                epochs=self.epochs,
                imgsz=self.imgsz,
                batch=self.batch_size,
                lr0=self.learning_rate,
                optimizer=self.optimizer,
                pretrained=True,
                project="runs",
                device=self.device,
                workers=self.num_workers,
                name=model_tracker.run_path,
            )

            logger.info(
                f"[TRAINING - {model_tracker.model_name}] Training completed successfully. Model saved at: runs/{model_tracker.run_path}"
            )
            model_tracker.status = TrainStatus.COMPLETED.value
            model_tracker.error_message = None
            model_tracker.train_metrics = train_metrics
            return model_tracker

        except Exception as e:
            logger.error(
                f"[TRAINING - {model_tracker.model_name}] Training failed with error: {e}"
            )
            model_tracker.status = TrainStatus.FAILED.value
            model_tracker.error_message = str(e)
            model_tracker.train_metrics = None
            return model_tracker
