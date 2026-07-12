import torch
from ultralytics import YOLO

from src.data_models.training_result import TrainStatus
from src.utils.logger import training_logger as logger


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

    def train(self, model_name: str, model_path: str) -> None:
        logger.info(
            f"[TRAINING - {model_name}] Starting training with the following parameters: epochs={self.epochs}, imgsz={self.imgsz}, batch_size={self.batch_size}, learning_rate={self.learning_rate}, optimizer={self.optimizer}, device={'GPU' if self.device is not None else 'CPU'}, num_workers={self.num_workers}"
        )

        try:
            model = YOLO(model_name)
            model.train(
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
                name=model_path,
            )
            logger.info(
                f"[TRAINING - {model_name}] Training completed successfully. Model saved at: runs/{model_path}"
            )
            return TrainStatus.COMPLETED.value, None

        except Exception as e:
            logger.error(f"[TRAINING - {model_name}] Training failed with error: {e}")
            return TrainStatus.FAILED.value, str(e)
