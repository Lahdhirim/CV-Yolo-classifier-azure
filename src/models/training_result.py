from dataclasses import dataclass
from enum import Enum


class TrainStatus(str, Enum):
    NOT_STARTED = "Not Started"
    FAILED = "Failed"
    COMPLETED = "Completed"


@dataclass
class TrainingTracker:
    model_name: str
    run_path: str
    time_taken: float = 0
    val_predictions: list = None
    val_metrics: dict = None
    test_predictions: list = None
    test_metrics: dict = None
    status: TrainStatus = TrainStatus.NOT_STARTED
    error_message: str = None
