from dataclasses import dataclass
from enum import Enum


class TrainStatus(str, Enum):
    NOT_STARTED = "Not Started"
    FAILED = "Failed"
    COMPLETED = "Completed"


@dataclass
class TrainingResult:
    model_name: str
    run_path: str
    time_taken: float = 0
    val_accuracy_best: float = None
    val_accuracy_last: float = None
    test_accuracy_best: float = None
    test_accuracy_last: float = None
    status: TrainStatus = TrainStatus.NOT_STARTED
    error_message: str = None
