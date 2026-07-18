import logging
import os


def create_logger(name: str, log_file: str, propagate: bool = True) -> logging.Logger:
    """Create and configure a logger writing to a specific file."""

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    if not logger.handlers:

        # Ensure log directory exists
        os.makedirs("logs", exist_ok=True)

        file_handler = logging.FileHandler(log_file, mode="w")
        file_handler.setLevel(logging.DEBUG)

        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )

        file_handler.setFormatter(formatter)

        logger.addHandler(file_handler)

        logger.propagate = propagate

    return logger


# Main application logger
logger = create_logger("main_logger", "logs/logs.log")

# Logger for training pipeline
training_logger = create_logger("training_logger", "logs/training.log", propagate=False)

# Logger for model registration pipeline
azure_logger = create_logger("azure_logger", "logs/azure.log", propagate=False)
