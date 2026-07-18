import time
from pathlib import Path

import typer
import yaml

from src.model_registration_pipeline import RegistrationPipeline
from src.training_pipeline import TrainingPipeline
from src.utils.logger import logger

app = typer.Typer(name="Yolo Classifier")
logger.info("Starting YOLO Classifier CLI application.")


@app.command(name="train")
def train(
    config: Path = typer.Option(
        ...,
        "--config",
        "-c",
        help="Path to the training configuration file.",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
    )
) -> None:
    """
    Train a YOLO model using the specified configuration file.
    """
    with open(config, "r") as f:
        training_config = yaml.safe_load(f)
    logger.info(f"Loaded training configuration: {training_config}")

    # Initialize and run the training pipeline
    start_time = time.time()
    logger.info("Running the training pipeline...")
    pipeline = TrainingPipeline(config=training_config)
    pipeline.run()
    end_time = time.time()
    logger.info(f"Training pipeline completed in {end_time - start_time:.2f} seconds.")


@app.command(name="register_models")
def register_models(
    config: Path = typer.Option(
        ...,
        "--config",
        "-c",
        help="Path to the model registration configuration file.",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
    )
) -> None:
    with open(config, "r") as f:
        registration_config = yaml.safe_load(f)
    logger.info(f"Loaded model registration configuration: {registration_config}")

    # Initialize and run the model registration pipeline
    start_time = time.time()
    logger.info("Running the model registration pipeline...")
    pipeline = RegistrationPipeline(config=registration_config)
    pipeline.run()
    end_time = time.time()
    logger.info(
        f"Model registration pipeline completed in {end_time - start_time:.2f} seconds."
    )


if __name__ == "__main__":
    app()
