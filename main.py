from pathlib import Path
import typer
import yaml
import time

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

if __name__ == "__main__":
    app()