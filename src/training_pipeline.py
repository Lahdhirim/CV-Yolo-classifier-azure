import datetime
import pickle
import time
from pathlib import Path

from src.base_pipeline import BasePipeline
from src.models.training_result import TrainingTracker, TrainStatus
from src.training.evaluate import Evaluator
from src.training.preprocess import DatasetProcessor
from src.training.train import YoloTrainer
from src.utils.logger import training_logger as logger


class TrainingPipeline(BasePipeline):
    """Training pipeline class"""

    def __init__(self, config: dict):
        super().__init__(config)

        # Load configuration parameters
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

        training_config = self.config.get("training", {})
        self.models = training_config.get("models", ["yolov8m-cls"])
        self.epochs = training_config.get("epochs", 10)
        self.batch_size = training_config.get("batch_size", 8)
        self.learning_rate = training_config.get("learning_rate", 1e-4)
        self.optimizer = training_config.get("optimizer", "adamW")

        device_config = self.config.get("device", {})
        self.enable_gpu = device_config.get("use_gpu", False)
        self.num_workers = device_config.get("num_workers", 0)
        self.trainer = YoloTrainer(
            data="./dataset",
            epochs=self.epochs,
            imgsz=self.img_size[0],
            batch_size=self.batch_size,
            learning_rate=self.learning_rate,
            optimizer=self.optimizer,
            enable_gpu=self.enable_gpu,
            num_workers=self.num_workers,
        )

        # Initialize output directory for training results
        self.training_results: dict[str, TrainingTracker] = {}
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_id = f"experiment_{timestamp}"
        self.output_dir = Path("outputs") / self.experiment_id
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize evaluator for model evaluation
        self.evaluator = Evaluator(data_path="./dataset", output_dir=self.output_dir)

    def _save_training_results(self) -> None:
        """Save all training results as a pickle file and a TXT summary file."""
        output_pkl_file = self.output_dir / "training_results.pkl"
        output_txt_file = self.output_dir / "training_results.txt"
        try:

            # Save training results as a pickle file
            with output_pkl_file.open("wb") as file:
                pickle.dump(self.training_results, file)

            logger.info(
                f"[TRAINING] Training results saved successfully: {output_pkl_file}"
            )

            # Save training results summary as a TXT file
            with output_txt_file.open("w", encoding="utf-8") as file:
                file.write(f"Experiment: {self.experiment_id}\n")
                file.write(
                    f"Time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                )
                file.write("=" * 60 + "\n\n")

                for model_name, tracker in self.training_results.items():
                    file.write(f"Model: {model_name}\n")
                    file.write(f"Status: {tracker.status}\n")
                    file.write(f"Training time: {tracker.time_taken:.2f} seconds\n")

                    if tracker.status == TrainStatus.COMPLETED.value:
                        test_accuracy = 0.0
                        if tracker.test_metrics:
                            test_accuracy = tracker.test_metrics.get(
                                "accuracy",
                                0.0,
                            )
                        file.write(f"Test accuracy: {test_accuracy:.2%}\n")

                    else:
                        error_message = (
                            tracker.error_message or "Unknown training error"
                        )
                        file.write(f"Error: {error_message}\n")
                    file.write("-" * 60 + "\n")

            logger.info(f"[TRAINING] Training summary saved: {output_txt_file}")

        except (OSError, pickle.PickleError) as error:
            logger.exception(f"[TRAINING] Failed to save training results: {error}")
            raise

    def run(self) -> None:
        """run the training pipeline."""
        logger.info(f"Running training pipeline with config: {self.config}")
        logger.info(f"Experiment ID: {self.experiment_id}")
        logger.info(f"Experiment output directory: {self.output_dir}")

        # Load and split the dataset
        self.dataset_processor.load_images()
        self.dataset_processor.split_dataset()

        # Train all the models
        assert Path(
            "./dataset"
        ).exists(), "Dataset directory does not exist. Cannot proceed with training."
        for model_name in self.models:
            logger.info(f"[TRAINING] Training model: {model_name}")

            # Create training tracker object
            training_tracker = TrainingTracker(
                model_name=model_name,
                run_path=f"{model_name}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}",
            )

            # Train the model
            start_time = time.time()
            training_tracker = self.trainer.train(model_tracker=training_tracker)
            end_time = time.time()
            training_tracker.time_taken = end_time - start_time

            # Evaluate the model
            if training_tracker.status == TrainStatus.COMPLETED.value:

                # Validation Set
                val_preds = self.evaluator.predict(
                    model_tracker=training_tracker, split="val"
                )
                val_metrics = self.evaluator.compute_metrics(predictions=val_preds)
                training_tracker.val_predictions = val_preds
                training_tracker.val_metrics = val_metrics

                # Test Set
                test_preds = self.evaluator.predict(
                    model_tracker=training_tracker, split="test"
                )
                test_metrics = self.evaluator.compute_metrics(predictions=test_preds)
                training_tracker.test_predictions = test_preds
                training_tracker.test_metrics = test_metrics

                # Save Results in Excel file
                self.evaluator.save_results_to_excel(model_tracker=training_tracker)

            # Store the training result
            self.training_results[model_name] = training_tracker

        # Save training results to pkl file
        self._save_training_results()
