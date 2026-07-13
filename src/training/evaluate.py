from pathlib import Path

import pandas as pd
from ultralytics import YOLO

from src.data_models.training_result import TrainingTracker
from src.utils.logger import training_logger as logger


class Evaluator:
    def __init__(self, data_path: str, output_dir: Path):
        self.data_path = data_path
        self.output_dir = output_dir

    def predict(self, model_tracker: TrainingTracker, split: str) -> float:

        # Prepare I/O paths
        model_path = Path(
            f"runs/classify/runs/{model_tracker.run_path}/weights/best.pt"
        )

        split_path = Path(self.data_path) / split

        if not model_path.is_file():
            logger.error(
                f"[EVALUATOR] Model file does not exist: {model_path.resolve()}"
            )
            raise FileNotFoundError(
                f"Model file does not exist: {model_path.resolve()}"
            )
        model = YOLO(model_path)

        if not split_path.is_dir():
            logger.error(
                f"[EVALUATOR] Split directory does not exist: {split_path.resolve()}"
            )
            raise FileNotFoundError(
                f"Split directory does not exist: {split_path.resolve()}"
            )

        supported_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".avif"}
        predictions_list = []
        for class_folder in sorted(split_path.iterdir()):
            if not class_folder.is_dir():
                continue

            true_class = class_folder.name

            for image_file in sorted(class_folder.iterdir()):
                if image_file.suffix.lower() not in supported_extensions:
                    logger.warning(
                        f"[EVALUATOR] Skipping unsupported file format: {image_file}"
                    )
                    continue

                prediction = model.predict(source=str(image_file), verbose=False)[0]

                predicted_id = int(prediction.probs.top1)
                predicted_class = prediction.names[predicted_id]
                confidence = float(prediction.probs.top1conf)
                predictions_list.append(
                    {
                        "image": str(image_file),
                        "true_class": true_class,
                        "prediction": predicted_class,
                        "confidence": confidence,
                    }
                )

        return predictions_list

    @staticmethod
    def compute_metrics(predictions: list[dict]) -> dict:
        """Compute accuracy metrics based on predictions."""
        if not predictions:
            logger.warning("[EVALUATOR] No predictions to compute metrics.")
            return {}

        # Overall accuracy calculation
        total_predictions = len(predictions)
        correct_predictions = 0

        # Per class accuracy calculation
        total_per_class = {}
        class_correct = {}

        for pred in predictions:
            true_class = pred["true_class"]
            predicted_class = pred["prediction"]

            if true_class not in total_per_class:
                total_per_class[true_class] = 0
                class_correct[true_class] = 0

            total_per_class[true_class] += 1
            if true_class == predicted_class:
                correct_predictions += 1
                class_correct[true_class] += 1

        # Overall accuracy calculation
        over_all_accuracy = (
            correct_predictions / total_predictions if total_predictions > 0 else 0.0
        )

        # Per class accuracy calculation
        per_class_accuracy = {}
        for class_name, total in total_per_class.items():
            correct = class_correct[class_name]
            per_class_accuracy[class_name] = correct / total if total > 0 else 0.0

        return {
            "total": total_predictions,
            "correct": correct_predictions,
            "accuracy": over_all_accuracy,
            "per_class_accuracy": per_class_accuracy,
        }

    @staticmethod
    def _compute_confusion_matrix(predictions: list[dict]) -> pd.DataFrame:
        """Compute confusion matrix from predictions."""

        predictions_df = pd.DataFrame(predictions)

        if predictions_df.empty:
            return pd.DataFrame()

        return pd.crosstab(
            predictions_df["true_class"],
            predictions_df["prediction"],
            rownames=["True class"],
            colnames=["Predicted class"],
            dropna=False,
        )

    def save_results_to_excel(
        self,
        model_tracker: TrainingTracker,
        val_predictions: list[dict],
        val_metrics: dict,
        test_predictions: list[dict],
        test_metrics: dict,
    ) -> None:
        """Save predictions and metrics to a single Excel file."""

        results_dir = self.output_dir / model_tracker.model_name
        results_dir.mkdir(parents=True, exist_ok=True)

        output_file = results_dir / "evaluation.xlsx"

        val_predictions_df = pd.DataFrame(val_predictions)
        test_predictions_df = pd.DataFrame(test_predictions)

        val_metrics_df = pd.DataFrame(
            {
                "Metric": ["Accuracy"],
                "Value": [val_metrics["accuracy"]],
            }
        )

        for class_name, accuracy in val_metrics["per_class_accuracy"].items():
            val_metrics_df.loc[len(val_metrics_df)] = [
                f"Accuracy - {class_name}",
                accuracy,
            ]

        test_metrics_df = pd.DataFrame(
            {
                "Metric": ["Accuracy"],
                "Value": [test_metrics["accuracy"]],
            }
        )

        for class_name, accuracy in test_metrics["per_class_accuracy"].items():
            test_metrics_df.loc[len(test_metrics_df)] = [
                f"Accuracy - {class_name}",
                accuracy,
            ]

        # Confusion matrix
        val_confusion_matrix = self._compute_confusion_matrix(val_predictions)

        test_confusion_matrix = self._compute_confusion_matrix(test_predictions)

        with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
            val_predictions_df.to_excel(
                writer,
                sheet_name="val_predictions",
                index=False,
            )

            val_metrics_df.to_excel(
                writer,
                sheet_name="val_metrics",
                index=False,
            )

            val_confusion_matrix.to_excel(
                writer,
                sheet_name="val_confusion",
            )

            test_predictions_df.to_excel(
                writer,
                sheet_name="test_predictions",
                index=False,
            )

            test_metrics_df.to_excel(
                writer,
                sheet_name="test_metrics",
                index=False,
            )

            test_confusion_matrix.to_excel(
                writer,
                sheet_name="test_confusion",
            )

        logger.info(f"[EVALUATOR] Results saved to: {output_file}")
