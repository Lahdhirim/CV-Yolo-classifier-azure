from pathlib import Path

import yaml
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image
from ultralytics import YOLO

from src.azure.azure_service import AzureService
from src.utils.logger import create_logger

logger = create_logger("inference_logger", "logs/inference.log", propagate=False)

# Initialize Azure Service
azure_config = Path("configs/azure_service.yaml")
with open(azure_config, "r") as f:
    azure_service_config = yaml.safe_load(f)
    logger.info(f"Loaded Azure service configuration: {azure_service_config}")
azure_service = AzureService(config=azure_service_config)
logger.info("Azure service initialized successfully.")

# Initialize FastAPI application
app = FastAPI(
    title="YOLO Classification API",
    version="1.0.0",
    description="REST API for YOLO image classification.",
)
logger.info("FastAPI application initialized successfully.")

# Load inference configuration
with open("configs/inference.yaml", "r") as f:
    inference_config = yaml.safe_load(f)
logger.info(f"Inference configuration loaded: {inference_config}")

# Download model from Azure Machine Learning workspace
model_config = inference_config["model"][0]
model_name = model_config["name"]
model_version = model_config["version"]
logger.info(f"Model name: {model_name}, version: {model_version}")

try:
    model_path = azure_service.download_model(model_name, model_version)
    model = YOLO(model_path)
    logger.info("Model successfully initialized.")
except Exception as e:
    logger.error(f"Error initializing model: {e}")
    raise e


# Health check and prediction endpoints for the YOLO Classification API
@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": "loaded",
    }


@app.post("/predict")
def predict(file: UploadFile = File(...)):

    if not file.content_type.startswith("image/"):
        logger.error("Uploaded file is not an image.")
        raise HTTPException(status_code=400, detail="File must be an image")

    try:
        image = Image.open(file.file)
        result = model(image)[0]

        class_id = int(result.probs.top1)
        confidence = result.probs.top1conf.item()
        class_name = result.names[class_id]

        logger.info(
            f"Prediction result - Class: {class_name}, Confidence: {confidence}"
        )
        return {"class": class_name, "confidence": confidence, "status": "ok"}

    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        raise HTTPException(status_code=500, detail="Error during prediction")
