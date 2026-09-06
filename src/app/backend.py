import yaml
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image
from ultralytics import YOLO

from src.utils.logger import inference_logger

inference_logger.info("Inference backend initialized.")


app = FastAPI(
    title="YOLO Classification API",
    version="1.0.0",
    description="REST API for YOLO image classification.",
)

# Load inference configuration
with open("configs/inference.yaml", "r") as f:
    inference_config = yaml.safe_load(f)
inference_logger.info(f"Inference configuration loaded: {inference_config}")

# Initialize model
model_path = inference_config["model"][0]["path"]
inference_logger.info(f"Model path: {model_path}")
try:
    model = YOLO(model_path)
    inference_logger.info("Model successfully initialized.")
except Exception as e:
    inference_logger.error(f"Error initializing model: {e}")
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
        raise HTTPException(status_code=400, detail="File must be an image")

    try:
        image = Image.open(file.file)
        result = model(image)[0]

        class_id = int(result.probs.top1)
        confidence = result.probs.top1conf.item()
        class_name = result.names[class_id]

        inference_logger.info(
            f"Prediction result - Class: {class_name}, Confidence: {confidence}"
        )
        return {"class": class_name, "confidence": confidence, "status": "ok"}

    except Exception as e:
        inference_logger.error(f"Error during prediction: {e}")
        raise HTTPException(status_code=500, detail="Error during prediction")
