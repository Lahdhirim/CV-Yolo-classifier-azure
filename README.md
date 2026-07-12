# YOLO Classification Fine-Tuning (Under Development 🚧)

This project provides a simple and reproducible pipeline for fine-tuning the **YOLO Classification (YOLO-CLS)** models on custom image classification datasets.

## Tunisian Food Dataset
The dataset used in this project is a collection of images of Tunisian food dishes found on the internet. It contains 11 classes, each with 100 images, totaling 1100 images.

# Installation

### 1. Install `uv`

#### Linux / macOS
```bash
curl -Ls https://astral.sh/uv/install.sh | sh
```
#### Windows
```bash
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### 2. Clone the repository and install dependencies
```bash
git clone https://github.com/Lahdhirim/CV-Yolo-classifier-azure.git
cd CV-Yolo-classifier-azure
uv sync
```
### 3. Run the training pipeline
```bash
uv run main.py train --config configs/train.yaml
```

## License

MIT License