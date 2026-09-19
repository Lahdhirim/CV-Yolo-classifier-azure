const imageInput = document.getElementById("image-input");
const predictButton = document.getElementById("predict-button");

const previewContainer = document.getElementById("preview-container");
const imagePreview = document.getElementById("image-preview");

const result = document.getElementById("result");
const prediction = document.getElementById("prediction");
const confidenceValue = document.getElementById("confidence-value");
const confidenceFill = document.getElementById("confidence-fill");

const errorMessage = document.getElementById("error");
const buttonText = document.getElementById("button-text");


imageInput.addEventListener("change", () => {
    const file = imageInput.files[0];

    result.classList.add("hidden");
    errorMessage.textContent = "";

    if (!file) {
        previewContainer.classList.add("hidden");
        return;
    }

    const imageUrl = URL.createObjectURL(file);

    imagePreview.src = imageUrl;
    previewContainer.classList.remove("hidden");
});


predictButton.addEventListener("click", async () => {
    const file = imageInput.files[0];

    if (!file) {
        errorMessage.textContent = "Please select an image first.";
        return;
    }

    const formData = new FormData();
    formData.append("file", file);

    errorMessage.textContent = "";
    result.classList.add("hidden");

    predictButton.disabled = true;
    buttonText.textContent = "Classifying...";

    try {
        const response = await fetch("/api/predict", {
            method: "POST",
            body: formData,
        });

        if (!response.ok) {
            throw new Error(`HTTP error: ${response.status}`);
        }

        const data = await response.json();

        const confidence = data.confidence * 100;

        prediction.textContent = data.class;

        confidenceValue.textContent =
            `${confidence.toFixed(2)}%`;

        confidenceFill.style.width =
            `${confidence}%`;

        result.classList.remove("hidden");

    } catch (error) {
        console.error(error);

        errorMessage.textContent =
            "Prediction failed. Please try again.";

    } finally {
        predictButton.disabled = false;
        buttonText.textContent = "Classify dish";
    }
});