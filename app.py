"""
FastAPI Inference Server — Self-Pruning Neural Network
Author: Ayush Paul (ayushpaul1805@gmail.com)
"""

from fastapi import FastAPI, UploadFile, File, HTTPException
import torch
import torchvision.transforms as transforms
from PIL import Image
import io
import os
from solution import PruningNet

app = FastAPI(
    title="CIFAR-10 Pruning Classifier",
    description="Inference API for a self-pruning neural network trained on CIFAR-10. Built by Ayush Paul.",
    version="1.0.0"
)

CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck']

MODEL_PATH = "model_0.0001.pth"

model = PruningNet()
if os.path.exists(MODEL_PATH):
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()
    print(f"[Ayush Paul] Model loaded from {MODEL_PATH}")
else:
    print(f"[Warning] Model file '{MODEL_PATH}' not found. Run solution.py first to train.")

transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])


@app.get("/")
def root():
    return {
        "message": "CIFAR-10 Pruning Classifier API",
        "author": "Ayush Paul",
        "usage": "POST /predict with an image file"
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Upload an image and get a CIFAR-10 class prediction.
    Supported formats: JPEG, PNG, BMP.
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")

    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    except Exception:
        raise HTTPException(status_code=400, detail="Could not read image. Ensure it's a valid image file.")

    input_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.softmax(output, dim=1)[0]
        prediction_idx = torch.argmax(probabilities).item()
        confidence = probabilities[prediction_idx].item()

    return {
        "class": CLASSES[prediction_idx],
        "confidence": round(confidence * 100, 2),
        "class_index": prediction_idx
    }


@app.get("/classes")
def get_classes():
    """Returns all CIFAR-10 class labels."""
    return {"classes": CLASSES}
