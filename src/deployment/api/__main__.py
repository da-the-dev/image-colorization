import os
import traceback
import numpy as np
import torch
from torch import nn
from PIL import Image
from flask import Flask, jsonify, request

from src.arch.proper_cgan.dataset import make_transforms
from src.arch.proper_cgan.utils import lab2rgb_denormalize
from src.inference import load_model

app = Flask(__name__)

# Torch device
device = "cuda" if torch.cuda.is_available() else "cpu"

model = None

# Initializing transforms
transforms = make_transforms(256)


@app.post("/colorize")
def classify():
    # Check if an image is part of the request
    if "image" not in request.files:
        return "No image part", 400

    file = request.files["image"]

    # If no file is selected
    if file.filename == "":
        return "No selected file", 400

    try:
        image = Image.open(file.stream)
        image = image.convert("RGB")
        image = transforms(image)
        image = image.unsqueeze(0)
        image = image.to(device)

        with torch.no_grad():
            outputs = model(image)

            color_image = Image.fromarray(lab2rgb_denormalize(outputs))

            return color_image.tobytes()

    except Exception as e:
        traceback.print_exc()

        return str(e), 500


if __name__ == "__main__":
    # Loading the model
    uri = "runs:/1c3deafe5165481c95237221b2085474/gan"
    model = load_model(uri).to(device)

    model.to(device)
    model.eval()

    if os.getenv("PROD"):
        from waitress import serve

        serve(app, host="0.0.0.0", port=8000)
    else:
        app.run(debug=True, port=8000)
