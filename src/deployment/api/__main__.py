import io
import os
import traceback
import numpy as np
import torch
from torch import nn
from PIL import Image
from flask import Flask, jsonify, request, send_file
from skimage.color import rgb2lab, lab2rgb

from src.arch.proper_cgan.dataset import make_transforms
from src.arch.proper_cgan.utils import lab2rgb_denormalize, rgb2lab_normalize
from src.inference import load_model


from torchvision import transforms

app = Flask(__name__)

# Torch device
device = "cuda" if torch.cuda.is_available() else "cpu"

model = None


def preprocess_image(img, img_size=256):
    trans = transforms.Resize((img_size, img_size), Image.BICUBIC)
    img = trans(img)
    img = np.array(img)
    img_to_lab = rgb2lab(img).astype("float32")
    img_to_lab = transforms.ToTensor()(img_to_lab)
    L = img_to_lab[[0], ...] / 50.0 - 1.0

    return L


@app.post("/colorize")
def colorize():
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
        L = preprocess_image(image)
        L = L.unsqueeze(0)
        L = L.to(device)

        with torch.no_grad():
            outputs = model(L)

            lab_image = (
                torch.concat([L, outputs], dim=1).detach().cpu().squeeze(0).numpy()
            )

            color_image = lab2rgb_denormalize(lab_image)

            color_image = color_image.astype(np.uint8)

            pil_image = Image.fromarray(color_image)
            print(pil_image.info)
            
            return send_file(io.BytesIO(pil_image.tobytes()), "image/png")

    except Exception as e:
        traceback.print_exc()

        return str(e), 500


if __name__ == "__main__":
    # Loading the model
    uri = "runs:/d840f361ea794eb29d2f7dbe873db4cd/gan"
    model = load_model(uri).to(device)

    model.to(device)
    model.eval()

    if os.getenv("PROD"):
        from waitress import serve

        serve(app, host="0.0.0.0", port=8000)
    else:
        app.run(debug=True, port=8000)
