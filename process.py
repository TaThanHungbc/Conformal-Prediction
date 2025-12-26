import os
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from rembg import remove
from torchvision import transforms

from src.model import get_model
import src.config as config

# -------- Load model + qhat once --------
def load_qhat():
    with open("outputs/qhat.txt", "r") as f:
        return float(f.read().strip())

QHAT = load_qhat()

# class names must match training order
CLASS_NAMES = sorted([
    d for d in os.listdir(config.TRAIN_DIR)
    if os.path.isdir(os.path.join(config.TRAIN_DIR, d))
])

MODEL = get_model(len(CLASS_NAMES))
MODEL.load_state_dict(torch.load(config.MODEL_SAVE_PATH, map_location=config.DEVICE))
MODEL.eval()

TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# -------- Main inference function --------
def predict_image(image: Image.Image):
    """
    image: PIL Image from frontend
    return: dict for JSON response
    """

    # 1. Remove background
    image_nobg = remove(image)
    image_nobg = Image.fromarray(image_nobg).convert("RGB")

    # 2. Transform
    x = TRANSFORM(image_nobg).unsqueeze(0).to(config.DEVICE)

    # 3. Predict
    with torch.no_grad():
        probs = F.softmax(MODEL(x), dim=1).cpu().numpy()[0]

    # 4. Argmax
    argmax_idx = int(np.argmax(probs))
    argmax_label = CLASS_NAMES[argmax_idx]
    argmax_conf = float(probs[argmax_idx])

    # 5. Conformal Prediction Set
    threshold = 1 - QHAT
    cp_indices = np.where(probs >= threshold)[0]

    conformal = {
        CLASS_NAMES[i]: float(probs[i])
        for i in cp_indices
    }

    return {
        "argmax": {
            "label": argmax_label,
            "confidence": argmax_conf
        },
        "conformal_prediction": conformal
    }
