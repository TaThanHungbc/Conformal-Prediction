#!/usr/bin/env python3
"""
process.py - FastAPI service for predicting a single uploaded image with:
 - optional background removal (rembg)
 - preprocess for ResNet18
 - load model (single file), load qhat
 - infer probabilities and compute conformal prediction set (prob >= 1 - qhat)

Run:
  uvicorn process:app --host 0.0.0.0 --port 8000

Config via environment variables or edit defaults below:
 - MODEL_FILE_NAME  (default: models/fruit_resnet18_1.pth)
 - QHAT_FILE        (default: qhat.txt)
 - TRAIN_DIR        (default: data/Training)
"""

import os
import io
import json
import logging
from typing import List, Dict, Any, Optional

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms, models

# try rembg
try:
    from rembg import remove as rembg_remove
    REMBG_AVAILABLE = True
except Exception:
    REMBG_AVAILABLE = False

# ---------- Configuration (edit if needed) ----------
MODEL_FILE_NAME = os.environ.get("MODEL_FILE_NAME", "models/fruit_resnet18_1.pth")
QHAT_FILE = os.environ.get("QHAT_FILE", "outputs/qhat.txt")
TRAIN_DIR = os.environ.get("TRAIN_DIR", "data/train/train")  # folder with subfolders per class
IMG_SIZE = int(os.environ.get("IMG_SIZE", "224"))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
UPLOAD_FOLDER = os.environ.get("UPLOAD_FOLDER", "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ---------- Logging ----------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("process")

# ---------- Globals set at startup ----------
app = FastAPI(title="Fruit CP Prediction")
MODEL: Optional[torch.nn.Module] = None
CLASS_NAMES: List[str] = []
QHAT: Optional[float] = None

# ---------- Utilities ----------
def load_qhat(qhat_path: str) -> float:
    if not os.path.exists(qhat_path):
        raise FileNotFoundError(f"qhat file not found: {qhat_path}")
    txt = open(qhat_path, "r", encoding="utf-8").read().strip()
    try:
        return float(txt)
    except Exception as e:
        raise ValueError(f"Cannot parse qhat from file {qhat_path}: '{txt}'") from e

def load_class_names_from_train(train_dir: str) -> List[str]:
    if not os.path.isdir(train_dir):
        raise FileNotFoundError(f"Training directory not found: {train_dir}")
    classes = [d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))]
    classes = sorted(classes)
    if len(classes) == 0:
        raise ValueError(f"No class subfolders found in training directory: {train_dir}")
    return classes

def build_model(num_classes: int, device: torch.device):
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    model.to(device)
    model.eval()
    return model

def load_model_weights(model: torch.nn.Module, model_file: str, device: torch.device):
    if not os.path.exists(model_file):
        raise FileNotFoundError(f"Model file not found: {model_file}")
    sd = torch.load(model_file, map_location=device)
    # If sd seems like state_dict or contains nested 'model'
    if isinstance(sd, dict):
        # common cases: direct state_dict, {'model':state_dict}, or saved with DataParallel prefix 'module.'
        if 'model' in sd and isinstance(sd['model'], dict):
            sd = sd['model']
        # remove "module." prefix if present
        new_sd = {}
        for k, v in sd.items():
            new_k = k.replace("module.", "") if isinstance(k, str) else k
            new_sd[new_k] = v
        sd = new_sd
    try:
        model.load_state_dict(sd)
    except Exception as e:
        raise RuntimeError(f"Failed to load model state dict: {e}") from e

def try_remove_background_bytes(image_bytes: bytes) -> Image.Image:
    """Remove background using rembg if available; else load image as RGB."""
    if REMBG_AVAILABLE:
        try:
            out = rembg_remove(image_bytes)
            img = Image.open(io.BytesIO(out)).convert("RGBA")
            bg = Image.new("RGB", img.size, (255,255,255))
            bg.paste(img, mask=img.split()[3])
            return bg
        except Exception as e:
            logger.warning("rembg failed, using original image: %s", e)
    # fallback
    return Image.open(io.BytesIO(image_bytes)).convert("RGB")

def preprocess_pil_image(img: Image.Image, img_size: int = IMG_SIZE):
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
    return transform(img).unsqueeze(0)  # (1,C,H,W)

def softmax_numpy(logits: np.ndarray) -> np.ndarray:
    ex = np.exp(logits - np.max(logits))
    return ex / ex.sum(axis=-1, keepdims=True)

# ---------- Startup: load model, qhat, classes ----------
@app.on_event("startup")
def startup():
    global MODEL, CLASS_NAMES, QHAT
    logger.info("Starting up - device=%s", DEVICE)
    # Classes
    try:
        CLASS_NAMES = load_class_names_from_train(TRAIN_DIR)
        logger.info("Loaded %d classes from %s", len(CLASS_NAMES), TRAIN_DIR)
    except Exception as e:
        logger.error("Failed to load class names: %s", e)
        # do not raise - allow startup but endpoints will respond with error
        CLASS_NAMES = []

    # qhat
    try:
        QHAT = load_qhat(QHAT_FILE)
        logger.info("Loaded qhat from %s: %s", QHAT_FILE, QHAT)
    except Exception as e:
        logger.error("Failed to load qhat: %s", e)
        QHAT = None

    # model
    if CLASS_NAMES:
        try:
            MODEL = build_model(len(CLASS_NAMES), DEVICE)
            load_model_weights(MODEL, MODEL_FILE_NAME, DEVICE)
            MODEL.eval()
            logger.info("Model loaded from %s", MODEL_FILE_NAME)
        except Exception as e:
            logger.error("Failed to load model: %s", e)
            MODEL = None
    else:
        MODEL = None

# ---------- Pydantic response model (optional) ----------
class CPItem(BaseModel):
    label: str
    prob: float
    prob_pct: str

class PredictResponse(BaseModel):
    argmax: Dict[str, Any]
    conformal_set: List[CPItem]
    all_probs: List[CPItem]
    qhat: Optional[float]
    threshold_prob: Optional[float]

# ---------- Endpoint ----------
@app.post("/predict", response_model=PredictResponse)
async def predict_endpoint(image: UploadFile = File(...)):
    """
    Accepts form file 'image'. Returns JSON with argmax, conformal_set and all_probs.
    """
    # basic checks
    if MODEL is None:
        raise HTTPException(status_code=500, detail="Model not loaded on server.")
    if QHAT is None:
        raise HTTPException(status_code=500, detail="qhat not loaded on server.")
    if not CLASS_NAMES:
        raise HTTPException(status_code=500, detail="Class names not available on server.")

    # read bytes
    try:
        contents = await image.read()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read uploaded file: {e}")

    # optional: save upload for debugging
    save_path = os.path.join(UPLOAD_FOLDER, image.filename)
    try:
        with open(save_path, "wb") as f:
            f.write(contents)
    except Exception:
        # non-fatal
        logger.warning("Could not save uploaded file.")

    # remove background if possible

    try:
        pil_img = try_remove_background_bytes(contents)

        # === SAVE DEBUG IMAGE ===
        debug_path = os.path.join("outputs/", "temp.jpg")
        pil_img.save(debug_path, format="JPEG")
        logger.info("Saved background-removed image to %s", debug_path)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Background removal failed: {e}")


    # preprocess and predict
    try:
        tensor = preprocess_pil_image(pil_img).to(DEVICE)
        with torch.no_grad():
            logits = MODEL(tensor)  # (1, K)
            logits_np = logits.cpu().numpy().ravel()
            probs = softmax_numpy(logits_np)  # (K,)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {e}")

    # argmax
    arg_idx = int(np.argmax(probs))
    arg_label = CLASS_NAMES[arg_idx]
    arg_conf = float(probs[arg_idx])

    # conformal set: keep classes with prob >= 1 - qhat
    threshold = 1.0 - float(QHAT)
    pred_idx = [int(i) for i in np.where(probs >= threshold)[0].tolist()]

    # prepare lists (sorted)
    items = []
    for i in range(len(CLASS_NAMES)):
        p = float(probs[i])
        items.append({"label": CLASS_NAMES[i], "prob": p, "prob_pct": f"{p*100:.2f}%"})
    items_sorted = sorted(items, key=lambda x: x["prob"], reverse=True)

    conformal_items = [items[i] for i in pred_idx]
    conformal_items_sorted = sorted(conformal_items, key=lambda x: x["prob"], reverse=True)

    resp = {
        "argmax": {"label": arg_label, "confidence": arg_conf},
        "conformal_set": conformal_items_sorted,
        "all_probs": items_sorted,
        "qhat": float(QHAT),
        "threshold_prob": threshold
    }
    
    #print(resp)
    
    return JSONResponse(content=resp)

# ---------- Simple health endpoint ----------
@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": MODEL is not None, "qhat_loaded": QHAT is not None, "num_classes": len(CLASS_NAMES)}
