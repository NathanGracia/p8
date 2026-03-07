"""
API FastAPI - Segmentation sémantique Cityscapes.

Endpoints :
  GET  /                         → health check + infos modèle
  GET  /images                   → liste des images val disponibles
  GET  /images/{image_id}/rgb    → image RGB originale (PNG)
  GET  /images/{image_id}/gt     → masque ground truth colorisé (PNG)
  POST /predict                  → upload image → masque prédit (PNG)
  GET  /predict/{image_id}       → prédiction sur image du dataset (PNG)

Lancement :
  cd api
  uvicorn app:app --reload --port 8000
"""

import io
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from PIL import Image

# Ajouter src/ au path pour importer les utilitaires du data loader
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from data_loader import (  # noqa: E402
    CATEGORY_NAMES,
    CATEGORY_TO_ID,
    LABEL_TO_CATEGORY,
    colorize_mask,
)
from predictor import Predictor  # noqa: E402

# ── Configuration ─────────────────────────────────────────────────────────────

import os

DATA_ROOT = Path(os.getenv("DATA_ROOT", str(Path(__file__).parent.parent / "data")))
MODEL_PATH = Path(os.getenv("MODEL_PATH", str(Path(__file__).parent.parent / "models" / "unet_mobilenet_20260209_132821.h5")))

# ── État global ───────────────────────────────────────────────────────────────

predictor: Predictor = None
val_images: List[dict] = []


# ── Helpers ───────────────────────────────────────────────────────────────────

def scan_val_images() -> List[dict]:
    """Parcourt data/leftImg8bit/val/ et retourne la liste des images disponibles."""
    val_rgb_dir = DATA_ROOT / "leftImg8bit" / "val"
    val_gt_dir = DATA_ROOT / "gtFine" / "val"

    images = []
    image_id = 0

    for city_dir in sorted(val_rgb_dir.iterdir()):
        if not city_dir.is_dir():
            continue
        city = city_dir.name
        for rgb_path in sorted(city_dir.glob("*_leftImg8bit.png")):
            stem = rgb_path.name.replace("_leftImg8bit.png", "")
            gt_path = val_gt_dir / city / f"{stem}_gtFine_labelIds.png"
            if gt_path.exists():
                images.append(
                    {
                        "id": image_id,
                        "city": city,
                        "filename": rgb_path.name,
                        "rgb_path": str(rgb_path),
                        "gt_path": str(gt_path),
                    }
                )
                image_id += 1

    return images


def labelids_to_colorized(gt_path: str) -> Image.Image:
    """Charge un masque labelIds et retourne l'image RGB colorisée."""
    label_mask = np.array(Image.open(gt_path), dtype=np.uint8)
    h, w = label_mask.shape
    category_mask = np.zeros((h, w), dtype=np.uint8)
    for label_id, category_name in LABEL_TO_CATEGORY.items():
        category_mask[label_mask == label_id] = CATEGORY_TO_ID[category_name]
    colored = colorize_mask(category_mask)
    return Image.fromarray(colored)


def pil_to_png_response(img: Image.Image) -> StreamingResponse:
    """Convertit une PIL Image en StreamingResponse PNG."""
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/png")


# ── Lifespan (startup / shutdown) ─────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    global predictor, val_images
    print(f"Chargement du modèle : {MODEL_PATH}")
    predictor = Predictor(str(MODEL_PATH))
    print(f"Modèle prêt — entrée : {predictor.img_height}×{predictor.img_width}")
    val_images = scan_val_images()
    print(f"Dataset val : {len(val_images)} images disponibles")
    yield


# ── Application ───────────────────────────────────────────────────────────────

app = FastAPI(
    title="Cityscapes Segmentation API",
    description=(
        "API de prédiction de segmentation sémantique sur le dataset Cityscapes. "
        "Entrée : une image. Sortie : masque de segmentation colorisé (PNG)."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/", summary="Health check")
def health_check():
    """Retourne l'état de l'API et les informations sur le modèle chargé."""
    return {
        "status": "ok",
        "model": "unet_mini",
        "model_file": MODEL_PATH.name,
        "input_size": f"{predictor.img_height}×{predictor.img_width}",
        "num_classes": 8,
        "categories": CATEGORY_NAMES,
        "val_images_count": len(val_images),
    }


@app.get("/images", summary="Liste des images val")
def list_images():
    """Retourne la liste des images de validation disponibles."""
    return [
        {"id": img["id"], "city": img["city"], "filename": img["filename"]}
        for img in val_images
    ]


@app.get("/images/{image_id}/rgb", summary="Image RGB originale")
def get_rgb_image(image_id: int):
    """Retourne l'image RGB originale du dataset (PNG)."""
    if image_id < 0 or image_id >= len(val_images):
        raise HTTPException(status_code=404, detail=f"Image {image_id} introuvable")
    img = Image.open(val_images[image_id]["rgb_path"]).convert("RGB")
    return pil_to_png_response(img)


@app.get("/images/{image_id}/gt", summary="Masque ground truth colorisé")
def get_gt_mask(image_id: int):
    """Retourne le masque ground truth colorisé (8 classes) en PNG."""
    if image_id < 0 or image_id >= len(val_images):
        raise HTTPException(status_code=404, detail=f"Image {image_id} introuvable")
    img = labelids_to_colorized(val_images[image_id]["gt_path"])
    return pil_to_png_response(img)


@app.post("/predict", summary="Prédiction sur image uploadée")
async def predict_upload(file: UploadFile = File(...)):
    """
    Reçoit une image (multipart), retourne le masque de segmentation prédit (PNG).

    - **file** : fichier image (JPEG, PNG, etc.)
    """
    contents = await file.read()
    try:
        predicted_img = predictor.predict_from_bytes(contents)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    return pil_to_png_response(predicted_img)


@app.get("/predict/{image_id}", summary="Prédiction sur image du dataset")
def predict_by_id(image_id: int):
    """
    Lance la prédiction sur une image du dataset val (identifiée par son ID).

    Retourne le masque de segmentation prédit colorisé (PNG).
    """
    if image_id < 0 or image_id >= len(val_images):
        raise HTTPException(status_code=404, detail=f"Image {image_id} introuvable")
    with open(val_images[image_id]["rgb_path"], "rb") as f:
        image_bytes = f.read()
    predicted_img = predictor.predict_from_bytes(image_bytes)
    return pil_to_png_response(predicted_img)
