"""
Application web Streamlit - Démonstration de segmentation Cityscapes.

Fonctionnalités :
- Liste des images val disponibles (via API)
- Sélection d'une image par ID ou par ville
- Affichage côte à côte : image RGB | masque GT | masque prédit
- Lancement de la prédiction via l'API FastAPI

Lancement :
  streamlit run app/streamlit_app.py
"""

import io

import requests
import streamlit as st
from PIL import Image

# ── Configuration ─────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Cityscapes Segmentation",
    page_icon="🚗",
    layout="wide",
)

import os
DEFAULT_API_URL = os.getenv("API_URL", "http://localhost:8000")

CATEGORY_COLORS = {
    "void":         "#000000",
    "flat":         "#804080",
    "construction": "#464646",
    "object":       "#DCDC00",
    "nature":       "#6B8E23",
    "sky":          "#46B4B4",
    "human":        "#DC143C",
    "vehicle":      "#00008E",
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def get_image_from_api(url: str) -> Image.Image | None:
    """Appelle l'API et retourne une PIL Image, ou None en cas d'erreur."""
    try:
        r = requests.get(url, timeout=60)
        r.raise_for_status()
        return Image.open(io.BytesIO(r.content))
    except Exception as e:
        st.error(f"Erreur API : {e}")
        return None


@st.cache_data(show_spinner="Chargement de la liste des images…")
def fetch_image_list(api_url: str) -> list[dict]:
    """Récupère la liste des images val depuis l'API (mis en cache)."""
    r = requests.get(f"{api_url}/images", timeout=10)
    r.raise_for_status()
    return r.json()


@st.cache_data(show_spinner=False)
def fetch_api_info(api_url: str) -> dict:
    r = requests.get(f"{api_url}/", timeout=5)
    r.raise_for_status()
    return r.json()


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("⚙️ Configuration")
    api_url = st.text_input("URL de l'API", value=DEFAULT_API_URL)

    # Vérification connexion API
    try:
        info = fetch_api_info(api_url)
        st.success("API connectée")
        st.caption(f"Modèle : `{info['model_file']}`")
        st.caption(f"Entrée : `{info['input_size']}`")
        st.caption(f"Images val : **{info['val_images_count']}**")
        api_ok = True
    except Exception as e:
        st.error(f"API inaccessible : {e}")
        st.info("Lancez l'API : `uvicorn api.app:app --port 8000`", icon="ℹ️")
        api_ok = False

    st.divider()
    st.subheader("Légende des classes")
    for name, color in CATEGORY_COLORS.items():
        st.markdown(
            f'<span style="display:inline-block;width:14px;height:14px;'
            f'background:{color};border-radius:2px;margin-right:6px;'
            f'vertical-align:middle"></span>{name}',
            unsafe_allow_html=True,
        )


# ── Main ──────────────────────────────────────────────────────────────────────

st.title("🚗 Segmentation sémantique — Cityscapes")
st.markdown(
    "Sélectionnez une image du dataset Cityscapes pour visualiser "
    "la **segmentation prédite** par le modèle U-Net."
)

if not api_ok:
    st.stop()

# Charger la liste des images
images = fetch_image_list(api_url)

# ── Sélection de l'image ──────────────────────────────────────────────────────

col_filter, col_select = st.columns([1, 2])

with col_filter:
    cities = sorted({img["city"] for img in images})
    selected_city = st.selectbox("Filtrer par ville", ["Toutes"] + cities)

filtered = images if selected_city == "Toutes" else [
    img for img in images if img["city"] == selected_city
]

with col_select:
    options = {
        f"#{img['id']} — {img['city']} — {img['filename']}": img["id"]
        for img in filtered
    }
    selected_label = st.selectbox(
        f"Image ({len(filtered)} disponibles)",
        list(options.keys()),
    )

image_id = options[selected_label]

# ── Bouton prédiction ─────────────────────────────────────────────────────────

predict_clicked = st.button("Lancer la prédiction", type="primary", use_container_width=True)

st.divider()

# ── Affichage ─────────────────────────────────────────────────────────────────

col_rgb, col_gt, col_pred = st.columns(3)

with col_rgb:
    st.subheader("Image RGB")
    with st.spinner("Chargement…"):
        rgb_img = get_image_from_api(f"{api_url}/images/{image_id}/rgb")
    if rgb_img:
        st.image(rgb_img, use_container_width=True)

with col_gt:
    st.subheader("Masque ground truth")
    with st.spinner("Chargement…"):
        gt_img = get_image_from_api(f"{api_url}/images/{image_id}/gt")
    if gt_img:
        st.image(gt_img, use_container_width=True)

with col_pred:
    st.subheader("Masque prédit")
    if predict_clicked:
        with st.spinner("Prédiction en cours…"):
            pred_img = get_image_from_api(f"{api_url}/predict/{image_id}")
        if pred_img:
            st.image(pred_img, use_container_width=True)
            st.success("Prédiction effectuée")
    else:
        st.info("Cliquez sur **Lancer la prédiction** pour afficher le masque prédit.", icon="👆")
