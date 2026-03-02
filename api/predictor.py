"""
Predictor - Chargement du modèle et inférence.

Classe Predictor :
- Charge le modèle Keras à l'initialisation
- Prétraite les images en entrée
- Retourne le masque prédit colorisé (PIL Image)
"""

import io
import sys
from pathlib import Path

import numpy as np
from PIL import Image

# Ajouter src/ au path pour importer les utilitaires du data loader
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from data_loader import decode_predictions, colorize_mask  # noqa: E402


class Predictor:
    """
    Charge un modèle de segmentation Keras et réalise des prédictions.

    Args:
        model_path (str): Chemin vers le fichier .h5 du modèle.
    """

    def __init__(self, model_path: str):
        import tensorflow as tf

        self.model = tf.keras.models.load_model(model_path, compile=False)

        # Déduire la taille d'entrée depuis le modèle
        input_shape = self.model.input_shape  # (None, H, W, 3)
        self.img_height: int = input_shape[1]
        self.img_width: int = input_shape[2]

    def preprocess(self, image_bytes: bytes) -> np.ndarray:
        """
        Charge et prétraite des bytes image pour l'inférence.

        Args:
            image_bytes: Contenu brut du fichier image.

        Returns:
            np.ndarray: Batch de taille (1, H, W, 3), normalisé 0-1.
        """
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image = image.resize((self.img_width, self.img_height), Image.BILINEAR)
        array = np.array(image, dtype=np.float32) / 255.0
        return np.expand_dims(array, axis=0)  # (1, H, W, 3)

    def predict(self, image_array: np.ndarray) -> Image.Image:
        """
        Lance l'inférence et retourne le masque colorisé.

        Args:
            image_array: np.ndarray de shape (1, H, W, 3).

        Returns:
            PIL Image RGB du masque prédit colorisé.
        """
        pred = self.model.predict(image_array, verbose=0)  # (1, H, W, 8)
        mask = decode_predictions(pred[0])                 # (H, W)
        colored = colorize_mask(mask)                      # (H, W, 3) RGB
        return Image.fromarray(colored)

    def predict_from_bytes(self, image_bytes: bytes) -> Image.Image:
        """
        Raccourci : bytes → masque colorisé.

        Args:
            image_bytes: Contenu brut du fichier image.

        Returns:
            PIL Image RGB du masque prédit colorisé.
        """
        image_array = self.preprocess(image_bytes)
        return self.predict(image_array)
