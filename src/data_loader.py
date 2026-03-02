"""
DataLoader Keras pour le dataset Cityscapes - Segmentation sémantique

Ce module contient la classe CityscapesSequence qui charge les images
et leurs masques de segmentation pour l'entraînement avec Keras.

Étapes du preprocessing :
1. Charger l'image RGB et le masque labelIds
2. Redimensionner (2048x1024 → taille configurable)
3. Convertir labelIds (0-33) → 8 catégories (0-7)
4. Normaliser les pixels (0-255 → 0-1)
5. Data augmentation (optionnel)
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from pathlib import Path
from PIL import Image
from typing import Tuple, Optional, List, Union
import albumentations as A

# Importer les pipelines d'augmentation avancés
try:
    from data_augmentation import get_augmentation_pipeline, RARE_CLASSES, MIN_PIXELS_THRESHOLD
except ImportError:
    # Fallback si data_augmentation.py n'est pas disponible
    RARE_CLASSES = [6, 7]  # human, vehicle
    MIN_PIXELS_THRESHOLD = 100
    def get_augmentation_pipeline(mode='standard', custom_pipeline=None):
        return None


# ============================================================================
# CONFIGURATION DES 8 CATÉGORIES CITYSCAPES
# ============================================================================

# Mapping labelId (0-33) → nom de catégorie
LABEL_TO_CATEGORY = {
    0: 'void', 1: 'void', 2: 'void', 3: 'void', 4: 'void', 5: 'void', 6: 'void',
    7: 'flat', 8: 'flat', 9: 'flat', 10: 'flat',
    11: 'construction', 12: 'construction', 13: 'construction',
    14: 'construction', 15: 'construction', 16: 'construction',
    17: 'object', 18: 'object', 19: 'object', 20: 'object',
    21: 'nature', 22: 'nature',
    23: 'sky',
    24: 'human', 25: 'human',
    26: 'vehicle', 27: 'vehicle', 28: 'vehicle', 29: 'vehicle',
    30: 'vehicle', 31: 'vehicle', 32: 'vehicle', 33: 'vehicle',
}

# Mapping nom de catégorie → ID (0-7)
CATEGORY_TO_ID = {
    'void': 0,
    'flat': 1,
    'construction': 2,
    'object': 3,
    'nature': 4,
    'sky': 5,
    'human': 6,
    'vehicle': 7,
}

# Couleurs pour visualisation (BGR pour OpenCV)
CATEGORY_COLORS = {
    0: (0, 0, 0),         # void - noir
    1: (128, 64, 128),    # flat - violet
    2: (70, 70, 70),      # construction - gris
    3: (220, 220, 0),     # object - jaune
    4: (107, 142, 35),    # nature - vert
    5: (70, 130, 180),    # sky - bleu ciel
    6: (220, 20, 60),     # human - rouge
    7: (0, 0, 142),       # vehicle - bleu foncé
}

# Noms des catégories
CATEGORY_NAMES = ['void', 'flat', 'construction', 'object', 'nature', 'sky', 'human', 'vehicle']


# ============================================================================
# CLASSE SEQUENCE KERAS
# ============================================================================

class CityscapesSequence(keras.utils.Sequence):
    """
    Générateur de données Keras pour Cityscapes.

    Hérite de keras.utils.Sequence pour un chargement efficace et multithread.

    Args:
        root_dir (str): Chemin vers le dossier data/
        split (str): 'train' ou 'test'
        batch_size (int): Taille des batchs
        img_size (tuple): Taille de sortie (height, width), ex: (256, 512)
        shuffle (bool): Mélanger les données à chaque epoch
        augmentation (bool or str): Appliquer data augmentation
            - False: pas d'augmentation
            - True: augmentation standard
            - 'light', 'standard', 'advanced', 'aggressive': pipelines specifiques
        preprocessing (callable, optional): Fonction de preprocessing (ex: pour encoder pré-entraîné)
        oversample_rare_classes (bool): Sur-echantillonner les images avec classes rares
        oversample_factor (int): Facteur de duplication pour classes rares (2 = doubler, 3 = tripler)
        rare_classes (list): Liste des IDs de classes rares a sur-echantillonner
        min_pixels_rare (int): Nombre minimum de pixels pour considerer la classe presente
    """

    def __init__(
        self,
        root_dir: str,
        split: str = 'train',
        batch_size: int = 8,
        img_size: Tuple[int, int] = (256, 512),
        shuffle: bool = True,
        augmentation: Union[bool, str] = False,
        preprocessing: Optional[callable] = None,
        oversample_rare_classes: bool = False,
        oversample_factor: int = 2,
        rare_classes: Optional[List[int]] = None,
        min_pixels_rare: int = 100
    ):
        self.root_dir = Path(root_dir)
        self.split = split
        self.batch_size = batch_size
        self.img_size = img_size  # (height, width)
        self.shuffle = shuffle
        self.augmentation = augmentation
        self.preprocessing = preprocessing
        self.oversample_rare_classes = oversample_rare_classes
        self.oversample_factor = oversample_factor
        self.rare_classes = rare_classes if rare_classes is not None else RARE_CLASSES
        self.min_pixels_rare = min_pixels_rare

        # Chemins des données
        self.gtfine_dir = self.root_dir / 'gtFine' / split
        self.images_dir = self.root_dir / 'leftImg8bit' / split

        # Récupérer tous les fichiers labelIds
        self.label_files = sorted(self.gtfine_dir.rglob('*_labelIds.png'))

        if len(self.label_files) == 0:
            raise ValueError(f"Aucune image trouvee dans {self.gtfine_dir}")

        # Initialiser les indices (sera modifie si oversample=True)
        self.indices = np.arange(len(self.label_files))
        self.rare_indices = []  # Indices des images avec classes rares

        # Sur-echantillonnage des classes rares
        if self.oversample_rare_classes:
            print(f"   Analyse des classes rares en cours...")
            self._identify_and_oversample_rare_classes()

        # Créer le pipeline d'augmentation
        self.augmentor = self._get_augmentation_pipeline()

        print(f"OK Dataset {split} initialise : {len(self.label_files)} images")
        print(f"   Batch size: {batch_size}, Image size: {img_size}")
        print(f"   Shuffle: {shuffle}, Augmentation: {augmentation}")
        if self.oversample_rare_classes:
            print(f"   Sur-echantillonnage: {len(self.rare_indices)} images avec classes rares (x{oversample_factor})")
            print(f"   Total effectif: {len(self.indices)} samples")


    def __len__(self) -> int:
        """Retourne le nombre de batchs par epoch (incluant sur-echantillonnage)."""
        return int(np.ceil(len(self.indices) / self.batch_size))


    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Génère un batch de données.

        Args:
            idx (int): Index du batch

        Returns:
            batch_images (np.ndarray): Images, shape (batch_size, H, W, 3)
            batch_masks (np.ndarray): Masques one-hot, shape (batch_size, H, W, 8)
        """
        # Récupérer les indices pour ce batch
        batch_indices = self.indices[idx * self.batch_size : (idx + 1) * self.batch_size]

        # Initialiser les arrays
        batch_images = np.zeros((len(batch_indices), *self.img_size, 3), dtype=np.float32)
        batch_masks = np.zeros((len(batch_indices), *self.img_size, 8), dtype=np.float32)

        # Charger chaque image du batch
        for i, file_idx in enumerate(batch_indices):
            image, mask = self._load_sample(file_idx)
            batch_images[i] = image
            batch_masks[i] = mask

        return batch_images, batch_masks


    def on_epoch_end(self):
        """Appelé à la fin de chaque epoch. Mélange les données si shuffle=True."""
        if self.shuffle:
            np.random.shuffle(self.indices)


    def _load_sample(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Charge et prétraite une image et son masque.

        Args:
            idx (int): Index de l'image

        Returns:
            image (np.ndarray): Image normalisée, shape (H, W, 3)
            mask_onehot (np.ndarray): Masque one-hot, shape (H, W, 8)
        """
        # 1. CHARGER LES FICHIERS
        label_path = self.label_files[idx]

        # Trouver l'image RGB correspondante
        # Exemple: jena_000000_000019_gtFine_labelIds.png
        #       -> jena_000000_000019_leftImg8bit.png
        city = label_path.parent.name
        filename = label_path.name.replace('_gtFine_labelIds.png', '_leftImg8bit.png')
        image_path = self.images_dir / city / filename

        # Charger
        image = Image.open(image_path).convert('RGB')
        label_mask = Image.open(label_path)


        # 2. REDIMENSIONNER
        image = image.resize((self.img_size[1], self.img_size[0]), Image.BILINEAR)
        label_mask = label_mask.resize((self.img_size[1], self.img_size[0]), Image.NEAREST)


        # 3. CONVERTIR EN NUMPY
        image = np.array(image, dtype=np.float32)  # Shape: (H, W, 3)
        label_mask = np.array(label_mask, dtype=np.uint8)  # Shape: (H, W)


        # 4. CONVERTIR labelIds → catégories (0-7)
        category_mask = self._convert_to_categories(label_mask)


        # 5. DATA AUGMENTATION (si activée)
        if self.augmentor is not None:
            augmented = self.augmentor(image=image, mask=category_mask)
            image = augmented['image']
            category_mask = augmented['mask']


        # 6. NORMALISER L'IMAGE (0-255 → 0-1)
        image = image / 255.0


        # 7. PREPROCESSING SPÉCIFIQUE (ex: pour encoder pré-entraîné)
        if self.preprocessing is not None:
            image = self.preprocessing(image)


        # 8. CONVERTIR MASQUE EN ONE-HOT ENCODING
        # category_mask: (H, W) avec valeurs 0-7
        # mask_onehot: (H, W, 8) avec 0/1
        mask_onehot = tf.keras.utils.to_categorical(category_mask, num_classes=8)


        return image, mask_onehot


    def _convert_to_categories(self, label_mask: np.ndarray) -> np.ndarray:
        """
        Convertit un masque labelIds (0-33) en catégories (0-7).

        Args:
            label_mask (np.ndarray): Masque avec labelIds, shape (H, W)

        Returns:
            category_mask (np.ndarray): Masque avec catégories 0-7, shape (H, W)
        """
        h, w = label_mask.shape
        category_mask = np.zeros((h, w), dtype=np.uint8)

        # Mapper chaque labelId à sa catégorie
        for label_id, category_name in LABEL_TO_CATEGORY.items():
            category_id = CATEGORY_TO_ID[category_name]
            category_mask[label_mask == label_id] = category_id

        return category_mask


    def _identify_and_oversample_rare_classes(self):
        """
        Identifie les images contenant des classes rares et sur-echantillonne ces images.

        Parcourt tous les masques, identifie ceux contenant les classes rares
        (human, vehicle par defaut), et duplique leurs indices selon oversample_factor.

        Modifie self.indices et self.rare_indices.
        """
        rare_indices = []

        # Parcourir tous les masques (avec un sous-echantillonnage pour accelerer)
        total_files = len(self.label_files)
        check_interval = max(1, total_files // 500)  # Verifier max 500 images

        for idx in range(0, total_files, check_interval):
            label_path = self.label_files[idx]

            try:
                # Charger et convertir le masque (taille reduite pour rapidite)
                label_mask = np.array(Image.open(label_path).resize((512, 256), Image.NEAREST))
                category_mask = self._convert_to_categories(label_mask)

                # Verifier presence des classes rares
                has_rare_class = False
                for rare_class_id in self.rare_classes:
                    pixel_count = np.sum(category_mask == rare_class_id)
                    if pixel_count >= self.min_pixels_rare:
                        has_rare_class = True
                        break

                if has_rare_class:
                    rare_indices.append(idx)

            except Exception as e:
                # Ignorer les erreurs de lecture
                continue

        # Extrapoler les indices rares a tout le dataset
        # (on suppose que la distribution est uniforme)
        if len(rare_indices) > 0:
            ratio = len(rare_indices) / max(1, (total_files // check_interval))
            estimated_rare_count = int(total_files * ratio)

            # Generer des indices rares estimes uniformement
            if estimated_rare_count < total_files:
                step = total_files / max(1, estimated_rare_count)
                self.rare_indices = [int(i * step) for i in range(estimated_rare_count)]
            else:
                self.rare_indices = rare_indices
        else:
            self.rare_indices = rare_indices

        # Dupliquer les indices des images rares
        if len(self.rare_indices) > 0:
            duplicated_indices = self.rare_indices * (self.oversample_factor - 1)
            self.indices = np.concatenate([self.indices, duplicated_indices])

            # Melanger pour eviter d'avoir toutes les images rares a la fin
            if self.shuffle:
                np.random.shuffle(self.indices)


    def _get_augmentation_pipeline(self) -> Optional[A.Compose]:
        """
        Retourne le pipeline d'augmentation Albumentations.

        Utilise les pipelines avances de data_augmentation.py si disponible.

        Returns:
            A.Compose ou None: Pipeline d'augmentation
        """
        if not self.augmentation:
            return None

        # Si augmentation est un string, utiliser le pipeline correspondant
        if isinstance(self.augmentation, str):
            try:
                return get_augmentation_pipeline(mode=self.augmentation)
            except:
                print(f"   Warning: Pipeline '{self.augmentation}' non trouve, utilisation pipeline standard")
                return get_augmentation_pipeline(mode='standard')

        # Si augmentation=True, utiliser le pipeline standard
        if self.augmentation is True:
            try:
                return get_augmentation_pipeline(mode='standard')
            except:
                # Fallback sur pipeline simple si data_augmentation.py non disponible
                return A.Compose([
                    A.HorizontalFlip(p=0.5),
                    A.Affine(
                        scale=(0.9, 1.1),
                        translate_percent=(-0.1, 0.1),
                        rotate=(-15, 15),
                        shear=(-5, 5),
                        mode=0,
                        p=0.5
                    ),
                    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
                    A.GaussianBlur(blur_limit=(3, 5), p=0.3),
                ])

        return None


    def get_sample_info(self, idx: int) -> dict:
        """
        Retourne les infos d'un échantillon (utile pour le debugging).

        Args:
            idx (int): Index de l'image

        Returns:
            dict: Informations sur l'échantillon
        """
        label_path = self.label_files[idx]

        return {
            'index': idx,
            'label_path': str(label_path),
            'city': label_path.parent.name,
            'filename': label_path.name,
        }


# ============================================================================
# FONCTIONS HELPER
# ============================================================================

def create_dataloaders(
    data_root: str,
    batch_size: int = 8,
    img_size: Tuple[int, int] = (256, 512),
    augmentation: Union[bool, str] = 'standard',
    preprocessing: Optional[callable] = None,
    oversample_rare_classes: bool = True,
    oversample_factor: int = 3,
    rare_classes: Optional[List[int]] = None,
    min_pixels_rare: int = 100
) -> Tuple[CityscapesSequence, CityscapesSequence]:
    """
    Cree les generateurs de donnees pour train et test.

    Args:
        data_root (str): Chemin vers le dossier data/
        batch_size (int): Taille des batchs
        img_size (tuple): Taille des images (height, width)
        augmentation (bool or str): Pipeline d'augmentation
            - False: pas d'augmentation
            - 'light', 'standard', 'advanced', 'aggressive': pipelines specifiques
        preprocessing (callable): Fonction de preprocessing (ex: pour encoder)
        oversample_rare_classes (bool): Sur-echantillonner les classes rares (human, vehicle)
        oversample_factor (int): Facteur de duplication (2=doubler, 3=tripler)
        rare_classes (list): Liste des IDs de classes rares (defaut: [6, 7])
        min_pixels_rare (int): Seuil minimum de pixels pour detection

    Returns:
        train_gen, test_gen
    """
    # Generateur train (avec shuffle, augmentation et sur-echantillonnage)
    train_gen = CityscapesSequence(
        root_dir=data_root,
        split='train',
        batch_size=batch_size,
        img_size=img_size,
        shuffle=True,
        augmentation=augmentation,
        preprocessing=preprocessing,
        oversample_rare_classes=oversample_rare_classes,
        oversample_factor=oversample_factor,
        rare_classes=rare_classes,
        min_pixels_rare=min_pixels_rare
    )

    # Generateur validation (pas de shuffle, pas d'augmentation, pas de sur-echantillonnage)
    # NOTE: Utiliser 'val' au lieu de 'test' car le test set Cityscapes n'a pas les vrais labels
    val_gen = CityscapesSequence(
        root_dir=data_root,
        split='val',
        batch_size=batch_size,
        img_size=img_size,
        shuffle=False,
        augmentation=False,
        preprocessing=preprocessing,
        oversample_rare_classes=False
    )

    return train_gen, val_gen


def decode_predictions(pred_mask: np.ndarray) -> np.ndarray:
    """
    Convertit un masque one-hot (ou probabilités) en masque de catégories.

    Args:
        pred_mask (np.ndarray): Masque one-hot ou proba, shape (H, W, 8)

    Returns:
        category_mask (np.ndarray): Masque de catégories, shape (H, W), valeurs 0-7
    """
    return np.argmax(pred_mask, axis=-1).astype(np.uint8)


def colorize_mask(mask: np.ndarray) -> np.ndarray:
    """
    Colorise un masque de catégories pour visualisation.

    Args:
        mask (np.ndarray): Masque de catégories, shape (H, W), valeurs 0-7

    Returns:
        colored_mask (np.ndarray): Image RGB colorée, shape (H, W, 3)
    """
    h, w = mask.shape
    colored_mask = np.zeros((h, w, 3), dtype=np.uint8)

    for category_id, color in CATEGORY_COLORS.items():
        # Convertir BGR → RGB
        colored_mask[mask == category_id] = color[::-1]

    return colored_mask


def calculate_class_weights(data_root: str, split: str = 'train') -> np.ndarray:
    """
    Calcule les poids de classe pour la loss pondérée.

    Parcourt toutes les images du split et compte les pixels par classe.

    Args:
        data_root (str): Chemin vers le dossier data/
        split (str): 'train' ou 'test'

    Returns:
        class_weights (np.ndarray): Poids par classe, shape (8,)
    """
    print(f"\nCalcul des poids de classe pour {split}...")

    # Initialiser les compteurs
    pixel_counts = np.zeros(8, dtype=np.int64)

    # Créer un générateur temporaire
    gen = CityscapesSequence(
        root_dir=data_root,
        split=split,
        batch_size=1,
        img_size=(256, 512),
        shuffle=False,
        augmentation=False
    )

    # Parcourir toutes les images
    for i in range(len(gen.label_files)):
        label_path = gen.label_files[i]
        label_mask = np.array(Image.open(label_path), dtype=np.uint8)

        # Convertir labelIds → catégories
        category_mask = gen._convert_to_categories(label_mask)

        # Compter les pixels par catégorie
        for cat_id in range(8):
            pixel_counts[cat_id] += np.sum(category_mask == cat_id)

        if (i + 1) % 500 == 0:
            print(f"   Traité {i + 1}/{len(gen.label_files)} images...")

    # Calculer les poids (inverse de la fréquence)
    total_pixels = pixel_counts.sum()
    class_frequencies = pixel_counts / total_pixels

    # Éviter division par zéro
    class_weights = np.where(
        class_frequencies > 0,
        1.0 / (class_frequencies * len(class_frequencies)),
        0.0
    )

    # Normaliser les poids
    class_weights = class_weights / class_weights.sum() * len(class_weights)

    print(f"\nOK Poids de classe calculés :")
    for i, (name, weight) in enumerate(zip(CATEGORY_NAMES, class_weights)):
        freq_pct = class_frequencies[i] * 100
        print(f"   {i} - {name:15s} : poids={weight:.4f}  (fréquence={freq_pct:.2f}%)")

    return class_weights


# ============================================================================
# EXEMPLE D'UTILISATION
# ============================================================================

if __name__ == '__main__':
    """
    Test rapide du DataLoader
    """
    import matplotlib.pyplot as plt

    print("=" * 70)
    print("TEST DU DATALOADER KERAS - CITYSCAPES")
    print("=" * 70)

    # Créer les générateurs
    train_gen, test_gen = create_dataloaders(
        data_root='../data',
        batch_size=4,
        img_size=(256, 512),
        augmentation=True
    )

    print(f"\n📊 Statistiques :")
    print(f"   Train : {len(train_gen.label_files)} images, {len(train_gen)} batchs")
    print(f"   Test  : {len(test_gen.label_files)} images, {len(test_gen)} batchs")

    # Charger un batch
    print(f"\n🔄 Chargement d'un batch...")
    batch_images, batch_masks = train_gen[0]

    print(f"\nOK Batch chargé :")
    print(f"   Images : shape={batch_images.shape}, dtype={batch_images.dtype}")
    print(f"   Min={batch_images.min():.3f}, Max={batch_images.max():.3f}")
    print(f"   Masques : shape={batch_masks.shape}, dtype={batch_masks.dtype}")
    print(f"   Sum={batch_masks[0].sum():.0f} (doit être ~{256*512})")

    # Décoder le premier masque
    mask_decoded = decode_predictions(batch_masks[0])
    print(f"\n   Masque décodé : shape={mask_decoded.shape}, valeurs uniques={np.unique(mask_decoded)}")

    # Visualiser
    print(f"\n🎨 Visualisation...")
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Image originale
    axes[0].imshow(batch_images[0])
    axes[0].set_title('Image RGB')
    axes[0].axis('off')

    # Masque one-hot (visualiser une classe)
    axes[1].imshow(batch_masks[0, :, :, 1], cmap='gray')  # Classe 'flat'
    axes[1].set_title('Masque classe "flat" (1)')
    axes[1].axis('off')

    # Masque colorisé
    mask_colored = colorize_mask(mask_decoded)
    axes[2].imshow(mask_colored)
    axes[2].set_title('Masque colorisé (8 classes)')
    axes[2].axis('off')

    plt.tight_layout()
    plt.savefig('../test_dataloader_keras.png', dpi=150, bbox_inches='tight')
    print(f"   OK Image sauvegardée : test_dataloader_keras.png")

    print("\n" + "=" * 70)
    print("OK TEST TERMINÉ AVEC SUCCÈS")
    print("=" * 70)
