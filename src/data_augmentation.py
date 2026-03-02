"""
Pipelines de Data Augmentation pour Cityscapes

Contient plusieurs pipelines d'augmentation adaptes a la conduite autonome :
- Pipeline standard : augmentations classiques
- Pipeline avance : conditions meteo, aberrations optiques, jour/nuit
- Pipeline agressif : pour sur-echantillonnage des classes rares

Utilise Albumentations pour les transformations.
"""

import albumentations as A
from typing import Optional


# ============================================================================
# CONFIGURATION DES AUGMENTATIONS
# ============================================================================

# Classes rares a privilegier (IDs des categories)
RARE_CLASSES = [6, 7]  # human, vehicle

# Seuils de pixels pour considerer qu'une classe est presente
MIN_PIXELS_THRESHOLD = 100  # Minimum 100 pixels pour considerer la classe presente


# ============================================================================
# PIPELINES D'AUGMENTATION
# ============================================================================

def get_light_augmentation() -> A.Compose:
    """
    Pipeline leger d'augmentation pour validation rapide.

    Utilise uniquement des transformations geometriques simples.

    Returns:
        A.Compose: Pipeline d'augmentation
    """
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.05,
            scale_limit=0.05,
            rotate_limit=5,
            border_mode=0,
            p=0.3
        ),
    ])


def get_standard_augmentation() -> A.Compose:
    """
    Pipeline standard d'augmentation.

    Transformations geometriques + ajustements photometriques classiques.
    Bon equilibre entre variete et realisme.

    Returns:
        A.Compose: Pipeline d'augmentation
    """
    return A.Compose([
        # Transformations geometriques
        A.HorizontalFlip(p=0.5),
        A.Affine(
            scale=(0.9, 1.1),
            translate_percent=(-0.1, 0.1),
            rotate=(-15, 15),
            shear=(-5, 5),
            mode=0,
            p=0.5
        ),

        # Ajustements photometriques
        A.OneOf([
            A.RandomBrightnessContrast(
                brightness_limit=0.2,
                contrast_limit=0.2,
                p=1
            ),
            A.RandomGamma(gamma_limit=(80, 120), p=1),
            A.HueSaturationValue(
                hue_shift_limit=15,
                sat_shift_limit=25,
                val_shift_limit=15,
                p=1
            ),
        ], p=0.6),

        # Flou et bruit
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 5), p=1),
            A.MotionBlur(blur_limit=5, p=1),
            A.MedianBlur(blur_limit=5, p=1),
        ], p=0.3),

        A.GaussNoise(var_limit=(10.0, 50.0), p=0.2),
    ])


def get_advanced_augmentation() -> A.Compose:
    """
    Pipeline avance avec conditions meteo et aberrations optiques.

    Simule differentes conditions de conduite :
    - Meteo (pluie, brouillard, neige)
    - Luminosite (jour, nuit, eblouissement)
    - Qualite image (bruit capteur, flou de mouvement)
    - Aberrations optiques (distorsion lentille)

    Returns:
        A.Compose: Pipeline d'augmentation avance
    """
    return A.Compose([
        # Transformations geometriques
        A.HorizontalFlip(p=0.5),
        A.Affine(
            scale=(0.85, 1.15),
            translate_percent=(-0.1, 0.1),
            rotate=(-20, 20),
            shear=(-8, 8),
            mode=0,
            p=0.5
        ),

        # Distorsions optiques (aberrations lentille camera)
        A.OneOf([
            A.OpticalDistortion(distort_limit=0.3, shift_limit=0.1, p=1),
            A.GridDistortion(num_steps=5, distort_limit=0.3, p=1),
        ], p=0.3),

        # Conditions meteo
        A.OneOf([
            A.RandomRain(
                slant_lower=-10,
                slant_upper=10,
                drop_length=15,
                drop_width=1,
                drop_color=(200, 200, 200),
                blur_value=3,
                brightness_coefficient=0.9,
                rain_type='drizzle',
                p=1
            ),
            A.RandomFog(
                fog_coef_lower=0.2,
                fog_coef_upper=0.5,
                alpha_coef=0.1,
                p=1
            ),
            A.RandomSnow(
                snow_point_lower=0.1,
                snow_point_upper=0.3,
                brightness_coeff=1.5,
                p=1
            ),
        ], p=0.2),

        # Conditions de luminosite (jour/nuit/crepuscule)
        A.OneOf([
            # Jour lumineux
            A.RandomBrightnessContrast(
                brightness_limit=(0.1, 0.3),
                contrast_limit=0.2,
                p=1
            ),
            # Nuit
            A.Compose([
                A.RandomBrightnessContrast(
                    brightness_limit=(-0.4, -0.2),
                    contrast_limit=(-0.2, 0),
                    p=1
                ),
                A.GaussNoise(var_limit=(20.0, 40.0), p=0.5),
            ], p=1),
            # Crepuscule/aube
            A.Compose([
                A.HueSaturationValue(
                    hue_shift_limit=10,
                    sat_shift_limit=30,
                    val_shift_limit=0,
                    p=1
                ),
                A.RandomBrightnessContrast(
                    brightness_limit=(-0.2, 0),
                    contrast_limit=0.1,
                    p=1
                ),
            ], p=1),
            # Normal
            A.RandomBrightnessContrast(
                brightness_limit=0.2,
                contrast_limit=0.2,
                p=1
            ),
            # Eblouissement soleil
            A.RandomSunFlare(
                flare_roi=(0, 0, 1, 0.5),
                angle_lower=0,
                angle_upper=1,
                num_flare_circles_lower=3,
                num_flare_circles_upper=6,
                src_radius=150,
                p=1
            ),
        ], p=0.7),

        # Degradation qualite image
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 7), p=1),
            A.MotionBlur(blur_limit=7, p=1),
            A.MedianBlur(blur_limit=7, p=1),
        ], p=0.4),

        # Bruit capteur (ISO eleve, faible luminosite)
        A.OneOf([
            A.GaussNoise(var_limit=(10.0, 60.0), p=1),
            A.ISONoise(
                color_shift=(0.01, 0.05),
                intensity=(0.1, 0.5),
                p=1
            ),
        ], p=0.3),

        # Amelioration contraste (CLAHE - simule post-traitement camera)
        A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.2),

        # Ombres (vehicules, batiments)
        A.RandomShadow(
            shadow_roi=(0, 0.5, 1, 1),
            num_shadows_lower=1,
            num_shadows_upper=2,
            shadow_dimension=5,
            p=0.2
        ),
    ])


def get_aggressive_augmentation() -> A.Compose:
    """
    Pipeline agressif pour sur-echantillonnage des classes rares.

    Utilise des transformations plus fortes pour augmenter drastiquement
    la variete des images contenant des classes minoritaires (human, vehicle).

    Returns:
        A.Compose: Pipeline d'augmentation agressif
    """
    return A.Compose([
        # Transformations geometriques plus prononcees
        A.HorizontalFlip(p=0.5),
        A.Affine(
            scale=(0.8, 1.2),
            translate_percent=(-0.15, 0.15),
            rotate=(-25, 25),
            shear=(-10, 10),
            mode=0,
            p=0.7
        ),

        # Distorsions optiques fortes
        A.OneOf([
            A.OpticalDistortion(distort_limit=0.5, shift_limit=0.2, p=1),
            A.GridDistortion(num_steps=5, distort_limit=0.5, p=1),
        ], p=0.5),

        # Conditions meteo variees
        A.OneOf([
            A.RandomRain(
                slant_lower=-15,
                slant_upper=15,
                drop_length=20,
                drop_width=1,
                drop_color=(200, 200, 200),
                blur_value=5,
                brightness_coefficient=0.85,
                rain_type='heavy',
                p=1
            ),
            A.RandomFog(
                fog_coef_lower=0.3,
                fog_coef_upper=0.7,
                alpha_coef=0.15,
                p=1
            ),
            A.RandomSnow(
                snow_point_lower=0.2,
                snow_point_upper=0.4,
                brightness_coeff=1.8,
                p=1
            ),
        ], p=0.4),

        # Variations de luminosite extremes
        A.OneOf([
            A.RandomBrightnessContrast(
                brightness_limit=(-0.5, 0.4),
                contrast_limit=(-0.3, 0.3),
                p=1
            ),
            A.RandomGamma(gamma_limit=(60, 140), p=1),
            A.RandomSunFlare(
                flare_roi=(0, 0, 1, 0.5),
                angle_lower=0,
                angle_upper=1,
                num_flare_circles_lower=4,
                num_flare_circles_upper=8,
                src_radius=200,
                p=1
            ),
        ], p=0.8),

        # Variations de couleur
        A.HueSaturationValue(
            hue_shift_limit=25,
            sat_shift_limit=40,
            val_shift_limit=25,
            p=0.6
        ),

        # Degradation qualite
        A.OneOf([
            A.GaussianBlur(blur_limit=(5, 9), p=1),
            A.MotionBlur(blur_limit=9, p=1),
            A.MedianBlur(blur_limit=9, p=1),
        ], p=0.5),

        # Bruit fort
        A.OneOf([
            A.GaussNoise(var_limit=(30.0, 80.0), p=1),
            A.ISONoise(
                color_shift=(0.01, 0.08),
                intensity=(0.2, 0.7),
                p=1
            ),
        ], p=0.5),

        # Ombres multiples
        A.RandomShadow(
            shadow_roi=(0, 0.3, 1, 1),
            num_shadows_lower=1,
            num_shadows_upper=3,
            shadow_dimension=6,
            p=0.4
        ),
    ])


# ============================================================================
# FONCTION HELPER POUR CHOISIR LE PIPELINE
# ============================================================================

def get_augmentation_pipeline(
    mode: str = 'standard',
    custom_pipeline: Optional[A.Compose] = None
) -> Optional[A.Compose]:
    """
    Retourne le pipeline d'augmentation selon le mode choisi.

    Args:
        mode (str): Type d'augmentation
            - 'none' : Pas d'augmentation
            - 'light' : Augmentation legere (flip, rotation minime)
            - 'standard' : Augmentation classique (recommande)
            - 'advanced' : Augmentation avancee (meteo, aberrations)
            - 'aggressive' : Augmentation agressive (classes rares)
            - 'custom' : Pipeline personnalise
        custom_pipeline (A.Compose): Pipeline personnalise si mode='custom'

    Returns:
        A.Compose ou None: Pipeline d'augmentation

    Raises:
        ValueError: Si le mode est invalide
    """
    if mode == 'none':
        return None
    elif mode == 'light':
        return get_light_augmentation()
    elif mode == 'standard':
        return get_standard_augmentation()
    elif mode == 'advanced':
        return get_advanced_augmentation()
    elif mode == 'aggressive':
        return get_aggressive_augmentation()
    elif mode == 'custom':
        if custom_pipeline is None:
            raise ValueError("custom_pipeline doit etre fourni si mode='custom'")
        return custom_pipeline
    else:
        raise ValueError(
            f"Mode invalide: {mode}. "
            f"Choix possibles: none, light, standard, advanced, aggressive, custom"
        )


# ============================================================================
# EXEMPLE D'UTILISATION
# ============================================================================

if __name__ == '__main__':
    """
    Test des differents pipelines d'augmentation
    """
    import numpy as np
    from PIL import Image
    import matplotlib.pyplot as plt

    # Charger une image de test
    print("Test des pipelines d'augmentation")
    print("=" * 70)

    # Creer une image dummy pour tester
    image = np.random.randint(0, 255, (256, 512, 3), dtype=np.uint8)
    mask = np.random.randint(0, 8, (256, 512), dtype=np.uint8)

    # Tester chaque pipeline
    pipelines = {
        'light': get_light_augmentation(),
        'standard': get_standard_augmentation(),
        'advanced': get_advanced_augmentation(),
        'aggressive': get_aggressive_augmentation(),
    }

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))

    for idx, (name, pipeline) in enumerate(pipelines.items()):
        # Appliquer augmentation
        augmented = pipeline(image=image, mask=mask)
        aug_image = augmented['image']
        aug_mask = augmented['mask']

        # Afficher
        axes[0, idx].imshow(aug_image)
        axes[0, idx].set_title(f'{name.capitalize()} - Image')
        axes[0, idx].axis('off')

        axes[1, idx].imshow(aug_mask, cmap='tab10', vmin=0, vmax=7)
        axes[1, idx].set_title(f'{name.capitalize()} - Mask')
        axes[1, idx].axis('off')

    plt.tight_layout()
    plt.savefig('../test_augmentations.png', dpi=150, bbox_inches='tight')
    print("OK Image sauvegardee : test_augmentations.png")
    print("=" * 70)
