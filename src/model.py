"""
Architecture U-Net avec backbone pre-entraine pour segmentation semantique

Cree un U-Net custom avec MobileNetV2 comme encoder pre-entraine (ImageNet).
Architecture encodeur-decodeur avec skip connections.

Loss : Categorical Cross-Entropy + Dice Loss (ponderes)
Metriques : IoU, Dice Coefficient, Pixel Accuracy
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from typing import Tuple, Optional
import numpy as np


# ============================================================================
# METRIQUES PERSONNALISEES
# ============================================================================

class MeanIoU(keras.metrics.MeanIoU):
    """
    Mean Intersection over Union adapte pour segmentation multiclasse.
    """
    def __init__(self, num_classes=8, name='mean_iou', **kwargs):
        super().__init__(num_classes=num_classes, name=name, **kwargs)

    def update_state(self, y_true, y_pred, sample_weight=None):
        # Convertir one-hot en indices de classe
        y_true = tf.argmax(y_true, axis=-1)
        y_pred = tf.argmax(y_pred, axis=-1)
        return super().update_state(y_true, y_pred, sample_weight)


def dice_coefficient(y_true, y_pred, smooth=1e-6):
    """
    Calcule le Dice Coefficient (metrique, pas loss).

    Args:
        y_true: Ground truth (one-hot), shape (batch, H, W, num_classes)
        y_pred: Predictions (softmax), shape (batch, H, W, num_classes)
        smooth: Facteur de lissage pour eviter division par zero

    Returns:
        dice: Coefficient de Dice moyen sur toutes les classes
    """
    # Aplatir les dimensions spatiales
    y_true_f = tf.reshape(y_true, [-1, tf.shape(y_true)[-1]])
    y_pred_f = tf.reshape(y_pred, [-1, tf.shape(y_pred)[-1]])

    # Calculer intersection et union
    intersection = tf.reduce_sum(y_true_f * y_pred_f, axis=0)
    union = tf.reduce_sum(y_true_f, axis=0) + tf.reduce_sum(y_pred_f, axis=0)

    # Dice coefficient par classe
    dice_per_class = (2.0 * intersection + smooth) / (union + smooth)

    # Moyenne sur toutes les classes
    return tf.reduce_mean(dice_per_class)


def pixel_accuracy(y_true, y_pred):
    """
    Calcule la precision pixel par pixel.

    Args:
        y_true: Ground truth (one-hot)
        y_pred: Predictions (softmax)

    Returns:
        accuracy: Proportion de pixels correctement classifies
    """
    y_true_class = tf.argmax(y_true, axis=-1)
    y_pred_class = tf.argmax(y_pred, axis=-1)

    correct = tf.cast(tf.equal(y_true_class, y_pred_class), tf.float32)
    return tf.reduce_mean(correct)


# ============================================================================
# LOSS FUNCTIONS
# ============================================================================

def categorical_crossentropy_weighted(class_weights):
    """
    Categorical Cross-Entropy ponderee par classe.

    Args:
        class_weights (np.ndarray): Poids par classe, shape (num_classes,)

    Returns:
        loss_fn: Fonction de loss
    """
    class_weights = tf.constant(class_weights, dtype=tf.float32)

    def loss(y_true, y_pred):
        # y_true: (batch, H, W, num_classes) one-hot
        # y_pred: (batch, H, W, num_classes) softmax

        # Calculer cross-entropy par pixel
        ce = -tf.reduce_sum(y_true * tf.math.log(y_pred + 1e-7), axis=-1)

        # Appliquer les poids de classe
        # Trouver la classe de chaque pixel
        class_indices = tf.argmax(y_true, axis=-1)
        weights = tf.gather(class_weights, class_indices)

        # Loss ponderee
        weighted_ce = ce * weights

        return tf.reduce_mean(weighted_ce)

    return loss


def dice_loss(y_true, y_pred, smooth=1e-6):
    """
    Dice Loss pour segmentation multiclasse.

    Args:
        y_true: Ground truth (one-hot)
        y_pred: Predictions (softmax)
        smooth: Facteur de lissage

    Returns:
        loss: 1 - Dice coefficient
    """
    return 1.0 - dice_coefficient(y_true, y_pred, smooth)


def combined_loss(class_weights, dice_weight=0.5):
    """
    Loss combinee : Categorical Cross-Entropy (ponderee) + Dice Loss.

    Args:
        class_weights (np.ndarray): Poids par classe
        dice_weight (float): Poids du Dice Loss (0-1)
            - 0.0 : 100% CE
            - 0.5 : 50% CE + 50% Dice (recommande)
            - 1.0 : 100% Dice

    Returns:
        loss_fn: Fonction de loss combinee
    """
    ce_loss = categorical_crossentropy_weighted(class_weights)

    def loss(y_true, y_pred):
        ce = ce_loss(y_true, y_pred)
        dice = dice_loss(y_true, y_pred)

        return (1 - dice_weight) * ce + dice_weight * dice

    return loss


# ============================================================================
# BLOCS DE CONSTRUCTION U-NET
# ============================================================================

def conv_block(x, filters, kernel_size=3, activation='relu', padding='same', name=None):
    """
    Bloc de convolution : Conv2D + BatchNorm + Activation.

    Args:
        x: Input tensor
        filters: Nombre de filtres
        kernel_size: Taille du kernel
        activation: Fonction d'activation
        padding: Padding
        name: Nom du bloc

    Returns:
        x: Output tensor
    """
    x = layers.Conv2D(filters, kernel_size, padding=padding, name=f'{name}_conv' if name else None)(x)
    x = layers.BatchNormalization(name=f'{name}_bn' if name else None)(x)
    x = layers.Activation(activation, name=f'{name}_act' if name else None)(x)
    return x


def upsampling_block(x, skip_features, filters, name=None):
    """
    Bloc d'upsampling avec skip connection.

    Args:
        x: Input tensor (du niveau inferieur)
        skip_features: Features du skip connection (meme niveau dans encoder)
        filters: Nombre de filtres
        name: Nom du bloc

    Returns:
        x: Output tensor upsample et concatene
    """
    # Upsampling (x2)
    x = layers.UpSampling2D(size=(2, 2), interpolation='bilinear', name=f'{name}_upsample' if name else None)(x)

    # Concatener avec skip connection
    x = layers.Concatenate(name=f'{name}_concat' if name else None)([x, skip_features])

    # Conv blocks
    x = conv_block(x, filters, name=f'{name}_conv1' if name else None)
    x = conv_block(x, filters, name=f'{name}_conv2' if name else None)

    return x


# ============================================================================
# CREATION DU MODELE U-NET
# ============================================================================

def create_unet_mobilenet(
    input_shape: Tuple[int, int, int] = (256, 512, 3),
    num_classes: int = 8,
    encoder_freeze: bool = False
) -> keras.Model:
    """
    Cree un modele U-Net avec MobileNetV2 comme encoder.

    Architecture :
    - Encoder : MobileNetV2 pre-entraine (ImageNet)
    - Decoder : Blocs d'upsampling avec skip connections
    - Output : Softmax pour classification multiclasse

    Args:
        input_shape (tuple): Forme de l'entree (height, width, channels)
        num_classes (int): Nombre de classes de segmentation
        encoder_freeze (bool): Geler l'encoder (pas de fine-tuning)

    Returns:
        model (keras.Model): Modele U-Net
    """
    # Input
    inputs = layers.Input(shape=input_shape, name='input')

    # ========================================================================
    # ENCODER : MobileNetV2 pre-entraine
    # ========================================================================

    # Charger MobileNetV2 pre-entraine (ImageNet)
    encoder = keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,  # Pas de classification finale
        weights='imagenet'
    )

    # Geler l'encoder si demande
    if encoder_freeze:
        encoder.trainable = False

    # Extraire les features a differents niveaux (skip connections)
    # MobileNetV2 layers names pour skip connections
    skip_names = [
        'block_1_expand_relu',   # 128x256x96   (stride 2)
        'block_3_expand_relu',   # 64x128x144   (stride 4)
        'block_6_expand_relu',   # 32x64x192    (stride 8)
        'block_13_expand_relu',  # 16x32x576    (stride 16)
        'out_relu'               # 8x16x1280    (stride 32)
    ]

    # Extraire les features
    x = inputs
    skip_features = []

    for i, layer_name in enumerate(skip_names):
        # Encoder jusqu'a cette couche
        if i == 0:
            # Premiere iteration : encoder complet
            encoder_model = keras.Model(
                inputs=encoder.input,
                outputs=[encoder.get_layer(name).output for name in skip_names]
            )
            encoder_outputs = encoder_model(x)
            skip_features = encoder_outputs[:-1]  # Toutes sauf la derniere
            x = encoder_outputs[-1]  # Derniere = bottleneck
        break

    # ========================================================================
    # BOTTLENECK (centre du U-Net)
    # ========================================================================

    # Le bottleneck est deja x (sortie de l'encoder)
    # On peut ajouter quelques conv si besoin
    x = conv_block(x, 1024, name='bottleneck_conv1')

    # ========================================================================
    # DECODER : Upsampling + Skip Connections
    # ========================================================================

    # Decoder : 5 niveaux (correspondant aux 5 skip connections)
    decoder_filters = [512, 256, 128, 64, 32]

    # Upsampling progressif avec skip connections
    for i, filters in enumerate(decoder_filters):
        # Recuperer le skip connection correspondant (en ordre inverse)
        skip_idx = len(skip_features) - 1 - i
        if skip_idx >= 0:
            skip = skip_features[skip_idx]
            x = upsampling_block(x, skip, filters, name=f'decoder_block{i+1}')
        else:
            # Pas de skip connection (premiers niveaux)
            x = layers.UpSampling2D(size=(2, 2), interpolation='bilinear')(x)
            x = conv_block(x, filters, name=f'decoder_block{i+1}_conv1')
            x = conv_block(x, filters, name=f'decoder_block{i+1}_conv2')

    # ========================================================================
    # OUTPUT HEAD
    # ========================================================================

    # Convolution finale pour produire les logits par classe
    outputs = layers.Conv2D(
        num_classes,
        kernel_size=1,
        activation='softmax',
        padding='same',
        name='output'
    )(x)

    # ========================================================================
    # MODELE FINAL
    # ========================================================================

    model = keras.Model(inputs=inputs, outputs=outputs, name='unet_mobilenetv2')

    return model


def create_unet_mini(
    input_shape: Tuple[int, int, int] = (256, 512, 3),
    num_classes: int = 8
) -> keras.Model:
    """
    Cree un U-Net mini leger (baseline sans transfer learning).

    Architecture simplifiee :
    - 4 niveaux d'encodeur/decodeur (vs 5 pour le gros U-Net)
    - Moins de filtres a chaque niveau
    - Pas de backbone pre-entraine
    - ~1-2M parametres (vs ~5-10M pour U-Net + MobileNetV2)

    Args:
        input_shape (tuple): Forme de l'entree (height, width, channels)
        num_classes (int): Nombre de classes de segmentation

    Returns:
        model (keras.Model): Modele U-Net mini
    """
    # Input
    inputs = layers.Input(shape=input_shape, name='input')

    # ========================================================================
    # ENCODER (4 niveaux)
    # ========================================================================

    # Niveau 1 : 256x512 -> 128x256
    e1 = conv_block(inputs, 32, name='encoder1_conv1')
    e1 = conv_block(e1, 32, name='encoder1_conv2')
    p1 = layers.MaxPooling2D((2, 2), name='encoder1_pool')(e1)

    # Niveau 2 : 128x256 -> 64x128
    e2 = conv_block(p1, 64, name='encoder2_conv1')
    e2 = conv_block(e2, 64, name='encoder2_conv2')
    p2 = layers.MaxPooling2D((2, 2), name='encoder2_pool')(e2)

    # Niveau 3 : 64x128 -> 32x64
    e3 = conv_block(p2, 128, name='encoder3_conv1')
    e3 = conv_block(e3, 128, name='encoder3_conv2')
    p3 = layers.MaxPooling2D((2, 2), name='encoder3_pool')(e3)

    # Niveau 4 : 32x64 -> 16x32
    e4 = conv_block(p3, 256, name='encoder4_conv1')
    e4 = conv_block(e4, 256, name='encoder4_conv2')
    p4 = layers.MaxPooling2D((2, 2), name='encoder4_pool')(e4)

    # ========================================================================
    # BOTTLENECK : 16x32
    # ========================================================================

    bottleneck = conv_block(p4, 512, name='bottleneck_conv1')
    bottleneck = conv_block(bottleneck, 512, name='bottleneck_conv2')

    # ========================================================================
    # DECODER (4 niveaux avec skip connections)
    # ========================================================================

    # Niveau 4 : 16x32 -> 32x64
    d4 = upsampling_block(bottleneck, e4, 256, name='decoder4')

    # Niveau 3 : 32x64 -> 64x128
    d3 = upsampling_block(d4, e3, 128, name='decoder3')

    # Niveau 2 : 64x128 -> 128x256
    d2 = upsampling_block(d3, e2, 64, name='decoder2')

    # Niveau 1 : 128x256 -> 256x512
    d1 = upsampling_block(d2, e1, 32, name='decoder1')

    # ========================================================================
    # OUTPUT HEAD
    # ========================================================================

    # Convolution finale pour produire les logits par classe
    outputs = layers.Conv2D(
        num_classes,
        kernel_size=1,
        activation='softmax',
        padding='same',
        name='output'
    )(d1)

    # ========================================================================
    # MODELE FINAL
    # ========================================================================

    model = keras.Model(inputs=inputs, outputs=outputs, name='unet_mini')

    return model


# ============================================================================
# COMPILATION DU MODELE
# ============================================================================

def compile_model(
    model: keras.Model,
    class_weights: Optional[np.ndarray] = None,
    learning_rate: float = 1e-4,
    dice_weight: float = 0.5
) -> keras.Model:
    """
    Compile le modele avec loss, optimiseur et metriques.

    Args:
        model (keras.Model): Modele a compiler
        class_weights (np.ndarray): Poids par classe pour loss ponderee
            Si None, utilise des poids uniformes
        learning_rate (float): Learning rate pour Adam
        dice_weight (float): Poids du Dice Loss (0-1)

    Returns:
        model (keras.Model): Modele compile
    """
    # Poids par defaut (uniformes) si non fournis
    if class_weights is None:
        class_weights = np.ones(8, dtype=np.float32)

    # Loss combinee
    loss_fn = combined_loss(class_weights, dice_weight=dice_weight)

    # Optimiseur
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)

    # Metriques
    metrics = [
        MeanIoU(num_classes=8, name='mean_iou'),
        dice_coefficient,
        pixel_accuracy
    ]

    # Compiler
    model.compile(
        optimizer=optimizer,
        loss=loss_fn,
        metrics=metrics
    )

    return model


# ============================================================================
# FONCTION COMPLETE : CREER + COMPILER
# ============================================================================

def build_model(
    input_shape: Tuple[int, int, int] = (256, 512, 3),
    num_classes: int = 8,
    class_weights: Optional[np.ndarray] = None,
    learning_rate: float = 1e-4,
    dice_weight: float = 0.5,
    encoder_freeze: bool = False
) -> keras.Model:
    """
    Cree et compile un modele U-Net pret a l'entrainement.

    Args:
        input_shape (tuple): Forme de l'entree (H, W, C)
        num_classes (int): Nombre de classes
        class_weights (np.ndarray): Poids par classe pour loss
        learning_rate (float): Learning rate
        dice_weight (float): Poids du Dice Loss dans la loss combinee
        encoder_freeze (bool): Geler l'encoder (pas de fine-tuning)

    Returns:
        model (keras.Model): Modele compile pret a l'entrainement
    """
    print("=" * 70)
    print("CREATION DU MODELE U-NET + MOBILENETV2")
    print("=" * 70)

    # Creer le modele
    print(f"\n1. Creation du modele...")
    print(f"   Backbone        : MobileNetV2")
    print(f"   Input shape     : {input_shape}")
    print(f"   Num classes     : {num_classes}")
    print(f"   Encoder weights : ImageNet")
    print(f"   Encoder freeze  : {encoder_freeze}")

    model = create_unet_mobilenet(
        input_shape=input_shape,
        num_classes=num_classes,
        encoder_freeze=encoder_freeze
    )

    # Compiler le modele
    print(f"\n2. Compilation du modele...")
    print(f"   Learning rate   : {learning_rate}")
    print(f"   Dice weight     : {dice_weight}")
    print(f"   Class weights   : {'Oui (ponderes)' if class_weights is not None else 'Non (uniformes)'}")

    model = compile_model(
        model=model,
        class_weights=class_weights,
        learning_rate=learning_rate,
        dice_weight=dice_weight
    )

    # Statistiques du modele
    print(f"\n3. Statistiques du modele...")
    total_params = model.count_params()
    trainable_params = sum([tf.size(w).numpy() for w in model.trainable_weights])
    non_trainable_params = total_params - trainable_params

    print(f"   Total params    : {total_params:,}")
    print(f"   Trainable       : {trainable_params:,}")
    print(f"   Non-trainable   : {non_trainable_params:,}")

    print("\n" + "=" * 70)
    print("MODELE PRET POUR L'ENTRAINEMENT")
    print("=" * 70)

    return model


def build_unet_mini(
    input_shape: Tuple[int, int, int] = (256, 512, 3),
    num_classes: int = 8,
    class_weights: Optional[np.ndarray] = None,
    learning_rate: float = 1e-4,
    dice_weight: float = 0.5
) -> keras.Model:
    """
    Cree et compile un U-Net Mini pret a l'entrainement.

    Version legere sans transfer learning pour baseline rapide.

    Args:
        input_shape (tuple): Forme de l'entree (H, W, C)
        num_classes (int): Nombre de classes
        class_weights (np.ndarray): Poids par classe pour loss
        learning_rate (float): Learning rate
        dice_weight (float): Poids du Dice Loss dans la loss combinee

    Returns:
        model (keras.Model): Modele compile pret a l'entrainement
    """
    print("=" * 70)
    print("CREATION DU MODELE U-NET MINI (BASELINE)")
    print("=" * 70)

    # Creer le modele
    print(f"\n1. Creation du modele...")
    print(f"   Architecture    : U-Net Mini (leger)")
    print(f"   Input shape     : {input_shape}")
    print(f"   Num classes     : {num_classes}")
    print(f"   Pre-entrainement: Aucun (from scratch)")

    model = create_unet_mini(
        input_shape=input_shape,
        num_classes=num_classes
    )

    # Compiler le modele
    print(f"\n2. Compilation du modele...")
    print(f"   Learning rate   : {learning_rate}")
    print(f"   Dice weight     : {dice_weight}")
    print(f"   Class weights   : {'Oui (ponderes)' if class_weights is not None else 'Non (uniformes)'}")

    model = compile_model(
        model=model,
        class_weights=class_weights,
        learning_rate=learning_rate,
        dice_weight=dice_weight
    )

    # Statistiques du modele
    print(f"\n3. Statistiques du modele...")
    total_params = model.count_params()
    trainable_params = sum([tf.size(w).numpy() for w in model.trainable_weights])
    non_trainable_params = total_params - trainable_params

    print(f"   Total params    : {total_params:,}")
    print(f"   Trainable       : {trainable_params:,}")
    print(f"   Non-trainable   : {non_trainable_params:,}")

    print("\n" + "=" * 70)
    print("MODELE PRET POUR L'ENTRAINEMENT")
    print("=" * 70)

    return model


# ============================================================================
# EXEMPLE D'UTILISATION
# ============================================================================

if __name__ == '__main__':
    """
    Test de creation du modele.
    """
    print("\nTest de creation du modele U-Net avec MobileNetV2...")

    # Creer le modele
    model = build_model(
        input_shape=(256, 512, 3),
        num_classes=8,
        learning_rate=1e-4,
        dice_weight=0.5,
        encoder_freeze=False
    )

    # Afficher le resume
    print("\n" + "=" * 70)
    print("RESUME DU MODELE")
    print("=" * 70)
    model.summary(line_length=100)

    print("\n" + "=" * 70)
    print("TEST TERMINE AVEC SUCCES")
    print("=" * 70)
