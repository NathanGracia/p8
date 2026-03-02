"""
Script d'entrainement du modele U-Net + MobileNetV2

Ce script entraine le modele de segmentation semantique sur Cityscapes.

Fonctionnalites :
- Chargement des donnees avec augmentation et sur-echantillonnage
- Calcul des poids de classe pour loss ponderee
- Callbacks : ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
- Sauvegarde du meilleur modele et de l'historique
- Visualisation des resultats d'entrainement
"""

import os
import sys
from pathlib import Path
import argparse
import json
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

# Importer nos modules
from data_loader import create_dataloaders, calculate_class_weights
from model import build_model, build_unet_mini


# ============================================================================
# CONFIGURATION PAR DEFAUT
# ============================================================================

DEFAULT_CONFIG = {
    # Donnees
    'data_root': '../data',
    'img_size': (256, 512),  # (height, width)
    'batch_size': 8,
    'augmentation': 'advanced',  # 'light', 'standard', 'advanced', 'aggressive'
    'oversample_rare_classes': True,
    'oversample_factor': 3,

    # Modele
    'model_type': 'unet_mobilenet',  # 'unet_mini' ou 'unet_mobilenet'
    'num_classes': 8,
    'learning_rate': 1e-4,
    'dice_weight': 0.5,
    'encoder_freeze': False,

    # Entrainement
    'epochs': 50,
    'validation_split': 0.0,  # On utilise test comme validation

    # Callbacks
    'early_stopping_patience': 10,
    'reduce_lr_patience': 5,
    'reduce_lr_factor': 0.5,

    # Sorties
    'model_dir': '../models',
    'logs_dir': '../logs',
    'save_best_only': True,
}


# ============================================================================
# CALLBACKS
# ============================================================================

def create_callbacks(config):
    """
    Cree les callbacks pour l'entrainement.

    Args:
        config (dict): Configuration de l'entrainement

    Returns:
        list: Liste des callbacks
    """
    callbacks = []

    # Creer les dossiers si necessaire
    model_dir = Path(config['model_dir'])
    logs_dir = Path(config['logs_dir'])
    model_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    # Timestamp pour les fichiers
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # 1. ModelCheckpoint - Sauvegarder le meilleur modele
    model_name = config.get('model_type', 'unet_mobilenet').replace('_', '')
    checkpoint_path = model_dir / f'{model_name}_{timestamp}.h5'
    checkpoint = keras.callbacks.ModelCheckpoint(
        filepath=str(checkpoint_path),
        monitor='val_mean_iou',
        mode='max',
        save_best_only=config['save_best_only'],
        save_weights_only=False,
        verbose=1
    )
    callbacks.append(checkpoint)

    # 2. EarlyStopping - Arreter si pas d'amelioration
    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_mean_iou',
        mode='max',
        patience=config['early_stopping_patience'],
        restore_best_weights=True,
        verbose=1
    )
    callbacks.append(early_stopping)

    # 3. ReduceLROnPlateau - Reduire le learning rate
    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        mode='min',
        factor=config['reduce_lr_factor'],
        patience=config['reduce_lr_patience'],
        min_lr=1e-7,
        verbose=1
    )
    callbacks.append(reduce_lr)

    # 4. TensorBoard - Monitoring
    tensorboard_dir = logs_dir / f'tensorboard_{timestamp}'
    tensorboard = keras.callbacks.TensorBoard(
        log_dir=str(tensorboard_dir),
        histogram_freq=0,
        write_graph=True,
        write_images=False,
        update_freq='epoch'
    )
    callbacks.append(tensorboard)

    # 5. CSVLogger - Historique en CSV
    csv_path = logs_dir / f'training_history_{timestamp}.csv'
    csv_logger = keras.callbacks.CSVLogger(
        filename=str(csv_path),
        separator=',',
        append=False
    )
    callbacks.append(csv_logger)

    print(f"\nCallbacks configures :")
    print(f"  - ModelCheckpoint  : {checkpoint_path}")
    print(f"  - EarlyStopping    : patience={config['early_stopping_patience']}")
    print(f"  - ReduceLROnPlateau: patience={config['reduce_lr_patience']}, factor={config['reduce_lr_factor']}")
    print(f"  - TensorBoard      : {tensorboard_dir}")
    print(f"  - CSVLogger        : {csv_path}")

    return callbacks


# ============================================================================
# VISUALISATION DES RESULTATS
# ============================================================================

def plot_training_history(history, save_path=None):
    """
    Visualise l'historique d'entrainement.

    Args:
        history: Historique Keras (history.history)
        save_path: Chemin pour sauvegarder la figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Loss
    axes[0, 0].plot(history['loss'], label='Train Loss')
    axes[0, 0].plot(history['val_loss'], label='Val Loss')
    axes[0, 0].set_title('Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Mean IoU
    axes[0, 1].plot(history['mean_iou'], label='Train mIoU')
    axes[0, 1].plot(history['val_mean_iou'], label='Val mIoU')
    axes[0, 1].set_title('Mean IoU')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('mIoU')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Dice Coefficient
    axes[1, 0].plot(history['dice_coefficient'], label='Train Dice')
    axes[1, 0].plot(history['val_dice_coefficient'], label='Val Dice')
    axes[1, 0].set_title('Dice Coefficient')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Dice')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Pixel Accuracy
    axes[1, 1].plot(history['pixel_accuracy'], label='Train Acc')
    axes[1, 1].plot(history['val_pixel_accuracy'], label='Val Acc')
    axes[1, 1].set_title('Pixel Accuracy')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Accuracy')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nGraphique sauvegarde : {save_path}")

    plt.show()


# ============================================================================
# FONCTION D'ENTRAINEMENT PRINCIPALE
# ============================================================================

def train(config):
    """
    Fonction principale d'entrainement.

    Args:
        config (dict): Configuration de l'entrainement

    Returns:
        model: Modele entraine
        history: Historique d'entrainement
    """
    print("=" * 70)
    model_name = "U-NET MINI" if config['model_type'] == 'unet_mini' else "U-NET + MOBILENETV2"
    print(f"ENTRAINEMENT {model_name} - CITYSCAPES")
    print("=" * 70)

    # ========================================================================
    # 1. CHARGEMENT DES DONNEES
    # ========================================================================

    print("\n1. CHARGEMENT DES DONNEES")
    print("-" * 70)

    train_gen, val_gen = create_dataloaders(
        data_root=config['data_root'],
        batch_size=config['batch_size'],
        img_size=config['img_size'],
        augmentation=config['augmentation'],
        oversample_rare_classes=config['oversample_rare_classes'],
        oversample_factor=config['oversample_factor']
    )

    print(f"\nDonnees chargees :")
    print(f"  Train : {len(train_gen.indices)} samples, {len(train_gen)} batchs")
    print(f"  Val   : {len(val_gen.indices)} samples, {len(val_gen)} batchs")

    # ========================================================================
    # 2. CALCUL DES POIDS DE CLASSE
    # ========================================================================

    print("\n2. CALCUL DES POIDS DE CLASSE")
    print("-" * 70)

    class_weights = calculate_class_weights(
        config['data_root'],
        split='train'
    )

    # ========================================================================
    # 3. CREATION DU MODELE
    # ========================================================================

    print("\n3. CREATION DU MODELE")
    print("-" * 70)

    # Choisir le bon modele
    if config['model_type'] == 'unet_mini':
        model = build_unet_mini(
            input_shape=(*config['img_size'], 3),
            num_classes=config['num_classes'],
            class_weights=class_weights,
            learning_rate=config['learning_rate'],
            dice_weight=config['dice_weight']
        )
    else:  # unet_mobilenet
        model = build_model(
            input_shape=(*config['img_size'], 3),
            num_classes=config['num_classes'],
            class_weights=class_weights,
            learning_rate=config['learning_rate'],
            dice_weight=config['dice_weight'],
            encoder_freeze=config['encoder_freeze']
        )

    # ========================================================================
    # 4. CALLBACKS
    # ========================================================================

    print("\n4. CONFIGURATION DES CALLBACKS")
    print("-" * 70)

    callbacks = create_callbacks(config)

    # ========================================================================
    # 5. ENTRAINEMENT
    # ========================================================================

    print("\n5. LANCEMENT DE L'ENTRAINEMENT")
    print("=" * 70)
    print(f"\nConfiguration :")
    print(f"  Epochs          : {config['epochs']}")
    print(f"  Batch size      : {config['batch_size']}")
    print(f"  Learning rate   : {config['learning_rate']}")
    print(f"  Augmentation    : {config['augmentation']}")
    print(f"  Oversample      : x{config['oversample_factor']}")
    print("\nDemarrage de l'entrainement...\n")

    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=config['epochs'],
        callbacks=callbacks,
        verbose=1
    )

    # ========================================================================
    # 6. SAUVEGARDE DES RESULTATS
    # ========================================================================

    print("\n6. SAUVEGARDE DES RESULTATS")
    print("-" * 70)

    # Timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Sauvegarder l'historique
    history_path = Path(config['logs_dir']) / f'history_{timestamp}.json'
    with open(history_path, 'w') as f:
        # Convertir les valeurs numpy en float
        history_dict = {k: [float(v) for v in vals] for k, vals in history.history.items()}
        json.dump(history_dict, f, indent=2)
    print(f"  Historique sauvegarde : {history_path}")

    # Sauvegarder la configuration
    config_path = Path(config['logs_dir']) / f'config_{timestamp}.json'
    with open(config_path, 'w') as f:
        # Convertir les tuples en listes pour JSON
        config_json = {k: list(v) if isinstance(v, tuple) else v for k, v in config.items()}
        json.dump(config_json, f, indent=2)
    print(f"  Configuration sauvegardee : {config_path}")

    # Visualiser et sauvegarder les courbes
    plot_path = Path(config['logs_dir']) / f'training_curves_{timestamp}.png'
    plot_training_history(history.history, save_path=plot_path)

    # ========================================================================
    # 7. RESULTATS FINAUX
    # ========================================================================

    print("\n7. RESULTATS FINAUX")
    print("=" * 70)

    # Meilleurs scores
    best_epoch = np.argmax(history.history['val_mean_iou'])
    best_miou = history.history['val_mean_iou'][best_epoch]
    best_dice = history.history['val_dice_coefficient'][best_epoch]
    best_acc = history.history['val_pixel_accuracy'][best_epoch]
    final_loss = history.history['val_loss'][best_epoch]

    print(f"\nMeilleurs resultats (epoch {best_epoch + 1}) :")
    print(f"  Val Loss          : {final_loss:.4f}")
    print(f"  Val Mean IoU      : {best_miou:.4f}")
    print(f"  Val Dice Coeff    : {best_dice:.4f}")
    print(f"  Val Pixel Acc     : {best_acc:.4f}")

    print("\n" + "=" * 70)
    print("ENTRAINEMENT TERMINE AVEC SUCCES")
    print("=" * 70)

    return model, history


# ============================================================================
# INTERFACE LIGNE DE COMMANDE
# ============================================================================

def parse_args():
    """Parse les arguments de ligne de commande."""
    parser = argparse.ArgumentParser(
        description='Entrainer le modele U-Net + MobileNetV2 sur Cityscapes'
    )

    # Donnees
    parser.add_argument('--data-root', type=str, default='../data',
                        help='Chemin vers le dossier data/')
    parser.add_argument('--batch-size', type=int, default=8,
                        help='Taille des batchs')
    parser.add_argument('--img-size', type=int, nargs=2, default=[256, 512],
                        help='Taille des images (height width)')
    parser.add_argument('--augmentation', type=str, default='advanced',
                        choices=['light', 'standard', 'advanced', 'aggressive', 'none'],
                        help='Type d\'augmentation')
    parser.add_argument('--no-oversample', action='store_true',
                        help='Desactiver le sur-echantillonnage')
    parser.add_argument('--oversample-factor', type=int, default=3,
                        help='Facteur de sur-echantillonnage')

    # Modele
    parser.add_argument('--model-type', type=str, default='unet_mobilenet',
                        choices=['unet_mini', 'unet_mobilenet'],
                        help='Type de modele (unet_mini=baseline leger, unet_mobilenet=transfer learning)')
    parser.add_argument('--learning-rate', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--dice-weight', type=float, default=0.5,
                        help='Poids du Dice Loss (0-1)')
    parser.add_argument('--encoder-freeze', action='store_true',
                        help='Geler l\'encoder (pas de fine-tuning)')

    # Entrainement
    parser.add_argument('--epochs', type=int, default=50,
                        help='Nombre d\'epochs')
    parser.add_argument('--early-stopping', type=int, default=10,
                        help='Patience pour early stopping')
    parser.add_argument('--reduce-lr-patience', type=int, default=5,
                        help='Patience pour ReduceLROnPlateau')

    # Sorties
    parser.add_argument('--model-dir', type=str, default='../models',
                        help='Dossier pour sauvegarder les modeles')
    parser.add_argument('--logs-dir', type=str, default='../logs',
                        help='Dossier pour les logs')

    args = parser.parse_args()
    return args


def args_to_config(args):
    """Convertit les arguments en configuration."""
    config = DEFAULT_CONFIG.copy()

    config['data_root'] = args.data_root
    config['batch_size'] = args.batch_size
    config['img_size'] = tuple(args.img_size)
    config['augmentation'] = args.augmentation if args.augmentation != 'none' else False
    config['oversample_rare_classes'] = not args.no_oversample
    config['oversample_factor'] = args.oversample_factor

    config['model_type'] = args.model_type
    config['learning_rate'] = args.learning_rate
    config['dice_weight'] = args.dice_weight
    config['encoder_freeze'] = args.encoder_freeze

    config['epochs'] = args.epochs
    config['early_stopping_patience'] = args.early_stopping
    config['reduce_lr_patience'] = args.reduce_lr_patience

    config['model_dir'] = args.model_dir
    config['logs_dir'] = args.logs_dir

    return config


# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    """
    Point d'entree du script.

    Exemples d'utilisation :
        # Entrainement par defaut
        python train.py

        # Entrainement avec parametres personnalises
        python train.py --epochs 100 --batch-size 16 --learning-rate 5e-4

        # Entrainement rapide (sans augmentation ni sur-echantillonnage)
        python train.py --augmentation none --no-oversample --epochs 10
    """
    # Parser les arguments
    args = parse_args()
    config = args_to_config(args)

    # Lancer l'entrainement
    model, history = train(config)

    print("\nPour visualiser avec TensorBoard :")
    print(f"  tensorboard --logdir={config['logs_dir']}")
