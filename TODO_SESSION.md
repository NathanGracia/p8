# TODO - Session entraînement baseline

## Objectif
Lancer l'entraînement du modèle U-Net Mini (baseline) en local sur CPU.

## Étapes à faire dans l'ordre

### 1. Installer les dépendances manquantes
```bash
pip install albumentations
```

### 2. Vérifier que toutes les dépendances sont OK
```bash
pip install -r C:\Users\nathan\Documents\OpenClassrooms\p8\requirements.txt
```

### 3. Lancer un test rapide (2 epochs pour vérifier que ça tourne)
```bash
cd C:\Users\nathan\Documents\OpenClassrooms\p8\src && python train.py --model-type unet_mini --epochs 2 --batch-size 2 --img-size 128 256 --augmentation none --no-oversample
```

### 4. Si le test passe → lancer l'entraînement complet
```bash
cd C:\Users\nathan\Documents\OpenClassrooms\p8\src && python train.py --model-type unet_mini --epochs 20 --batch-size 4 --img-size 128 256 --augmentation light
```

## Notes
- Les warnings `oneDNN` et `MessageFactory` sont normaux, pas bloquants
- Images en 128x256 pour CPU (plus petit = plus rapide)
- Durée estimée : ~13-14h pour 20 epochs en CPU
- Les modèles sont sauvegardés dans `models/`
- Les logs dans `logs/`
