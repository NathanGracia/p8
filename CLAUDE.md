# Claude Notes - Projet P8 OpenClassrooms

## Vue d'ensemble du projet

**Entreprise** : Future Vision Transport (systèmes embarqués de vision pour véhicules autonomes)

**Rôle** : Ingénieur IA - Équipe R&D, spécialisé sur la **segmentation d'images**

**Mission** : Concevoir un modèle de segmentation d'images qui s'intègre dans la chaîne complète du système embarqué de vision.

---

## Architecture du système embarqué

Le système complet se décompose en 4 blocs :

1. **Acquisition des images** en temps réel
2. **Traitement des images** (Franck)
3. **Segmentation des images** (VOUS) ← Notre mission
4. **Système de décision** (Laura)

Votre bloc (3) est alimenté par le bloc (2) et alimente le bloc (4).

---

## Contraintes et exigences du projet

### Contraintes de Franck (traitement des images)

- **Dataset** : Cityscapes
  - Lien 1 : https://www.cityscapes-dataset.com/
  - Téléchargement direct : [Lien 1](https://s3-eu-west-1.amazonaws.com/static.oc-static.com/prod/courses/files/AI+Engineer/Project+8+-+Cityscapes/gtFine.zip) | [Lien 2](https://s3-eu-west-1.amazonaws.com/static.oc-static.com/prod/courses/files/AI+Engineer/Project+8+-+Cityscapes/leftImg8bit.zip)
- **IMPORTANT** : On a uniquement besoin des **8 catégories principales** (et non pas des 32 sous-catégories)

### Contraintes de Laura (système de décision)

- **API simple** à utiliser
- **Entrée** : une image
- **Sortie** : la segmentation de l'image (mask prédit)

### Contraintes techniques

- **Framework** : **Keras** (framework commun à toute l'équipe)
- **Déploiement** : Cloud (Azure, Heroku, PythonAnywhere ou autre)
- **Performance** : Modèle léger et rapide pour système embarqué

---

## Les 8 catégories principales Cityscapes

Les 8 catégories à utiliser (au lieu des 32 sous-catégories) :

1. **void** (0) - Pixels non classifiés
2. **flat** (1) - Surfaces planes (route, trottoir)
3. **construction** (2) - Bâtiments, murs, clôtures
4. **object** (3) - Poteaux, panneaux de signalisation
5. **nature** (4) - Végétation, terrain
6. **sky** (5) - Ciel
7. **human** (6) - Personnes, cyclistes
8. **vehicle** (7) - Voitures, camions, bus, trains, motos, vélos

**Mapping labelIds (0-33) → Catégories (0-7)** :
- labelIds 0-6 → void (0)
- labelIds 7-10 → flat (1)
- labelIds 11-16 → construction (2)
- labelIds 17-20 → object (3)
- labelIds 21-22 → nature (4)
- labelId 23 → sky (5)
- labelIds 24-25 → human (6)
- labelIds 26-33 → vehicle (7)

---

## Plan d'action défini

### 1. Modèle de segmentation
- Entraîner un modèle sur les **8 catégories principales**
- Framework : **Keras**
- Architecture recommandée : U-Net avec backbone pré-entraîné

### 2. API de prédiction
- Framework : Flask ou FastAPI
- Déployer sur le Cloud (Azure, Heroku, PythonAnywhere, etc.)
- **Entrée** : image
- **Sortie** : mask prédit (segments identifiés)

### 3. Application web de démonstration
- Framework : Flask ou Streamlit
- Déployer sur le Cloud
- **Fonctionnalités** :
  - Affichage de la liste des ID d'images disponibles
  - Lancement de la prédiction pour un ID sélectionné
  - Affichage de l'image réelle, du mask réel et du mask prédit

---

## Livrables attendus

1. **Scripts développés sur notebook** - Pipeline complet
   - Démontre le caractère "industrialisable" du travail
   - Générateur de données (data augmentation)

2. **API déployée** (Flask/FastAPI)
   - Permet à Laura d'utiliser facilement le modèle

3. **Application web déployée** (Flask/Streamlit)
   - Illustre le travail auprès des collègues

4. **Note technique** (10 pages environ)
   - Présentation des différentes approches
   - État de l'art
   - Modèle et architecture retenue
   - Synthèse des résultats (gains avec data augmentation)
   - Conclusion et pistes d'amélioration

5. **Support de présentation** (30 slides max)
   - Démarche méthodologique
   - Présentation des résultats à Laura

### Nomenclature des livrables

Format : `Nom_Prénom_n°_nom_livrable_mmaaaa`

Exemple : `Dupont_Jean_1_scripts_012024`

---

## Soutenance (30 minutes)

### Présentation (20 minutes)

1. **Contexte et principes** (5 min)
   - Contexte, objectifs
   - Principes de segmentation
   - Mesures de performance

2. **Modèles et comparaisons** (10 min)
   - Différents modèles testés
   - Simulations et comparaisons

3. **Mise en production** (5 min)
   - Architecture API et application web
   - Démarche de mise en production
   - **Démonstration** de l'application et prédiction d'un mask

### Discussion (5 minutes)
- Challenge sur les choix techniques

### Débriefing (5 minutes)

**Important** : Présentations < 15 min ou > 25 min peuvent être refusées.

**Note sur le déploiement** : Heroku est devenu payant (nov 2022). Enregistrer une démo vidéo pendant la soutenance pour éviter de maintenir l'application en ligne après.

---

## Dataset Cityscapes

### Statistiques

- **Train** : 2 975 images (18 villes allemandes)
- **Test** : 1 525 images (6 villes)

### Villes

**Train** : aachen, bochum, bremen, cologne, darmstadt, dusseldorf, erfurt, hamburg, hanover, jena, krefeld, monchengladbach, strasbourg, stuttgart, tubingen, ulm, weimar, zurich

**Test** : berlin, bielefeld, bonn, leverkusen, mainz, munich

### Fichiers par image

1. `*_gtFine_color.png` - Visualisation colorée
2. `*_gtFine_labelIds.png` - Masque de segmentation (IDs)
3. `*_gtFine_instanceIds.png` - IDs d'instance
4. `*_gtFine_polygons.json` - Annotations polygones

**Format** : `ville_numero_sequenceId_gtFine_type.ext`

Exemple : `jena_000000_000019_gtFine_labelIds.png`

---

## Structure du projet

```
p8/
├── data/                    # Dataset Cityscapes
│   ├── gtFine/             # Annotations (train/test)
│   └── leftImg8bit/        # Images RGB (train/test)
├── notebooks/               # Notebooks Jupyter
│   ├── 01_EDA_CLEAN.ipynb          # Exploration des données
│   ├── 02_Modeling.ipynb           # Modélisation (à créer)
│   └── test_dataloader.ipynb      # Test du data loader
├── src/                     # Code source Python
│   ├── data_loader.py              # Pipeline de chargement (à adapter pour Keras)
│   ├── model.py                    # Architecture (à créer)
│   ├── train.py                    # Script d'entraînement (à créer)
│   └── utils.py                    # Utilitaires (à créer)
├── models/                  # Modèles sauvegardés (à créer)
├── api/                     # API Flask/FastAPI (à créer)
├── app/                     # Application web (à créer)
├── requirements.txt         # Dépendances Python
├── CLAUDE.md                # Ce fichier
└── README.md                # Documentation projet (à créer)
```

---

## État d'avancement

### ✅ Réalisé

**Session 2026-02-02** :
1. **Dataset Cityscapes** installé et structuré
2. **Notebook EDA** (`01_EDA_CLEAN.ipynb`)
   - Exploration du dataset
   - Distribution des classes
   - Détection du déséquilibre

**Session 2026-02-09** :
3. **Data loader Keras** (`src/data_loader.py`)
   - ✅ Classe `CityscapesSequence` (hérite de `keras.utils.Sequence`)
   - ✅ Charge images + masques
   - ✅ Convertit labelIds (0-33) → 8 catégories (0-7)
   - ✅ One-hot encoding automatique
   - ✅ Data augmentation (Albumentations)
   - ✅ Calcul des poids de classe pour loss pondérée
   - ✅ Fonctions utilitaires (decode, colorize, etc.)
4. **Notebook de test** (`notebooks/test_dataloader.ipynb`)
   - Test complet du data loader
   - Visualisations
   - Intégration avec Keras
5. **Requirements.txt** avec dépendances Keras/TensorFlow

### ❌ À faire

1. **Créer dossier models/**
2. **Architecture du modèle** (`src/model.py`)
   - U-Net avec backbone pré-entraîné (Keras)
   - Utiliser `segmentation_models` library
3. **Script d'entraînement** (`src/train.py`)
   - Loss pondérée (utiliser les poids calculés par le data loader)
   - Métriques : mIoU, Dice, Pixel Accuracy
   - Callbacks : ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
4. **Notebook de modélisation** (`notebooks/02_Modeling.ipynb`)
   - Comparaison de plusieurs backbones (VGG16, ResNet50, EfficientNet)
   - Expérimentations avec/sans data augmentation
5. **Utilitaires** (`src/utils.py`)
   - Visualisation des prédictions
   - Calcul des métriques (IoU, Dice)
   - Courbes d'apprentissage
6. **API de prédiction** (Flask/FastAPI)
7. **Application web** (Flask/Streamlit)
8. **Déploiement Cloud** (API + App)
9. **Note technique** (10 pages)
10. **Support de présentation** (30 slides)

---

## Pipeline technique prévu

### 1. Preprocessing
- Normalisation (0-1)
- Redimensionnement (2048x1024 → 256x512 ou 512x1024)
- Conversion labelIds → 8 catégories
- Data augmentation (rotation, flip, brightness, etc.)

### 2. Modélisation
- **Architecture** : U-Net avec encoder pré-entraîné (VGG16, ResNet50, EfficientNet)
- **Loss** : Categorical Cross-Entropy + Dice Loss (pondérée)
- **Optimiseur** : Adam
- **Métriques** : mIoU, Dice Coefficient, Pixel Accuracy

### 3. Évaluation
- Mean IoU par classe
- IoU global
- Temps d'inférence (contrainte embarquée)
- Courbes d'apprentissage

### 4. Déploiement
- Sauvegarder le meilleur modèle (.h5 ou .keras)
- API Flask/FastAPI avec endpoint `/predict`
- Application Streamlit avec interface utilisateur
- Hébergement Cloud

---

## Notes importantes

### Déséquilibre des classes

D'après l'EDA :
- **Classe dominante** : flat (route, trottoir)
- **Classes minoritaires** : human, vehicle (critiques pour sécurité !)

**Solutions** :
- Weighted Loss (pénaliser plus les erreurs sur classes minoritaires)
- Focal Loss
- Data augmentation ciblée
- Sur-échantillonnage des classes rares

### Performance attendue

- **mIoU** : Objectif > 60% (état de l'art ~80%)
- **Temps d'inférence** : < 100ms par image (contrainte embarquée)
- **Taille du modèle** : < 50 MB si possible

### Licence Cityscapes

- Usage académique et non-commercial uniquement
- Pas de redistribution
- Propriété : Daimler AG, MPI Informatics, TU Darmstadt

---

## Ressources utiles

### Architectures de segmentation
- U-Net (Ronneberger et al., 2015)
- SegNet (Badrinarayanan et al., 2017)
- DeepLabV3+ (Chen et al., 2018)

### Keras/TensorFlow
- `tf.keras.applications` : Encoders pré-entraînés
- `segmentation_models` : Bibliothèque Keras pour segmentation
- `albumentations` : Data augmentation

### Métriques
- IoU (Intersection over Union)
- Dice Coefficient = 2 * IoU / (1 + IoU)
- Pixel Accuracy

---

---

## Session 2026-02-16 : Configuration GPU + Entraînement U-Net Mini

### État actuel
- ✅ U-Net Mini créé (7.86M paramètres)
- ✅ CUDA installé (à vérifier après redémarrage terminal)
- ⏳ En attente : Vérification GPU + Lancement entraînement

### Prochaines étapes (après redémarrage terminal)

#### 1. Vérifier CUDA
```bash
# Vérifier CUDA Toolkit
nvcc --version

# Vérifier que TensorFlow détecte le GPU
python -c "import tensorflow as tf; print('GPUs:', tf.config.list_physical_devices('GPU'))"
```

**Résultat attendu** :
- nvcc doit afficher : `Cuda compilation tools, release 12.x`
- TensorFlow doit afficher : `[PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]`

#### 2. Lancer l'entraînement U-Net Mini avec GPU
```bash
cd src
python train.py --model-type unet_mini --epochs 30 --batch-size 8 --augmentation light
```

**Configuration** :
- Modèle : U-Net Mini (baseline)
- Epochs : 30
- Batch size : 8 (optimisé GPU)
- Augmentation : light
- Durée estimée : **10-15 min avec GPU** (vs 2h par epoch sur CPU)

#### 3. Si le GPU ne fonctionne pas

**Plan B - Entraînement CPU optimisé** :
```bash
cd src
python train.py --model-type unet_mini --epochs 20 --batch-size 4 --img-size 128 256 --augmentation light
```

(Images réduites 128x256 → 4x plus rapide sur CPU)

### Fichiers de sortie attendus
- `models/unetmini_YYYYMMDD_HHMMSS.h5` - Meilleur modèle
- `logs/tensorboard_YYYYMMDD_HHMMSS/` - Logs TensorBoard
- `logs/training_history_YYYYMMDD_HHMMSS.csv` - Historique CSV

### Monitoring pendant l'entraînement
```bash
# Dans un autre terminal
tensorboard --logdir=logs
```

Puis ouvrir : http://localhost:6006

---

## Session 2026-02-16 01:04 : Entraînement U-Net Mini LANCÉ 🚀

### Configuration GPU
- ❌ TensorFlow Windows = CPU uniquement (pas de support CUDA)
- ✅ TensorFlow 2.18.0 installé
- ✅ CUDA 12.6 + Driver NVIDIA détectés (mais non utilisés par TF Windows)

### Entraînement en cours
- **Modèle** : U-Net Mini (7.86M paramètres)
- **Démarré** : 2026-02-16 à 01:04
- **Configuration** :
  - Epochs : 20
  - Batch size : 4
  - Image size : 128x256 (réduit pour CPU)
  - Augmentation : light
- **Durée estimée** : ~13-14 heures
- **PID** : 1684
- **Log** : `training_output.log`

### Monitoring
```bash
# Vérifier la progression
bash monitor_training.sh

# Suivre en temps réel
tail -f training_output.log

# Arrêter
kill $(cat training.pid | cut -d'=' -f2)
```

### Fichiers créés
- ✅ `monitor_training.sh` - Script de monitoring
- ✅ `TRAINING_GUIDE.md` - Guide complet
- ✅ `training.pid` - PID du processus
- ✅ `training_output.log` - Log en temps réel

### Prochaines étapes (après entraînement)
1. Analyser les résultats (TensorBoard, CSV)
2. Évaluer le modèle sur test set
3. Générer des prédictions
4. ~~Considérer WSL2 pour entraînement GPU si nécessaire~~ ✅ FAIT

---

## Session 2026-02-16 12:35 : Migration WSL2 + GPU 🚀

### Objectif
Migrer l'entraînement vers WSL2 pour exploiter la RTX 5070 12GB.

### Actions réalisées

#### 1. Arrêt entraînement CPU
- ✅ Processus 1684 arrêté

#### 2. Configuration WSL2
- ✅ Ubuntu-22.04 déjà installé (WSL2)
- ✅ Système mis à jour (`apt update && upgrade`)
- ✅ Python 3.10 + outils de développement installés
- ✅ Environnement virtuel créé (`~/p8_gpu/venv`)

#### 3. Installation TensorFlow GPU
- ✅ TensorFlow 2.16.2 avec CUDA 12.3 installé
- ✅ Toutes les bibliothèques CUDA téléchargées (2.3 GB)
  - nvidia-cublas-cu12, nvidia-cudnn-cu12, nvidia-cufft-cu12, etc.
- ✅ Dépendances installées :
  - numpy 1.26.4 (compatible TF)
  - pandas, matplotlib, seaborn
  - albumentations 2.0.8
  - opencv-python 4.9.0.80 (compatible numpy<2)
  - scikit-learn, scikit-image

#### 4. Vérification GPU
```bash
GPUs détectés: [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
TensorFlow version: 2.16.2
GPU disponible: True
```
⚠️ **Note** : Kernels CUDA compilés JIT (compute capability 12.0 pour RTX 5070)

#### 5. Copie du projet
- ✅ Projet copié vers `~/p8_gpu/p8/` (883 MB)
- ✅ Dossiers créés : `models/`, `logs/`
- ✅ Dataset Cityscapes copié

#### 6. Lancement entraînement GPU
- **PID** : 6616
- **Commande** :
  ```bash
  python train.py --model-type unet_mobilenet --epochs 30 --batch-size 8 --img-size 256 512 --augmentation light
  ```
- **Configuration** :
  - Modèle : U-Net + MobileNetV2
  - Dataset : 2975 train (sur-échantillonné à 8895) + 500 val
  - Batch size : 8
  - Image size : 256x512
  - Augmentation : light
  - Epochs : 30
- **Durée estimée** : **1h - 2h30** (vs 13-14h CPU)

### Fichiers créés

#### Scripts Windows
- ✅ `monitor_gpu_training.bat` - Monitoring depuis Windows
- ✅ `copy_results_from_wsl.bat` - Copie des résultats WSL → Windows

#### Scripts WSL2
- ✅ `~/p8_gpu/p8/launch_training.sh` - Lancement entraînement
- ✅ `~/p8_gpu/p8/monitor_gpu.sh` - Monitoring WSL

#### Documentation
- ✅ `WSL2_GPU_GUIDE.md` - Guide complet WSL2 + GPU

### État actuel (12:50)
- ✅ Entraînement GPU en cours
- ✅ Calcul des poids de classe : 500/2975 images
- ✅ CPU : 117% | RAM : 588 MB
- ⏳ En attente : Démarrage de l'epoch 1/30

### Monitoring

**Option 1** : Double-clic sur `monitor_gpu_training.bat`

**Option 2** : Commande manuelle
```bash
wsl -d Ubuntu-22.04 bash ~/p8_gpu/p8/monitor_gpu.sh
```

**Option 3** : Log complet
```bash
wsl -d Ubuntu-22.04 tail -f ~/p8_gpu/p8/training_gpu.log
```

### Récupération des résultats

Double-clic sur `copy_results_from_wsl.bat` pour copier :
- `models/*.h5` → Modèles entraînés
- `logs/*.csv` → Historique d'entraînement
- `training_gpu.log` → Log complet

### Performances attendues

| Métrique | GPU (RTX 5070) | CPU (Baseline) |
|----------|----------------|----------------|
| Temps/epoch | 3-5 min | ~40 min |
| Temps total (30 epochs) | **1-2h30** | 13-14h |
| mIoU val | **70-75%** | 67% |
| Gain | **10x** | 1x |

### Prochaines étapes
1. ⏳ Attendre fin entraînement (~1h30)
2. Récupérer résultats avec `copy_results_from_wsl.bat`
3. Analyser performances (TensorBoard, CSV)
4. Comparer avec baseline CPU
5. Tester avec U-Net + ResNet50 si besoin

---

**Date de création** : 2026-02-02
**Dernière mise à jour** : 2026-02-16 12:50
**Auteur** : Claude Code Assistant
