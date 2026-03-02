# Projet P8 - Segmentation Sémantique pour Véhicule Autonome

**Projet OpenClassrooms - Ingénieur IA**

---

## 📋 Description

Projet de **segmentation sémantique d'images** pour un système embarqué de véhicule autonome. Le modèle doit analyser en temps réel le flux vidéo d'une caméra embarquée et identifier pixel par pixel les éléments de l'environnement urbain (route, trottoirs, véhicules, piétons, obstacles, etc.).

**Entreprise** : Future Vision Transport (conception de systèmes embarqués de vision pour véhicules autonomes)

**Mission** : Concevoir et déployer un modèle de segmentation léger et rapide qui s'intègre dans la chaîne complète du système embarqué.

---

## 🎯 Objectifs

1. **Entraîner un modèle de segmentation** sur le dataset Cityscapes (8 catégories principales)
2. **Développer une API de prédiction** (Flask/FastAPI) déployée sur le Cloud
3. **Créer une application web de démonstration** (Flask/Streamlit) déployée sur le Cloud
4. **Garantir des performances adaptées** à un système embarqué (temps d'inférence < 100ms)

---

## 🗂️ Structure du projet

```
p8/
├── data/                           # Dataset Cityscapes
│   ├── gtFine/                    # Annotations (train/test)
│   │   ├── train/                 # 2 975 images d'entraînement
│   │   └── test/                  # 1 525 images de test
│   └── leftImg8bit/               # Images RGB
│       ├── train/
│       └── test/
│
├── notebooks/                      # Notebooks Jupyter
│   ├── 01_EDA_CLEAN.ipynb         # Exploration et analyse des données
│   ├── 02_Modeling.ipynb          # Modélisation (à créer)
│   └── test_dataloader.ipynb     # Test du data loader
│
├── src/                            # Code source Python
│   ├── data_loader.py             # Pipeline de chargement des données
│   ├── model.py                   # Architecture du modèle (à créer)
│   ├── train.py                   # Script d'entraînement (à créer)
│   └── utils.py                   # Fonctions utilitaires (à créer)
│
├── models/                         # Modèles sauvegardés (à créer)
│   └── best_model.h5              # Meilleur modèle entraîné
│
├── api/                            # API de prédiction (à créer)
│   ├── app.py                     # API Flask/FastAPI
│   └── requirements.txt
│
├── app/                            # Application web (à créer)
│   ├── streamlit_app.py           # Interface Streamlit
│   └── requirements.txt
│
├── docs/                           # Documentation (à créer)
│   ├── note_technique.pdf         # Note technique (10 pages)
│   └── presentation.pptx          # Support de présentation (30 slides max)
│
├── requirements.txt                # Dépendances Python principales
├── CLAUDE.md                       # Notes de développement
└── README.md                       # Ce fichier
```

---

## 📊 Dataset : Cityscapes

### Statistiques

- **Entraînement** : 2 975 images haute résolution (2048x1024)
- **Test** : 1 525 images haute résolution
- **18 villes allemandes** pour le train
- **6 villes** pour le test

### Les 8 catégories principales

Le projet utilise les **8 catégories principales** (au lieu des 32 sous-catégories) :

| ID | Catégorie      | Description                                      | Couleur   |
|----|----------------|--------------------------------------------------|-----------|
| 0  | void           | Pixels non classifiés                            | Noir      |
| 1  | flat           | Route, trottoir, parking                         | Violet    |
| 2  | construction   | Bâtiments, murs, clôtures                        | Gris      |
| 3  | object         | Poteaux, panneaux de signalisation, feux         | Jaune     |
| 4  | nature         | Végétation, terrain                              | Vert      |
| 5  | sky            | Ciel                                             | Bleu ciel |
| 6  | human          | Piétons, cyclistes                               | Rouge     |
| 7  | vehicle        | Voitures, camions, bus, motos, vélos, trains     | Bleu      |

### Mapping labelIds → Catégories

Cityscapes fournit 34 labelIds (0-33) qu'il faut convertir en 8 catégories :

- **labelIds 0-6** → void (0)
- **labelIds 7-10** → flat (1)
- **labelIds 11-16** → construction (2)
- **labelIds 17-20** → object (3)
- **labelIds 21-22** → nature (4)
- **labelId 23** → sky (5)
- **labelIds 24-25** → human (6)
- **labelIds 26-33** → vehicle (7)

---

## 🚀 Installation

### Prérequis

- Python 3.8+
- GPU recommandé (CUDA compatible)
- 8 GB RAM minimum

### Installation des dépendances

```bash
# Cloner le dépôt
git clone <url-du-repo>
cd p8

# Créer un environnement virtuel
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
.venv\Scripts\activate     # Windows

# Installer les dépendances
pip install -r requirements.txt
```

### Télécharger le dataset

Le dataset Cityscapes est déjà inclus dans le dossier `data/`.

Si besoin, télécharger depuis :
- [gtFine.zip](https://s3-eu-west-1.amazonaws.com/static.oc-static.com/prod/courses/files/AI+Engineer/Project+8+-+Cityscapes/gtFine.zip) (annotations)
- [leftImg8bit.zip](https://s3-eu-west-1.amazonaws.com/static.oc-static.com/prod/courses/files/AI+Engineer/Project+8+-+Cityscapes/leftImg8bit.zip) (images)

---

## 🔧 Utilisation

### 1. Exploration des données

```bash
jupyter notebook notebooks/01_EDA_CLEAN.ipynb
```

Ce notebook contient :
- Visualisation des images et masques
- Distribution des classes
- Analyse du déséquilibre
- Statistiques du dataset

### 2. Entraînement du modèle (à venir)

```bash
python src/train.py --epochs 50 --batch-size 8 --img-size 256 512
```

### 3. Évaluation (à venir)

```bash
python src/evaluate.py --model models/best_model.h5
```

### 4. API de prédiction (à venir)

```bash
cd api
python app.py
```

L'API sera accessible à `http://localhost:5000/predict`

### 5. Application web (à venir)

```bash
cd app
streamlit run streamlit_app.py
```

---

## 🏗️ Architecture technique

### Modèle de segmentation

- **Architecture** : U-Net avec encoder pré-entraîné
- **Encoders possibles** : VGG16, ResNet50, EfficientNetB0 (ImageNet weights)
- **Framework** : Keras/TensorFlow
- **Bibliothèque** : `segmentation_models` (Pavel Yakubovskiy)

### Pipeline d'entraînement

1. **Preprocessing**
   - Redimensionnement : 2048x1024 → 256x512 (ou 512x1024)
   - Normalisation : pixels [0, 255] → [0, 1]
   - Conversion labelIds → 8 catégories

2. **Data Augmentation**
   - Flip horizontal
   - Rotation (±15°)
   - Brightness/Contrast adjustment
   - Gaussian blur

3. **Entraînement**
   - Loss : Categorical Cross-Entropy + Dice Loss (pondérée)
   - Optimiseur : Adam (lr=1e-4)
   - Batch size : 8-16
   - Epochs : 50-100

4. **Évaluation**
   - Mean IoU (mIoU)
   - Dice Coefficient
   - Pixel Accuracy
   - Temps d'inférence

### API de prédiction

- **Framework** : Flask ou FastAPI
- **Endpoint** : `POST /predict`
- **Input** : Image (upload ou URL)
- **Output** : Mask prédit (image PNG + JSON avec classes détectées)
- **Déploiement** : Azure / Heroku / PythonAnywhere

### Application web

- **Framework** : Streamlit ou Flask
- **Fonctionnalités** :
  - Liste des images disponibles (dropdown)
  - Upload d'image personnalisée
  - Affichage : Image originale | Mask réel | Mask prédit
  - Métriques : IoU par classe, temps d'inférence
- **Déploiement** : Streamlit Cloud / Heroku

---

## 📈 Performance attendue

### Métriques cibles

- **Mean IoU** : > 60% (état de l'art ~80%)
- **Temps d'inférence** : < 100ms par image (256x512)
- **Taille du modèle** : < 50 MB

### Déséquilibre des classes

⚠️ **Problème identifié** : Classes très déséquilibrées

- **Classe dominante** : flat (route, trottoir) - ~50% des pixels
- **Classes minoritaires** : human, vehicle - critiques pour la sécurité !

**Solutions implémentées** :
- Weighted Loss (poids inversement proportionnels à la fréquence)
- Data augmentation ciblée
- Focal Loss (optionnel)

---

## 📦 Livrables

1. ✅ **Scripts notebook** - Pipeline complet d'entraînement
2. ⏳ **API déployée** - Flask/FastAPI sur le Cloud
3. ⏳ **Application web** - Streamlit déployée
4. ⏳ **Note technique** - 10 pages (état de l'art, résultats, pistes)
5. ⏳ **Support de présentation** - 30 slides max

Format de nommage : `Nom_Prénom_n°_livrable_mmaaaa`

---

## 🧪 Tests

### Test du data loader

```bash
jupyter notebook notebooks/test_dataloader.ipynb
```

### Test de l'API (à venir)

```bash
pytest tests/test_api.py
```

---

## 📚 Ressources

### Papers

- [U-Net](https://arxiv.org/abs/1505.04597) (Ronneberger et al., 2015)
- [SegNet](https://arxiv.org/abs/1511.00561) (Badrinarayanan et al., 2017)
- [DeepLabV3+](https://arxiv.org/abs/1802.02611) (Chen et al., 2018)
- [Cityscapes Dataset](https://arxiv.org/abs/1604.01685) (Cordts et al., 2016)

### Bibliothèques utiles

- [segmentation_models](https://github.com/qubvel/segmentation_models) - Architectures Keras
- [albumentations](https://albumentations.ai/) - Data augmentation
- [tensorflow-addons](https://www.tensorflow.org/addons) - Métriques avancées

### Tutoriels

- [Keras Semantic Segmentation](https://keras.io/examples/vision/oxford_pets_image_segmentation/)
- [TensorFlow Image Segmentation](https://www.tensorflow.org/tutorials/images/segmentation)

---

## 🔍 État d'avancement

### ✅ Complété

- [x] Dataset Cityscapes installé et exploré
- [x] Notebook EDA complet (01_EDA_CLEAN.ipynb)
- [x] Data loader fonctionnel (à adapter pour Keras)
- [x] Requirements.txt avec dépendances

### 🚧 En cours

- [ ] Adapter data_loader.py pour Keras
- [ ] Créer l'architecture U-Net (src/model.py)
- [ ] Script d'entraînement (src/train.py)

### ⏳ À faire

- [ ] Entraîner le modèle baseline
- [ ] Optimiser les hyperparamètres
- [ ] Développer l'API Flask/FastAPI
- [ ] Créer l'application Streamlit
- [ ] Déployer sur le Cloud
- [ ] Rédiger la note technique
- [ ] Préparer la présentation

---

## ⚠️ Points d'attention

1. **Framework Keras** : Le projet utilise Keras (pas PyTorch)
2. **8 catégories** : Ne pas utiliser les 32 sous-catégories (consigne Franck)
3. **Déséquilibre** : Implémenter une loss pondérée obligatoirement
4. **Performance** : Optimiser pour temps d'inférence (système embarqué)
5. **Déploiement** : Enregistrer une démo vidéo (Heroku est payant)

---

## 👥 Équipe

- **Vous** : Segmentation des images
- **Franck** : Traitement des images (en amont)
- **Laura** : Système de décision (en aval)

---

## 📄 Licence

Ce projet utilise le dataset **Cityscapes** :
- Usage académique et non-commercial uniquement
- Pas de redistribution autorisée
- © Daimler AG, MPI Informatics, TU Darmstadt

---

## 📞 Contact

Projet OpenClassrooms - Formation Ingénieur IA

---

**Date de création** : 2026-02-02
**Dernière mise à jour** : 2026-02-09
**Version** : 1.0
