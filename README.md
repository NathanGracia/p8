# P8 — Segmentation sémantique pour véhicule autonome

**Projet OpenClassrooms — Ingénieur IA**
**Entreprise** : Future Vision Transport
**Dataset** : Cityscapes · **Framework** : TensorFlow / Keras 2.16

---

## Contexte

Ce module est le bloc 3 d'une chaîne embarquée de vision pour véhicule autonome :

```
[Capteurs] → [Prétraitement (Franck)] → [Segmentation (ce module)] → [Décision (Laura)]
```

Le modèle classifie chaque pixel d'une image en 8 catégories (route, bâtiments, piétons, véhicules...) et expose le résultat via une API REST.

---

## Résultats

| Expérience | Modèle | Augmentation | val mIoU |
|-----------|--------|-------------|---------|
| Baseline | U-Net Mini | Non | 70,1 % |
| Transfer learning | U-Net + MobileNetV2 | Non | 72,4 % |
| Meilleur | U-Net Mini | Oui + Oversample ×3 | **74,0 %** |

Objectif fixé : > 60 % mIoU ✅

---

## Services déployés

| Service | URL |
|---------|-----|
| Application web (Streamlit) | https://p8.nathangracia.com |
| API de prédiction (FastAPI) | https://p8-api.nathangracia.com |
| Documentation API (Swagger) | https://p8-api.nathangracia.com/docs |

---

## Structure

```
p8/
├── api/                  # API FastAPI
│   ├── app.py            # Endpoints REST
│   ├── predictor.py      # Chargement modèle + inférence
│   ├── Dockerfile
│   └── requirements.txt
│
├── app/                  # Application web
│   ├── streamlit_app.py  # Interface de démonstration
│   ├── Dockerfile
│   └── requirements.txt
│
├── src/                  # Code source
│   ├── data_loader.py    # CityscapesSequence (Keras)
│   ├── model.py          # Architecture U-Net + métriques + losses
│   ├── train.py          # Script d'entraînement
│   └── data_augmentation.py
│
├── notebooks/
│   ├── 01_EDA_CLEAN.ipynb    # Exploration du dataset
│   ├── 02_Modeling.ipynb     # Comparaison des expériences
│   └── test_dataloader.ipynb
│
├── models/               # Modèles entraînés
│   ├── unet_mobilenet_20260209_132821.h5   # U-Net + MobileNetV2 (déployé)
│   ├── unetmini_20260216_110245.h5         # U-Net Mini baseline
│   └── unetmini_20260302_021229.h5         # U-Net Mini + augmentation
│
├── logs/                 # Résultats des 3 expériences
│   ├── config_*.json     # Hyperparamètres
│   ├── history_*.json    # Courbes d'apprentissage
│   └── *.png             # Figures (distribution, métriques, prédictions)
│
├── samples/              # Images de démonstration (val Cityscapes)
├── data/                 # Dataset Cityscapes complet
│
├── docs/
│   ├── note_technique_P8.docx
│   ├── generate_note_technique.py
│   └── DEPLOY.md         # Guide de déploiement VPS
│
├── docker-compose.yml
├── requirements.txt
└── CLAUDE.md
```

---

## Lancement local

### Avec Docker (recommandé)

```bash
docker compose up -d
```

- App : http://localhost:8501
- API : http://localhost:8000
- Docs : http://localhost:8000/docs

### Sans Docker

```bash
# API
cd api && uvicorn app:app --reload --port 8000

# App (dans un autre terminal)
cd app && streamlit run streamlit_app.py
```

### Entraînement

```bash
cd src
python train.py --model-type unet_mobilenet --epochs 30 --batch-size 8 --augmentation light
```

Options disponibles :
- `--model-type` : `unet_mini` ou `unet_mobilenet`
- `--augmentation` : `none`, `light`, `heavy`
- `--img-size` : ex. `128 256` ou `256 512`
- `--no-oversample` : désactive l'oversampling des classes rares

---

## API — Endpoints

| Méthode | Endpoint | Description |
|---------|----------|-------------|
| GET | `/` | Health check + infos modèle |
| GET | `/images` | Liste des images de validation disponibles |
| GET | `/images/{id}/rgb` | Image RGB originale (PNG) |
| GET | `/images/{id}/gt` | Masque ground truth colorisé (PNG) |
| POST | `/predict` | Upload image → masque prédit (PNG) |
| GET | `/predict/{id}` | Prédiction par ID (PNG) |

---

## Les 8 catégories

| ID | Classe | Éléments |
|----|--------|---------|
| 0 | void | Pixels non classifiés |
| 1 | flat | Route, trottoir, parking |
| 2 | construction | Bâtiments, murs, clôtures |
| 3 | object | Poteaux, panneaux, feux |
| 4 | nature | Végétation, terrain |
| 5 | sky | Ciel |
| 6 | human | Piétons, cyclistes |
| 7 | vehicle | Voitures, camions, bus, motos |

---

## Déploiement continu

Chaque push sur `master` déclenche un workflow GitHub Actions qui se connecte au VPS en SSH, tire le code et rebuilde les containers Docker automatiquement.

---

*Dataset Cityscapes — usage académique uniquement © Daimler AG, MPI Informatics, TU Darmstadt*
