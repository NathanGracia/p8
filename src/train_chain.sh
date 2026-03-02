#!/bin/bash
# Enchaîne deux entraînements : U-Net Mini sans aug, puis MobileNet sans aug

echo "========================================"
echo "RUN 1/2 : U-Net Mini (déjà lancé — skip)"
echo "========================================"
echo "Ce script enchaîne après le run en cours."
echo ""

echo "========================================"
echo "RUN 2/2 : U-Net + MobileNetV2 (sans aug)"
echo "========================================"
python train.py \
  --model-type unet_mobilenet \
  --img-size 128 256 \
  --augmentation none \
  --epochs 20 \
  --batch-size 4 \
  --no-oversample \
  >> ../logs/training_mobilenet_noaug.log 2>&1

echo "========================================"
echo "DONE — MobileNetV2 terminé"
echo "========================================"
