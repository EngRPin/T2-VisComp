#!/usr/bin/env python3
"""Exemplo minimo de uso do kit fornecido.

    python exemplo.py
"""

import torch
import torch.nn as nn

from src.gtsrb import NUM_CLASSES, get_dataloaders, save_predictions

# 1) Dados — o split ja vem fixo, basta chamar:
train_loader, val_loader, test_loader = get_dataloaders(img_size=32, batch_size=128)

# 2) Seu modelo — implemente aqui
model = nn.Sequential(

    nn.Conv2d(
        in_channels=3,
        out_channels=32,
        kernel_size=3,
        padding=1
    ),
    nn.ReLU(),
    nn.MaxPool2d(2),

    nn.Conv2d(
        in_channels=32,
        out_channels=64,
        kernel_size=3,
        padding=1
    ),
    nn.ReLU(),
    nn.MaxPool2d(2),

    nn.Flatten(),

    nn.Linear(64 * 8 * 8, 128),
    nn.ReLU(),

    nn.Linear(128, NUM_CLASSES)
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 3) Seu treino — implemente aqui
#    Use train_loader para treinar e val_loader para validar.
#    NAO use test_loader durante o treino.

# 4) Avaliacao final e entrega
model.eval()
all_preds = []
with torch.no_grad():
    for images, _ in test_loader:
        outputs = model(images.to(device))
        all_preds.append(outputs.argmax(dim=1).cpu())
y_pred = torch.cat(all_preds)

# Gera o CSV para entrega — um por experimento
save_predictions(y_pred, "results/predicoes_baseline.csv", experiment_name="Baseline")
print(f"Predicoes salvas ({len(y_pred)} imagens)")
