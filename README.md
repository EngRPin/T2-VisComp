# GTSRB — Kit Inicial

## Conteudo

```
alunos/
├── src/
│   └── gtsrb.py       # Carregamento de dados + exportacao de predicoes
├── exemplo.py          # Uso minimo
├── requirements.txt
└── README.md
```

## Regras

1. **NAO modifique** `src/gtsrb.py` — ele garante o mesmo split e formato para todos.
2. O test set so deve ser usado para **avaliacao final**.
3. Todos os modelos devem ser **treinados do zero** (sem pesos pre-treinados).

## Uso

```python
from src.gtsrb import get_dataloaders, save_predictions, NUM_CLASSES, GTSRB_CLASSES

# Carregar dados (split fixo: seed=42, 80% treino / 20% validacao)
train_loader, val_loader, test_loader = get_dataloaders(img_size=32, batch_size=128)

# Para usar data augmentation, passe seu proprio transform de treino:
from torchvision import transforms
meu_transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.RandomAffine(degrees=10, translate=(0.1, 0.1)),
    transforms.ColorJitter(brightness=0.3, contrast=0.3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.3403, 0.3121, 0.3214],
                         std=[0.2724, 0.2608, 0.2669]),
])
train_loader, val_loader, test_loader = get_dataloaders(
    img_size=32, batch_size=128, train_transform=meu_transform,
)
```

## Entrega

Cada grupo deve entregar **um CSV por experimento** com as predicoes no test set.

```python
save_predictions(y_pred, "results/predicoes_baseline.csv", experiment_name="Baseline CNN SGD")
```

Formato do CSV (12.630 linhas, uma por imagem de teste):

```
image_index,predicted_class
0,14
1,1
2,38
...
12629,35
```

Nomeie cada arquivo indicando o experimento:

```
results/
├── predicoes_baseline.csv
├── predicoes_adam.csv
├── predicoes_bn.csv
├── predicoes_bn_aug.csv
└── predicoes_resnet.csv
```
