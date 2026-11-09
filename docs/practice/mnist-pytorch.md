# 실습: MNIST 분류기 (PyTorch)

**목표**: MLP와 CNN으로 손글씨 숫자를 분류하고, 정확도 **99%** 를 달성합니다.
**관련 노트**: [PyTorch 기초](../notes/08-pytorch.md), [CNN](../notes/09-cnn.md)

> GPU가 없어도 되지만, 있으면 훨씬 빠릅니다. 로컬에 설치가 번거로우면 [Google Colab](https://colab.research.google.com/) → 런타임 → GPU를 쓰세요.

## 1. 데이터

```python
import torch, torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

device = "cuda" if torch.cuda.is_available() else "cpu"

tf = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),     # MNIST 평균/표준편차
])
train_ds = datasets.MNIST("data", train=True, download=True, transform=tf)
test_ds = datasets.MNIST("data", train=False, download=True, transform=tf)

train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=512)

x, y = next(iter(train_loader))
print(x.shape, y.shape)          # (128,1,28,28), (128,)
```

먼저 이미지 몇 장을 직접 그려서 눈으로 확인하세요. (`plt.imshow(x[0,0], cmap="gray")`)

## 2. 학습/평가 함수

```python
def train_epoch(model, loader, opt, loss_fn):
    model.train()
    total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        loss = loss_fn(model(x), y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        total += loss.item() * len(x)
    return total / len(loader.dataset)

@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    correct = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        correct += (model(x).argmax(1) == y).sum().item()
    return correct / len(loader.dataset)
```

## 3. 모델 1: MLP (기준선)

```python
mlp = nn.Sequential(
    nn.Flatten(),
    nn.Linear(784, 256), nn.ReLU(),
    nn.Linear(256, 128), nn.ReLU(),
    nn.Linear(128, 10),
).to(device)

opt = torch.optim.Adam(mlp.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()

for epoch in range(5):
    loss = train_epoch(mlp, train_loader, opt, loss_fn)
    print(epoch, round(loss, 4), "test acc:", evaluate(mlp, test_loader))
```

보통 **97.5~98%** 가 나옵니다.

## 4. 모델 2: CNN

```python
cnn = nn.Sequential(
    nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),   # 14x14
    nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),  # 7x7
    nn.Flatten(),
    nn.Dropout(0.3),
    nn.Linear(64 * 7 * 7, 10),
).to(device)

opt = torch.optim.Adam(cnn.parameters(), lr=1e-3)
sched = torch.optim.lr_scheduler.StepLR(opt, step_size=3, gamma=0.5)

for epoch in range(8):
    loss = train_epoch(cnn, train_loader, opt, loss_fn)
    sched.step()
    print(epoch, round(loss, 4), "test acc:", evaluate(cnn, test_loader))
```

보통 **99.0~99.4%** 에 도달합니다.

## 5. 틀린 샘플 살펴보기

```python
import matplotlib.pyplot as plt

cnn.eval()
x, y = next(iter(test_loader))
with torch.no_grad():
    pred = cnn(x.to(device)).argmax(1).cpu()
wrong = (pred != y).nonzero().flatten()[:8]

fig, axes = plt.subplots(1, len(wrong), figsize=(2 * len(wrong), 2.4))
for ax, i in zip(axes, wrong):
    ax.imshow(x[i, 0], cmap="gray"); ax.axis("off")
    ax.set_title(f"true {y[i].item()}\npred {pred[i].item()}")
plt.show()
```

사람이 봐도 헷갈리는 숫자가 많습니다. 모델의 한계를 이해하는 중요한 단계입니다.

## 6. 과제

- [ ] MLP와 CNN의 **파라미터 수**와 **정확도**를 표로 비교하세요. (`sum(p.numel() for p in m.parameters())`)
- [ ] 학습/테스트 정확도를 epoch마다 기록해 그래프로 그리고, 과적합 징후가 있는지 보세요.
- [ ] 데이터 증강(`RandomAffine`, `RandomRotation`)을 train에만 적용하고 변화를 측정하세요.
- [ ] 학습률을 `1e-1`, `1e-3`, `1e-5`로 바꿔 보고 손실 곡선을 비교하세요.
- [ ] 학습한 모델을 `state_dict`로 저장하고, 새 세션에서 불러와 같은 정확도가 나오는지 확인하세요.
- [ ] (도전) 직접 그린 숫자 사진을 28×28 흑백으로 바꿔 예측해 보세요. 왜 잘 안 되는지(전처리 차이)를 분석하세요.

??? success "힌트"
    - 증강은 **test에는 절대 적용하지 않습니다.** (`transforms.Compose([...RandomAffine(10, translate=(0.1,0.1)), ToTensor(), Normalize(...)])` 를 train용으로만)
    - 직접 그린 숫자는 MNIST처럼 **검은 배경 + 흰 글씨, 중앙 정렬, 정규화**가 되어 있어야 합니다.

## 7. 정리 노트에 적을 것

- MLP vs CNN 정확도·파라미터 차이와 그 이유
- 학습률/배치 크기/에폭이 결과에 미친 영향
- 모델이 틀린 사례의 공통점
