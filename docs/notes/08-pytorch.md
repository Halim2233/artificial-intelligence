# PyTorch 기초

!!! abstract "한 줄 요약"
    PyTorch = **GPU에서 도는 NumPy + 자동미분 + 신경망 부품**. 학습 루프는 항상 같은 5줄이고, 외우면 모든 모델에 적용됩니다.

선행: [NumPy](01-numpy.md), [신경망과 역전파](07-backprop.md)

```bash
pip install torch torchvision
```

## 1. 텐서

NumPy 배열과 거의 같고, GPU로 보낼 수 있습니다.

```python
import torch

x = torch.tensor([[1., 2.], [3., 4.]])
x.shape, x.dtype, x.device          # (2, 2), float32, cpu

torch.zeros(2, 3); torch.randn(2, 3); torch.arange(6).reshape(2, 3)

x @ x                                # 행렬곱
x * x                                # 원소별
x.sum(dim=0)                         # NumPy의 axis -> PyTorch의 dim
x.view(4)                            # reshape (메모리 연속일 때)

# NumPy와 변환
import numpy as np
t = torch.from_numpy(np.ones(3))
n = t.numpy()
```

### 장치(device)

```python
device = "cuda" if torch.cuda.is_available() else "cpu"
x = x.to(device)
```

!!! warning "자주 나는 에러"
    모델과 데이터가 **서로 다른 device**에 있으면 에러가 납니다. 모델과 모든 입력을 `.to(device)` 하세요.

## 2. 자동미분 (autograd)

```python
w = torch.tensor(3.0, requires_grad=True)
L = (w * 2 - 4) ** 2         # L = (2w-4)^2
L.backward()
print(w.grad)                # dL/dw = 2(2w-4)*2 = 8
```

- `requires_grad=True`: 이 텐서의 기울기를 추적
- `.backward()`: 연쇄법칙으로 기울기를 계산해 `.grad`에 **누적**
- 추론 때는 기울기가 필요 없으니 `with torch.no_grad():`

!!! danger "기울기는 누적됩니다"
    매 스텝 `optimizer.zero_grad()`로 비우지 않으면 이전 스텝의 기울기가 더해집니다. [micrograd의 `+=`](07-backprop.md)와 같은 이유입니다.

## 3. 모델 정의: nn.Module

```python
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, d_in, d_hidden, d_out):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_hidden),
            nn.ReLU(),
            nn.Linear(d_hidden, d_out),
        )

    def forward(self, x):
        return self.net(x)

model = MLP(784, 128, 10).to(device)
print(sum(p.numel() for p in model.parameters()))   # 파라미터 수: 101,770
```

`__init__`에서 부품을 만들고 `forward`에서 연결한다. 이 패턴이 전부입니다.

## 4. 학습 루프 (5줄 공식)

```python
loss_fn = nn.CrossEntropyLoss()
opt = torch.optim.Adam(model.parameters(), lr=1e-3)

for x, y in loader:
    x, y = x.to(device), y.to(device)
    logits = model(x)             # 1. 순전파
    loss = loss_fn(logits, y)     # 2. 손실
    opt.zero_grad()               # 3. 기울기 초기화
    loss.backward()               # 4. 역전파
    opt.step()                    # 5. 파라미터 갱신
```

!!! note "CrossEntropyLoss"
    입력은 **softmax 전의 logits**이고 정답은 **클래스 번호(정수)** 입니다. 내부에서 softmax + log + NLL을 합쳐 계산합니다. 모델 끝에 softmax를 또 붙이지 마세요.

## 5. 데이터 파이프라인

```python
from torch.utils.data import Dataset, DataLoader

class MyDataset(Dataset):
    def __init__(self, X, y):
        self.X, self.y = X, y
    def __len__(self):
        return len(self.X)
    def __getitem__(self, i):
        return self.X[i], self.y[i]

loader = DataLoader(MyDataset(X, y), batch_size=64, shuffle=True)
```

**미니배치**를 쓰는 이유: 전체 데이터로 한 번에 계산하면 메모리가 부족하고, 조금씩 자주 갱신하는 편이 빠르게 수렴하며 노이즈가 일반화에도 도움이 됩니다.

## 6. 평가 모드

```python
model.eval()                       # 드롭아웃/배치정규화를 평가용으로
with torch.no_grad():
    correct = 0
    for x, y in test_loader:
        x, y = x.to(device), y.to(device)
        correct += (model(x).argmax(dim=1) == y).sum().item()
print(correct / len(test_loader.dataset))
model.train()                      # 다시 학습 모드로
```

`model.eval()`을 빼먹으면 드롭아웃이 켜진 채라 정확도가 흔들립니다. 단골 실수입니다.

## 7. 저장과 불러오기

```python
torch.save(model.state_dict(), "model.pt")

model = MLP(784, 128, 10)
model.load_state_dict(torch.load("model.pt"))
model.eval()
```

## 8. 옵티마이저 선택

| 옵티마이저 | 특징 |
|---|---|
| SGD (+momentum) | 단순, 튜닝 잘하면 강력 |
| **Adam** | 기본값. 학습률 `1e-3` 근처에서 시작 |
| **AdamW** | Adam + 올바른 weight decay. **트랜스포머/LLM 표준** |

## 연습문제

1. `nn.Linear(10, 5)`의 파라미터 수는?
2. `loss.backward()` 전에 `opt.zero_grad()`를 빼면 어떻게 되나요?
3. 정확도 계산 시 `with torch.no_grad()`를 쓰는 이유는?
4. 입력 (32, 3, 28, 28) 배치를 `nn.Linear(784, 10)`에 넣으려면 어떻게 해야 하나요?

??? success "정답"
    1. 가중치 10×5 + 편향 5 = **55**
    2. 기울기가 스텝마다 누적되어 사실상 매우 큰 배치/잘못된 방향으로 갱신됩니다. 학습이 불안정해집니다.
    3. 계산 그래프를 만들지 않아 **메모리와 시간을 절약**합니다.
    4. `x.view(32, -1)`로 (32, 2352)로 펴야 하는데, 이는 784가 아니므로 `Linear(2352, 10)`이 필요합니다. 흑백 1채널이라면 (32,1,28,28) → `view(32, -1)` = (32, 784).

## 참고

- [PyTorch 공식 튜토리얼 · Learn the Basics](https://pytorch.org/tutorials/beginner/basics/intro.html)
- 실습: [MNIST 분류기](../practice/mnist-pytorch.md)
- 다음: [CNN](09-cnn.md)
