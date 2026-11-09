# CNN (합성곱 신경망)

!!! abstract "한 줄 요약"
    CNN은 **작은 필터를 이미지 전체에 밀면서(가중치 공유)** 가장자리 → 질감 → 사물 순으로 특징을 쌓아 올리는 구조입니다. 이미지처럼 **공간 구조가 있는 데이터**에 맞춤 설계된 모델입니다.

선행: [PyTorch 기초](08-pytorch.md)

## 왜 필요한가

28×28 이미지를 일렬로 펴서 MLP에 넣으면 문제가 두 가지입니다.

1. 픽셀의 **이웃 관계**가 사라집니다.
2. 같은 고양이가 **위치만 달라져도** 완전히 다른 입력이 됩니다.

CNN은 **국소성**(이웃 픽셀끼리 의미가 있다)과 **평행이동 등변성**(어디 있든 같은 필터로 찾는다)을 구조에 내장합니다.

## 1. 합성곱 연산

필터(커널)를 이미지 위에서 한 칸씩 밀며 **원소별 곱의 합**을 구합니다.

$$
y_{i,j}=\sum_{m}\sum_{n} x_{i+m,\,j+n}\;k_{m,n}
$$

```python
import numpy as np

def conv2d(x, k):
    H, W = x.shape
    kh, kw = k.shape
    out = np.zeros((H - kh + 1, W - kw + 1))
    for i in range(out.shape[0]):
        for j in range(out.shape[1]):
            out[i, j] = (x[i:i+kh, j:j+kw] * k).sum()
    return out

img = np.zeros((6, 6)); img[:, 3:] = 1          # 왼쪽 검정, 오른쪽 흰색
edge = np.array([[-1, 0, 1]] * 3)               # 세로 경계 검출 필터
print(conv2d(img, edge))                        # 경계 위치에서만 값이 큼
```

**학습이란 이 필터 값들을 데이터로부터 찾아내는 것**입니다.

## 2. 출력 크기 공식

입력 $W$, 필터 $K$, 패딩 $P$, 스트라이드 $S$ 일 때

$$
W_{\text{out}}=\left\lfloor\frac{W-K+2P}{S}+1\right\rfloor
$$

| 설정 | 28×28, K=3 | 의미 |
|---|---|---|
| P=0, S=1 | 26×26 | 크기가 줄어듦 |
| P=1, S=1 | 28×28 | **same** 패딩, 크기 유지 |
| P=1, S=2 | 14×14 | 절반으로 다운샘플 |

## 3. 채널과 파라미터 수

입력 채널 $C_{in}$, 출력 채널 $C_{out}$, 필터 $K\times K$ 이면

$$
\text{파라미터}=C_{out}\times(C_{in}\times K\times K+1)
$$

예: `Conv2d(3, 16, 3)` → 16×(3×3×3+1) = **448개**. 이미지 크기와 무관하다는 점이 MLP와 크게 다릅니다.

```python
import torch, torch.nn as nn

conv = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
x = torch.randn(8, 3, 32, 32)       # (배치, 채널, 높이, 너비)
print(conv(x).shape)                # (8, 16, 32, 32)
```

PyTorch 이미지 텐서의 순서는 **(N, C, H, W)** 입니다.

## 4. 풀링

작은 영역을 하나의 값으로 요약해서 크기를 줄입니다.

```python
pool = nn.MaxPool2d(2)              # 2x2 최댓값 -> 가로세로 절반
print(pool(torch.randn(8, 16, 32, 32)).shape)   # (8, 16, 16, 16)
```

- 계산량 감소
- 작은 위치 변화에 둔감해짐
- 층이 깊어질수록 **수용 영역(receptive field)** 이 넓어져 큰 구조를 봄

## 5. 전형적인 구조

```mermaid
flowchart LR
    I["입력 1x28x28"] --> C1["Conv+ReLU 16ch"] --> P1[MaxPool] --> C2["Conv+ReLU 32ch"] --> P2[MaxPool] --> F[Flatten] --> FC["Linear -> 10"]
```

```python
class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),   # 14x14
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # 7x7
        )
        self.head = nn.Linear(32 * 7 * 7, 10)

    def forward(self, x):
        x = self.features(x)
        return self.head(x.flatten(1))

print(SmallCNN()(torch.randn(4, 1, 28, 28)).shape)   # (4, 10)
```

**특징 맵은 깊어질수록 공간은 작아지고 채널은 많아집니다.** 이 패턴이 거의 모든 CNN의 공통점입니다.

## 6. 성능을 올리는 도구들

| 기법 | 역할 |
|---|---|
| **BatchNorm** | 층 출력을 정규화해 학습을 빠르고 안정적으로 |
| **Dropout** | 학습 중 일부 뉴런을 무작위로 끔 → 과적합 방지 |
| **데이터 증강** | 뒤집기, 자르기, 색 변형으로 데이터를 늘림 |
| **잔차 연결 (ResNet)** | $y=F(x)+x$. 기울기가 지름길로 흘러 **매우 깊은 망** 학습 가능 |

```python
class ResBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.f = nn.Sequential(
            nn.Conv2d(c, c, 3, padding=1), nn.BatchNorm2d(c), nn.ReLU(),
            nn.Conv2d(c, c, 3, padding=1), nn.BatchNorm2d(c),
        )
    def forward(self, x):
        return torch.relu(self.f(x) + x)     # 잔차 연결
```

!!! note "잔차 연결은 트랜스포머에도 그대로 씁니다"
    [트랜스포머](transformer.md)의 모든 블록이 `x + Sublayer(x)` 구조입니다.

## 7. 전이 학습

이미지넷으로 미리 학습된 모델의 특징 추출기를 가져다 내 데이터에 맞게 마지막 층만 바꿔 학습합니다. **데이터가 적을 때 가장 효과적인 방법**입니다.

```python
from torchvision import models

model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
for p in model.parameters():
    p.requires_grad = False                      # 특징 추출기 고정
model.fc = nn.Linear(model.fc.in_features, 5)    # 내 클래스 수 5로 교체 (새 층만 학습됨)
```

데이터가 충분해지면 일부 층을 풀어서 낮은 학습률로 함께 미세조정합니다.

## 연습문제

1. 입력 32×32, `Conv2d(k=5, padding=0, stride=1)` 출력 크기는?
2. `Conv2d(16, 32, 3)`의 파라미터 수는?
3. 3×3 합성곱 두 층의 수용 영역은? (5×5 한 층과 비교해 장점은?)
4. 28×28 흑백 이미지 CNN에서 `Linear(32*7*7, 10)`의 `7`은 어디서 나온 숫자인가요?

??? success "정답"
    1. (32-5)/1+1 = **28×28**
    2. 32×(16×3×3+1) = **4,640**
    3. **5×5**. 5×5 한 층은 파라미터 25개, 3×3 두 층은 18개로 더 적고, 중간에 비선형이 하나 더 들어가 표현력이 높습니다.
    4. 28 → (풀링) 14 → (풀링) 7. `padding=1`의 3×3 합성곱은 크기를 유지하고 풀링이 두 번 절반으로 줄입니다.

## 참고

- [Dive into Deep Learning · 합성곱 신경망](https://d2l.ai/)
- Stanford CS231n 강의 노트
- 다음: [RNN과 어텐션의 등장](10-rnn-attention.md)
