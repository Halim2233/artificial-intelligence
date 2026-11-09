# 미분과 경사하강법

!!! abstract "한 줄 요약"
    학습이란 **손실(loss)이 줄어드는 방향으로 파라미터를 조금씩 움직이는 것**이고, 그 방향을 알려 주는 것이 미분(기울기)입니다.

## 왜 필요한가

모델은 파라미터 $w$를 가지고, 손실 $L(w)$는 "얼마나 틀렸나"를 나타냅니다. 우리는 $L$을 가장 작게 만드는 $w$를 찾고 싶습니다. 수백억 개의 $w$를 한 번에 찾는 방법이 바로 경사하강법입니다.

## 1. 미분 = 순간 변화율

$$
f'(x) = \lim_{h\to 0}\frac{f(x+h)-f(x)}{h}
$$

- $f'(x) > 0$: $x$를 키우면 $f$가 커집니다.
- $f'(x) < 0$: $x$를 키우면 $f$가 작아집니다.
- $f'(x) = 0$: 평평한 곳(최솟값 후보)입니다.

자주 쓰는 공식:

| $f(x)$ | $f'(x)$ |
|---|---|
| $x^n$ | $n x^{n-1}$ |
| $e^x$ | $e^x$ |
| $\ln x$ | $1/x$ |
| $\sigma(x)=\frac{1}{1+e^{-x}}$ | $\sigma(x)(1-\sigma(x))$ |

## 2. 편미분과 기울기(gradient)

변수가 여러 개이면 하나씩 따로 미분(편미분)합니다. 이를 모은 벡터가 **기울기**입니다.

$$
\nabla L = \left[\frac{\partial L}{\partial w_1}, \frac{\partial L}{\partial w_2}, \dots\right]
$$

!!! important "핵심"
    기울기는 **손실이 가장 가파르게 커지는 방향**입니다. 그러니 **반대 방향**으로 가면 손실이 줄어듭니다.

## 3. 경사하강법

$$
w \leftarrow w - \eta \,\nabla L(w)
$$

$\eta$는 학습률(learning rate)입니다.

```python
import numpy as np

# L(w) = (w - 3)^2  -> 최솟값은 w = 3
def grad(w):
    return 2 * (w - 3)

w = 10.0
lr = 0.1
for step in range(30):
    w = w - lr * grad(w)
    if step % 5 == 0:
        print(step, round(w, 4))
# w가 3에 수렴
```

### 학습률의 영향

| 학습률 | 현상 |
|---|---|
| 너무 작음 | 수렴이 매우 느림 |
| 적당함 | 안정적으로 수렴 |
| 너무 큼 | 진동하거나 **발산** |

위 코드에서 `lr = 1.1`로 바꿔 보면 발산하는 것을 직접 볼 수 있습니다.

## 4. 연쇄법칙 (chain rule)

합성함수의 미분은 **각 단계의 미분을 곱합니다.**

$$
\frac{dL}{dx} = \frac{dL}{dy}\cdot\frac{dy}{dx}
$$

예: $y = x^2$, $L = 3y + 1$ 이면 $\frac{dL}{dx} = 3 \cdot 2x = 6x$.

**역전파(backpropagation)는 이 연쇄법칙을 층마다 뒤에서부터 반복 적용하는 것**일 뿐입니다. 이것만 이해하면 딥러닝의 가장 큰 산을 넘은 것입니다.

## 5. 수치 미분으로 검증하기

직접 구한 기울기가 맞는지 확인하는 표준 방법입니다. (gradient check)

```python
def numerical_grad(f, w, h=1e-5):
    return (f(w + h) - f(w - h)) / (2 * h)

f = lambda w: (w - 3) ** 2
print(numerical_grad(f, 10.0))   # 약 14.0
print(2 * (10.0 - 3))            # 14.0
```

## 6. 선형 회귀의 기울기

데이터 $(X, y)$, 예측 $\hat{y} = Xw$, 손실 $L = \frac{1}{N}\lVert Xw - y\rVert^2$ 이면

$$
\nabla_w L = \frac{2}{N} X^\top (Xw - y)
$$

이 식은 [실습: 경사하강법으로 선형 회귀](../practice/gradient-descent.md)에서 직접 구현합니다.

## 연습문제

1. $f(x) = x^3 - 3x$의 $f'(x)$를 구하고, $f'(x)=0$인 점을 찾으세요.
2. $L(w) = (w-3)^2$에서 $w=10$, $\eta=0.1$일 때 한 번 업데이트한 $w$는?
3. $y = \sigma(2x)$일 때 $\frac{dy}{dx}$를 연쇄법칙으로 쓰세요.
4. 학습률 $\eta=1.1$이면 왜 $(w-3)^2$에서 발산하나요?

??? success "정답"
    1. $f'(x)=3x^2-3=0 \Rightarrow x=\pm 1$
    2. $w \leftarrow 10 - 0.1 \times 14 = 8.6$
    3. $\frac{dy}{dx} = \sigma(2x)\,(1-\sigma(2x)) \cdot 2$
    4. 업데이트가 $w-3 \leftarrow (1-2\eta)(w-3)$이고, $\eta=1.1$이면 $|1-2.2|=1.2>1$이라 오차가 매번 1.2배로 커집니다. 수렴 조건은 $|1-2\eta|<1$, 즉 $0<\eta<1$입니다.

## 참고

- 3Blue1Brown, *Essence of Calculus*, *Neural Networks* (4편: Backpropagation)
- 다음: [확률과 통계](05-probability.md)
