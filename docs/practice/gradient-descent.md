# 실습: 경사하강법으로 선형 회귀

**목표**: scikit-learn 없이 NumPy만으로 선형 회귀를 학습시키고, 닫힌 해와 결과를 비교합니다.
**관련 노트**: [선형대수](../notes/03-linear-algebra.md), [미분과 경사하강법](../notes/04-calculus-gradient.md)

## 1. 가짜 데이터 만들기

정답을 알고 있는 데이터를 만들면 모델이 제대로 학습했는지 확인할 수 있습니다.

```python
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(0)
N = 200
X = rng.uniform(-3, 3, size=(N, 1))
y = 2.5 * X[:, 0] + 1.0 + rng.normal(0, 0.5, size=N)   # 정답: w=2.5, b=1.0

plt.scatter(X, y, s=10); plt.show()
```

## 2. 직접 구현

$$
\hat y = Xw + b,\quad
L=\frac1N\sum(\hat y - y)^2,\quad
\nabla_w L=\frac2N X^\top(\hat y-y),\quad
\nabla_b L=\frac2N\sum(\hat y-y)
$$

```python
def train(X, y, lr=0.05, epochs=200):
    n, d = X.shape
    w = np.zeros(d)
    b = 0.0
    history = []
    for _ in range(epochs):
        y_hat = X @ w + b
        err = y_hat - y
        loss = (err ** 2).mean()

        grad_w = 2 / n * X.T @ err
        grad_b = 2 / n * err.sum()

        w -= lr * grad_w
        b -= lr * grad_b
        history.append(loss)
    return w, b, history

w, b, history = train(X, y)
print("w =", w, " b =", b)       # 약 [2.5], 1.0

plt.plot(history); plt.xlabel("epoch"); plt.ylabel("loss"); plt.show()
```

## 3. 닫힌 해와 비교

정규방정식 $w=(X^\top X)^{-1}X^\top y$ 로 한 번에 구할 수도 있습니다.

```python
Xb = np.hstack([X, np.ones((N, 1))])          # 편향용 1열 추가
theta = np.linalg.solve(Xb.T @ Xb, Xb.T @ y)
print(theta)                                   # [w, b] 가 위 결과와 거의 같아야 함
```

## 4. 과제

- [ ] 학습률을 `0.001`, `0.05`, `0.5`, `1.1`로 바꿔 손실 곡선 4개를 한 그림에 그리세요. 어떤 일이 생기나요?
- [ ] 특징을 3개로 늘려 보세요. (`y = 1.5*x1 - 2*x2 + 0.5*x3 + 4`) 학습된 `w`가 맞는지 확인하세요.
- [ ] 특징의 크기가 크게 다를 때(예: x1은 0~1, x2는 0~1000) 학습이 어떻게 되는지 보고, **표준화** 후 다시 비교하세요.
- [ ] 수치 미분으로 `grad_w`가 맞는지 검증하는 gradient check를 작성하세요.

??? success "힌트 / 예상 결과"
    - `lr=0.001`은 200번으로는 부족하게 느리고, `lr=1.1`은 손실이 발산합니다. (이 데이터에서는 안정 범위의 상한이 대략 1 안팎입니다.)
    - 스케일이 큰 특징이 있으면 안정적인 학습률이 그 특징 때문에 아주 작아져서 다른 특징이 거의 학습되지 않습니다. 표준화하면 해결됩니다.
    - gradient check: `(L(w+h) - L(w-h)) / (2h)`를 각 원소마다 계산해서 `grad_w`와 `np.allclose`로 비교합니다.

## 5. 정리 노트에 적을 것

- 학습률이 학습에 미치는 영향
- 스케일링이 왜 필요한지
- 손실 곡선을 어떻게 읽는지
