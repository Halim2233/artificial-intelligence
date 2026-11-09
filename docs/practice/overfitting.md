# 실습: 과적합 직접 만들고 해결하기

**목표**: 다항식 차수를 올려 가며 과적합을 눈으로 확인하고, 정규화(Ridge)로 잡습니다.
**관련 노트**: [ML 기초](../notes/06-ml-basics.md)

## 1. 데이터

정답은 사인 곡선이고, 거기에 노이즈를 얹은 적은 수의 데이터입니다.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error

rng = np.random.default_rng(1)
def make(n):
    x = rng.uniform(0, 1, size=(n, 1))
    y = np.sin(2 * np.pi * x[:, 0]) + rng.normal(0, 0.2, size=n)
    return x, y

X_tr, y_tr = make(20)      # 일부러 적게
X_va, y_va = make(200)
```

## 2. 차수별 훈련/검증 오차

```python
degrees = range(1, 16)
tr_err, va_err = [], []
for d in degrees:
    m = make_pipeline(PolynomialFeatures(d), StandardScaler(), LinearRegression())
    m.fit(X_tr, y_tr)
    tr_err.append(mean_squared_error(y_tr, m.predict(X_tr)))
    va_err.append(mean_squared_error(y_va, m.predict(X_va)))

plt.plot(degrees, tr_err, label="train")
plt.plot(degrees, va_err, label="validation")
plt.yscale("log"); plt.xlabel("degree"); plt.legend(); plt.show()
```

**기대되는 모양**: 훈련 오차는 계속 내려가지만 검증 오차는 어느 차수부터 다시 올라갑니다. 그 U자 곡선의 바닥이 적절한 복잡도입니다.

## 3. 적합 곡선 눈으로 보기

```python
xs = np.linspace(0, 1, 300).reshape(-1, 1)
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
for ax, d in zip(axes, [1, 4, 15]):
    m = make_pipeline(PolynomialFeatures(d), StandardScaler(), LinearRegression()).fit(X_tr, y_tr)
    ax.scatter(X_tr, y_tr, s=15)
    ax.plot(xs, m.predict(xs), "r")
    ax.set_ylim(-2, 2); ax.set_title(f"degree={d}")
plt.show()
```

차수 1은 과소적합, 4는 적절, 15는 데이터 점을 지나려고 심하게 출렁입니다(과적합).

## 4. 정규화로 해결

차수 15를 유지한 채 Ridge의 `alpha`(벌점 세기)를 키워 봅니다.

```python
for alpha in [0, 1e-3, 1e-1, 1, 10]:
    model = make_pipeline(PolynomialFeatures(15), StandardScaler(),
                          Ridge(alpha=alpha) if alpha > 0 else LinearRegression())
    model.fit(X_tr, y_tr)
    print(alpha, round(mean_squared_error(y_va, model.predict(X_va)), 4))
```

## 5. 과제

- [ ] 훈련 데이터를 20개에서 200개로 늘리면 차수 15의 과적합이 어떻게 변하나요?
- [ ] `alpha`를 로그 스케일로 촘촘히 훑어 검증 오차가 최소인 값을 찾으세요.
- [ ] 이 과정을 `GridSearchCV`로 자동화하세요.
- [ ] 차수 15 모델의 계수 크기를 `alpha`별로 비교하고, 정규화가 계수에 무슨 일을 하는지 설명하세요.

??? success "힌트"
    - 데이터가 많아지면 같은 모델 용량으로도 과적합이 줄어듭니다.
    - 계수 확인: `model[-1].coef_` (Pipeline의 마지막 단계). `alpha`가 커질수록 계수의 절댓값이 작아집니다.

## 6. 정리 노트에 적을 것

- 편향-분산 트레이드오프를 이 그림으로 설명해 보기
- "데이터를 늘리는 것"과 "정규화"가 각각 언제 효과적인지
