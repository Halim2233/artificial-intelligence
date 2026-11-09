# 머신러닝 기초

!!! abstract "한 줄 요약"
    ML은 **데이터에서 규칙을 학습**하는 일이고, 핵심 질문은 항상 하나입니다. **"처음 보는 데이터에서도 잘 맞는가?"** (일반화)

## 1. 학습의 종류

| 종류 | 입력 | 예시 |
|---|---|---|
| 지도학습 | 데이터 + 정답 | 스팸 분류, 집값 예측 |
| 비지도학습 | 데이터만 | 군집화, 차원 축소 |
| 자기지도학습 | 데이터에서 정답을 스스로 만듦 | **LLM의 다음 단어 예측** |
| 강화학습 | 행동 + 보상 | 게임, RLHF |

지도학습은 정답이 숫자면 **회귀**, 범주면 **분류**입니다.

## 2. 학습 과정 한눈에

```mermaid
flowchart LR
    A[데이터] --> B[모델 예측]
    B --> C[손실 계산]
    C --> D[기울기 계산]
    D --> E[파라미터 갱신]
    E --> B
```

[경사하강법](04-calculus-gradient.md)을 반복하는 것이 학습입니다.

## 3. 데이터 분할

| 세트 | 용도 | 비율(예) |
|---|---|---|
| train | 파라미터 학습 | 60~80% |
| validation | 하이퍼파라미터 조정, 모델 선택 | 10~20% |
| test | **마지막에 딱 한 번** 최종 성능 확인 | 10~20% |

!!! danger "데이터 누수 (data leakage)"
    test 데이터 정보가 학습에 새어 들어가면 성능이 거짓으로 좋아 보입니다.
    예: 전체 데이터로 평균을 구해 표준화한 뒤 분할하기. **분할을 먼저, 전처리는 train 기준으로** 해야 합니다.

## 4. 과적합과 과소적합

| | 훈련 성능 | 검증 성능 | 원인 |
|---|---|---|---|
| 과소적합 | 낮음 | 낮음 | 모델이 너무 단순 |
| 적절 | 높음 | 높음 | |
| **과적합** | 매우 높음 | **낮음** | 훈련 데이터를 외워버림 |

과적합 대응책:

1. 데이터를 더 모은다.
2. 모델을 단순하게 한다.
3. **정규화** (L2: 가중치가 커지지 않게 벌점)
4. 조기 종료 (검증 손실이 오르기 시작하면 멈춤)
5. 드롭아웃, 데이터 증강 (딥러닝)

$$
L_{\text{total}} = L_{\text{data}} + \lambda \lVert w\rVert_2^2
$$

### 편향-분산 트레이드오프

- **편향(bias)**: 모델이 단순해서 생기는 체계적 오차
- **분산(variance)**: 데이터가 조금만 바뀌어도 모델이 크게 변하는 정도
- 모델이 복잡해질수록 편향↓ 분산↑

## 5. 평가 지표

### 분류: 혼동행렬

|  | 예측 양성 | 예측 음성 |
|---|---|---|
| 실제 양성 | TP | FN |
| 실제 음성 | FP | TN |

$$
\text{정밀도}=\frac{TP}{TP+FP},\quad
\text{재현율}=\frac{TP}{TP+FN},\quad
F_1=\frac{2PR}{P+R}
$$

- **정밀도**: 양성이라 한 것 중 진짜 비율 (오탐을 피하고 싶을 때)
- **재현율**: 진짜 양성 중 잡아낸 비율 (놓치면 안 될 때, 예: 암 진단)
- **정확도의 함정**: 양성이 1%인 데이터에서 "전부 음성"이라 해도 정확도 99%입니다.

### 회귀

$$
\mathrm{MSE}=\frac1N\sum(y-\hat y)^2,\qquad
\mathrm{MAE}=\frac1N\sum|y-\hat y|
$$

## 6. scikit-learn 기본 틀

모든 모델이 같은 인터페이스를 씁니다: `fit` -> `predict` -> 평가.

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

X, y = load_iris(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler().fit(X_tr)        # train으로만 fit!
X_tr, X_te = scaler.transform(X_tr), scaler.transform(X_te)

model = LogisticRegression(max_iter=200).fit(X_tr, y_tr)
pred = model.predict(X_te)

print(accuracy_score(y_te, pred))
print(classification_report(y_te, pred))
```

### 교차 검증

데이터가 적을 때 분할 운에 휘둘리지 않도록 여러 번 나눠서 평균을 냅니다.

```python
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline

pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=200))
scores = cross_val_score(pipe, X, y, cv=5)
print(scores.mean(), scores.std())
```

`Pipeline`을 쓰면 각 fold마다 전처리가 train에만 fit되어 **누수가 자동으로 방지**됩니다.

## 연습문제

1. 양성 10명, 음성 990명 데이터에서 모델이 양성 5명을 맞히고(TP=5), 음성 10명을 양성으로 잘못 예측(FP=10)했습니다. 정확도, 정밀도, 재현율은?
2. 훈련 정확도 99%, 검증 정확도 70%입니다. 어떤 문제이고 무엇을 시도하겠습니까?
3. 왜 `StandardScaler`를 분할 전에 전체 데이터에 fit하면 안 되나요?

??? success "정답"
    1. TP=5, FN=5, FP=10, TN=980 → 정확도 = (5+980)/1000 = **98.5%**, 정밀도 = 5/15 = **33.3%**, 재현율 = 5/10 = **50%**. 정확도는 높지만 실제로는 쓸모가 별로입니다.
    2. **과적합**. 정규화 강화, 모델 단순화, 데이터 추가, 조기 종료 등.
    3. test 데이터의 평균·표준편차 정보가 학습 단계에 새어 들어가는 **데이터 누수**이기 때문입니다.

## 참고

- Andrew Ng, *Machine Learning Specialization*
- 실습: [과적합 직접 만들어 보기](../practice/overfitting.md)
