# 실습: Titanic 생존 예측

**목표**: 실제 데이터를 받아 탐색, 결측치 처리, 인코딩, 모델 학습, 평가까지 한 바퀴 돕니다.
**관련 노트**: [Pandas](../notes/02-pandas.md), [ML 기초](../notes/06-ml-basics.md)

```bash
pip install pandas seaborn scikit-learn matplotlib
```

## 1. 데이터 불러오기

```python
import pandas as pd
import seaborn as sns

df = sns.load_dataset("titanic")     # 인터넷 연결 필요
df.info()
df.head()
```

주요 열: `survived`(정답), `pclass`(객실 등급), `sex`, `age`, `sibsp`, `parch`, `fare`, `embarked`.

## 2. 탐색 (EDA)

```python
print(df["survived"].mean())                       # 전체 생존율 약 0.38
print(df.groupby("sex")["survived"].mean())        # 성별 생존율
print(df.groupby("pclass")["survived"].mean())     # 등급별 생존율
print(df.isna().sum())                             # age, deck 등 결측
```

먼저 가설을 세워 보세요. "여성과 1등석이 더 많이 살았을 것이다." 숫자로 맞는지 확인합니다.

## 3. 전처리

```python
data = df[["survived", "pclass", "sex", "age", "sibsp", "parch", "fare", "embarked"]].copy()

data["sex"] = (data["sex"] == "female").astype(int)
data["embarked"] = data["embarked"].fillna(data["embarked"].mode()[0])
data = pd.get_dummies(data, columns=["embarked"], drop_first=True)
```

`age`는 나중에 파이프라인 안에서 채웁니다. (누수 방지)

## 4. 모델 학습

```python
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

X = data.drop(columns="survived")
y = data["survived"]
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

logit = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                      LogisticRegression(max_iter=500))
forest = make_pipeline(SimpleImputer(strategy="median"),
                       RandomForestClassifier(n_estimators=200, random_state=42))

for name, m in [("logistic", logit), ("forest", forest)]:
    cv = cross_val_score(m, X_tr, y_tr, cv=5).mean()
    print(name, "CV accuracy:", round(cv, 3))

best = forest.fit(X_tr, y_tr)
print(classification_report(y_te, best.predict(X_te)))
```

일반적으로 정확도는 0.78~0.83 정도가 나옵니다. (분할과 설정에 따라 달라집니다.)

## 5. 과제

- [ ] 로지스틱 회귀의 계수(`coef_`)를 보고 어떤 특징이 생존에 가장 크게 영향을 주는지 해석하세요.
- [ ] `family_size = sibsp + parch + 1`, `is_alone` 같은 **새 특징**을 만들고 성능이 변하는지 보세요.
- [ ] 정확도 대신 **정밀도·재현율**을 보고, 어느 쪽이 약한지 확인하세요.
- [ ] 랜덤 포레스트의 `feature_importances_`를 막대그래프로 그리세요.
- [ ] **test 세트는 마지막에 한 번만** 사용했는지 점검하세요.

## 6. 정리 노트에 적을 것

- 어떤 특징이 중요했고, 그게 직관과 맞았는지
- 결측치를 어떻게 처리했고 왜 그렇게 했는지
- 교차 검증 점수와 test 점수의 차이
