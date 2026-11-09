# Pandas 기초

!!! abstract "한 줄 요약"
    Pandas는 **표 형태 데이터(DataFrame)** 를 읽고, 거르고, 합치고, 요약하는 도구입니다. 모델 학습 전 데이터 준비의 대부분이 여기서 이루어집니다.

## 1. 기본 구조

- `Series`: 1차원 (열 하나)
- `DataFrame`: 2차원 (표)

```python
import pandas as pd

df = pd.DataFrame({
    "name": ["철수", "영희", "민수", "지아"],
    "age": [23, 31, None, 27],
    "city": ["서울", "부산", "서울", "대구"],
    "score": [88, 92, 75, 60],
})
```

## 2. 데이터 훑어보기

데이터를 받으면 **가장 먼저** 이 다섯 줄을 실행합니다.

```python
df.head()          # 앞 5행
df.shape           # (행, 열)
df.info()          # 열별 타입, 결측치 개수
df.describe()      # 수치형 요약 통계
df["city"].value_counts()   # 범주별 개수
```

파일 읽기/쓰기:

```python
df = pd.read_csv("data.csv")
df.to_csv("out.csv", index=False)
```

## 3. 선택하기

```python
df["age"]                    # 열 하나 (Series)
df[["name", "score"]]        # 열 여러 개 (DataFrame)

df.loc[0, "name"]            # 라벨 기준: 0번 행의 name
df.iloc[0, 1]                # 위치 기준: 0행 1열
df.loc[:, "name":"age"]      # 라벨 슬라이스는 끝 포함!
```

!!! warning "loc vs iloc"
    `loc`은 **라벨**, `iloc`은 **정수 위치**입니다. 슬라이스 끝은 `loc`은 포함, `iloc`은 제외입니다.

## 4. 조건 필터링

```python
df[df["score"] >= 80]
df[(df["city"] == "서울") & (df["score"] > 80)]   # and는 &, or은 |
df[df["city"].isin(["서울", "부산"])]
```

!!! danger "괄호 필수"
    조건마다 `( )`로 감싸야 합니다. `df["a"] > 1 & df["b"] < 2`는 에러가 납니다.

## 5. 결측치 처리

```python
df.isna().sum()                       # 열별 결측 개수
df["age"].fillna(df["age"].median())  # 중앙값으로 채우기
df.dropna(subset=["age"])             # 결측 행 제거
```

채울지 버릴지는 **데이터가 얼마나 비었는지, 그 열이 얼마나 중요한지**로 판단합니다.

## 6. 새 열 만들기

```python
df["passed"] = df["score"] >= 70
df["grade"] = pd.cut(df["score"], bins=[0, 70, 90, 100],
                     labels=["C", "B", "A"], right=True)
df["city_code"] = df["city"].map({"서울": 0, "부산": 1, "대구": 2})
```

## 7. 그룹 집계

```python
df.groupby("city")["score"].mean()
df.groupby("city").agg(avg=("score", "mean"), n=("name", "count"))
```

"~별 ~의 평균/개수"라는 질문이 나오면 `groupby`입니다.

## 8. 합치기

```python
extra = pd.DataFrame({"city": ["서울", "부산"], "region": ["수도권", "영남"]})
df.merge(extra, on="city", how="left")     # SQL의 LEFT JOIN
pd.concat([df1, df2], axis=0)              # 위아래로 이어 붙이기
```

## 9. 시각화 맛보기

```python
import matplotlib.pyplot as plt

df["score"].plot(kind="hist", bins=5)
df.groupby("city")["score"].mean().plot(kind="bar")
plt.show()
```

## 연습문제

위 `df`를 사용하세요.

1. 나이가 비어 있는 행은 몇 개인가요?
2. 결측 나이를 평균으로 채운 새 열 `age_filled`를 만드세요.
3. 도시별 평균 점수를 높은 순으로 정렬해 출력하세요.
4. 서울에 살면서 점수가 80 이상인 사람의 이름만 리스트로 뽑으세요.

??? success "정답"
    ```python
    df["age"].isna().sum()                                    # 1: 1개
    df["age_filled"] = df["age"].fillna(df["age"].mean())     # 2
    df.groupby("city")["score"].mean().sort_values(ascending=False)   # 3
    df[(df["city"] == "서울") & (df["score"] >= 80)]["name"].tolist() # 4: ['철수']
    ```

## 다음에 볼 것

- [x] Pandas 기초
- [ ] [Titanic 실습](../practice/titanic.md)에서 실제 데이터에 적용
- [ ] [선형대수](03-linear-algebra.md)
