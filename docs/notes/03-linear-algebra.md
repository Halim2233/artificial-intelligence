# 선형대수

!!! abstract "한 줄 요약"
    신경망의 한 층은 결국 **행렬곱 하나 + 덧셈 하나**입니다. 선형대수는 그 계산이 무엇을 의미하는지 알려 주는 언어입니다.

## 왜 필요한가

데이터는 벡터, 모델의 파라미터는 행렬, 학습은 행렬 연산의 반복입니다. 수식을 읽고 shape를 맞추려면 기본기가 필요합니다.

## 1. 벡터

크기와 방향을 가진 숫자 묶음입니다. ML에서는 **"특징(feature)들의 목록"** 으로 봅니다.

$$
\mathbf{x} = \begin{bmatrix} 3 \\ 4 \end{bmatrix}
$$

**노름(크기)**:

$$
\lVert \mathbf{x} \rVert_2 = \sqrt{3^2 + 4^2} = 5
$$

## 2. 내적 (dot product)

$$
\mathbf{a}\cdot\mathbf{b} = \sum_i a_i b_i = \lVert\mathbf{a}\rVert \lVert\mathbf{b}\rVert \cos\theta
$$

- 값이 크면 **방향이 비슷**하고, 0이면 **수직**, 음수면 **반대 방향**입니다.
- 어텐션의 $QK^\top$, 코사인 유사도, 선형 모델의 $\mathbf{w}\cdot\mathbf{x}$가 모두 내적입니다.

```python
import numpy as np

a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

np.dot(a, b)                     # 32
np.linalg.norm(a)                # 3.7416...
cos = a @ b / (np.linalg.norm(a) * np.linalg.norm(b))   # 코사인 유사도
```

## 3. 행렬곱

$(m \times n)$ 행렬과 $(n \times p)$ 행렬을 곱하면 $(m \times p)$입니다.

!!! important "shape 규칙"
    **안쪽 숫자가 같아야 하고, 바깥 숫자가 결과 shape가 됩니다.**
    `(2, 3) @ (3, 4) -> (2, 4)` / `(2, 3) @ (2, 4) -> 에러`

결과의 $(i, j)$ 원소는 **A의 i행과 B의 j열의 내적**입니다.

```python
A = np.array([[1, 2, 3],
              [4, 5, 6]])         # (2, 3)
B = np.array([[1, 0],
              [0, 1],
              [1, 1]])            # (3, 2)

A @ B
# [[ 4  5]
#  [10 11]]
```

`*`는 원소별 곱, `@`는 행렬곱입니다. 헷갈리지 마세요.

## 4. 선형 변환으로 보기

행렬 $W$를 곱하는 것은 **공간을 늘리고, 돌리고, 기울이는 변환**입니다. 3Blue1Brown의 *Essence of Linear Algebra* 영상 1~4편을 꼭 보세요. 이 직관이 있으면 신경망이 "데이터를 보기 좋은 공간으로 옮기는 기계"로 보입니다.

```python
theta = np.pi / 2                         # 90도 회전
R = np.array([[np.cos(theta), -np.sin(theta)],
              [np.sin(theta),  np.cos(theta)]])
R @ np.array([1, 0])                      # 약 [0, 1]
```

## 5. 전치, 단위행렬, 역행렬

| 개념 | 기호 | NumPy |
|---|---|---|
| 전치 | $A^\top$ | `A.T` |
| 단위행렬 | $I$ | `np.eye(n)` |
| 역행렬 | $A^{-1}$ ($AA^{-1}=I$) | `np.linalg.inv(A)` |

실제로는 역행렬을 직접 구하기보다 `np.linalg.solve(A, b)`를 씁니다. 더 빠르고 안정적입니다.

## 6. 고유값과 고유벡터 (맛보기)

$$
A\mathbf{v} = \lambda \mathbf{v}
$$

"변환해도 **방향이 안 바뀌는** 벡터 $\mathbf{v}$와, 그때 늘어나는 비율 $\lambda$"입니다. PCA(차원 축소)에서 쓰입니다.

```python
A = np.array([[2, 0], [0, 3]])
vals, vecs = np.linalg.eig(A)   # vals = [2, 3]
```

## 7. ML과의 연결

선형 회귀와 신경망의 한 층은 같은 모양입니다.

$$
\hat{Y} = X W + \mathbf{b}
$$

| 기호 | shape | 의미 |
|---|---|---|
| $X$ | (샘플 수 $N$, 특징 수 $D$) | 입력 |
| $W$ | ($D$, 출력 수 $K$) | 가중치 |
| $\mathbf{b}$ | ($K$,) | 편향 (브로드캐스팅됨) |
| $\hat{Y}$ | ($N$, $K$) | 예측 |

```python
rng = np.random.default_rng(0)
X = rng.normal(size=(5, 3))     # 샘플 5개, 특징 3개
W = rng.normal(size=(3, 2))
b = np.zeros(2)
Y = X @ W + b
print(Y.shape)                  # (5, 2)
```

## 연습문제

1. $\mathbf{a}=[1,0]$, $\mathbf{b}=[1,1]$ 사이의 각도(도 단위)를 구하세요.
2. `(4, 3) @ (3, 5) @ (5, 2)`의 결과 shape는?
3. $A=\begin{bmatrix}2&1\\1&3\end{bmatrix}$, $\mathbf{b}=[3,5]$일 때 $A\mathbf{x}=\mathbf{b}$를 푸세요.
4. 이미지 한 장(28×28)을 펴서 784차원 벡터로 만들고, 784×10 가중치를 곱하면 shape는?

??? success "정답"
    ```python
    import numpy as np
    a, b = np.array([1, 0]), np.array([1, 1])
    cos = a @ b / (np.linalg.norm(a) * np.linalg.norm(b))
    print(np.degrees(np.arccos(cos)))                  # 1: 45.0

    # 2: (4, 2)

    A = np.array([[2, 1], [1, 3]]); bb = np.array([3, 5])
    print(np.linalg.solve(A, bb))                      # 3: [0.8 1.4]

    # 4: (784,) @ (784, 10) -> (10,)   (배치로 N장이면 (N,784)@(784,10)=(N,10))
    ```

## 참고

- 3Blue1Brown, *Essence of Linear Algebra*
- 다음: [미분과 경사하강법](04-calculus-gradient.md)
