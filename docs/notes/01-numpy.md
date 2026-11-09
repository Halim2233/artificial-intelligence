# NumPy 기초

!!! abstract "한 줄 요약"
    NumPy는 **배열 단위로 한꺼번에 계산**하게 해 주는 라이브러리이고, 딥러닝의 모든 계산(텐서)의 뿌리입니다.

## 왜 필요한가

파이썬 `list`로 반복문을 돌리면 느립니다. NumPy는 C로 구현된 연산을 배열 전체에 적용해서 수십~수백 배 빠릅니다. PyTorch의 `Tensor`도 NumPy와 거의 같은 사용법입니다.

## 1. 배열 만들기

```python
import numpy as np

a = np.array([1, 2, 3])               # 1차원
b = np.array([[1, 2, 3], [4, 5, 6]])  # 2차원 (2행 3열)

print(b.shape)   # (2, 3)
print(b.ndim)    # 2
print(b.dtype)   # int64 (환경에 따라 int32)

np.zeros((2, 3))        # 0으로 채움
np.ones((2, 3))         # 1로 채움
np.arange(0, 10, 2)     # [0 2 4 6 8]
np.linspace(0, 1, 5)    # [0. 0.25 0.5 0.75 1.]
np.eye(3)               # 단위행렬
```

!!! note "shape가 가장 중요합니다"
    딥러닝 에러의 절반은 shape 불일치입니다. 코드를 짤 때마다 **"이 배열의 shape가 뭐지?"** 를 습관처럼 확인하세요.

## 2. 인덱싱과 슬라이싱

```python
x = np.array([[10, 11, 12],
              [20, 21, 22],
              [30, 31, 32]])

x[0, 1]       # 11        (0행 1열)
x[1]          # [20 21 22] (1행 전체)
x[:, 2]       # [12 22 32] (2열 전체)
x[0:2, 1:3]   # [[11 12], [21 22]]
x[x > 20]     # [21 22 30 31 32]  (불리언 마스크)
```

## 3. 벡터화 연산

```python
a = np.array([1, 2, 3])
b = np.array([10, 20, 30])

a + b      # [11 22 33]
a * b      # [10 40 90]   (원소별 곱, 행렬곱 아님!)
a ** 2     # [1 4 9]
np.sqrt(a)
```

반복문 대신 이렇게 쓰는 것이 NumPy의 핵심입니다.

## 4. 브로드캐스팅

shape가 달라도 규칙에 맞으면 자동으로 늘려서 계산합니다.

```python
m = np.array([[1, 2, 3],
              [4, 5, 6]])     # (2, 3)
v = np.array([10, 20, 30])    # (3,)

m + v
# [[11 22 33]
#  [14 25 36]]
```

**규칙**: shape를 오른쪽부터 비교해서, 각 차원이 (1) 같거나 (2) 둘 중 하나가 1이면 됩니다.

```text
(2, 3) + (3,)   -> OK   ((3,)은 (1, 3)으로 취급되어 행 방향으로 복사)
(2, 3) + (2,)   -> 에러  (3 vs 2 불일치)
(2, 3) + (2, 1) -> OK   (열 방향으로 복사)
```

## 5. axis와 집계

```python
m = np.array([[1, 2, 3],
              [4, 5, 6]])

m.sum()          # 21
m.sum(axis=0)    # [5 7 9]   (행 방향으로 합쳐서 열별 합)
m.sum(axis=1)    # [ 6 15]   (열 방향으로 합쳐서 행별 합)
m.mean(axis=0)
m.argmax(axis=1) # [2 2]
```

!!! tip "axis 외우는 법"
    `axis=k`는 **"k번째 차원이 사라진다"** 로 기억하세요. (2, 3)에서 `axis=0`이면 결과는 (3,)입니다.

## 6. reshape와 전치

```python
x = np.arange(6)         # [0 1 2 3 4 5]
x.reshape(2, 3)          # (2, 3)
x.reshape(3, -1)         # -1은 자동 계산 -> (3, 2)
x.reshape(2, 3).T        # 전치 -> (3, 2)
```

## 7. 난수

```python
rng = np.random.default_rng(42)   # 시드 고정 -> 재현 가능
rng.random((2, 3))                # 0~1 균등
rng.normal(0, 1, size=(2, 3))     # 표준정규분포
rng.integers(0, 10, size=5)
```

## 연습문제

1. `np.arange(1, 13)`을 (3, 4)로 바꾸고, 각 행의 합을 구하세요.
2. 위 배열에서 짝수만 뽑으세요.
3. (3, 4) 배열의 각 열에서 그 열의 평균을 뺀 배열을 만드세요. (브로드캐스팅)
4. 길이 1,000,000인 배열의 제곱합을 `for` 루프와 NumPy로 각각 구하고 시간을 비교하세요.

??? success "정답"
    ```python
    import numpy as np, time

    x = np.arange(1, 13).reshape(3, 4)
    print(x.sum(axis=1))             # 1번: [10 26 42]
    print(x[x % 2 == 0])             # 2번
    print(x - x.mean(axis=0))        # 3번: (3,4) - (4,) 브로드캐스팅

    a = np.random.default_rng(0).random(1_000_000)   # 4번
    t = time.time(); s1 = sum(v * v for v in a);  print(time.time() - t)
    t = time.time(); s2 = (a ** 2).sum();         print(time.time() - t)
    ```

## 다음에 볼 것

- [x] NumPy 기초
- [ ] [Pandas 기초](02-pandas.md)
