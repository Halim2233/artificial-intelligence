# 확률과 통계

!!! abstract "한 줄 요약"
    ML 모델의 출력은 대개 **확률**이고, 손실 함수(교차 엔트로피)는 **"정답에 얼마나 높은 확률을 줬나"** 를 재는 도구입니다.

## 1. 기본 개념

- **확률변수**: 결과가 확률적으로 정해지는 값 (주사위 눈 $X$)
- **확률분포**: 각 값이 나올 확률 $P(X=x)$, 합은 1
- **기댓값**: 평균적으로 기대되는 값

$$
E[X] = \sum_x x\,P(x), \qquad \mathrm{Var}(X) = E[(X-E[X])^2]
$$

```python
import numpy as np
rng = np.random.default_rng(0)

rolls = rng.integers(1, 7, size=100_000)   # 주사위 10만 번
print(rolls.mean())                        # 약 3.5  (이론값 E[X] = 3.5)
print(rolls.var())                         # 약 2.92
```

**큰 수의 법칙**: 시행이 많아질수록 평균은 기댓값에 가까워집니다.

## 2. 조건부 확률과 베이즈 정리

$$
P(A\mid B)=\frac{P(A\cap B)}{P(B)}, \qquad
P(A\mid B)=\frac{P(B\mid A)\,P(A)}{P(B)}
$$

### 예제: 검사의 함정

- 병에 걸린 사람: 1% ($P(D)=0.01$)
- 걸렸을 때 양성: 99% ($P(+\mid D)=0.99$)
- 안 걸렸는데 양성: 5% ($P(+\mid \neg D)=0.05$)

양성이 나왔을 때 실제로 병에 걸렸을 확률은?

$$
P(D\mid +)=\frac{0.99\times0.01}{0.99\times0.01+0.05\times0.99}\approx 0.167
$$

```python
p_d, p_pos_d, p_pos_nd = 0.01, 0.99, 0.05
p_pos = p_pos_d * p_d + p_pos_nd * (1 - p_d)
print(p_pos_d * p_d / p_pos)    # 0.1667
```

!!! note "직관"
    양성이어도 17%뿐입니다. **병 자체가 드물어서(사전확률이 낮아서)** 오탐이 진짜 환자보다 많기 때문입니다. 정확도만 보면 안 되는 이유와 같습니다. ([ML 기초](06-ml-basics.md)의 정밀도/재현율)

## 3. 정규분포

$$
\mathcal{N}(x;\mu,\sigma^2)=\frac{1}{\sqrt{2\pi\sigma^2}}\exp\!\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)
$$

- 평균 $\mu$ 근처에 몰리고 $\sigma$가 퍼짐 정도를 정합니다.
- $\mu\pm\sigma$ 안에 약 68%, $\mu\pm2\sigma$ 안에 약 95%가 들어갑니다.
- 가중치 초기화, 노이즈, 데이터 표준화 등 곳곳에 등장합니다.

```python
x = rng.normal(loc=170, scale=10, size=100_000)   # 키: 평균 170, 표준편차 10
print(((x > 160) & (x < 180)).mean())             # 약 0.68
```

**표준화**: $z = (x-\mu)/\sigma$ 로 평균 0, 표준편차 1로 맞추면 학습이 안정됩니다.

## 4. 소프트맥스: 점수를 확률로

모델이 내는 점수(logit)를 합이 1인 확률로 바꿉니다.

$$
\mathrm{softmax}(z)_i=\frac{e^{z_i}}{\sum_j e^{z_j}}
$$

```python
def softmax(z):
    z = z - z.max()          # 수치 안정화: 오버플로 방지
    e = np.exp(z)
    return e / e.sum()

print(softmax(np.array([2.0, 1.0, 0.1])))   # [0.659 0.242 0.099]
```

## 5. 교차 엔트로피와 최대우도

정답 클래스가 $y$일 때 모델이 준 확률이 $p_y$이면

$$
L = -\log p_y
$$

- $p_y=1$ 이면 손실 0
- $p_y\to0$ 이면 손실 $\to\infty$

```python
p = softmax(np.array([2.0, 1.0, 0.1]))
print(-np.log(p[0]))   # 정답이 0번일 때: 0.417
print(-np.log(p[2]))   # 정답이 2번일 때: 2.317 (틀린 쪽에 확신 -> 큰 손실)
```

!!! important "연결"
    **교차 엔트로피를 최소화 = 정답 데이터의 우도(likelihood)를 최대화(MLE).** LLM이 다음 토큰을 맞추도록 학습하는 손실이 바로 이것입니다.

## 연습문제

1. 공정한 동전을 10번 던져 앞면이 정확히 5번 나올 확률은? (`math.comb` 사용)
2. 베이즈 예제에서 병의 유병률이 10%라면 $P(D\mid+)$는?
3. `softmax([1, 1, 1])`은? 이유는?
4. 정답 확률이 0.9일 때와 0.1일 때 교차 엔트로피 손실을 각각 구하세요.

??? success "정답"
    ```python
    import math, numpy as np
    print(math.comb(10, 5) / 2**10)                       # 1: 0.2461

    p = 0.10                                              # 2
    print(0.99*p / (0.99*p + 0.05*(1-p)))                 # 0.6875

    # 3: [1/3, 1/3, 1/3]  (모든 점수가 같으니 균등)
    print(-np.log(0.9), -np.log(0.1))                     # 4: 0.105, 2.303
    ```

## 참고

- Khan Academy 확률과 통계
- 다음: [머신러닝 기초](06-ml-basics.md)
