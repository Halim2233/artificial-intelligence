# 토큰화와 임베딩

!!! abstract "한 줄 요약"
    LLM은 글자가 아니라 **토큰(정수 ID)** 을 읽고, 각 ID를 **임베딩 벡터**로 바꿔 계산합니다. 비슷한 의미의 토큰은 벡터 공간에서 가까이 놓입니다.

선행: [선형대수](03-linear-algebra.md), [PyTorch 기초](08-pytorch.md)

## 파이프라인

```mermaid
flowchart LR
    T["'안녕 AI'"] --> K[토크나이저] --> I["[1832, 4021, ...]"] --> E[임베딩 테이블] --> V["(길이, d_model) 행렬"] --> M[트랜스포머]
```

## 1. 왜 토큰화인가

| 단위 | 장점 | 단점 |
|---|---|---|
| 글자 | 어휘가 작음, OOV 없음 | 시퀀스가 너무 길어짐 |
| 단어 | 의미 단위로 짧음 | 어휘가 폭발, 처음 보는 단어 처리 불가 |
| **서브워드** | 둘의 절충 | (현재 표준) |

서브워드는 자주 나오는 덩어리는 한 토큰으로, 드문 단어는 조각으로 쪼갭니다. 예: `unhappiness` → `un` + `happiness`

## 2. BPE (Byte Pair Encoding)

**가장 자주 붙어 나오는 쌍을 반복해서 합치는** 방식입니다. GPT 계열이 사용합니다.

1. 모든 글자를 개별 토큰으로 시작
2. 가장 빈번한 인접 쌍을 찾아 새 토큰으로 병합
3. 원하는 어휘 크기까지 반복

```python
from collections import Counter

def get_pairs(ids):
    return Counter(zip(ids, ids[1:]))

def merge(ids, pair, new_id):
    out, i = [], 0
    while i < len(ids):
        if i < len(ids) - 1 and (ids[i], ids[i + 1]) == pair:
            out.append(new_id); i += 2
        else:
            out.append(ids[i]); i += 1
    return out

text = "low lower lowest low low newer newest"
ids = list(text.encode("utf-8"))            # 바이트 단위(0~255)로 시작
merges = {}

for step in range(10):
    pairs = get_pairs(ids)
    if not pairs:
        break
    best = pairs.most_common(1)[0][0]
    new_id = 256 + step
    ids = merge(ids, best, new_id)
    merges[best] = new_id
    print(f"merge {best} -> {new_id}, 길이 {len(ids)}")
```

문장 길이가 합칠 때마다 줄어드는 것을 확인하세요. 실제 토크나이저는 이 병합 규칙표를 저장해 두었다가 새 텍스트에 같은 순서로 적용합니다.

## 3. 실제 토크나이저 써보기

```bash
pip install transformers tiktoken
```

```python
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("gpt2")
ids = tok.encode("Hello, tokenization!")
print(ids)
print(tok.convert_ids_to_tokens(ids))     # ['Hello', ',', 'Ġtoken', 'ization', '!']
print(tok.decode(ids))
```

`Ġ`는 "앞에 공백이 있음"을 뜻합니다.

### 한국어는 왜 비쌀까?

```python
for s in ["I love you", "나는 너를 사랑해"]:
    print(s, len(tok.encode(s)))
```

영어 위주로 학습된 토크나이저는 한글을 **바이트 단위로 잘게** 쪼개서 같은 뜻도 토큰이 훨씬 많아집니다. 토큰 수는 **비용·속도·컨텍스트 길이**에 직결됩니다.

!!! note "토큰이 LLM의 이상한 행동을 설명합니다"
    - 글자 수 세기, 철자 뒤집기에 약함 → 모델은 글자가 아니라 **토큰 덩어리**를 봄
    - 숫자 계산에서 실수 → 숫자가 이상한 경계로 쪼개짐
    - 앞뒤 공백 하나로 결과가 달라짐

## 4. 임베딩: ID → 벡터

토큰 ID를 학습 가능한 벡터로 바꿉니다. 사실상 **행렬에서 행 하나를 뽑는 룩업 테이블**입니다.

```python
import torch, torch.nn as nn

vocab_size, d_model = 50_000, 768
emb = nn.Embedding(vocab_size, d_model)
ids = torch.tensor([[15, 2301, 7]])         # (배치=1, 길이=3)
print(emb(ids).shape)                        # (1, 3, 768)
print(vocab_size * d_model)                  # 38,400,000 파라미터
```

**원-핫 벡터에 행렬을 곱하는 것과 수학적으로 같고**, 룩업이 훨씬 효율적입니다. 이 벡터들은 학습하는 동안 다음 토큰 예측에 도움이 되도록 조정됩니다.

## 5. 의미 공간과 코사인 유사도

학습된 임베딩에서는 의미가 비슷한 단어가 가까워집니다. 거리는 보통 코사인 유사도로 잽니다.

$$
\cos(\mathbf{a},\mathbf{b})=\frac{\mathbf{a}\cdot\mathbf{b}}{\lVert\mathbf{a}\rVert\,\lVert\mathbf{b}\rVert}
$$

문장 단위 임베딩을 쓰면 **의미 검색**이 가능합니다. ([RAG](14-rag.md)의 기반)

```python
from sentence_transformers import SentenceTransformer, util

m = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
sents = ["고양이가 소파에서 잔다", "A cat is sleeping on the couch", "주식 시장이 하락했다"]
e = m.encode(sents, convert_to_tensor=True)
print(util.cos_sim(e, e).round(decimals=2))
# 1번과 2번(같은 의미, 다른 언어)이 높고 3번은 낮음
```

## 6. 위치 정보

어텐션은 단어 순서를 모릅니다. "개가 사람을 문다"와 "사람이 개를 문다"를 구분하려면 **위치 정보**를 더해야 합니다. 구현 방식은 [트랜스포머](transformer.md)에서 다룹니다.

## 7. 컨텍스트 윈도우

한 번에 볼 수 있는 최대 토큰 수입니다. 입력과 출력이 모두 포함됩니다. 길이가 늘수록 [어텐션 비용이 제곱으로 증가](10-rnn-attention.md)하므로 이를 줄이는 연구가 활발합니다.

## 연습문제

1. 어휘 크기 32,000, 임베딩 차원 4,096이면 임베딩 테이블의 파라미터 수는?
2. BPE 코드에서 `text = "aaabdaaabac"`로 3번 병합하면 가장 먼저 합쳐지는 쌍은?
3. 글자 단위 모델 대비 서브워드의 장점 두 가지는?
4. 코사인 유사도가 -1, 0, 1일 때 각각 의미는?

??? success "정답"
    1. 32,000 × 4,096 = **131,072,000** (약 1.3억)
    2. 바이트 `a`가 가장 많으니 **(97, 97)** 즉 `aa`가 먼저입니다. (`aaa`에서 `aa`가 가장 빈번)
    3. 시퀀스가 짧아져 계산이 줄고, 자주 쓰는 의미 단위가 하나의 토큰으로 묶여 학습이 쉬워집니다.
    4. 1: 같은 방향(매우 유사), 0: 직교(무관), -1: 반대 방향.

## 참고

- Karpathy, *Let's build the GPT Tokenizer*
- OpenAI `tiktoken`, Hugging Face Tokenizers 문서
- 다음: [트랜스포머](transformer.md)
