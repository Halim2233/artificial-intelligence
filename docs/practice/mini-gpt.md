# 실습: 미니 GPT를 처음부터 학습시키기

**목표**: [트랜스포머 노트](../notes/transformer.md)의 `TinyGPT`로 셰익스피어 텍스트를 학습해 글을 생성합니다.
**관련 노트**: [트랜스포머](../notes/transformer.md), [LLM의 학습과 추론](../notes/12-llm-training.md)

> GPU(Colab 무료 T4 정도)를 권장합니다. CPU에서는 모델을 줄이고 step을 낮추세요.

## 0. 준비

[트랜스포머 노트 7절](../notes/transformer.md)의 `CausalSelfAttention`, `Block`, `TinyGPT` 코드를 `model.py`로 저장합니다. 그리고 데이터를 받습니다.

```python
import urllib.request
url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
urllib.request.urlretrieve(url, "input.txt")
text = open("input.txt", encoding="utf-8").read()
print(len(text), text[:200])        # 약 110만 글자
```

## 1. 글자 단위 토크나이저

```python
import torch

chars = sorted(set(text))
vocab = len(chars)                           # 약 65
stoi = {c: i for i, c in enumerate(chars)}
itos = {i: c for c, i in stoi.items()}
encode = lambda s: [stoi[c] for c in s]
decode = lambda ids: "".join(itos[i] for i in ids)

data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data, val_data = data[:n], data[n:]
```

## 2. 배치 만들기

입력 `x`와 정답 `y`는 **한 칸 어긋난** 같은 구간입니다.

```python
block_size, batch_size = 128, 64
device = "cuda" if torch.cuda.is_available() else "cpu"

def get_batch(split):
    d = train_data if split == "train" else val_data
    ix = torch.randint(len(d) - block_size - 1, (batch_size,))
    x = torch.stack([d[i:i + block_size] for i in ix])
    y = torch.stack([d[i + 1:i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)

x, y = get_batch("train")
print(decode(x[0][:30].tolist()))
print(decode(y[0][:30].tolist()))     # x를 한 글자 민 것
```

## 3. 학습

```python
import torch.nn.functional as F
from model import TinyGPT

model = TinyGPT(vocab, d_model=192, n_head=6, n_layer=6, max_len=block_size).to(device)
print(sum(p.numel() for p in model.parameters()) / 1e6, "M params")

opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)

@torch.no_grad()
def estimate_loss(iters=20):
    model.eval()
    out = {}
    for split in ("train", "val"):
        ls = torch.zeros(iters)
        for k in range(iters):
            x, y = get_batch(split)
            ls[k] = F.cross_entropy(model(x).view(-1, vocab), y.view(-1)).item()
        out[split] = ls.mean().item()
    model.train()
    return out

for step in range(3001):
    x, y = get_batch("train")
    loss = F.cross_entropy(model(x).view(-1, vocab), y.view(-1))
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)   # 기울기 폭발 방지
    opt.step()
    if step % 300 == 0:
        print(step, estimate_loss())
```

**기대하는 모습**
- 시작 손실 ≈ $\ln(65)\approx4.17$ (무작위 추측)
- step 3000 즈음 train/val 손실 약 1.5 안팎 (설정에 따라 다름)
- val 손실이 오르기 시작하면 [과적합](../notes/06-ml-basics.md)

## 4. 글 생성

```python
@torch.no_grad()
def generate(prompt="\n", n_new=500, temperature=0.8, top_k=40):
    model.eval()
    idx = torch.tensor([encode(prompt)], device=device)
    for _ in range(n_new):
        logits = model(idx[:, -block_size:])[:, -1, :] / temperature
        v, _ = torch.topk(logits, top_k)
        logits[logits < v[:, [-1]]] = float("-inf")
        nxt = torch.multinomial(F.softmax(logits, dim=-1), 1)
        idx = torch.cat([idx, nxt], dim=1)
    return decode(idx[0].tolist())

print(generate("ROMEO:"))
```

학습 초반(step 0)과 후반의 출력을 비교하세요. 처음에는 무작위 글자였다가, 단어 → 철자 → 대본 형식 → 말투 순으로 나아지는 것을 볼 수 있습니다.

## 5. 과제

- [ ] 학습 곡선(train/val 손실)을 그래프로 그리세요.
- [ ] `n_layer`, `d_model`을 바꿔 가며 **val 손실**을 비교하고, 크기와 성능의 관계를 표로 만드세요.
- [ ] 어텐션 층을 제거(MLP만)하거나 위치 임베딩을 제거했을 때 결과가 어떻게 달라지는지 실험하세요.
- [ ] `temperature` 0.3 / 0.8 / 1.5로 생성해 비교하세요.
- [ ] 첫 번째 블록의 어텐션 가중치를 히트맵으로 시각화하세요. (인과 마스크 확인)
- [ ] (도전) 다른 텍스트(좋아하는 소설, 한국어 위키 문서 등)로 학습해 보세요.
- [ ] (도전) BPE 토크나이저(`tiktoken` 또는 `tokenizers`)로 바꿔 글자 단위와 비교하세요.

??? success "힌트"
    - 히트맵: `CausalSelfAttention.forward`에서 `att`를 `self.last_att = att.detach()`로 저장한 뒤 `plt.imshow(model.blocks[0].attn.last_att[0, 0].cpu())`
    - 어텐션 제거 시 val 손실이 눈에 띄게 높아지는 것을 보면, 토큰 간 정보 교환의 역할을 체감할 수 있습니다.
    - 손실이 안 내려가면 학습률을 `1e-3`/`1e-4`로 바꾸고, 먼저 **작은 데이터 한 배치에 과적합**이 되는지 확인하세요. ([디버깅 요령](../notes/07-backprop.md))

## 6. 정리 노트에 적을 것

- 학습 단계별 생성 품질의 변화
- 모델 크기와 val 손실의 관계
- GPT-2 같은 실제 모델과 이 모델의 차이점 (데이터 양, 토크나이저, 규모)
