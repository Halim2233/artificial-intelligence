# 실습: Hugging Face로 LLM 만져보기

**목표**: 사전학습된 모델을 불러와 토큰, 확률, 샘플링을 눈으로 확인합니다.
**관련 노트**: [토큰화와 임베딩](../notes/11-tokenization-embedding.md), [LLM의 학습과 추론](../notes/12-llm-training.md)

```bash
pip install transformers torch accelerate
```

GPU가 없어도 `gpt2`(1.2억 파라미터)는 CPU에서 충분히 돕니다.

## 1. 한 줄로 생성해 보기

```python
from transformers import pipeline

gen = pipeline("text-generation", model="gpt2")
out = gen("Artificial intelligence is", max_new_tokens=40, do_sample=True)
print(out[0]["generated_text"])
```

여러 번 실행해서 **매번 결과가 달라지는지** 확인하세요. (샘플링)

## 2. 토크나이저와 모델을 따로 쓰기

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

tok = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2").eval()

prompt = "The capital of France is"
inputs = tok(prompt, return_tensors="pt")
print(inputs["input_ids"], tok.convert_ids_to_tokens(inputs["input_ids"][0]))
```

## 3. 다음 토큰 확률 직접 보기

모델이 실제로 하는 일은 이것뿐입니다.

```python
with torch.no_grad():
    logits = model(**inputs).logits          # (1, T, vocab)

next_logits = logits[0, -1]                  # 마지막 위치 = 다음 토큰 예측
probs = torch.softmax(next_logits, dim=-1)

top = torch.topk(probs, 10)
for p, i in zip(top.values, top.indices):
    print(f"{tok.decode(i)!r:>12}  {p.item():.3f}")
```

`' Paris'`가 상위에 있는지, 확률이 얼마나 되는지 보세요. 다른 프롬프트(`"2 + 2 ="`, `"Once upon a"`)로도 분포가 어떻게 달라지는지 비교해 보세요.

## 4. 직접 만든 생성 루프

`model.generate`가 내부에서 하는 일을 손으로 구현합니다. (greedy)

```python
ids = inputs["input_ids"]
for _ in range(15):
    with torch.no_grad():
        logits = model(ids).logits[:, -1, :]
    next_id = logits.argmax(dim=-1, keepdim=True)
    ids = torch.cat([ids, next_id], dim=1)
print(tok.decode(ids[0]))
```

## 5. 샘플링 파라미터 실험

```python
def run(**kw):
    out = model.generate(**inputs, max_new_tokens=30, pad_token_id=tok.eos_token_id, **kw)
    return tok.decode(out[0], skip_special_tokens=True)

print("greedy      :", run(do_sample=False))
print("temp 0.3    :", run(do_sample=True, temperature=0.3))
print("temp 1.5    :", run(do_sample=True, temperature=1.5))
print("top_k=5     :", run(do_sample=True, top_k=5))
print("top_p=0.9   :", run(do_sample=True, top_p=0.9))
print("repetition  :", run(do_sample=False, repetition_penalty=1.3))
```

greedy가 같은 말을 반복하는지, 온도가 높으면 어떻게 무너지는지 관찰하세요.

## 6. 같은 말, 다른 토큰 수

```python
for s in ["Hello world", "안녕하세요 세계", "🙂🙂🙂"]:
    print(s, len(tok.encode(s)), tok.convert_ids_to_tokens(tok.encode(s))[:8])
```

## 7. 과제

- [ ] 프롬프트 5개로 3번의 결과 **top-5 확률**을 표로 정리하세요.
- [ ] 위 4번의 직접 만든 루프에 **temperature + top-k** 를 구현해 `generate`와 비슷한 결과를 내세요. ([샘플 함수](../notes/12-llm-training.md))
- [ ] `distilgpt2`와 `gpt2-medium`을 비교하세요. 크기가 커지면 문장 품질과 속도는 어떻게 바뀌나요?
- [ ] 퍼플렉서티를 계산해 보세요: 영어 문장과 뒤섞은 문장의 값을 비교하세요.
- [ ] (도전) 같은 프롬프트를 한국어로 넣었을 때 `gpt2`가 왜 못하는지 토큰 관점에서 설명하세요.

??? success "퍼플렉서티 힌트"
    ```python
    enc = tok("The quick brown fox jumps over the lazy dog.", return_tensors="pt")
    with torch.no_grad():
        loss = model(**enc, labels=enc["input_ids"]).loss
    print(torch.exp(loss).item())
    ```
    자연스러운 문장일수록 낮고, 단어를 섞으면 크게 올라갑니다.

## 8. 정리 노트에 적을 것

- 모델의 "생각"이 결국 확률분포라는 것을 직접 확인한 소감
- 샘플링 설정별 출력 특성
- 토크나이저가 언어별로 달라지는 비용
