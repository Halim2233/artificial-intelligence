# 실습: LoRA로 감성 분류 모델 미세조정

**목표**: DistilBERT를 LoRA로 미세조정해 영화 리뷰 감성 분류 정확도를 올리고, 학습 파라미터가 전체의 1% 정도인지 확인합니다.
**관련 노트**: [파인튜닝과 LoRA](../notes/15-finetuning-lora.md)

> GPU 권장입니다. **Google Colab 무료 GPU(T4)** 에서 몇 분이면 끝납니다. CPU에서는 데이터를 500개 정도로 줄이세요.

```bash
pip install transformers peft datasets accelerate evaluate scikit-learn
```

## 1. 데이터

```python
from datasets import load_dataset
from transformers import AutoTokenizer

ds = load_dataset("imdb")
train = ds["train"].shuffle(seed=42).select(range(2000))
test = ds["test"].shuffle(seed=42).select(range(1000))

tok = AutoTokenizer.from_pretrained("distilbert-base-uncased")
prep = lambda b: tok(b["text"], truncation=True, max_length=256)
train = train.map(prep, batched=True)
test = test.map(prep, batched=True)

print(train[0]["text"][:200], train[0]["label"])       # 0: 부정, 1: 긍정
```

## 2. 기준선: 미세조정 전

학습하지 않은 모델은 분류 머리가 무작위 초기화 상태이므로 정확도가 약 50%(동전 던지기)에 가깝습니다. 이 숫자를 **먼저** 기록해 두세요.

```python
import numpy as np
from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments, DataCollatorWithPadding

def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=1)
    return {"accuracy": float((preds == p.label_ids).mean())}

base = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)
trainer0 = Trainer(
    model=base,
    args=TrainingArguments(output_dir="tmp", per_device_eval_batch_size=64, report_to="none"),
    eval_dataset=test, data_collator=DataCollatorWithPadding(tok), compute_metrics=compute_metrics,
)
print("before:", trainer0.evaluate())
```

## 3. LoRA 적용

```python
from peft import LoraConfig, get_peft_model, TaskType

base = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)
cfg = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    r=8, lora_alpha=16, lora_dropout=0.05,
    target_modules=["q_lin", "v_lin"],
)
model = get_peft_model(base, cfg)
model.print_trainable_parameters()      # 학습 가능한 비율이 약 1% 내외인지 확인
```

`print_trainable_parameters`의 출력으로 **전체 대비 학습 파라미터 비율**을 기록하세요.

## 4. 학습

```python
args = TrainingArguments(
    output_dir="lora-out",
    learning_rate=2e-4,                  # LoRA는 전체 미세조정보다 높은 학습률
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    num_train_epochs=2,
    eval_strategy="epoch",
    save_strategy="no",
    logging_steps=20,
    report_to="none",
)
trainer = Trainer(
    model=model, args=args,
    train_dataset=train, eval_dataset=test,
    data_collator=DataCollatorWithPadding(tok),
    compute_metrics=compute_metrics,
)
trainer.train()
print("after:", trainer.evaluate())
```

보통 정확도가 **85~90%** 근처로 올라갑니다. (데이터 크기와 설정에 따라 다름)

## 5. 어댑터 저장과 불러오기

```python
model.save_pretrained("imdb-lora")           # 어댑터만 저장 (수 MB)
```

```python
from peft import PeftModel
base = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)
loaded = PeftModel.from_pretrained(base, "imdb-lora").eval()

import torch
text = "A beautiful film with brilliant acting."
inputs = tok(text, return_tensors="pt", truncation=True)
with torch.no_grad():
    print(loaded(**inputs).logits.softmax(-1))   # [부정, 긍정] 확률
```

저장된 폴더 크기를 확인해 보세요. 66M 모델 전체가 아니라 **작은 어댑터만** 들어 있습니다.

## 6. 과제

- [ ] **기준선 → LoRA 학습 후** 정확도를 표로 정리하세요.
- [ ] `r`을 `1 / 4 / 8 / 32`로 바꿔 학습 파라미터 수와 정확도의 관계를 그래프로 그리세요.
- [ ] `target_modules`를 `["q_lin"]` 하나만 / `["q_lin","k_lin","v_lin","out_lin"]` 전부로 바꿔 비교하세요.
- [ ] 학습 데이터를 100 / 500 / 2000개로 바꿔 **데이터량의 영향**을 보세요.
- [ ] 직접 만든 리뷰 문장 10개(반어법 포함)로 예측해 보고 틀린 이유를 분석하세요.
- [ ] 같은 작업을 **프롬프트만으로 LLM에게** 시키고 (`claude`, 분류 프롬프트), 정확도·비용·속도를 LoRA 모델과 비교하세요. 언제 어느 쪽이 나은가요?
- [ ] (도전) 전체 미세조정과 비교해 학습 시간과 GPU 메모리(`nvidia-smi`)를 측정하세요.

??? success "힌트"
    - 데이터가 매우 적으면 LoRA도 과적합합니다. 검증 정확도가 내려가기 시작하면 에폭을 줄이세요.
    - 프롬프트 방식은 학습 없이 시작할 수 있고, LoRA는 **대량 처리 시 건당 비용과 지연이 매우 낮다**는 것이 장점입니다. 처리량이 많고 작업이 고정이라면 파인튜닝한 작은 모델이 유리해집니다.

## 7. 정리 노트에 적을 것

- 학습 파라미터 비율과 성능의 관계
- `r`, 대상 모듈, 데이터량이 결과에 미친 영향
- 프롬프트 방식 vs 파인튜닝: 내 결론
