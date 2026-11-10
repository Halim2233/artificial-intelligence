# 노트

개념별 학습 노트입니다. 순서대로 읽으면 [로드맵](../roadmap.md) 전체를 따라갈 수 있습니다.

## 1단계 · 기초

| # | 노트 | 핵심 |
|---|---|---|
| 1 | [NumPy](01-numpy.md) | 배열, 브로드캐스팅, axis |
| 2 | [Pandas](02-pandas.md) | DataFrame, 필터, groupby |
| 3 | [선형대수](03-linear-algebra.md) | 내적, 행렬곱, shape |
| 4 | [미분과 경사하강법](04-calculus-gradient.md) | 기울기, 연쇄법칙 |
| 5 | [확률과 통계](05-probability.md) | 베이즈, 소프트맥스, 교차 엔트로피 |
| 6 | [머신러닝 기초](06-ml-basics.md) | 과적합, 평가 지표, sklearn |

## 2단계 · 딥러닝

| # | 노트 | 핵심 |
|---|---|---|
| 7 | [신경망과 역전파](07-backprop.md) | 활성화 함수, 역전파, micrograd |
| 8 | [PyTorch 기초](08-pytorch.md) | 텐서, autograd, 학습 루프 |
| 9 | [CNN](09-cnn.md) | 합성곱, 풀링, ResNet, 전이 학습 |
| 10 | [RNN과 어텐션의 등장](10-rnn-attention.md) | LSTM, seq2seq 병목, 어텐션 |

## 3단계 · LLM

| # | 노트 | 핵심 |
|---|---|---|
| 11 | [토큰화와 임베딩](11-tokenization-embedding.md) | BPE, 임베딩, 코사인 유사도 |
| 12 | [트랜스포머](transformer.md) | 셀프 어텐션, 인과 마스크, 블록 구조 |
| 13 | [LLM의 학습과 추론](12-llm-training.md) | 사전학습, SFT, RLHF, 샘플링, KV 캐시 |
| 14 | [프롬프트 엔지니어링](13-prompt-engineering.md) | 프롬프트 구조, API, 평가 |

## 4단계 · 응용

| # | 노트 | 핵심 |
|---|---|---|
| 15 | [RAG](14-rag.md) | 청킹, 검색, 평가, 실패 대응 |
| 16 | [파인튜닝과 LoRA](15-finetuning-lora.md) | 저랭크 적응, QLoRA, 데이터 |
| 17 | [에이전트와 도구 사용](16-agents.md) | 루프, 도구 설계, 안전 |
| 18 | [MCP와 API 호출 구조](17-mcp.md) | 호스트·클라이언트·서버, tool_use 루프, JSON-RPC |

## 공부 방법

1. 노트를 읽으면서 **코드를 직접 타이핑해서 실행**합니다. (복붙 금지)
2. 연습문제를 먼저 풀고 나서 정답을 펼칩니다.
3. 막힌 부분은 템플릿(`templates/note.md`)으로 **내 말로 다시 정리**합니다.
4. 단계 끝의 [실습](../practice/index.md)을 합니다.

## 새 노트 추가하기

1. `templates/note.md`를 `docs/notes/새이름.md`로 복사합니다.
2. 내용을 채웁니다.
3. `mkdocs.yml`의 `nav` → 노트 아래에 등록합니다.
