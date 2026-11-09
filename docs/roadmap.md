# 학습 로드맵

체크박스를 `[x]`로 바꾸면 완료 표시가 됩니다. 기간은 하루 1~2시간 기준의 대략적인 추정입니다.

## 1단계 · 기초 <small>(약 4~6주)</small>

### 파이썬 / 데이터
- [ ] 파이썬 문법 — [점프 투 파이썬](https://wikidocs.net/book/1)
- [ ] NumPy, Pandas — [NumPy 노트](notes/01-numpy.md), [Pandas 노트](notes/02-pandas.md) / 참고서: *Python for Data Analysis*
- [ ] **과제**: [Titanic 실습](practice/titanic.md)

### 수학
- [ ] 선형대수 — [노트](notes/03-linear-algebra.md) + 3Blue1Brown *Essence of Linear Algebra*
- [ ] 미적분 / 연쇄법칙 — [노트](notes/04-calculus-gradient.md) + 3Blue1Brown *Essence of Calculus*
- [ ] 확률·통계 기초 — [노트](notes/05-probability.md) + Khan Academy
- [ ] **과제**: [경사하강법 선형 회귀](practice/gradient-descent.md)

### 머신러닝 개념
- [ ] [ML 기초 노트](notes/06-ml-basics.md) + Andrew Ng *Machine Learning Specialization* (Coursera)
- [ ] scikit-learn으로 분류·회귀 한 번씩
- [ ] **과제**: [과적합 실험](practice/overfitting.md)

## 2단계 · 딥러닝 <small>(약 6~8주)</small>

- [ ] 신경망과 역전파 — 3Blue1Brown *Neural Networks*
- [ ] 밑바닥부터 구현 — Karpathy [Neural Networks: Zero to Hero](https://karpathy.ai/zero-to-hero.html) (micrograd)
- [ ] PyTorch 기초 — [공식 튜토리얼](https://pytorch.org/tutorials/)
- [ ] CNN, RNN — [Dive into Deep Learning](https://d2l.ai/)
- [ ] **과제**: MNIST 분류기를 PyTorch로 학습시키고 정확도 99% 달성

## 3단계 · LLM <small>(약 6~8주)</small>

- [ ] 트랜스포머 구조 — Jay Alammar *The Illustrated Transformer*
- [ ] 논문 읽기 — *Attention Is All You Need*
- [ ] 직접 만들기 — Karpathy *Let's build GPT* (nanoGPT)
- [ ] Hugging Face — [LLM Course](https://huggingface.co/learn/llm-course)
- [ ] 프롬프트 엔지니어링 — [Anthropic 프롬프트 가이드](https://docs.claude.com/en/docs/build-with-claude/prompt-engineering/overview)
- [ ] **과제**: 작은 GPT를 학습시켜 셰익스피어풍 텍스트 생성

## 4단계 · 응용 <small>(계속)</small>

- [ ] RAG — 내 노트(이 사이트)를 검색해 답하는 챗봇 만들기
- [ ] 파인튜닝 — LoRA로 소형 모델 조정
- [ ] 에이전트와 도구 사용 — API로 도구 호출하는 에이전트 만들기
- [ ] **과제**: 위 중 하나를 골라 [실습](practice/index.md)에 기록

!!! tip "공부 팁"
    강의를 보는 것보다 **직접 구현하고 노트로 정리**하는 쪽이 오래 남습니다. 단계마다 과제 하나는 꼭 끝내고 넘어가세요.
