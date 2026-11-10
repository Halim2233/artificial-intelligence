# 실습

노트에서 배운 것을 직접 손으로 구현하는 과제입니다.

| 단계 | 실습 | 배우는 것 | 선행 노트 |
|---|---|---|---|
| 1 | [경사하강법 선형 회귀](gradient-descent.md) | NumPy로 학습 루프 구현, 학습률 | 선형대수, 미분 |
| 1 | [Titanic 생존 예측](titanic.md) | EDA, 결측치, sklearn 파이프라인 | Pandas, ML 기초 |
| 1 | [과적합 실험](overfitting.md) | 과적합 시각화, 정규화 | ML 기초 |
| 2 | [MNIST 분류기](mnist-pytorch.md) | PyTorch 학습 루프, MLP vs CNN | PyTorch, CNN |
| 3 | [Hugging Face로 LLM 만져보기](hf-transformers.md) | 토큰, 확률, 샘플링 | 토큰화, LLM 학습·추론 |
| 3 | [미니 GPT 학습](mini-gpt.md) | 트랜스포머 직접 학습·생성 | 트랜스포머 |
| 4 | [RAG 챗봇](rag-chatbot.md) | 청킹, 임베딩 검색, 평가 | RAG, 프롬프트 |
| 4 | [LoRA 미세조정](lora-finetune.md) | PEFT, 전/후 비교 | 파인튜닝과 LoRA |
| 4 | [도구 쓰는 에이전트](tool-agent.md) | 도구 설계, 에이전트 루프, 안전 | 에이전트 |
| 4 | [MCP 서버와 에이전트 연결](mcp-agent.md) | MCP 서버 제작, 클라이언트 호출, API 연동 | MCP |

## 환경 준비

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install numpy pandas matplotlib seaborn scikit-learn jupyter
jupyter lab
```

2단계부터는 추가로 설치합니다.

```bash
pip install torch torchvision                       # 2단계
pip install transformers datasets accelerate peft   # 3~4단계
pip install sentence-transformers anthropic         # RAG, 에이전트
```

코드는 Jupyter 노트북에서 셀 단위로 실행하면 결과를 바로 볼 수 있어 편합니다. 로컬 설치가 번거롭거나 **GPU가 필요한 실습(MNIST CNN, 미니 GPT, LoRA)** 은 [Google Colab](https://colab.research.google.com/)을 쓰세요.

## API 키가 필요한 실습

RAG 챗봇과 에이전트는 Anthropic API를 호출합니다.

- 키는 코드에 쓰지 말고 **환경 변수**(`ANTHROPIC_API_KEY`)로 설정합니다.
- 키가 들어간 파일을 **공개 저장소에 올리지 마세요.** (`.env`는 `.gitignore`에 추가)
- 사용량은 `response.usage`로 확인하고, 에이전트에는 **최대 반복 횟수**를 둡니다.

!!! tip "기록 습관"
    실습을 끝내면 결과 그래프와 느낀 점을 노트에 한 단락씩 남기세요. 나중에 가장 값진 자료가 됩니다.
