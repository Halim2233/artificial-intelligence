# 실습

노트에서 배운 것을 직접 손으로 구현하는 과제입니다.

| 실습 | 배우는 것 | 선행 노트 |
|---|---|---|
| [경사하강법 선형 회귀](gradient-descent.md) | NumPy로 학습 루프 구현, 학습률 | 선형대수, 미분 |
| [Titanic 생존 예측](titanic.md) | EDA, 결측치, sklearn 파이프라인 | Pandas, ML 기초 |
| [과적합 실험](overfitting.md) | 과적합 시각화, 정규화 | ML 기초 |

## 환경 준비

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install numpy pandas matplotlib seaborn scikit-learn jupyter
jupyter lab
```

코드는 Jupyter 노트북에서 셀 단위로 실행하면 결과를 바로 볼 수 있어 편합니다. 로컬 설치가 번거로우면 [Google Colab](https://colab.research.google.com/)을 쓰세요.

!!! tip "기록 습관"
    실습을 끝내면 결과 그래프와 느낀 점을 노트에 한 단락씩 남기세요. 나중에 가장 값진 자료가 됩니다.
