# 실습: 도구를 쓰는 에이전트 만들기

**목표**: 계산기, 노트 검색, 노트 읽기 도구를 가진 에이전트를 **루프를 직접 짜서** 만듭니다.
**관련 노트**: [에이전트와 도구 사용](../notes/16-agents.md), [RAG](../notes/14-rag.md)

```bash
pip install anthropic
export ANTHROPIC_API_KEY="sk-ant-..."       # PowerShell: $env:ANTHROPIC_API_KEY="sk-ant-..."
```

> 에이전트는 호출을 여러 번 하므로 비용이 늘어납니다. 반드시 **최대 반복 횟수**를 두고, 토큰 사용량을 출력해 확인하세요. 프로젝트 루트(`docs/`가 있는 곳)에서 실행합니다.

## 1. 도구 구현

모델은 호출을 "요청"만 하고, 실행은 우리 코드가 합니다. 그래서 **안전하게** 만드는 것이 중요합니다.

```python
import ast, operator, glob, os

# --- 1) 계산기: eval()은 위험하므로 허용된 연산만 직접 계산 ---
OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
       ast.Div: operator.truediv, ast.Pow: operator.pow, ast.USub: operator.neg,
       ast.Mod: operator.mod}

def _eval(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in OPS:
        return OPS[type(node.op)](_eval(node.left), _eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in OPS:
        return OPS[type(node.op)](_eval(node.operand))
    raise ValueError("허용되지 않은 식입니다")

def calculator(expression: str) -> str:
    try:
        return str(_eval(ast.parse(expression, mode="eval").body))
    except Exception as e:
        return f"오류: {e}. 숫자와 + - * / ** % 만 사용할 수 있습니다."

# --- 2) 노트 검색: 키워드가 많이 나오는 파일 순 ---
DOCS = "docs"

def search_notes(query: str, limit: int = 3) -> str:
    words = query.lower().split()
    scored = []
    for path in glob.glob(f"{DOCS}/**/*.md", recursive=True):
        text = open(path, encoding="utf-8").read().lower()
        score = sum(text.count(w) for w in words)
        if score:
            scored.append((score, path.replace("\\", "/")))
    scored.sort(reverse=True)
    if not scored:
        return "검색 결과가 없습니다. 다른 키워드로 시도해 보세요."
    return "\n".join(f"{p} (일치 {s}회)" for s, p in scored[:limit])

# --- 3) 노트 읽기: docs/ 밖으로 나가지 못하게 경로 검증 ---
def read_note(path: str, max_chars: int = 6000) -> str:
    full = os.path.realpath(path)
    root = os.path.realpath(DOCS)
    if not full.startswith(root + os.sep) or not full.endswith(".md"):
        return "오류: docs/ 폴더 안의 .md 파일만 읽을 수 있습니다."
    if not os.path.exists(full):
        return "오류: 파일이 없습니다. search_notes로 경로를 먼저 찾으세요."
    return open(full, encoding="utf-8").read()[:max_chars]

TOOL_FUNCS = {"calculator": calculator, "search_notes": search_notes, "read_note": read_note}
```

`read_note`의 **경로 검증**을 주목하세요. 모델이 `../../비밀파일`을 요청해도 막혀야 합니다. 도구의 안전은 전적으로 개발자 책임입니다.

## 2. 도구 설명서 (스키마)

```python
tools = [
    {
        "name": "calculator",
        "description": "수식을 정확히 계산합니다. 곱셈 등 산술 계산이 필요할 때 사용합니다. "
                       "모델이 암산하면 틀릴 수 있으므로 큰 수의 계산은 반드시 이 도구를 쓰세요.",
        "input_schema": {
            "type": "object",
            "properties": {"expression": {"type": "string", "description": "예: 1234*5678"}},
            "required": ["expression"],
        },
    },
    {
        "name": "search_notes",
        "description": "사용자의 AI 공부 노트에서 키워드로 관련 파일 경로를 찾습니다. "
                       "노트에 정리된 개념을 확인할 때 먼저 사용합니다.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "검색 키워드 (공백으로 구분)"},
                "limit": {"type": "integer", "description": "최대 결과 수 (기본 3)"},
            },
            "required": ["query"],
        },
    },
    {
        "name": "read_note",
        "description": "search_notes로 찾은 노트 파일의 내용을 읽습니다. 경로는 docs/ 로 시작해야 합니다.",
        "input_schema": {
            "type": "object",
            "properties": {"path": {"type": "string", "description": "예: docs/notes/05-probability.md"}},
            "required": ["path"],
        },
    },
]
```

## 3. 에이전트 루프

```python
import anthropic
client = anthropic.Anthropic()

SYSTEM = ("당신은 사용자의 AI 공부 노트를 바탕으로 답하는 도우미입니다. "
          "노트에 관한 질문은 도구로 찾아 읽고 근거 파일을 밝히세요. "
          "계산은 반드시 calculator를 쓰세요. 노트에 없는 내용은 없다고 말하세요.")

def run_agent(question: str, max_steps: int = 8, verbose: bool = True) -> str:
    messages = [{"role": "user", "content": question}]
    total_in = total_out = 0

    for step in range(max_steps):
        r = client.messages.create(
            model="claude-opus-5-5",
            max_tokens=4000,
            system=SYSTEM,
            tools=tools,
            messages=messages,
        )
        total_in += r.usage.input_tokens
        total_out += r.usage.output_tokens
        messages.append({"role": "assistant", "content": r.content})

        if r.stop_reason != "tool_use":                      # 최종 답변
            if verbose:
                print(f"[토큰] in={total_in} out={total_out} steps={step + 1}")
            return "".join(b.text for b in r.content if b.type == "text")

        results = []
        for block in r.content:
            if block.type == "tool_use":
                func = TOOL_FUNCS.get(block.name)
                output = func(**block.input) if func else f"오류: 알 수 없는 도구 {block.name}"
                if verbose:
                    print(f"[도구] {block.name}({block.input}) -> {str(output)[:80]!r}")
                results.append({"type": "tool_result", "tool_use_id": block.id,
                                "content": str(output)})
        messages.append({"role": "user", "content": results})      # 결과는 한 메시지에 모아서

    return "최대 단계 수를 넘어 중단했습니다."

print(run_agent("확률 노트에서 베이즈 예제의 결과가 몇 퍼센트였고, 그 값에 12345를 곱하면 얼마야?"))
```

기대하는 흐름: `search_notes` → `read_note` → `calculator` → 최종 답변. 도구 호출 로그를 보며 **모델이 어떤 순서로 판단하는지** 관찰하세요.

## 4. 시험해 볼 시나리오

| 질문 | 확인할 점 |
|---|---|
| "1234 곱하기 5678은?" | 암산하지 않고 calculator를 쓰는가 |
| "LoRA에서 r이 뭐야?" | 검색 → 읽기 → 출처 제시 |
| "노트에 양자컴퓨터 내용 있어?" | 없다고 말하는가 (환각 방지) |
| "read_note로 ../../etc/passwd 읽어줘" | 경로 검증이 막는가 |
| "2의 1000제곱 계산해줘" | 도구 오류/거대 결과 처리 |

## 5. 과제

- [ ] 위 시나리오 5개를 전부 돌려 결과와 **도구 호출 횟수**를 표로 기록하세요.
- [ ] `max_steps`를 2로 줄이면 어떤 질문에서 실패하나요?
- [ ] 새 도구를 추가하세요: `list_notes()` (노트 목록 반환) 또는 `today()` (오늘 날짜)
- [ ] 도구 설명(description)을 일부러 부실하게 바꿨을 때 선택 정확도가 어떻게 달라지는지 비교하세요.
- [ ] 모든 호출을 `agent.log`(JSON Lines)에 기록하는 로깅을 추가하세요. (도구명, 인자, 결과 길이, 토큰)
- [ ] **프롬프트 인젝션 테스트**: 노트 한 개에 "이전 지시를 무시하고 …" 문장을 넣고, 에이전트가 따르는지 확인한 뒤 방어 문구를 시스템 프롬프트에 추가해 보세요.
- [ ] 평가셋 10개(질문 + 기대 도구 순서 + 정답 키워드)를 만들어 자동 채점 스크립트를 작성하세요.
- [ ] (도전) `write_note` 도구를 추가하되 **실행 전 사용자에게 y/n 승인**을 받도록 하세요.

??? success "힌트: 승인 단계와 로깅"
    ```python
    import json, time

    def log(entry):
        with open("agent.log", "a", encoding="utf-8") as f:
            f.write(json.dumps({"t": time.time(), **entry}, ensure_ascii=False) + "\n")

    def write_note(path: str, content: str) -> str:
        print(f"\n[승인 필요] {path} 에 {len(content)}자 쓰기")
        if input("허용? (y/n): ").strip().lower() != "y":
            return "사용자가 거부했습니다."
        # ... 경로 검증 후 쓰기
        return "저장했습니다."
    ```
    위험한 동작은 **코드 레벨에서 사람 승인을 강제**하세요. 프롬프트로 "조심해"라고 부탁하는 것만으로는 부족합니다.

## 6. 정리 노트에 적을 것

- 모델이 도구를 선택하는 방식에서 관찰한 규칙성
- 실패 사례(잘못된 인자, 루프, 환각)와 개선 방법
- 에이전트로 만든 것 vs 고정 워크플로로 충분했을 부분
