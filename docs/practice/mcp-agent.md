# MCP 서버 만들고 에이전트에 연결하기

!!! abstract "목표"
    직접 MCP 서버를 만들고, 클라이언트로 `tools/list`·`tools/call`을 호출해 본 뒤, Claude 에이전트 루프에 연결합니다.

선행 노트: [MCP와 API 호출 구조](../notes/17-mcp.md), [에이전트와 도구 사용](../notes/16-agents.md)
환경: Google Colab 권장. API 키가 필요합니다(Colab 왼쪽 열쇠 아이콘 Secrets에 `ANTHROPIC_API_KEY` 저장).

!!! warning "검증 안내"
    이 코드는 작성자 환경에서 실행해 보지 못했습니다. SDK 버전에 따라 이름이 다를 수 있으니, 에러가 나면 메시지를 그대로 가지고 질문하세요.

## 0. 준비

```python
!pip install -q mcp anthropic
```

```python
import os
from google.colab import userdata
os.environ["ANTHROPIC_API_KEY"] = userdata.get("ANTHROPIC_API_KEY")
```

## 1. MCP 서버 만들기

노트북 셀에서 `server.py` 파일을 만듭니다.

```python
%%writefile server.py
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("calc")

@mcp.tool()
def add(a: int, b: int) -> int:
    """두 정수를 더한다."""
    return a + b

@mcp.tool()
def word_count(text: str) -> int:
    """문자열의 단어 수를 센다. 공백 기준."""
    return len(text.split())

@mcp.tool()
def get_weather(city: str) -> str:
    """도시의 현재 날씨를 반환한다. (연습용 가짜 데이터)"""
    data = {"서울": "맑음, 18도", "부산": "흐림, 21도"}
    return data.get(city, f"{city}: 데이터 없음")

if __name__ == "__main__":
    mcp.run()   # stdio. print()는 쓰지 말 것!
```

## 2. 클라이언트로 직접 호출해 보기

에이전트를 붙이기 전에 프로토콜이 어떻게 동작하는지 눈으로 확인합니다.

```python
import sys
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

params = StdioServerParameters(command=sys.executable, args=["server.py"])

async def explore():
    # 노트북에서 stderr에 fileno가 없다는 오류가 나면 errlog=open(os.devnull, "w") 를 stdio_client에 넘긴다
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            tools = await session.list_tools()
            for t in tools.tools:
                print(t.name, "|", t.description)
                print("   schema:", t.inputSchema)

            result = await session.call_tool("add", {"a": 3, "b": 4})
            print("결과:", result.content[0].text, "| isError:", result.isError)

await explore()   # Colab/Jupyter는 최상위에서 await 가능
```

### 과제 1
- [ ] 위 출력에서 `inputSchema`가 함수의 타입 힌트와 어떻게 대응하는지 확인한다.
- [ ] `word_count`와 `get_weather`도 `call_tool`로 호출해 본다.
- [ ] 존재하지 않는 도구 이름이나 잘못된 인자 타입을 넣고 어떤 오류가 나는지 본다.

## 3. MCP 도구를 에이전트에 연결하기

핵심은 두 변환입니다. (1) MCP 도구 목록 → Messages API `tools=`, (2) `tool_use` → `call_tool`.

```python
import anthropic

client = anthropic.Anthropic()

async def mcp_agent(question: str, max_steps: int = 10) -> str:
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            # (1) MCP 도구 -> Messages API 형식
            listed = await session.list_tools()
            tools = [{
                "name": t.name,
                "description": t.description or "",
                "input_schema": t.inputSchema,
            } for t in listed.tools]

            messages = [{"role": "user", "content": question}]
            for _ in range(max_steps):
                resp = client.messages.create(
                    model="claude-opus-5-5", max_tokens=1024,
                    tools=tools, messages=messages,
                )
                messages.append({"role": "assistant", "content": resp.content})

                if resp.stop_reason != "tool_use":
                    return "".join(b.text for b in resp.content if b.type == "text")

                results = []
                for block in resp.content:
                    if block.type != "tool_use":
                        continue
                    # (2) tool_use -> MCP tools/call
                    r = await session.call_tool(block.name, block.input)
                    text = "".join(c.text for c in r.content if c.type == "text")
                    item = {"type": "tool_result", "tool_use_id": block.id,
                            "content": text}
                    if r.isError:
                        item["is_error"] = True
                    results.append(item)
                messages.append({"role": "user", "content": results})
            return "최대 반복 횟수를 넘었습니다."

print(await mcp_agent("서울과 부산 날씨를 알려주고, 두 도시 이름의 글자 수를 더해줘."))
```

### 과제 2
- [ ] 위 질문에서 모델이 도구를 **몇 번, 어떤 순서로** 호출하는지 출력해서 확인한다. (`print(block.name, block.input)` 추가)
- [ ] 한 턴에 `tool_use`가 **여러 개** 오는 경우가 있는지 본다. 그때 `tool_result`가 한 user 메시지에 모여 있는지 `messages`를 출력해 확인한다.
- [ ] `resp.usage`를 매 단계 출력해서 입력 토큰이 단계마다 어떻게 늘어나는지 기록한다.

## 4. 확장 과제

- [ ] 도구를 하나 더 만든다. 예: `read_note(name)`에서 허용된 폴더 안의 `.md`만 읽게 한다. **경로 검증**을 반드시 넣는다.
- [ ] 의도적으로 나쁜 `description`("뭔가 계산함")으로 바꿔 모델이 도구를 잘못 고르는지 비교한다. 좋은 설명과 나쁜 설명의 차이를 표로 정리한다.
- [ ] 도구에서 예외를 일으켜 `isError`가 `True`일 때 모델이 어떻게 복구하는지 관찰한다.
- [ ] 같은 에이전트에 서버를 두 개 연결해 본다. (서버별 `ClientSession`을 만들고 도구 이름이 겹칠 때 어떻게 구분할지 설계한다.)

## 힌트

??? tip "`unsupported operation: fileno` 오류가 나요"
    노트북의 stderr에는 파일 번호가 없어서 자식 프로세스에 연결하지 못하는 경우입니다. `stdio_client(params, errlog=open(os.devnull, "w"))`처럼 파일을 넘기거나, 서버를 로컬 터미널에서 실행해 보세요.

??? tip "`await`에서 SyntaxError가 나요"
    일반 `.py` 스크립트에서는 최상위 `await`를 쓸 수 없습니다. `import asyncio; asyncio.run(mcp_agent("..."))`로 실행하세요. Colab/Jupyter 셀에서는 `await`를 바로 쓸 수 있습니다.

??? tip "도구를 아예 안 불러요"
    `description`이 모호하거나 질문에 도구가 필요 없어 보이는 경우입니다. 설명에 "언제 쓰는지"를 적어 보세요. `tool_choice`로 강제하기 전에 설명부터 고칩니다.

## 정리 노트에 적을 것

- 모델, 호스트, MCP 서버 사이에서 **각 메시지가 누구에게서 누구에게** 가는지 그림으로 다시 그리기
- `tools/list` 결과와 Messages API `tools=`의 필드 대응표
- 좋은 도구 설명과 나쁜 도구 설명이 호출 정확도에 준 영향
- 단계가 늘수록 입력 토큰이 늘어나는 이유와 줄이는 방법
