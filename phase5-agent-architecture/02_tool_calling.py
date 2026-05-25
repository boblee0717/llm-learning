"""
第五阶段 · 第 2 课：Tool Use / Function Calling
================================================

本课目标：理解 function calling 的分工——

    模型负责：决定"调用哪个工具、传什么参数"（选择 + 填参）
    runtime 负责：校验参数、执行工具、处理错误、把结果回灌（执行 + 安全）

模型本身**不会**执行任何工具。它只是输出一个结构化的"我想调用 X(参数)"的请求，
真正动手的是你的 runtime。这是初学者最容易混淆的一点。

本文件实现：
  1. 一个 tool 注册表（name + description + JSON-Schema 风格的参数说明）
  2. 一个 dispatcher：解析模型的 tool 请求 -> 校验 -> 执行 -> 回灌
  3. 错误处理：工具报错/参数非法/未知工具，都变成模型能读懂的 observation

同样用 FakeLLM 模拟模型决策，无需 GPU/网络/API key。

运行：
    python3 phase5-agent-architecture/02_tool_calling.py
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Callable


# ---------------------------------------------------------------------------
# 1. Tool 定义：每个工具有 name、description、参数 schema、和执行函数。
#    模型靠 description 决定"何时用这个工具"，所以 description 写得好不好，
#    直接影响模型的调用准确率（这是第 2 课最重要的工程直觉之一）。
# ---------------------------------------------------------------------------

@dataclass
class Tool:
    name: str
    description: str
    parameters: dict          # JSON-Schema 风格，告诉模型每个参数的含义
    fn: Callable[..., str]    # 真正执行的函数，由 runtime 调用


def _calculator(expression: str) -> str:
    import re
    if not re.fullmatch(r"[0-9+\-*/(). ]+", expression):
        return "ERROR: 表达式包含不允许的字符"
    try:
        return str(eval(expression))  # noqa: S307 - 白名单字符已限制
    except Exception as exc:  # noqa: BLE001
        return f"ERROR: {exc}"


def _get_weather(city: str) -> str:
    # mock 数据：真实系统里这会是一次外部 API 调用。
    fake = {"beijing": "晴, 22°C", "shanghai": "多云, 25°C", "shenzhen": "雷阵雨, 28°C"}
    key = city.strip().lower()
    if key not in fake:
        return f"ERROR: 没有 {city} 的天气数据"
    return fake[key]


def _word_count(text: str) -> str:
    return str(len(text.split()))


# ---------------------------------------------------------------------------
# 2. Tool 注册表：runtime 持有所有可用工具。新增工具 = 往这里注册一项。
# ---------------------------------------------------------------------------

class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def names(self) -> list[str]:
        return list(self._tools)

    def schema_for_prompt(self) -> str:
        """把工具清单格式化成可以放进 system prompt 的文本。
        真实 function-calling API 会把这个结构化地传给模型；这里打印出来，
        让你直观看到"模型看到的工具说明书"长什么样。"""
        lines = []
        for t in self._tools.values():
            params = ", ".join(
                f"{k}: {v['type']} // {v['desc']}"
                for k, v in t.parameters.items()
            )
            lines.append(f"- {t.name}({params}): {t.description}")
        return "\n".join(lines)


def build_registry() -> ToolRegistry:
    reg = ToolRegistry()
    reg.register(Tool(
        name="calculator",
        description="计算一个算术表达式，支持 + - * / 和括号。",
        parameters={"expression": {"type": "string", "desc": "要计算的算式"}},
        fn=_calculator,
    ))
    reg.register(Tool(
        name="get_weather",
        description="查询某个城市当前天气。支持 beijing/shanghai/shenzhen。",
        parameters={"city": {"type": "string", "desc": "城市名（英文）"}},
        fn=_get_weather,
    ))
    reg.register(Tool(
        name="word_count",
        description="统计一段文本里的单词数（按空格分隔）。",
        parameters={"text": {"type": "string", "desc": "要统计的文本"}},
        fn=_word_count,
    ))
    return reg


# ---------------------------------------------------------------------------
# 3. Dispatcher：解析模型的 tool 请求 -> 校验 -> 执行 -> 把结果/错误回灌。
#    所有失败都被翻译成 observation 字符串，绝不让 loop 崩掉。
# ---------------------------------------------------------------------------

def dispatch(registry: ToolRegistry, raw_request: str) -> str:
    """raw_request 是模型输出的工具调用，约定为 JSON：
        {"tool": "calculator", "args": {"expression": "1+1"}}
    返回值是 observation（成功结果或错误说明），它会被回灌给模型。
    """
    # (a) 解析：模型可能输出格式错误的 JSON，这本身要当成可恢复的错误。
    try:
        req = json.loads(raw_request)
    except json.JSONDecodeError as exc:
        return f"ERROR: 工具调用不是合法 JSON：{exc}"

    tool_name = req.get("tool")
    args = req.get("args", {})

    # (b) 未知工具
    tool = registry.get(tool_name)
    if tool is None:
        return f"ERROR: 未知工具 {tool_name!r}，可用工具：{registry.names()}"

    # (c) 参数校验：缺参数 / 多参数都拦下来，给模型清晰的反馈。
    expected = set(tool.parameters)
    got = set(args)
    if missing := expected - got:
        return f"ERROR: 工具 {tool_name} 缺少参数 {sorted(missing)}"
    if extra := got - expected:
        return f"ERROR: 工具 {tool_name} 收到多余参数 {sorted(extra)}"

    # (d) 执行：工具内部抛异常也要兜住，变成 observation。
    try:
        return tool.fn(**args)
    except Exception as exc:  # noqa: BLE001
        return f"ERROR: 工具 {tool_name} 执行失败：{exc}"


# ---------------------------------------------------------------------------
# 4. FakeLLM：根据问题决定输出哪个 tool 调用（JSON），或给最终答案。
# ---------------------------------------------------------------------------

class FakeLLM:
    def decide(self, question: str, last_obs: str | None) -> str:
        # 如果已经有观察结果，就收尾。
        if last_obs is not None:
            if last_obs.startswith("ERROR:"):
                return "FINAL: 抱歉，工具调用出错了：" + last_obs
            return f"FINAL: 结果是 {last_obs}。"

        q = question.lower()
        if any(c.isdigit() for c in question) and any(op in question for op in "+-*/"):
            import re
            expr = re.search(r"[\d(][\d+\-*/(). ]*\d", question)
            return json.dumps({"tool": "calculator", "args": {"expression": expr.group(0)}})
        if "天气" in question or "weather" in q:
            for city in ("beijing", "shanghai", "shenzhen", "北京", "上海", "深圳"):
                if city in q or city in question:
                    name = {"北京": "beijing", "上海": "shanghai", "深圳": "shenzhen"}.get(city, city)
                    return json.dumps({"tool": "get_weather", "args": {"city": name}})
            return json.dumps({"tool": "get_weather", "args": {"city": "unknown"}})
        if "几个词" in question or "word" in q:
            text = question.split("：", 1)[-1] if "：" in question else question
            return json.dumps({"tool": "word_count", "args": {"text": text}})
        return "FINAL: 我没有合适的工具处理这个问题。"


# ---------------------------------------------------------------------------
# 5. 把 tool calling 接进一个最小 loop。
# ---------------------------------------------------------------------------

def run(question: str, registry: ToolRegistry, max_steps: int = 4) -> str:
    llm = FakeLLM()
    print(f"\n[问题] {question}")
    last_obs: str | None = None

    for step in range(1, max_steps + 1):
        out = llm.decide(question, last_obs)
        print(f"--- Step {step} ---")

        if out.startswith("FINAL:"):
            answer = out[len("FINAL:"):].strip()
            print(f"[最终答案] {answer}")
            return answer

        # 否则 out 是一个工具调用请求（JSON）。
        print(f"[模型请求调用] {out}")
        last_obs = dispatch(registry, out)
        print(f"[观察/回灌] {last_obs}")

    print("[终止] 达到最大步数。")
    return "(未完成)"


def main() -> None:
    registry = build_registry()

    print("=" * 60)
    print("模型看到的工具说明书（system prompt 里的工具清单）：")
    print("=" * 60)
    print(registry.schema_for_prompt())

    run("帮我算 (5 + 7) * 2", registry)
    run("北京今天天气怎么样？", registry)
    run("这句话有几个词：the quick brown fox jumps", registry)

    print("\n" + "=" * 60)
    print("演示错误回灌：故意构造一个会失败的工具调用")
    print("=" * 60)
    bad = json.dumps({"tool": "get_weather", "args": {"city": "mars"}})
    print(f"[模型请求调用] {bad}")
    print(f"[观察/回灌] {dispatch(registry, bad)}  <- 注意：loop 没有崩，错误变成了 observation")

    print("\n" + "=" * 60)
    print("小练习：")
    print("- 新增一个工具（如 reverse_text），注册进 build_registry 并让 FakeLLM 用上。")
    print("- 把某个工具的 description 写得很差，体会它如何影响'模型该不该用它'的判断。")
    print("- 给 dispatch 加一个'工具白名单'参数，模拟第 7 课的安全边界。")
    print("=" * 60)


if __name__ == "__main__":
    main()
