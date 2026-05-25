"""
第五阶段 · 第 1 课：最小 Agent Loop
====================================

本课目标：理解一个「Agent」的本质，就是一个带终止条件的循环：

    观察(observation) -> 思考(LLM) -> 行动(action) -> 新观察 -> ...

直到模型给出最终答案，或触发终止条件（最大步数 / 超时 / 报错）。

为了让你无需 GPU、无需联网、无需 API key 就能跑通，本文件用一个规则驱动的
`FakeLLM` 模拟「模型在每一步决定：调用工具，还是给出最终答案」。

把 FakeLLM 读懂，你就理解了所有 agent 框架（LangGraph、AutoGen、OpenAI Agents
SDK……）最核心的那个循环。等理解了骨架，把 FakeLLM 换成真实的 function-calling
模型即可。

运行：
    python3 phase5-agent-architecture/01_minimal_agent_loop.py
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# 1. 工具（tool）：真正"做事"的部分。模型只决定调用谁、传什么，执行的是它们。
# ---------------------------------------------------------------------------

def tool_calculator(expression: str) -> str:
    """计算一个简单算术表达式。真实系统里这会是受限解释器或沙箱。"""
    # 仅允许数字和基本运算符，避免 eval 任意代码（第 7 课会专门讲安全）。
    if not re.fullmatch(r"[0-9+\-*/(). ]+", expression):
        return "ERROR: 表达式包含不允许的字符"
    try:
        return str(eval(expression))  # noqa: S307 - 已用白名单字符限制
    except Exception as exc:  # noqa: BLE001
        return f"ERROR: {exc}"


TOOLS = {
    "calculator": tool_calculator,
}


# ---------------------------------------------------------------------------
# 2. FakeLLM：模拟"模型的决策"。真实模型会读整个上下文后输出结构化决定，
#    这里用关键词规则代替，让循环可以确定性地跑起来。
# ---------------------------------------------------------------------------

@dataclass
class Decision:
    """模型每一步的决定：要么调用工具，要么给最终答案。"""
    kind: str          # "tool" 或 "final"
    tool: str = ""     # kind == "tool" 时有效
    tool_input: str = ""
    answer: str = ""   # kind == "final" 时有效
    thought: str = ""  # ReAct 里的"思考"，便于观察


class FakeLLM:
    """规则驱动的假模型。它看 observation 历史，决定下一步动作。"""

    def decide(self, question: str, history: list[str]) -> Decision:
        last = history[-1] if history else ""

        # 如果上一步已经拿到了计算结果（observation 里有 "结果="），就收尾。
        if last.startswith("结果="):
            value = last.split("=", 1)[1]
            return Decision(
                kind="final",
                thought="已经拿到工具结果，可以回答了。",
                answer=f"答案是 {value}。",
            )

        # 否则：如果问题里含算式，就决定调用计算器。
        match = re.search(r"[\d(][\d+\-*/(). ]*\d", question)
        if match:
            return Decision(
                kind="tool",
                thought="这是一个算术问题，应该调用 calculator。",
                tool="calculator",
                tool_input=match.group(0),
            )

        # 兜底：无法处理就直接回答。
        return Decision(
            kind="final",
            thought="不需要工具，直接回答。",
            answer="我暂时无法用现有工具回答这个问题。",
        )


# ---------------------------------------------------------------------------
# 3. Agent Loop：把"思考-行动-观察"串成一个带终止条件的循环。
# ---------------------------------------------------------------------------

@dataclass
class AgentResult:
    answer: str
    steps: int
    transcript: list[str] = field(default_factory=list)
    stopped_reason: str = "final"  # "final" / "max_steps"


def run_agent(question: str, max_steps: int = 5, verbose: bool = True) -> AgentResult:
    """运行最小 agent loop。

    max_steps 是关键的安全阀：没有它，模型可能永远循环下去。这是 agent 与
    普通一次性 prompt 最根本的区别——循环 + 终止条件。
    """
    llm = FakeLLM()
    history: list[str] = []          # 累积的 observation（工具结果等）
    transcript: list[str] = []       # 完整轨迹，用于观测/调试

    def log(line: str) -> None:
        transcript.append(line)
        if verbose:
            print(line)

    log(f"[问题] {question}")

    for step in range(1, max_steps + 1):
        decision = llm.decide(question, history)
        log(f"\n--- Step {step} ---")
        log(f"[思考] {decision.thought}")

        if decision.kind == "final":
            log(f"[最终答案] {decision.answer}")
            return AgentResult(
                answer=decision.answer,
                steps=step,
                transcript=transcript,
                stopped_reason="final",
            )

        # decision.kind == "tool"：runtime 负责执行工具（模型不执行）。
        log(f"[行动] 调用工具 {decision.tool}({decision.tool_input!r})")
        tool_fn = TOOLS.get(decision.tool)
        if tool_fn is None:
            observation = f"ERROR: 未知工具 {decision.tool}"
        else:
            observation = tool_fn(decision.tool_input)

        # 把工具结果作为新的 observation 回灌给模型——这是循环得以继续的关键。
        obs_line = f"结果={observation}"
        history.append(obs_line)
        log(f"[观察] {obs_line}")

    # 走到这里说明达到了 max_steps 还没收尾：终止并如实报告。
    log(f"\n[终止] 达到最大步数 {max_steps}，未得到最终答案。")
    return AgentResult(
        answer="(未完成：达到最大步数)",
        steps=max_steps,
        transcript=transcript,
        stopped_reason="max_steps",
    )


# ---------------------------------------------------------------------------
# 4. 演示
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 60)
    print("演示 1：需要调用工具的算术问题")
    print("=" * 60)
    run_agent("帮我算一下 (12 + 8) * 3 等于多少？")

    print("\n" + "=" * 60)
    print("演示 2：不需要工具的问题")
    print("=" * 60)
    run_agent("你好，今天过得怎么样？")

    print("\n" + "=" * 60)
    print("小练习思路：")
    print("- 把 max_steps 改成 1，观察会发生什么。")
    print("- 给 FakeLLM 增加对'减法/百分比'问题的处理。")
    print("- 画出这个 loop 的状态流转图：observation -> think -> act -> observation。")
    print("=" * 60)


if __name__ == "__main__":
    main()
