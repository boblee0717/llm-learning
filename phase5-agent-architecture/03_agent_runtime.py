"""
第五阶段 · 第 6 课：Agent Runtime 与状态管理
=============================================

本课目标：理解线上的 agent **不是一次函数调用**，而是一个有状态、可中断、
可恢复的长流程。这是 agent infra 和 demo 脚本最大的区别。

本文件实现一个最小但完整的 agent runtime，包含三件 infra 关心的事：

  1. run / step 抽象：一次任务是一个 run，由多个 step 组成。
  2. checkpoint / resume：每个 step 后把状态落盘；进程崩了能从断点续跑
     （durable execution 的核心思想）。
  3. trace + 成本核算：每步记录耗时、工具、（假）token 数；run 结束给出报告。

为了无需 GPU/网络/API key，依旧用 FakeLLM。状态用 JSON 落到本地文件，模拟真实
系统里的数据库/对象存储。

运行：
    python3 phase5-agent-architecture/03_agent_runtime.py

它会演示：正常跑完一个 run；然后模拟「第 2 步崩溃」，再从 checkpoint resume 续跑。
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field, asdict
from typing import Callable


# ---------------------------------------------------------------------------
# 工具
# ---------------------------------------------------------------------------

def _calculator(expression: str) -> str:
    if not re.fullmatch(r"[0-9+\-*/(). ]+", expression):
        return "ERROR: 表达式包含不允许的字符"
    try:
        return str(eval(expression))  # noqa: S307
    except Exception as exc:  # noqa: BLE001
        return f"ERROR: {exc}"


TOOLS: dict[str, Callable[..., str]] = {"calculator": _calculator}


# ---------------------------------------------------------------------------
# 1. RunState：一个 run 的全部状态。能被序列化 = 能 checkpoint = 能 resume。
# ---------------------------------------------------------------------------

@dataclass
class StepTrace:
    """单个 step 的 trace：可观测性的最小单元（第 9 课会展开）。"""
    index: int
    action: str            # "tool:<name>" 或 "final"
    detail: str
    latency_ms: float
    fake_tokens: int       # 模拟该 step 的 token 消耗


@dataclass
class RunState:
    run_id: str
    question: str
    max_steps: int = 6
    step: int = 0                              # 已完成的 step 数
    history: list[str] = field(default_factory=list)   # observation 累积
    trace: list[dict] = field(default_factory=list)    # StepTrace 的字典形式
    status: str = "running"                    # running / done / failed
    answer: str = ""

    # ---- 序列化：状态可被存到文件/DB/对象存储，是 resume 的前提 ----
    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, indent=2)

    @classmethod
    def from_json(cls, text: str) -> "RunState":
        return cls(**json.loads(text))


# ---------------------------------------------------------------------------
# 2. CheckpointStore：把 RunState 落盘并能读回。真实系统里换成 DB/对象存储即可。
# ---------------------------------------------------------------------------

class CheckpointStore:
    def __init__(self, directory: str) -> None:
        self.directory = directory
        os.makedirs(directory, exist_ok=True)

    def _path(self, run_id: str) -> str:
        return os.path.join(self.directory, f"{run_id}.json")

    def save(self, state: RunState) -> None:
        with open(self._path(state.run_id), "w", encoding="utf-8") as f:
            f.write(state.to_json())

    def load(self, run_id: str) -> RunState | None:
        path = self._path(run_id)
        if not os.path.exists(path):
            return None
        with open(path, encoding="utf-8") as f:
            return RunState.from_json(f.read())

    def exists(self, run_id: str) -> bool:
        return os.path.exists(self._path(run_id))


# ---------------------------------------------------------------------------
# 3. FakeLLM：和前两课同思路，根据 observation 历史决定下一步。
# ---------------------------------------------------------------------------

class FakeLLM:
    def decide(self, question: str, history: list[str]) -> dict:
        last = history[-1] if history else ""
        if last.startswith("结果="):
            return {"kind": "final", "answer": f"答案是 {last.split('=', 1)[1]}。"}
        match = re.search(r"[\d(][\d+\-*/(). ]*\d", question)
        if match:
            return {"kind": "tool", "tool": "calculator", "input": match.group(0)}
        return {"kind": "final", "answer": "无法用现有工具回答。"}


# ---------------------------------------------------------------------------
# 4. Runtime：驱动一个 run，逐 step 执行，每步 checkpoint。支持 resume。
# ---------------------------------------------------------------------------

class AgentRuntime:
    def __init__(self, store: CheckpointStore) -> None:
        self.store = store
        self.llm = FakeLLM()

    def start(self, run_id: str, question: str, max_steps: int = 6) -> RunState:
        """开新 run：建初始状态并落盘第 0 个 checkpoint。"""
        state = RunState(run_id=run_id, question=question, max_steps=max_steps)
        self.store.save(state)
        return state

    def run(self, run_id: str, crash_at_step: int | None = None) -> RunState:
        """从 checkpoint 加载状态并继续执行。

        crash_at_step：仅用于演示——在指定 step 抛异常，模拟进程崩溃，
        以验证「崩溃后能 resume」。真实系统里崩溃来自 OOM、抢占、重启等。
        """
        state = self.store.load(run_id)
        if state is None:
            raise ValueError(f"run {run_id} 不存在，请先 start")

        if state.status == "done":
            print(f"[runtime] run {run_id} 已完成，直接返回缓存结果。")
            return state

        print(f"[runtime] 加载 run {run_id}，从 step {state.step + 1} 开始 "
              f"（已完成 {state.step} 步）")

        while state.step < state.max_steps and state.status == "running":
            next_step = state.step + 1

            # 模拟崩溃：注意此时**之前的 step 已 checkpoint**，所以能续跑。
            if crash_at_step is not None and next_step == crash_at_step:
                raise RuntimeError(f"💥 模拟在 step {next_step} 崩溃（状态已保存到上一步）")

            t0 = time.perf_counter()
            decision = self.llm.decide(state.question, state.history)

            if decision["kind"] == "final":
                latency = (time.perf_counter() - t0) * 1000
                state.trace.append(asdict(StepTrace(
                    index=next_step, action="final",
                    detail=decision["answer"], latency_ms=round(latency, 2),
                    fake_tokens=20,
                )))
                state.answer = decision["answer"]
                state.status = "done"
                state.step = next_step
                self.store.save(state)   # 收尾也要 checkpoint
                print(f"  step {next_step}: [final] {decision['answer']}")
                break

            # 工具调用：执行 -> 回灌 observation。
            tool = TOOLS.get(decision["tool"])
            observation = tool(decision["input"]) if tool else f"ERROR: 未知工具 {decision['tool']}"
            time.sleep(0.01)  # 假装工具有耗时
            latency = (time.perf_counter() - t0) * 1000

            state.history.append(f"结果={observation}")
            state.trace.append(asdict(StepTrace(
                index=next_step, action=f"tool:{decision['tool']}",
                detail=f"{decision['input']} -> {observation}",
                latency_ms=round(latency, 2), fake_tokens=35,
            )))
            state.step = next_step
            self.store.save(state)   # ★ 关键：每个 step 后 checkpoint
            print(f"  step {next_step}: [tool:{decision['tool']}] "
                  f"{decision['input']} -> {observation}  (checkpoint saved)")

        if state.status == "running":   # 用尽 max_steps 还没收尾
            state.status = "failed"
            state.answer = "(未完成：达到最大步数)"
            self.store.save(state)

        return state


# ---------------------------------------------------------------------------
# 5. run 级别报告：把 trace 汇总成可读的成本/性能小结（第 9 课主题的预演）。
# ---------------------------------------------------------------------------

def print_report(state: RunState) -> None:
    total_tokens = sum(s["fake_tokens"] for s in state.trace)
    total_latency = sum(s["latency_ms"] for s in state.trace)
    n_tool = sum(1 for s in state.trace if s["action"].startswith("tool:"))
    print("\n  ── run 报告 ──────────────────────────────")
    print(f"  run_id   : {state.run_id}")
    print(f"  status   : {state.status}")
    print(f"  steps    : {state.step}（其中工具调用 {n_tool} 次）")
    print(f"  tokens   : {total_tokens}（模拟值）")
    print(f"  latency  : {total_latency:.2f} ms")
    print(f"  answer   : {state.answer}")
    print("  ──────────────────────────────────────────")


# ---------------------------------------------------------------------------
# 6. 演示
# ---------------------------------------------------------------------------

def main() -> None:
    store = CheckpointStore(directory="/tmp/phase5_agent_runs")

    print("=" * 60)
    print("演示 A：正常跑完一个 run")
    print("=" * 60)
    rt = AgentRuntime(store)
    rt.start("run-A", "帮我算 (12 + 8) * 3")
    state_a = rt.run("run-A")
    print_report(state_a)

    print("\n" + "=" * 60)
    print("演示 B：模拟崩溃 -> 从 checkpoint resume 续跑")
    print("=" * 60)
    rt.start("run-B", "帮我算 (12 + 8) * 3")
    try:
        rt.run("run-B", crash_at_step=2)   # 第 1 步会先 checkpoint，第 2 步崩
    except RuntimeError as exc:
        print(f"  捕获到崩溃：{exc}")
        print("  进程'重启'... 现在用同一个 run_id 重新 run，应该从 step 2 续上：")

    state_b = rt.run("run-B")   # 不传 crash_at_step，正常续跑到结束
    print_report(state_b)

    print("\n" + "=" * 60)
    print("小练习：")
    print("- 打开 /tmp/phase5_agent_runs/run-B.json，看 checkpoint 长什么样。")
    print("- 给 RunState 加一个 plan 字段（第 3 课），让 runtime 先规划再执行。")
    print("- 把 CheckpointStore 换成 sqlite，体会真实持久化的差别。")
    print("- 思考：为什么'每步 checkpoint'让扩缩容、抢占、重启都变得安全？")
    print("=" * 60)


if __name__ == "__main__":
    main()
