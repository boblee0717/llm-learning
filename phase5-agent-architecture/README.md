# 第五阶段：Agent 架构与 Agent 基础设施

> 从「会聊天的模型」走到「会做事的系统」。本阶段把 Agent 拆成两条线一起学：**怎么搭一个 Agent（应用范式）** 和 **怎么把 Agent 跑稳、跑快、跑省（基础设施）**。

面向 Agent infra 方向打基础。目标不是会用某个框架，而是建立一张地图：一个请求从用户进来，经过规划、工具调用、记忆、上下文管理，最后稳定返回，这中间每一层在解决什么瓶颈、付出什么代价。

本阶段刻意**应用与 infra 均衡**：

- **Agent 应用范式**：ReAct、Planning、Reflexion、Tool Use、Memory、RAG、Multi-Agent 编排、Context/Prompt 工程。让你知道「一个 Agent 是怎么被搭出来的」。
- **Agent 基础设施**：Agent runtime、状态管理、工具沙箱、并发与调度、KV/prefix cache 复用、流式、可观测性、成本与延迟、Agent serving。让你知道「线上跑 100 万个 Agent 会卡在哪」。

## 前置要求

完成第二阶段（Transformer/GPT）即可开始，但建议先了解第三、四阶段的内容，因为 infra 部分会用到：

- GPT 自回归生成、KV Cache、采样策略（第三阶段第 5 课、第四阶段第 3-6 课）
- 推理服务概念：TTFT、TPOT、throughput、PagedAttention、prefix cache、continuous batching（第四阶段第 5 课）
- function calling / tool use 的基本直觉（本阶段第 2 课会补）

如果你只学完第二阶段，先看第 1-4 课（应用范式），第 6-10 课（infra）建议补完第四阶段第 5 课后再深入。

## 快速索引

| 入口 | 用途 |
|------|------|
| [课程结构](#course-structure) | 10 课主线，一课一个工程问题 |
| [学习节奏](#rhythm) | 沿用前面阶段「读 → 跑代码 → 复盘」节奏 |
| [动手代码](#code) | 本阶段配套的可运行 `.py` 文件 |
| [每课详细大纲](#outline) | 每课的一句话目标、核心内容、材料、动手任务 |
| [必要学习材料](#materials) | 必读论文、博客、框架文档 |
| [实践项目](#projects) | 学完后能做的 4 个项目 |
| [两条线对照表](#two-lines) | 应用范式 ↔ infra 的映射，面试/onboard 速记 |

<a id="course-structure"></a>

## 课程结构

按顺序学习，每课约 90-150 分钟。第 1-5 课偏 Agent 应用范式，第 6-10 课偏 Agent infra。

| 课程 | 主题 | 核心问题 | 关键概念 | 实战训练 |
|------|------|----------|----------|----------|
| 第 1 课 | Agent 总览与 Agent Loop | 什么让一个 LLM 调用变成一个「Agent」？ | perception-plan-act、ReAct、agent loop、终止条件 | 跑通最小 agent loop |
| 第 2 课 | Tool Use / Function Calling | 模型怎么「调用」外部世界？谁来执行？ | tool schema、function calling、parse/dispatch、错误回灌 | 实现 tool 注册与分发 |
| 第 3 课 | 规划与反思 | 任务很长时，怎么不让模型走丢？ | ReAct、Plan-and-Execute、Reflexion、self-critique | 给 agent 加 plan 与 retry |
| 第 4 课 | 记忆系统 | 上下文窗口装不下历史怎么办？ | short-term vs long-term、向量检索、记忆写入/召回 | 接一个最小 memory store |
| 第 5 课 | RAG 与上下文工程 | 怎么把对的信息、在对的时刻塞进上下文？ | chunking、embedding、retrieval、context assembly、context rot | 做一个最小 RAG 管线 |
| 第 6 课 | Agent Runtime 与状态管理 | 一个 agent 在线上是什么形态？崩了怎么续？ | run state、step、checkpoint/resume、graph、durable execution | 给 runtime 加状态与重放 |
| 第 7 课 | 工具沙箱与安全 | 让模型执行代码/命令，怎么不出事？ | sandbox、权限边界、超时、注入、人审 | 给 tool 加超时与权限白名单 |
| 第 8 课 | 并发、调度与上下文复用 | 同时跑很多 agent，GPU 和上下文怎么省？ | continuous batching、prefix/KV cache 复用、调度、限流 | 估算 agent 场景的 cache 命中 |
| 第 9 课 | 可观测性、评测与成本 | agent 答错/变慢/烧钱了，怎么定位？ | tracing、span、token/latency/cost 核算、agent eval | 给 runtime 加 trace 与成本统计 |
| 第 10 课 | Multi-Agent 编排与框架对比 | 单 agent 不够时，多个 agent 怎么协作？ | orchestrator-worker、handoff、共享状态、框架对比 | 用现有框架复现自己的 runtime |

<a id="rhythm"></a>

## 学习节奏

每课沿用前面阶段的架构：

| 步骤 | 做什么 | 产出 |
|------|--------|------|
| 1 | 先读本课「一句话目标」和关键概念 | 明确这节课解决哪个瓶颈 |
| 2 | 读对应的论文/博客/框架文档 | 建立概念主线 |
| 3 | 对照本阶段 `.py` 代码或开源框架源码 | 把概念落到实现位置 |
| 4 | 改一处代码或画一张流程图 | 不只会背名词 |
| 5 | 写 3 条复盘 | 记录「瓶颈 → 技术 → 代价」 |

复盘固定回答：

1. 这个技术优化的是 **正确率、延迟、成本、可控性、可靠性** 中哪一个？
2. 它属于 **应用范式（prompt/loop/编排）** 还是 **infra（runtime/调度/缓存/沙箱/观测）**？
3. 它牺牲了什么：实现复杂度、token 成本、延迟、安全边界，还是通用性？

<a id="code"></a>

## 动手代码

本阶段配套可运行的 `.py` 文件，**无需 GPU、无需联网、无需 API key**——用一个可替换的「假 LLM」把 Agent 的骨架跑通，重点在结构而非模型质量。等你理解了骨架，再把假 LLM 换成真实的 function-calling 模型即可。

| 文件 | 对应课程 | 内容 |
|------|----------|------|
| `01_minimal_agent_loop.py` | 第 1 课 | 最小 agent loop：observe → think → act → observe，含终止条件 |
| `02_tool_calling.py` | 第 2 课 | tool 注册表、schema、parse/dispatch、把工具结果回灌给模型 |
| `03_agent_runtime.py` | 第 6 课 | 带 run state、step、checkpoint/resume 的最小 agent runtime |

运行方式：

```bash
python3 phase5-agent-architecture/01_minimal_agent_loop.py
python3 phase5-agent-architecture/02_tool_calling.py
python3 phase5-agent-architecture/03_agent_runtime.py
```

每个文件顶部都有一个 `FakeLLM`，它用规则模拟「模型决定调用哪个工具 / 何时停止」。把它读懂，你就理解了所有 agent 框架的核心循环。

<a id="outline"></a>

## 每课详细大纲

### 第 1 课：Agent 总览与 Agent Loop

**一句话目标**：理解 Agent 的本质是一个「带终止条件的循环」，而不是一次性的 prompt。

**核心内容**

- 从 chat 到 agent：单次 completion → 多轮「思考-行动-观察」循环。
- Agent loop 的最小骨架：`观察(observation) → 思考(LLM) → 行动(tool/answer) → 新观察`。
- ReAct 的直觉：让模型显式交替输出 Thought / Action / Observation。
- 终止条件：模型主动给出最终答案、达到最大步数、超时、报错。
- Agent ≈ LLM(大脑) + 工具(手) + 记忆 + 循环控制 + 环境反馈。

**材料**

- [ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629)
- [Anthropic - Building Effective Agents](https://www.anthropic.com/research/building-effective-agents)
- 复习：项目内 [harness-engineering-li-hongyi.md](../harness-engineering-li-hongyi.md)（agent-first、反馈回路）

**动手任务**

- 运行 `01_minimal_agent_loop.py`，画出它的状态流转图。
- 改一改终止条件（最大步数、是否要求显式 `FINAL:` 标记），观察行为变化。
- 复盘：为什么「循环 + 终止条件」是 agent 与普通 prompt 的根本区别？

### 第 2 课：Tool Use / Function Calling

**一句话目标**：理解模型只是「决定调用哪个工具、传什么参数」，真正执行工具的是你的 runtime。

**核心内容**

- Tool schema：name、description、参数 JSON Schema——模型靠 description 决定何时用。
- Function calling 协议：模型输出结构化的 tool 调用请求，runtime 解析并分发。
- 执行与回灌：runtime 执行工具 → 把结果作为新的 observation 塞回上下文 → 模型继续。
- 错误处理：工具报错、参数非法、超时，都要变成模型能读懂的 observation。
- 谁负责什么：模型负责「选择与填参」，runtime 负责「校验、执行、回灌、安全」。

**材料**

- [OpenAI Function Calling 文档](https://platform.openai.com/docs/guides/function-calling)
- [Anthropic Tool Use 文档](https://docs.anthropic.com/en/docs/build-with-claude/tool-use)
- [Model Context Protocol (MCP) 介绍](https://modelcontextprotocol.io/introduction)

**动手任务**

- 运行 `02_tool_calling.py`，看 tool 注册表与分发逻辑。
- 新增一个工具（比如 `word_count`），注册进去并让 agent 用上。
- 故意让工具抛异常，确认错误能作为 observation 回灌而不是让整个 loop 崩。
- 复盘：tool 的 description 写得好不好，如何直接影响模型的调用准确率？

### 第 3 课：规划与反思

**一句话目标**：任务一长，单步反应式 agent 会走丢；用规划和反思把长任务结构化。

**核心内容**

- 反应式（ReAct）的局限：每步只看当前，缺乏全局计划，容易绕圈、重复。
- Plan-and-Execute：先让模型产出一个 step 列表，再逐步执行，可中途重规划。
- Reflexion / self-critique：执行失败后让模型反思、生成改进策略再重试。
- 何时该规划：步骤多、有依赖、可验证的任务收益大；简单问答不需要。
- 代价：规划增加 token 与延迟；过度反思会陷入「想太多」。

**材料**

- [Plan-and-Solve Prompting](https://arxiv.org/abs/2305.04091)
- [Reflexion: Language Agents with Verbal Reinforcement Learning](https://arxiv.org/abs/2303.11366)
- [LangGraph - Plan-and-Execute 教程](https://langchain-ai.github.io/langgraph/tutorials/plan-and-execute/plan-and-execute/)

**动手任务**

- 在 `01_minimal_agent_loop.py` 基础上，加一个「先列计划再执行」的变体。
- 给 agent 加一个简单 retry：工具失败后带着错误信息重试一次。
- 复盘：你的任务真的需要 planning 吗？什么情况下 ReAct 就够了？

### 第 4 课：记忆系统

**一句话目标**：上下文窗口有限，agent 需要把「该记住的」写出去、用时再召回。

**核心内容**

- short-term memory：当前对话/任务的上下文，受窗口大小限制。
- long-term memory：跨会话持久化，常用向量库做语义检索。
- 写入策略：什么值得记（事实、偏好、任务进展），怎么压缩（摘要、要点）。
- 召回策略：按相关性检索 top-k，注意召回噪声会污染上下文。
- 记忆 ≠ 全量历史：盲目把所有历史塞进上下文会拖慢、变贵、变差。

**材料**

- [MemGPT: Towards LLMs as Operating Systems](https://arxiv.org/abs/2310.08560)
- [Generative Agents (斯坦福小镇)](https://arxiv.org/abs/2304.03442)
- [LangGraph Memory 概念文档](https://langchain-ai.github.io/langgraph/concepts/memory/)

**动手任务**

- 给 `03_agent_runtime.py` 加一个内存版 memory store（dict 即可）：写入与按 key 召回。
- 思考一个「摘要压缩」策略：当历史超过 N 条就压成一段摘要。
- 复盘：short-term 和 long-term 各解决什么问题？为什么不能只靠加大窗口？

### 第 5 课：RAG 与上下文工程

**一句话目标**：Agent 的质量上限，常常取决于「上下文里放了什么」，而不是模型本身。

**核心内容**

- RAG 管线：chunk → embed → 存向量库 → 检索 top-k → 拼进 prompt → 生成。
- chunking 策略：粒度太大召回不准，太小丢失语境。
- context assembly：系统提示、工具结果、检索片段、历史，如何排布与裁剪。
- context rot / lost-in-the-middle：上下文太长、关键信息在中间会被忽略。
- 上下文工程是 infra 与应用的交界：既是 prompt 设计，也是检索/缓存系统。

**材料**

- [Retrieval-Augmented Generation (原始 RAG 论文)](https://arxiv.org/abs/2005.11401)
- [Lost in the Middle](https://arxiv.org/abs/2307.03172)
- [Anthropic - Effective Context Engineering for AI Agents](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents)

**动手任务**

- 写一个最小 RAG：把几段文本切块、用简单 TF-IDF 或关键词匹配检索（先不引入重依赖）。
- 对同一个问题，比较「不给上下文 / 给对的上下文 / 给一堆噪声上下文」三种结果。
- 复盘：为什么说「检索质量」常比「模型大小」更影响 agent 表现？

### 第 6 课：Agent Runtime 与状态管理

**一句话目标**：线上的 agent 不是一个函数调用，而是一个有状态、可中断、可恢复的长流程。

**核心内容**

- run / step 抽象：一次 agent 任务是一个 run，由多个 step 组成。
- 状态包含什么：消息历史、当前计划、工具结果、step 计数、状态机阶段。
- checkpoint / resume：每个 step 后持久化状态，崩溃后能从断点续跑（durable execution）。
- graph 表示：把 agent 流程建模成节点+边（LangGraph、状态机），便于控制与可视化。
- 为什么 infra 关心这个：长任务可能跑几分钟到几小时，进程重启、扩缩容、抢占都需要状态可恢复。

**材料**

- [LangGraph 概念：Persistence / Checkpointing](https://langchain-ai.github.io/langgraph/concepts/persistence/)
- [Temporal - Durable Execution 概念](https://docs.temporal.io/temporal)（agent 编排常借鉴工作流引擎）
- [OpenAI Agents SDK 文档](https://openai.github.io/openai-agents-python/)

**动手任务**

- 运行 `03_agent_runtime.py`，看它如何把状态在每步落盘并能 resume。
- 故意在第 2 步「杀掉」（抛异常），然后从 checkpoint 重新 resume，确认能续上。
- 复盘：为什么有状态 + 可恢复，是 agent infra 和 demo 脚本最大的区别？

### 第 7 课：工具沙箱与安全

**一句话目标**：让模型执行代码、命令、网络请求时，必须假设它会出错或被攻击。

**核心内容**

- 工具风险等级：只读查询 < 写操作 < 执行代码/命令 < 访问外部网络/支付。
- 沙箱：容器/子进程/受限解释器，限制文件、网络、CPU、内存。
- 超时与配额：每个工具调用都要有超时和资源上限。
- prompt injection：检索内容或工具结果里藏的指令可能劫持 agent。
- 人审（human-in-the-loop）：高危操作前插入确认步骤。

**材料**

- [OWASP Top 10 for LLM Applications](https://owasp.org/www-project-top-10-for-large-language-model-applications/)
- [Simon Willison - Prompt injection 系列](https://simonwillison.net/series/prompt-injection/)
- [Anthropic - Tool use 安全建议](https://docs.anthropic.com/en/docs/build-with-claude/tool-use)

**动手任务**

- 给 `02_tool_calling.py` 的工具执行加上超时和「只允许白名单工具」的检查。
- 设计一个「危险工具需人审」的开关：执行前打印确认提示。
- 复盘：哪些工具应该默认禁止？为什么不能信任检索/工具返回的文本？

### 第 8 课：并发、调度与上下文复用

**一句话目标**：单个 agent 跑通不难，难的是同时跑成千上万个还省 GPU。

**核心内容**

- agent 负载特点：一个 run 内有多次 LLM 调用，长 prompt（系统提示+工具描述+历史）反复出现。
- prefix / KV cache 复用：相同系统提示、相同历史前缀，可复用 KV，省 prefill（接第四阶段第 5 课）。
- continuous batching：把不同 agent 的 decode step 动态拼批，提升 GPU 利用率。
- 调度与限流：每个 run 占多个并发槽，需要排队、优先级、超时回收。
- 成本结构：agent 场景 token 消耗远大于单轮 chat，prefix 复用与缓存命中率直接决定成本。

**材料**

- [PagedAttention / vLLM](https://arxiv.org/abs/2309.06180)（复习第四阶段第 5 课）
- [SGLang RadixAttention](https://docs.sglang.io/)（前缀复用对 agent 尤其关键）
- [DeepSeek API Context Caching](https://api-docs.deepseek.com/guides/kv_cache)

**动手任务**

- 估算一个典型 agent run：系统提示 1.5K token + 工具描述 1K + 每步追加 0.5K，跑 6 步，总 prefill/decode token 大致是多少。
- 算一笔账：如果系统提示+工具描述能被 prefix cache 命中，省下多少重复 prefill。
- 复盘：为什么「稳定不变的前缀」对 agent infra 的成本影响这么大？

### 第 9 课：可观测性、评测与成本

**一句话目标**：agent 是黑盒长流程，没有 trace 就无法定位它为什么慢/错/贵。

**核心内容**

- tracing / span：把一个 run 拆成可视化的 step span（每次 LLM 调用、每次工具调用）。
- 关键指标：每 run 的总 token、总延迟、step 数、工具调用次数、成功率、成本。
- agent 评测难点：没有唯一正确答案，常用 task success rate、轨迹评估、LLM-as-judge。
- 失败归因：模型选错工具、参数错、规划错、检索差、超时、循环不终止。
- 成本核算：把 token × 单价 × 调用次数累加到 run 级别，做成 dashboard。

**材料**

- [LangSmith / Tracing 概念](https://docs.smith.langchain.com/)
- [OpenTelemetry for LLM/GenAI 语义约定](https://opentelemetry.io/docs/specs/semconv/gen-ai/)
- [τ-bench: agent 任务评测](https://arxiv.org/abs/2406.12045)

**动手任务**

- 给 `03_agent_runtime.py` 加一个简单 trace：每步记录耗时、（假）token 数、工具名。
- 在 run 结束时打印一份小报告：总 step、总 token、总耗时、是否成功。
- 复盘：如果一个 agent 偶尔答错，你会先看 trace 里的哪一层？

### 第 10 课：Multi-Agent 编排与框架对比

**一句话目标**：当单 agent 上下文过载或职责太杂时，用多个专职 agent 协作。

**核心内容**

- 何时需要 multi-agent：职责分离（规划/执行/审查）、并行子任务、上下文隔离。
- 编排模式：orchestrator-worker（主管派活）、handoff（移交）、辩论/投票。
- 共享状态 vs 消息传递：子 agent 之间怎么交换信息，谁持有全局状态。
- 代价：多 agent 协调开销大、token 翻倍、调试更难——不要为了多而多。
- 框架对比：把前面自己写的 runtime 概念，对应到主流框架。

| 框架 | 定位 | 核心抽象 | 适合 |
|------|------|----------|------|
| LangGraph | 有状态 agent 图 | graph、node、checkpoint | 需要持久化、复杂控制流 |
| OpenAI Agents SDK | 轻量 agent + handoff | agent、tool、handoff、guardrail | 快速搭多 agent |
| AutoGen | 多 agent 对话 | conversable agent、group chat | 多 agent 协作研究 |
| CrewAI | 角色化多 agent | crew、role、task | 角色分工明确的流程 |
| 自写 runtime | 学习/可控 | 自定义 loop+state | 理解原理、定制 infra |

**材料**

- [Anthropic - Building Effective Agents（multi-agent 部分）](https://www.anthropic.com/research/building-effective-agents)
- [AutoGen 文档](https://microsoft.github.io/autogen/)
- [LangGraph Multi-Agent 文档](https://langchain-ai.github.io/langgraph/concepts/multi_agent/)

**动手任务**

- 选一个框架（推荐先 LangGraph 或 OpenAI Agents SDK），用它复现你在第 1-6 课写的 loop+state+tool。
- 对比：框架替你做了哪些事（状态、重试、并发、trace）？你自己写时漏了什么？
- 复盘：你手上的真实任务，是该用单 agent 还是 multi-agent？理由是什么？

<a id="materials"></a>

## 必要学习材料

### A. Agent 范式（应用线）

| 优先级 | 材料 | 读法 |
|--------|------|------|
| 必读 | [ReAct](https://arxiv.org/abs/2210.03629) | 抓「思考-行动-观察」交替 |
| 必读 | [Anthropic - Building Effective Agents](https://www.anthropic.com/research/building-effective-agents) | 工程视角的 agent 模式总览 |
| 必读 | [Reflexion](https://arxiv.org/abs/2303.11366) | 看失败反思与重试 |
| 选读 | [Plan-and-Solve](https://arxiv.org/abs/2305.04091) | 看显式规划 |
| 选读 | [Toolformer](https://arxiv.org/abs/2302.04761) | 看模型如何学会调用工具 |

### B. 记忆与上下文（交界线）

| 优先级 | 材料 | 读法 |
|--------|------|------|
| 必读 | [MemGPT](https://arxiv.org/abs/2310.08560) | 把上下文当作分页内存来管理 |
| 必读 | [Lost in the Middle](https://arxiv.org/abs/2307.03172) | 理解长上下文的注意力衰减 |
| 必读 | [Anthropic - Effective Context Engineering](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents) | 上下文工程的工程实践 |
| 选读 | [RAG 原始论文](https://arxiv.org/abs/2005.11401) | 检索增强的起点 |

### C. Agent 基础设施（infra 线）

| 优先级 | 材料 | 读法 |
|--------|------|------|
| 必读 | [LangGraph Persistence](https://langchain-ai.github.io/langgraph/concepts/persistence/) | 看 checkpoint/resume |
| 必读 | [PagedAttention / vLLM](https://arxiv.org/abs/2309.06180) | 复习 KV/prefix 复用 |
| 必读 | [Model Context Protocol](https://modelcontextprotocol.io/introduction) | 工具/资源接入的标准协议 |
| 选读 | [Temporal Durable Execution](https://docs.temporal.io/temporal) | 工作流引擎对 agent 编排的启发 |
| 选读 | [OpenTelemetry GenAI](https://opentelemetry.io/docs/specs/semconv/gen-ai/) | agent 可观测性标准 |

### D. 安全与评测

| 优先级 | 材料 | 读法 |
|--------|------|------|
| 必读 | [OWASP Top 10 for LLM Apps](https://owasp.org/www-project-top-10-for-large-language-model-applications/) | agent 安全清单 |
| 必读 | [Prompt injection 系列](https://simonwillison.net/series/prompt-injection/) | 理解注入攻击面 |
| 选读 | [τ-bench](https://arxiv.org/abs/2406.12045) | agent 任务评测方法 |

<a id="projects"></a>

## 实践项目

### 项目 1：从零写一个带工具的 Agent

在本阶段 `01`/`02` 代码基础上，做一个能调用 2-3 个真实工具（计算器、查天气 mock、读文件）的 agent，跑通完整 loop。

**目标**：彻底理解 agent loop + tool dispatch。

**验收标准**

- 能正确选择并调用工具，把结果回灌后给出最终答案。
- 工具报错时不崩，错误作为 observation 回灌。
- 有最大步数与超时保护。

### 项目 2：可恢复的 Agent Runtime

在 `03` 基础上，做一个支持 checkpoint/resume 的 runtime：每步落盘状态，进程被杀后能从断点续跑。

**目标**：理解有状态、可恢复的 agent 是 infra 的核心。

**验收标准**

- 状态序列化到文件/sqlite。
- 中途中断后能 resume 并完成任务。
- 有一份 run 级别的 trace（step、耗时、工具、token）。

### 项目 3：最小 RAG + 上下文对照实验

做一个最小 RAG，针对同一组问题，对比「无上下文 / 精准上下文 / 噪声上下文」的回答质量。

**目标**：用数据说明上下文质量对 agent 的影响。

**验收标准**

- 能切块、检索、拼接上下文。
- 至少 5 个问题、3 种上下文配置的对照结果。
- 一段复盘：什么时候检索帮了倒忙。

### 项目 4：成本与缓存账本

针对一个典型 agent 任务，估算 token、延迟、成本，并分析 prefix cache 命中能省多少。

**目标**：建立 agent infra 的成本直觉。

**验收标准**

- 拆出系统提示、工具描述、历史、检索各占多少 token。
- 算出「有/无 prefix cache」两种情况下的 prefill 成本差。
- 一页结论：这个任务的成本瓶颈在哪、怎么优化。

<a id="two-lines"></a>

## 两条线对照表（onboard / 面试速记）

面对任何一个 agent 概念，先把它归到「应用范式」还是「infra」，再说清它优化什么、代价是什么。

| 应用范式问题 | 对应 infra 问题 |
|--------------|-----------------|
| agent 怎么决定调哪个工具 | 工具怎么注册、分发、沙箱、限流 |
| 怎么让 agent 记住历史 | memory 存哪、怎么检索、怎么压缩、怎么持久化 |
| 怎么把对的信息塞进上下文 | 检索系统、向量库、prefix cache、上下文裁剪 |
| 怎么让 agent 规划长任务 | run/step 状态机、checkpoint/resume、durable execution |
| 怎么让多个 agent 协作 | 编排调度、共享状态、消息传递、并发 |
| 怎么知道 agent 答得好不好 | tracing、eval、成本/延迟核算、dashboard |

## 完成后你将理解

- Agent 的本质是「带终止条件、有状态、能调用工具的循环」，而不是一次 prompt。
- function calling 中，模型只负责「选工具填参数」，执行与安全由 runtime 负责。
- 记忆、RAG、上下文工程为什么常比换更大模型更影响 agent 表现。
- 线上 agent runtime 为什么必须有状态、可恢复、可观测。
- agent 场景下 prefix/KV cache 复用为什么直接决定成本。
- 面对一个新 agent 框架或需求，能用「应用范式 vs infra」「正确率/延迟/成本/可控/可靠」两套坐标快速定位它在解决什么。
