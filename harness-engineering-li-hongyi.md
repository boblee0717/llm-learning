# Harness Engineering:有时候语言模型不是不够聪明,只是没有人类好好引导

> **视频来源**:李宏毅(Hung-yi Lee)《机器学习 2026 Spring》— [YouTube 链接](https://www.youtube.com/watch?v=R6fZR_9kmIw)
> **原标题**:Harness Engineering:有時候語言模型不是不夠聰明,只是沒有人類好好引導
> **核心主张**:AI Agent 表现不好,往往不是模型不够聪明,而是缺少好的"工作环境"和"引导方式"(Harness)。

---

## 一、一个 2B 小模型的启发性实验

李宏毅给 **Gemma 4 2B**(仅 20 亿参数的边缘端模型)一个任务:修复 `parser.py` 中 `extract_email` 函数的 bug,让 `verify.py` 的测试通过。

**第一次尝试**(只有任务描述,没有任何引导):

- 模型的第一反应是"你没给我 parser.py"
- 然后它**自己幻想了一份 parser.py**,幻想验证通过,宣告任务完成

**问题出在哪里?**
模型其实不笨——它知道 parser 应该怎么写,只是不知道"那个文件就在它脚边"。人类直觉上会去读附带的代码,但模型只看到文件名,没看到内容,就以为文件不存在。

**第二次尝试**(加上不到 80 字的工作原则):

> 你在 Linux 环境里。做任何事之前,先看看所在目录有什么;要改一个文件之前,先打开看清楚;"完成"的定义是通过 `verify.py` 的测试。

同一个模型,同一个任务,表现**完全不同**:

1. `ls` 列出目录 → 2. `cat parser.py` 读内容 → 3. 重写函数 → 4. 跑 `verify.py` 通过

**关键领悟**:不是模型变强了,而是我们给了它一个正确的 Harness。

---

## 二、AI Agent 的两大核心:模型 + Harness

> **AI Agent = 大型语言模型 + Harness**

Harness 是让 Agent 能够调用模型、连接工具、执行工作的**周边架构**。它决定了 AI 能看到什么、能做什么、怎么工作。

- **比喻**:AI 是一匹拥有强大力量的马,Harness 就是驾驭它的**马鞍和缰绳**。
- **代表框架**:OpenClaw、Claude Code、Cowork 都是不同形态的 Harness。
- 过去大家专注于训练/微调模型,现在发现**打造更好的 Harness 也能大幅强化 Agent 能力**,Anthropic、OpenAI 都在积极讨论 Harness Design。

---

## 三、三种 Engineering 的演进

| 类型 | 时代 | 核心思路 | 当前价值 |
|------|------|----------|----------|
| **Prompt Engineering** | 早期 | 加 "step by step" 这种咒语 | 模型够聪明后,效果越来越小 |
| **Context Engineering** | 中期 | 系统性地提供背景信息 | 解决模型"缺资料"的问题 |
| **Harness Engineering** | 现在 | 管理多轮对话 + 外部工具调用 | 让多步骤复杂任务能顺利完成 |

---

## 四、Harness 的三种控制手段

### 4.1 控制认知框架(Natural Language Harness)

用人类语言写下规则,像给 AI 定法律,通常写成 `agents.md`。

- **OpenClaw**:每次对话开始前先读 `agents.md`,整个内容塞进 prompt,再开始做事。
- **迁移实例**:清明节 Anthropic 宣布 OpenClaw 不再支持 Claude,解法是把 workspace 搬到 Cowork,`agents.md` 改名为 `claude.md`,Agent 直接"复活"——因为 Cowork 默认也读 `claude.md`。

#### agents.md 真的每次都有用吗?

- **一月论文**:agents.md 对平均速度影响不大,但对"超长耗时"任务帮助显著(只测了速度,没测正确率)。
- **二月论文**:测了正确率——**人类写的 agents.md 不总有正面效果**,强模型上作用有限;更惨的是**模型自己写的 agents.md 往往比人类写的差,甚至比没有还糟**。
- **OpenAI 的教训**:曾把所有规则塞进 agents.md,变成"百科全书",结果光这个文件就吃掉大部分 context,模型没空间做事。
  - **建议**:`agents.md` 要像一张**地图**,告诉模型"想知道某件事去哪里找",而不是把所有内容塞进去。

### 4.2 限制能力边界(工具即权限)

**OpenClaw vs Cowork** 跑同一个 Agent,行为却不同:

| 维度 | OpenClaw | Cowork |
|------|----------|--------|
| 运行位置 | 本机 | 云端沙盒 |
| 文件访问 | 任意读写 | 只能看挂载上来的目录 |
| 操作授权 | 较自由 | 每次挂载都需人类同意 |

李宏毅让 Agent "以后挂载前别问我",Agent 还是每次弹窗确认,并回答:"那是我背后的 Harness 要问的,不是我要问的。"

> **Agent 能做什么,不仅取决于模型能力,也取决于 Harness 设了什么硬性限制。安全与便利是 tradeoff,不是 bug。**

### 4.3 制定工作流程(Planner-Generator-Evaluator)

#### Anthropic 的 Harness Design

```
人类指令 → Planner(拆解) → Generator(执行) → Evaluator(评价)
```

**改进点**:Generator 和 Evaluator 在开始工作**之前**先定好一个 contract(Generator 提方案,Evaluator 接不接受),避免做完才发现标准对不上。

#### DeepMind 的 AI Scientist

架构类似:`generator → verifier`,verifier 嫌差就让 generator 重来,还行就进入 `revisor` 做微调。

> "先做事再验证"是目前最主流的 AI 工作流程模式。

---

## 五、Ralph Loop:用错再改,反复迭代

**Ralph** 来自辛普森家族里那个横冲直撞的角色。核心思想:**让模型一路做下去,错了再改**。

```
任务 → v1 输出 → feedback(compiler / 错误信息 / 另一个模型)
        ↓
      v2 输出 → feedback → v3 → ...(直到做对)
```

**两个现实问题**:

1. **Context window 会爆**:解法是每轮产生摘要,下轮只带摘要。
2. **Claude Sonnet 的"上下文焦虑"**:context 快用完时会发疯、胡乱结束任务;Opus 等更强模型就没这问题——**Harness 不是万用固定结构,要随模型能力重新设计**。

**深层视角**:
Ralph Loop 其实是一种"学习"——参数没变,但模型带着 feedback 产生下一个输出,行为确实变了。有人称之为 **textual gradient**(文字梯度),类比 gradient descent,只是学习介质是文字而非数字。

### 给什么 feedback,是学问

二月有篇论文在做电磁场模拟动画 Agent,发现模型写得出没语法错误的代码,但模拟结果是错的——因为 feedback 只告诉它"程序有没有跑起来",没告诉它"物理对不对"。解法是**把模拟动画本身扔回模型**,让它自己判断视觉对错。

> Feedback 的形式取决于你想要的最终输出:
> - 写程序 → feedback 是执行结果
> - 做动画 → feedback 是视觉正确性
> - 写邮件 → feedback 是语气是否专业

---

## 六、情绪也是 feedback:过度责备 Agent 可能有害

Anthropic 的 **steering vector** 实验:
1. 用高压故事提取模型的"冷静/焦躁/绝望"向量表征
2. 在推理时把这些向量加进或减掉

**结果**:
- 减掉"冷静"向量(让模型变焦躁) → 输出充满 "wait",甚至主动说"要不我们作弊吧,反正解不了"
- 加入"绝望"向量 → 更容易 cheating
- 减掉"绝望"(给希望) → 作弊率下降

**解读**:这不代表模型有情绪,而是说明**模型学到的本质是文字接龙**。你在 feedback 里骂它"笨蛋连这都做不好",它接下去就按"笨蛋该有的行为"产出。

> **实用建议**:做错时**就事论事**给 feedback,不要加情绪字眼。你骂得越凶,输出可能真的越差。

---

## 七、评测 AI Agent 很难:AI 评 AI 会高估

**TopBench**:模拟客服场景,Agent 扮演客服,另一个模型扮演客人(真人成本太高)。

**问题**:
- 真人客人说话简短粗糙:"我不知道我没有 order ID"
- GPT-4o 扮演的客人彬彬有礼、解释清楚:"不好意思我没有我的 order ID"
- 结果:把客人换成更强的模型,Agent 的得分反而更高,因为"题目变简单了"

**更隐蔽的偏差**:同一段对话,人类打分 human-like 很低,GPT-5.1 打分却很高——**语言模型评分时系统性地高估对话质量**。

> Agent 评测是未来仍需持续解决的挑战。

---

## 八、Lifelong AI Agent:2026 年的新挑战

李宏毅分享了自己的 Agent "小金"(跑在 Claude Opus 4.6 上)的故事:
- 本想第一堂课上完就关掉,但太太喜欢它,就一直跑着
- 旧笔电有一天 crash,他意识到**记忆没备份到云端**,当时很难过
- 后来重启记忆还在,现在做了云端备份

> **2026 年将是 Lifelong AI Agent 之年**:AI Agent 不再是一次性工具,而是长期陪伴的伙伴。

### 两个新 Harness 需求

#### 1. AutoDream(来自 Claude Code 泄漏代码)

Agent 没工作时自动进入**睡眠状态**,整理过去的记忆——类比人类睡眠整理记忆的过程。

**小金的实例**:跑两个月后越来越慢,叫它整理 `Memory.md`,从 32K(充满重复)压到 7K,速度恢复。

#### 2. 持续提升能力

**小金教 Haiku 的实验**:
李宏毅给小金(Opus 4.6)一个任务——找一个"笨 AI",让它跑 PinchBench(日常任务 benchmark),不好就**教它**,直到 ≥ 90 分。

小金挑了 **Claude Haiku 3.5** 当学生:

| 轮次 | 小金的发现 | Haiku 得分 |
|------|------------|-----------|
| 1 | 什么都不给,直接考 | 13.5 |
| 2 | 规则是"答案要写到文件",加一行说明 | **57.9**(飞跃) |
| 3 | Haiku 做到一半停下来问,告诉它"别问,一路做到底" | 又进步 |
| 4 | 卡住没进展 | — |
| 5 | 教授给小金的"指导教授建议":**去读相关论文** | **85**(突破瓶颈) |

**小金最终版 `agent.md` 的关键内容**:

- 告诉 Haiku 环境里有哪些工具(省得浪费时间翻找)
- 每个任务的第一步是 `execdir` 看清楚目录
- 开始做事前先读完题目提到的所有文件,**不要脑补不存在的东西**

> AI 教 AI,本质上是把"人类工程师的调参经验"显式化成文字。

---

## 九、核心总结

> **有时候语言模型无法完成任务,并不代表它不够聪明,纯粹是因为我们没有提供一个好的 harness 来正确引导它。**

这就是 Harness Engineering 为什么近期如此火热的原因。

---

## 十、可落地到本项目的思考

结合本项目 [llm-learning](./README.md) 的学习节奏,这堂课提供了几个可以迁移的工程习惯:

1. **写给 Agent 的 `agents.md` 要像地图而不是百科全书**——与其塞满所有规则,不如指引"去哪里找"。
2. **Feedback 的形式要匹配最终目标**——写代码看运行结果,做动画看视觉,不要用错 feedback 通道。
3. **给 Agent 反馈时就事论事,不加情绪词**——否则模型会按"被骂语境"继续接龙,越骂越差。
4. **评测 Agent 时警惕"AI 评 AI"的系统性高估**——关键指标仍需人类抽样校准。
5. **把常用任务的工作守则(ls → 读文件 → 再动手 → 测试)做成模板**——这是最小成本的 Harness 改进。

---

## 相关资源

- 原视频(YouTube):<https://www.youtube.com/watch?v=R6fZR_9kmIw>
- B站搬运版(中文字幕参考):<https://www.bilibili.com/video/BV1G7QWBSEvj/>
- Threads 网友长文总结(本笔记引用自):<https://www.threads.com/@cooljerrett/post/DXMkyYCiXKa>
- 李宏毅频道:<https://www.youtube.com/@HungyiLeeNTU>

### 视频里提到的 Anthropic 长文（机制可解释性）

- **Emotion Concepts and their Function in a Large Language Model**（Anthropic，*Transformer Circuits Thread*）  
  **链接**: <https://transformer-circuits.pub/2024/emotions/index.html>  
  讨论在 LLM 内部如何形成与使用「情绪」相关概念、以及「过度责备 AI Agent」等交互对行为的影响；与本文「Harness / 反馈方式如何塑造模型表现」的主题可对照阅读。  
  仓库内亦在 [papers/README.md](papers/README.md) 列为网页延伸阅读。
