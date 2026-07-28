# InstructGPT Related Work 精选阅读地图

> 来源：InstructGPT Related Work 中的 RLHF、跨任务指令泛化、对齐目标、语言模型风险与行为修改引用链。
> 原则：**按当前问题定点读，不把所有引用变成全文债**。状态均为「资料已就位、尚未精读」，读完后再在 `learning-progress.md` 记录领悟。

## 先说结论：现在该读哪几篇

按与第三阶段第 3 课的距离分三层：

1. **当前必读：RLHF 方法链**
   Christiano 2017 → Ziegler 2019 → Stiennon 2020 → InstructGPT 2022 → DPO 2023。
   它回答「人类 A/B 选择怎样一路变成能训练语言模型的对齐方法」。
2. **紧接着读：指令微调对照组**
   FLAN + T0。
   它回答「多任务监督微调已经会遵循指令，为什么 InstructGPT 还需要人类偏好」。
3. **DPO 后选读：目标、风险与替代干预**
   Askell + Gabriel + Kenton + Weidinger；Carlini / Xu / PALMS 用作具体案例。
   它回答「对齐到底对齐谁、风险怎样分类、善意干预为何也会产生副作用」。

## Session A：RLHF 方法链（当前执行，约 90 分钟）

| 顺序 | 论文 | 定点范围 | 时间 | 读完必须回答 |
|------|------|----------|------|--------------|
| 1 | [Deep RL from Human Preferences](efficient-training/Deep_RL_from_Human_Preferences_2017.pdf)（Christiano et al., 2017） | Figure 1 + §2.2.3 | 10 分钟 | 人只选 A/B，为什么能拟合标量奖励？ |
| 2 | [Fine-Tuning Language Models from Human Preferences](efficient-training/Fine_Tuning_Language_Models_from_Human_Preferences_2019.pdf)（Ziegler et al., 2019） | Figure 1 + §2（PDF pp. 2–4）；§4.3–4.4（pp. 11–13） | 20 分钟 | 方法怎样从轨迹迁移到文本？标注捷径和实现 bug 为什么会被策略放大？ |
| 3 | [Learning to Summarize from Human Feedback](efficient-training/Learning_to_Summarize_from_Human_Feedback_2020.pdf)（Stiennon et al., 2020） | §3.1 + Figure 2 + §3.4 + §4.3 Figure 5 | 25 分钟 | RM 预测奖励继续上涨时，人类偏好为什么可能下降？KL 在防什么？ |
| 4 | [InstructGPT](core-transformers/InstructGPT_Training_LMs_to_Follow_Instructions_2022.pdf)（Ouyang et al., 2022） | Figure 2 + §3 | 20 分钟 | 单任务摘要的 RLHF 怎样扩展到 broad class of written instructions？三阶段分别吃什么数据？ |
| 5 | 画一张演进图并写四行对照 | `轨迹偏好 → 文本偏好 → 高质量摘要 → 广泛指令 → DPO` | 15 分钟 | 每篇新增了什么，而不是重复了什么？ |

完成 Session A 后，立即回到 [`phase3-training/03_rlhf_self_write.py`](../phase3-training/03_rlhf_self_write.py) TODO-1，再读 [DPO](efficient-training/Direct_Preference_Optimization_2023.pdf) §3–4。PPO 论文只按需回查 §3 式 (7)，不另开全文线。

### 方法链延伸：可扩展监督（25 分钟，非当前阻塞项）

- [Recursively Summarizing Books with Human Feedback](efficient-training/Recursively_Summarizing_Books_with_Human_Feedback_2021.pdf)（Wu et al., 2021）
  - **读法**：Figure 1 + §2.1 / §2.4（PDF pp. 3–7）+ §6.1–6.2（pp. 12–13）。
  - **问题**：当人无法直接评价一本书的完整摘要时，如何把任务递归拆成仍可由人监督的小问题？
  - **安排**：学到 scalable oversight、长上下文评测或过程监督时再读。

## Session B：Instruction Tuning 与 RLHF 的边界（DPO 后，约 45 分钟）

先带着这条区分去读：

| 维度 | Instruction Tuning（FLAN / T0） | RLHF（InstructGPT） |
|------|----------------------------------|---------------------|
| 训练信号 | 多个公开 NLP 任务的输入—目标答案 | 人类示范 + 同一提示下回答排序 |
| 核心作用 | 学会把自然语言 instruction 映射到任务与答案格式 | 在多个可行回答中优化人类更偏好的行为 |
| 泛化重点 | held-out task 的 zero/few-shot 泛化 | helpful / honest / harmless 与用户意图 |
| 主要风险 | 任务与模板分布决定泛化边界 | 奖励代理、标注者偏差、reward hacking |

1. **[FLAN: Finetuned Language Models Are Zero-Shot Learners](instruction-tuning/FLAN_Finetuned_Language_Models_Are_Zero_Shot_Learners_2021.pdf)**（Wei et al., 2021，25 分钟）
   - 读 Figure 1–2 + §2（PDF pp. 1–3）+ §4.1–4.2（pp. 6–8）。
   - 观察三件事：训练任务数量、模型规模、自然语言指令格式怎样共同影响 held-out task。
2. **[T0: Multitask Prompted Training Enables Zero-Shot Task Generalization](instruction-tuning/T0_Multitask_Prompted_Training_Enables_Zero_Shot_Task_Generalization_2021.pdf)**（Sanh et al., 2021，20 分钟）
   - 只读主文 PDF pp. 1–10，重点看 Figure 1、Figure 2–3 与 §6；**跳过约 190 页 prompt 附录**。
   - 对照 FLAN：同一数据集使用多种 prompt template，是否等价于教模型对表述变化保持鲁棒？

读完写一个不超过 100 字的答案：**“Instruction tuning 已经提升指令遵循，RLHF 额外解决的是什么？”**

## Session C：Alignment 目标与风险（完成 DPO 后，约 70 分钟）

这组不是算法实现前置，而是防止把「对齐」误解为一个无歧义标量。

| 论文 | 定点范围 | 时间 | 作用 |
|------|----------|------|------|
| [A General Language Assistant as a Laboratory for Alignment](alignment/A_General_Language_Assistant_as_a_Laboratory_for_Alignment_2021.pdf)（Askell et al., 2021） | §1.1 + Figure 1；§3.1；Appendix E（PDF pp. 44–45） | 25 分钟 | HHH 的直接来源；区分 preference modeling 与 imitation learning，并看 H/H/H 发生冲突时怎么办 |
| [Artificial Intelligence, Values, and Alignment](alignment/Artificial_Intelligence_Values_and_Alignment_2020.pdf)（Gabriel, 2020） | §3（PDF pp. 7–13） | 15 分钟 | 区分 instructions、intentions、revealed / ideal preferences、interests 与 values |
| [Alignment of Language Agents](alignment/Alignment_of_Language_Agents_2021.pdf)（Kenton et al., 2021） | §3.1、§4、§5.4 | 15 分钟 | 把目标错设、欺骗/操纵、伤害与 objective gaming 放进语言 agent 语境 |
| [Ethical and Social Risks of Harm from Language Models](alignment/Ethical_and_Social_Risks_of_Harm_from_Language_Models_2021.pdf)（Weidinger et al., 2021） | 先看 pp. 63–64 风险总表，再从 §2 六类风险里选一类回读 | 15 分钟 | 建立六大领域、21 类风险的检查表，不要求通读 64 页 |

读完画一条因果链：`想要的价值 → 写下的目标 / 收集的偏好 → 学到的代理奖励 → 部署行为 → 下游伤害`，并在每条箭头旁写一种可能失真。

## 三个风险与干预案例（按需，各 15–20 分钟）

- **隐私泄漏**：[Extracting Training Data from Large Language Models](alignment/Extracting_Training_Data_from_Large_Language_Models_2021.pdf)（Carlini et al., 2021）
  看 Figure 1、§3–5、§8–10：黑盒生成加成员推断如何从 GPT-2 提取训练样本，以及可做哪些缓解。
- **干预副作用**：[Detoxifying Language Models Risks Marginalizing Minority Voices](alignment/Detoxifying_Language_Models_Risks_Marginalizing_Minority_Voices_2021.pdf)（Xu et al., 2021）
  全文仅 8 页，重点 §4–6：训练数据里的虚假毒性相关性如何让 detoxification 对少数群体语言造成不成比例的损害。
- **非 RLHF 替代方案**：[PALMS: Process for Adapting Language Models to Society](alignment/PALMS_Process_for_Adapting_Language_Models_to_Society_2021.pdf)（Solaiman & Dennison, 2021）
  只读 Figure 1 + §3（PDF pp. 3–5）+ §4–9（pp. 6–9）：小规模 values-targeted 数据微调能改变行为，但必须同时检查能力完整性、限制与更广影响。

## 这轮先不读的引用

| 引用簇 | 当前处理 |
|--------|----------|
| Ibarz；Böhm；Jaques / Yi / Hancock；Kreutzer / Bahdanau；Lawrence；Zhou / Cho / Perez | 作为偏好学习在 Atari、摘要、对话、翻译、语义解析等领域的历史证据；当前 LLM 主线由 Christiano → Ziegler → Stiennon 覆盖 |
| Mishra (Natural Instructions)、Khashabi (UnifiedQA)、Aribandi (ExT5)，以及 Yi 的跨任务工作 | 都是指令/多任务泛化的重要变体；先用 FLAN + T0 建主框架，需要比较数据集设计时再回查 |
| Bahdanau / Abramson / Zhao 的导航工作；Nahian 的文本环境工作 | 偏 embodied / agent RL；留到第五阶段 agent 或环境交互学习 |
| Madaan 的 prompt memory | 留到第五阶段记忆与上下文工程 |
| Bender、Bommasani、Tamkin 等广义社会技术综述 | 先用 Weidinger 的风险分类建立索引；做治理专题时再系统读 |
| Dhamala、Liang、Manela、Caliskan、Kirk；Gehman、Nadeem、Nangia、Rudinger 等偏见/毒性 benchmark | 建立评测流水线时按指标选读；现在不逐篇囤 benchmark |
| Henderson、Dinan 等特定对话系统风险 | 做聊天系统安全专题时再读 |
| Ngo 的预训练数据过滤；Keskar / Dinan 的控制 token；Liu / Huang / Qian / Vig 等去偏；Dathathri (PPLM)、Krause (GeDi)、Schick 等生成控制 | 都属于行为修改工具箱；先用 PALMS + Xu + Weidinger 理解「干预—评测—副作用」框架，再按技术需要深入 |
| Solaiman / Buchanan 的误信息与恶意使用研究；Welbl / Blodgett 的干预与公平性讨论 | 作为风险案例索引保留，当前用 Carlini + Xu 各吃透一个具体机制即可 |

## 这轮的完成标准

不是读完 PDF 数量，而是能独立解释以下四组区别：

1. **偏好建模 vs 模仿学习**：排序反馈提供了什么，示范数据又缺什么？
2. **Instruction tuning vs RLHF**：前者解决任务泛化，后者为何仍有必要？
3. **代理奖励 vs 人类真实意图**：为什么 RM 分数上涨不保证人类更喜欢？
4. **降低一种伤害 vs 重新分配伤害**：为什么 detoxification 可能让少数群体承担额外代价？

完成 Session A / B / C 时，各在 `learning-progress.md` 写一条真实领悟；**未读前不要把“资料已整理”写成“论文已掌握”**。

## 在线原文入口

- RLHF 方法链：[Christiano 2017](https://arxiv.org/abs/1706.03741) · [Ziegler 2019](https://arxiv.org/abs/1909.08593) · [Stiennon 2020](https://arxiv.org/abs/2009.01325) · [InstructGPT 2022](https://arxiv.org/abs/2203.02155) · [DPO 2023](https://arxiv.org/abs/2305.18290)
- 可扩展监督：[Wu et al. 2021](https://arxiv.org/abs/2109.10862)
- 指令微调：[FLAN](https://arxiv.org/abs/2109.01652) · [T0](https://arxiv.org/abs/2110.08207)
- 对齐目标与风险：[Askell et al.](https://arxiv.org/abs/2112.00861) · [Gabriel](https://arxiv.org/abs/2001.09768) · [Kenton et al.](https://arxiv.org/abs/2103.14659) · [Weidinger et al.](https://arxiv.org/abs/2112.04359)
- 具体案例：[Carlini et al.](https://arxiv.org/abs/2012.07805) · [Xu et al.](https://arxiv.org/abs/2104.06390) · [PALMS](https://arxiv.org/abs/2106.10328)
