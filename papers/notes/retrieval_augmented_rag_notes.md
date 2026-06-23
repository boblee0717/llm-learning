# 精读骨架：检索增强生成 / RAG（REALM → RAG → RETRO）

> 本文件是**待填的精读骨架**（AI 协助搭建，正文留白待亲手精读后回填）。
> 三篇 PDF：
> - REALM：[`papers/retrieval-augmented/REALM_Retrieval_Augmented_Language_Model_PreTraining_2020.pdf`](../retrieval-augmented/REALM_Retrieval_Augmented_Language_Model_PreTraining_2020.pdf)（Guu et al., Google, 2020, arXiv:2002.08909）
> - RAG：[`papers/retrieval-augmented/RAG_Retrieval_Augmented_Generation_for_Knowledge_Intensive_NLP_2020.pdf`](../retrieval-augmented/RAG_Retrieval_Augmented_Generation_for_Knowledge_Intensive_NLP_2020.pdf)（Lewis et al., FAIR/UCL, 2020, arXiv:2005.11401）
> - RETRO：[`papers/retrieval-augmented/RETRO_Improving_Language_Models_by_Retrieving_from_Trillions_of_Tokens_2021.pdf`](../retrieval-augmented/RETRO_Improving_Language_Models_by_Retrieving_from_Trillions_of_Tokens_2021.pdf)（Borgeaud et al., DeepMind, 2021, arXiv:2112.04426）
> 关联课程：第五阶段第 5 课「RAG 与上下文工程」[`phase5-agent-architecture/README.md`](../../phase5-agent-architecture/README.md)；与第四阶段 DeepSeek 的「参数化知识」形成对照。

为什么把这三篇放一起读：它们是检索增强（retrieval-augmented）这条线的奠基三件套，按时间线正好是一条递进——
**检索进预训练（REALM）→ 检索进生成（RAG）→ 检索 scale 到万亿 token（RETRO）**。
读它们的核心目的，是吃透一个判断：**模型的知识不一定要全部「压」进参数里，也可以在用的时候从外部知识库「取」回来。**

---

## 读前先抓三个共性问题（带着问题读三篇）

无论哪一篇，都在回答同样的三件事，读的时候逐篇填它们的答案：

1. **检索什么？**（检索单元的粒度：document / passage / chunk？）
2. **怎么检索？**（检索器怎么实现：dense retriever / 最近邻？检索器和生成器是否一起训练？）
3. **检索到的内容怎么喂给模型？**（拼接进输入 prompt？还是用 cross-attention 融进解码？）

> 一句话提示：这三篇最大的差异恰恰在第 1 和第 3 个问题上——RAG 偏「拼进输入」，RETRO 偏「cross-attention 融合」，REALM 把检索器做进了预训练目标本身。

---

## 一、REALM（Guu et al., 2020）

> 一句话（读前先记，读后修正）：第一个把 neural retriever 端到端做进**预训练**的工作。

### 待填要点
- [ ] REALM 的预训练目标是什么？检索这一步是怎么获得梯度、被「学」出来的？
- [ ] knowledge retriever 怎么实现？（embedding + 最大内积搜索 MIPS？）
- [ ] 为什么检索器的更新会带来「索引过期」问题，论文怎么处理（异步刷新索引）？
- [ ] inference 时怎么用？和预训练时一致吗？

### 我的理解（读后回填）
> （留白）

---

## 二、RAG（Lewis et al., 2020）

> 一句话（读前先记，读后修正）：「RAG」名字的出处。DPR 稠密检索器 + BART seq2seq 生成器，**生成时**取回 Top-K 文档当条件。

### 待填要点
- [ ] RAG-Sequence 和 RAG-Token 两个变体的区别是什么？各自怎么把 K 篇文档边缘化（marginalize）？
- [ ] 检索器（DPR）和生成器（BART）哪部分参数训练、哪部分冻结？
- [ ] 为什么说 RAG 让知识「可更新、可溯源」？（换掉知识库 ≠ 重新训练模型）
- [ ] 在开放域问答等任务上相对纯参数化模型（如 T5/BART closed-book）强在哪？

### 我的理解（读后回填）
> （留白）

---

## 三、RETRO（Borgeaud et al., 2021）

> 一句话（读前先记，读后修正）：把检索增强 scale 到万亿 token 知识库，用 chunked cross-attention 把检索到的近邻 chunk 融进解码。

### 待填要点
- [ ] RETRO 的检索单元是什么粒度？（chunk，多大？）检索库规模量级？
- [ ] 什么是 chunked cross-attention（CCA）？它和 RAG「把文档拼进输入」有何本质不同？
- [ ] 为什么 RETRO 检索用的是「冻结的 BERT embedding 做最近邻」而不是端到端学检索器？这样做的取舍是什么？
- [ ] 论文主张「7.5B 的 RETRO 在部分指标逼近 25× 大的模型」，这对「检索 vs 堆参数」意味着什么？
- [ ] 注意潜在的数据泄漏/评测公平问题（检索库与测试集去重 leakage）论文怎么讨论的？

### 我的理解（读后回填）
> （留白）

---

## 四、三篇对照（读完后回填表格）

| 维度 | REALM (2020) | RAG (2020) | RETRO (2021) |
|---|---|---|---|
| 检索发生在 | 预训练 + 推理 | （待填） | （待填） |
| 检索单元粒度 | （待填） | passage/document | chunk |
| 检索器是否端到端训练 | 是 | （待填） | 否（冻结 BERT 近邻） |
| 检索内容怎么进模型 | （待填） | 拼进输入 / 边缘化 | chunked cross-attention |
| 知识库规模量级 | （待填） | （待填） | 万亿 token |
| 一句话定位 | （待填） | （待填） | （待填） |

---

## 五、和本仓库其它线索的连接（读后回填）

- [ ] **参数化知识 vs 非参数化知识**：DeepSeek 那条线（MoE/MLA）是在「把更多知识更高效地塞进参数」；RAG 这条线是「把知识放到外部、用时再取」。两者解决的痛点分别是什么？
- [ ] **和第五阶段第 4/5 课的关系**：现代 agent 的「记忆系统 + RAG + 上下文工程」本质上是这三篇思想的工程化延伸。对照 [`papers/agents/Decision-Aware_Memory_Cards_CICL_2026.pdf`](../agents/Decision-Aware_Memory_Cards_CICL_2026.pdf)（CICL：按「对下一步动作的影响」而非语义相似度排证据）——它对经典 RAG「按相似度 top-k 取」提出了什么批评？
- [ ] **和 KV Cache / 长上下文的关系**：「把所有资料塞进超长上下文」vs「检索少量相关片段」是两条互补路线，各自的成本在哪？

---

## TODO（读完后回填）
- [ ] 用一句话分别说清 REALM / RAG / RETRO 各自「检索什么、怎么检索、怎么喂给模型」
- [ ] 能解释 RAG-Sequence 与 RAG-Token 的区别
- [ ] 能解释 RETRO 的 chunked cross-attention 与 RAG「拼输入」的本质差异
- [ ] 填完第四节对照表
- [ ] 回到第五阶段第 5 课，把这三篇和「chunking / retrieval / context assembly」对应起来
