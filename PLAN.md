# 项目总体规划 (Project Plan)

## 📌 当前进度 (Current Progress)

- **当前所处阶段**：`Phase 2 & Phase 3 进行中（Task 1 统一训练评测引擎与单向模型组已合入 — 待完成 Task 2 BERT4Rec 与 Task 3 SR-GNN）`
- **整体完成度**：`60%`（已完成 Phase 1 数据底座，以及 Task 1 / PR #4：模型抽象基类 `base.py`、统一训练器 `trainer.py`、统一评测器 `evaluator.py`、`Pop` / `GRU4Rec` / `SASRec` 三模型实现，以及两大数据集上 3 模型 × 9 窗口配置共 27 组全库/采样与分桶基准实验）
- **最近完成事项**：
  - 合入 **PR #4 (Task 1 / Issue #1)**：
    - 完成 `src/seqrec/models/base.py`（`SequentialRecommender` 抽象基类、`score_position` 打分位置契约、分块 `ce` / `bce` / `bpr_max` 损失函数）与 `src/seqrec/models/__init__.py`（支持 `bert4rec` / `srgnn` 惰性导入）；
    - 完成 `src/seqrec/models/baselines.py` (`Pop`)、`src/seqrec/models/gru4rec.py` (`GRU4Rec`)、`src/seqrec/models/sasrec.py` (`SASRec`)；
    - 完成 `src/seqrec/engine/evaluator.py`（严格同分 item ID 升序打破并列、全库 Full-Ranking 与合法候选分布 1正100负均匀/流行度采样评测、序列长度分桶、目标热度分桶、`ARP@10`、`Coverage@10` 及 `oracle_middle_mask` 评测钩子）；
    - 完成 `src/seqrec/engine/trainer.py` 与 `src/seqrec/cli.py`（支持 AMP、早停与最优权重回滚、全字段 `reproduce` 往返校验，以及 `uv run seqrec train` / `uv run seqrec sweep`）；
    - 产出 `Pop`、`GRU4Rec`、`SASRec` 在 `Amazon-Beauty`（$N \in \{5, 10, 20, 50\}$）与 `MovieLens-1M`（$N \in \{10, 20, 50, 100, 200\}$）上的 **27 份完整评测 JSON**（持久化于 `reports/metrics/`）。
- **下一步行动 (Next Actions)**：
  1. 推进 **Task 2 (Issue #3)**：实现 `src/seqrec/models/bert4rec.py` 与 `src/seqrec/analysis/leakage_exp.py`，完成 BERT4Rec 双数据集主实验、窗口消融与专题三（Pure Cloze / +Last Mask / Oracle Middle Mask 及 LOO vs Global Temporal Split）实验；
  2. 推进 **Task 3 (Issue #2)**：实现 `src/seqrec/models/srgnn.py` 与 `src/seqrec/analysis/plot_results.py`，完成 SR-GNN 双数据集主实验与窗口消融，汇总全部模型 JSON 产出“序列长度—模型类型—指标”三维对照表及 Vega-Altair 全套矢量图表；
  3. 撰写 `reports/main.typ` 最终研究报告。

### 里程碑状态总览

| 阶段 | 里程碑名称 | 状态 | 核心产出 |
| :--- | :--- | :---: | :--- |
| **Phase 0** | 项目立项与架构规划 | ✅ 已完成 | `AGENTS.md`, `PLAN.md` |
| **Phase 1** | 工程初始化与数据底座搭建 | ✅ 已完成 | `config.py`, `cli.py`, `preprocess.py`, `dataset.py`, `phase1_data_explore.py` |
| **Phase 2** | 统一训练评测框架与四大模型实现 | 🔄 进行中 | 已完成 `Trainer`, `Evaluator`, `Pop`, `GRU4Rec`, `SASRec`；待合入 `BERT4Rec`, `SR-GNN` |
| **Phase 3** | 两大数据集主实验与基准对齐 | 🔄 进行中 | 已产出 `Pop`/`GRU4Rec`/`SASRec` 27 组 `orjson` 评测结果；待补充 `BERT4Rec`, `SR-GNN` |
| **Phase 4** | 三大专题深度实验与三维对照表产出 | 🔄 进行中 | 已产出单向模型组窗口 $N$、长度/热度分桶与采样偏差数据；待完成信息泄露实验、全模型三维表、矢量图表与 Typst 报告 |

---

## 一、 核心研究问题与项目目标

围绕序列推荐的**归纳偏置（Inductive Bias）**与**训练/评测范式**，本项目重点回答以下四个问题：

1. **单向 vs 双向（SASRec vs BERT4Rec）**：
   双向 Cloze 掩码在训练时可利用右侧未来上下文预测中间掩码项，但在测试推理时只能在序列末尾追加 `[MASK]` 预测下一项。这种 **训练-推理不一致（Train-Test Mismatch）** 对不同长度的序列有何影响？
2. **纯序列 vs 图结构（GRU4Rec/SASRec vs SR-GNN）**：
   将序列转化为有向会话图（Session Graph）并通过门控图神经网络（GGNN）做消息传递，在**短序列、包含重复点击或局部回跳**的场景下是否优于纯序列位置建模？
3. **序列长度与长尾分布敏感性**：
   不同模型在不同用户历史长度（极短/中短/中长/长）与不同物品流行度分桶（头部/腰部/长尾）下的表现如何？
4. **“未来信息泄露”与评测陷阱**：
   Leave-One-Out（LOO）划分下的跨序列全局时间穿越、双向掩码训练策略、以及负采样评测（Sampled Metrics）是否存在系统性偏差？

### 预期核心交付物
1. **“序列长度—模型类型—指标”三维对照表**（覆盖用户有效序列长度分桶 + 模型最大截断窗口 $N$ 敏感性两个维度）；
2. **短序列场景建模选型结论报告**（以 Amazon-Beauty 及短序列分桶为核心依据）；
3. **长尾物品召回与流行度偏差（Popularity Bias）量化分析图表**；
4. **高可复现的模块化代码库**（`uv` + `PyTorch` + `Polars` + `Vega-Altair`）。

---

## 二、 数据集与统一评测协议规范

### 1. 数据集配置与 Hugging Face 数据源
遵循**“优先复用成熟现代资产，避免重复造轮子”**的原则，直接通过 Hugging Face `datasets` 库（`load_dataset`）拉取 Hub 上已转换为 Parquet 的标准数据集，并调用 `.to_polars()` 无缝接入 `Polars` 处理管线：

| 数据集 | HF Hub 仓库 ID | 原始规模与字段 | 序列长度与分布特征 | 最大截断长度 $N$ (默认) |
| :--- | :--- | :--- | :--- | :---: |
| **MovieLens-1M** | [`wwbrannon/ml-interview-examples-movielens-1m`](https://huggingface.co/datasets/wwbrannon/ml-interview-examples-movielens-1m) | 合并 `train/val/test` 还原完整官方 `1,000,209` 条交互 (`user_id`, `item_id`, `rating`, `timestamp`) | 长序列为主（平均长度 > 100），相对稠密 | $200$（消融 $20, 50, 100$） |
| **Amazon-Beauty** | [`ttb06/amazon_beauty_5_2014`](https://huggingface.co/datasets/ttb06/amazon_beauty_5_2014)（辅以 [`milistu/Amazon_Beauty_2014`](https://huggingface.co/datasets/milistu/Amazon_Beauty_2014) 元数据） | SASRec/BERT4Rec 论文同源 2014 5-core 版，共 `198,502` 条交互 (`reviewerID`, `asin`, `overall`, `unixReviewTime`) | **典型短序列**（平均长度 < 10），极度稀疏，强长尾 | $50$（消融 $10, 20, 50$） |

### 2. 数据加载与预处理标准（`datasets` + `Polars` 实现）
- **声明式下载与缓存**：利用 `datasets.load_dataset` 自动管理下载、分片缓存与校验，通过 `.to_polars()` 直接零拷贝转换为 Polars DataFrame，省去手动解析 `.dat` 或解压 `.json.gz` 的胶水代码。
- **5-core 校验与过滤**：统一规范化列名为 `(user_id, item_id, rating, timestamp)`，执行 5-core 校验（确保所有用户和物品至少有 5 条交互记录）。
- **时序排序**：按 `(user_id, timestamp, item_id)` 严格升序排列。
- **数据划分方案**：
  - **主方案（Leave-One-Out, LOO）**：对每个用户序列 $S_u = [i_1, i_2, \dots, i_{|S_u|}]$，取 $i_{|S_u|}$ 为测试集目标，$i_{|S_u|-1}$ 为验证集目标，$[i_1, \dots, i_{|S_u|-2}]$ 为训练集。
  - **对照方案（Global Temporal Split，用于专题 3 信息泄露实验）**：按全局时间戳 80% / 10% / 10% 切分训练/验证/测试时间窗口，检验跨用户未来流行度泄露的影响。

### 3. 评测指标体系（基于 `torchmetrics`）
- **准确性指标**：通过 `torchmetrics` 计算 `HR@K` (Hit Rate / Recall, $K \in \{5, 10, 20\}$)、`NDCG@K` ($K \in \{5, 10, 20\}$)、`MRR`。
- **流行度偏差与多样性指标**：
  - `ARP@10`（Average Recommendation Popularity，推荐列表前 10 项在训练集中的平均频次）；
  - `Coverage@10`（所有测试用户 Top-10 推荐结果覆盖的物品占总物品库的比例）。
- **排序候选集协议**：
  - **Full-Ranking（主协议）**：对全部候选物品打分，屏蔽用户历史已交互物品后计算指标（消除负采样偏差）。
  - **Sampled-Ranking（辅助对照）**：每个正样本搭配 100 个负样本（对比均匀随机采样 vs 按流行度采样），用于与原论文报告数值对齐并揭示采样评测的夸大效应。

---

## 三、 四大模型复现技术方案（轻量级 PyTorch 原生层 + 全量 Type Hints）

兼顾跨平台兼容性与实验改造灵活性，所有模型均采用**轻量级 PyTorch 2.x 原生层**构建（不依赖重型外部图计算库或推荐黑盒库），全面配备类型标注（Type Hints），并通过 `tyro` + `dataclass` 统一管理超参数（默认隐层维度 $d = 64$，消融 $32, 128$，优化器 AdamW）：

### 1. GRU4Rec（循环网络代表）
- **核心架构**：Item Embedding $\rightarrow$ Embedding Dropout $\rightarrow$ 单层/多层 GRU $\rightarrow$ 投影层与物品嵌入矩阵点积。
- **关键细节**：支持序列逐步预测与末尾预测；对比原始论文的排序损失（BPR-max / TOP1）与现代交叉熵损失（Cross-Entropy Loss），验证损失函数对 RNN 基线的决定性影响。

### 2. SASRec（单向自注意力代表）
- **核心架构**：Item Embedding + 可学习位置编码（Learnable Positional Embedding）$\rightarrow$ $L$ 层带 **因果掩码（Causal Attention Mask）** 的 Transformer Block（Self-Attention + Point-wise FFN + LayerNorm + Residual）。
- **关键细节**：位置 $t$ 仅能关注 $\le t$ 的历史项；对比原始论文的二元交叉熵（BCE，1正1负采样）与全库/多负样本交叉熵（CE）损失。

### 3. BERT4Rec（双向 Transformer 代表）
- **核心架构**：双向 Transformer Encoder（移除因果掩码，允许双向上下文交互）+ 专用 `[MASK]` Token（ID 为 `num_items + 1`）。
- **关键细节**：
  - **训练阶段**：对输入序列以概率 $p_{\text{mask}} \in [0.15, 0.2]$ 进行 Cloze 随机掩码（80% 替换为 `[MASK]`，10% 随机替换，10% 保持不变），仅对被掩码位置计算 CE Loss；
  - **缓解 Train-Test Mismatch 策略**：按设定比例在训练批次中追加“仅掩码序列最后一项”的样本（Last-item Masking Augmentation），并在实验中量化该技巧的必要性。

### 4. SR-GNN（图神经网络代表）
- **核心架构**：
  - **构图**：将用户序列 $S = [i_1, i_2, \dots, i_l]$ 转化为有向会话图 $\mathcal{G}_s = (\mathcal{V}_s, \mathcal{E}_s)$，构建归一化入度与出度矩阵 $A^{\text{in}}, A^{\text{out}}$；
  - **消息传递**：通过门控图神经网络（GGNN）迭代更新节点隐状态；
  - **读出与预测**：将最后一个点击物品表征 $s_{\text{local}} = v_{i_l}$ 与软注意力聚合的全局图表示 $s_{\text{global}}$ 拼接后线性变换，对候选物品打分。

---

## 四、 三大核心专题实验规划

### 专题一：“序列长度—模型类型—指标”三维对照实验
1. **维度 A：测试用户有效序列长度分桶（Sequence Length Bucketing）**
   - 将测试集样本按输入序列实际长度 $|S_u|$ 划分为 4 个区间：
     - **极短序列**：$|S_u| \le 5$
     - **短序列**：$6 \le |S_u| \le 15$
     - **中长序列**：$16 \le |S_u| \le 50$
     - **长序列**：$|S_u| > 50$
   - 产出 `序列长度分桶 × 模型类型 (Pop/GRU4Rec/SASRec/BERT4Rec/SR-GNN) × 指标 (HR@10, NDCG@10)` 三维对照总表。
2. **维度 B：模型截断窗口长度 $N$ 超参敏感性**
   - 在 MovieLens-1M 上测试 $N \in \{10, 20, 50, 100, 200\}$，在 Amazon-Beauty 上测试 $N \in \{5, 10, 20, 50\}$，比较各模型随窗口增大时的收益曲线与计算开销。

### 专题二：长尾物品与流行度偏差（Popularity Bias）分析
1. **目标物品热度分桶（Target Item Popularity Bucketing）**：
   - 统计训练集各物品交互频次，按累计交互量或物品频次分位数划分为 **Head（头部热门 20%）**、**Mid（腰部 30%）**、**Tail（长尾 50%）**。
   - 分别计算当测试集真实下一项属于 Head / Mid / Tail 时，各模型的 `HR@10` 与 `NDCG@10`。
2. **模型推荐倾向分析**：
   - 对比各模型推荐结果的 `ARP@10`（平均流行度）与 `Coverage@10`（覆盖率），量化哪些模型过度依赖流行度捷径。

### 专题三：“未来信息泄露”与双向训练评测陷阱剖析
1. **陷阱 1：BERT4Rec 的训练-推理不一致与未来信息依赖**
   - 对比三种设定：
     - `BERT4Rec (Pure Cloze)`：仅用随机 Cloze 掩码训练，测试时在末尾加 `[MASK]`；
     - `BERT4Rec (+Last Mask)`：训练混入末尾掩码样本；
     - `BERT4Rec (Oracle Middle Mask)`：假设测试时掩码位于序列倒数第 2 项（可见最后 1 项真实未来交互），量化双向模型在“真正拥有未来上下文”与“末尾盲猜”之间的性能鸿沟。
2. **陷阱 2：Leave-One-Out 划分的跨序列时间泄露**
   - 对比 **Leave-One-Out (LOO)** 与 **Global Temporal Split（全局时间线切分）** 下各模型表现与流行度基线（Pop）的强弱变化，揭示 LOO 划分中“利用未来年份的全局流行趋势预测过去年份交互”的潜在信息泄露。
3. **陷阱 3：负采样评测偏差**
   - 对比 Full-Ranking 与 100-Negative Sampled-Ranking，展示采样评测如何掩盖长尾劣势并扭曲模型间的相对优劣。

---

## 五、 分阶段任务清单 (Task Checklist)

### Phase 1：工程初始化与数据底座搭建
- [x] 使用 `uv init --package --name seqrec` 初始化项目与 `pyproject.toml`，配置 `torch`, `torchmetrics`, `datasets`, `polars`, `orjson`, `altair`, `vl-convert-python`, `tyro`, `rich`, `tqdm`, `marimo`, `ruff`, `ty` 并建立完整子目录
- [x] 编写 `src/seqrec/config.py` 与 `src/seqrec/cli.py`：定义强类型实验配置 `dataclass` 与统一 CLI 入口（`uv run seqrec ...`）
- [x] 编写 `src/seqrec/data/preprocess.py`：使用 `datasets.load_dataset` 拉取 HF 现成数据集 (`wwbrannon/ml-interview-examples-movielens-1m` 与 `ttb06/amazon_beauty_5_2014`)，转 `polars` 完成 5-core 校验、连续 ID 映射、LOO 划分与全局时间切分，导出标准化 Parquet 与统计摘要
- [x] 编写 `src/seqrec/data/dataset.py`：实现支持 Causal 序列、Cloze 掩码序列、以及 SR-GNN 会话图结构的 Dataset 与 DataLoader（全程严格添加 Type Hints）

### Phase 2：统一训练评测框架与四大模型实现
- [x] 实现 `src/seqrec/models/base.py` 与 `src/seqrec/models/baselines.py` (`Pop`)
- [x] 实现 `src/seqrec/models/gru4rec.py` (`GRU4Rec`)
- [x] 实现 `src/seqrec/models/sasrec.py` (`SASRec`)
- [ ] 实现 `src/seqrec/models/bert4rec.py` (`BERT4Rec`)（Task 2 / Issue #3）
- [ ] 实现 `src/seqrec/models/srgnn.py` (`SR-GNN`)（Task 3 / Issue #2）
- [x] 实现 `src/seqrec/engine/evaluator.py`（基于 `torchmetrics` 支持全库/采样评测、按序列长度分桶、按物品热度分桶、ARP/Coverage 统计及 `oracle_middle_mask` 评测钩子）
- [x] 实现 `src/seqrec/engine/trainer.py`（支持 AMP、梯度裁剪、Early Stopping、完整实验元数据与可复现命令导出）并通过 `ruff` / `ty` 静态检查

### Phase 3：两大数据集主实验与基准对齐
- [x] 在 Amazon-Beauty 上完成单向模型组（`Pop`, `GRU4Rec`, `SASRec`）训练、窗口 $N \in \{5, 10, 20, 50\}$ 消融与全量评测（12 份指标已落盘 `reports/metrics/`）
- [x] 在 MovieLens-1M 上完成单向模型组（`Pop`, `GRU4Rec`, `SASRec`）训练、窗口 $N \in \{10, 20, 50, 100, 200\}$ 消融与全量评测（15 份指标已落盘 `reports/metrics/`）
- [ ] 在 Amazon-Beauty 与 MovieLens-1M 上完成 `BERT4Rec`（Task 2）与 `SR-GNN`（Task 3）训练及全量评测
- [ ] 汇总全模型主实验指标（Full-Ranking vs Sampled-Ranking）并持久化至 `reports/metrics/`

### Phase 4：三大专题深度实验与结论产出
- [ ] 完成专题一：汇总全模型“序列长度—模型类型—指标”三维对照表及截断窗口 $N$ 敏感性实验数据（`Pop`/`GRU4Rec`/`SASRec` 数据已就绪，待合入 `BERT4Rec`/`SR-GNN`）
- [ ] 完成专题二：汇总全模型 Head/Mid/Tail 长尾召回率对比与流行度偏差（ARP / Coverage）实验数据（`Pop`/`GRU4Rec`/`SASRec` 数据已就绪，待合入 `BERT4Rec`/`SR-GNN`）
- [ ] 完成专题三：编写 `src/seqrec/analysis/leakage_exp.py`，完成 BERT4Rec 掩码消融/Oracle 实验与 Global Temporal Split 时间泄露对照实验（Task 2 / Issue #3）
- [ ] 编写 `src/seqrec/analysis/plot_results.py` 使用 `vega-altair` + `vl-convert-python` 导出全套 `.svg` / `.pdf` 图表至 `reports/figures/`（Task 3 / Issue #2）
- [ ] 在 `reports/main.typ` 中撰写最终实验分析报告（重点总结**短序列场景选型结论**与**单向 vs 双向之争的核心发现**）
