# Project Context & Agent Instructions

## 1. 项目概述

本项目是《深度学习原理及其应用》课程项目：**序列推荐：GRU、SASRec 与 BERT4Rec 的“单向 vs 双向”之争（含图视角 SR-GNN 对照）**。

核心目标：
1. 在统一的数据预处理与评测框架下，从零复现四类代表性序列推荐模型：**GRU4Rec**（循环神经网络）、**SASRec**（单向因果自注意力）、**BERT4Rec**（双向 Transformer + Cloze 掩码）、**SR-GNN**（有向会话图 + 门控图神经网络），并辅以 **Pop** 流行度基线。
2. 在长序列稠密数据集 **MovieLens-1M** 与短序列高稀疏数据集 **Amazon-Beauty** 上开展系统实验。
3. 重点探究：
   - “下一项预测（Next-Item）”与“掩码项预测（Cloze）”两种训练范式的差异及训练-推理不一致（Train-Test Mismatch）问题；
   - 序列长度分布、长尾物品与流行度偏差对各类方法的影响；
   - 双向训练及 Leave-One-Out 划分是否存在“未来信息泄露”的评测陷阱；
   - 产出 **“序列长度—模型类型—指标”三维对照表** 与 **短序列场景建模选型结论**。

---

## 2. 关键文档索引

每次对话开始时，请先阅读以下文件以对齐上下文：
- `@PLAN.md`：**项目总体规划与当前进度面板**（必读，开头声明了当前所处阶段与待办任务）。
- `@任务介绍.md`：课程原始题目要求、背景说明与参考论文链接。

> **进度同步规范**：每完成一个子任务或阶段性里程碑后，必须同步更新 `PLAN.md` 开头的「当前进度」区块以及正文中的任务复选框。

---

## 3. 技术栈与工具链规范

本项目严格遵循现代、高性能、可复现的 Python 工程规范：

- **设计哲学**：如果有现成、维护良好且未过时的现代工具/组件，优先直接采用，避免重复造轮子。
- **环境与包管理**：唯一使用 `uv`。
  - 运行脚本或工具必须通过 `uv run <command>`（例如 `uv run seqrec ...`, `uv run ruff check`, `uv run ty check`, `uv run marimo edit`），严禁直接裸调 `python` 或 `pip`。
- **代码风格、静态检查与交互式验证**：
  - 全项目所有函数签名、类方法、数据结构与配置类（`dataclass`）**必须配备完整、现代的 Python Type Hints**（遵循 Python 3.12+ 语法，如 `list[int]`, `dict[str, float]`, `torch.Tensor | None`，避免使用过时的 `typing.List` / `typing.Optional`）。
  - 使用 `ruff` 与 `ty` 进行代码规范与静态类型检查（`ty` 检查不必过于严苛，已知 `ty` 对 `altair` 存在误报 bug，可忽略或局部 `# type: ignore`）；临时实验验证、数据探索与快速调试统一在 `marimo` 笔记本（`notebooks/`）中进行。
  - **Marimo 笔记本编写规范**：
    1. 所有导入的包必须统一写在 `setup cell`（即 `with app.setup:` 块 / 名为 `setup` 的单元格）里；
    2. 每个类（`class`）和函数（`def`）的定义必须各自单独占一个单元格（如 `@app.function` 或 `@app.class_definition`）。
- **数据集获取与加载**：使用 Hugging Face `datasets` 库（`load_dataset`）直接拉取 Hub 上的 Parquet 数据集（`wwbrannon/ml-interview-examples-movielens-1m` 与 `ttb06/amazon_beauty_5_2014`），并通过 `.to_polars()` 交由 `polars` 处理。
- **数据处理与分桶分析**：默认使用 `polars`（优先使用 Lazy API 处理数据过滤、序列聚合与指标分桶统计），中间数据持久化采用 `Parquet` 格式。
- **深度学习框架**：`torch`（PyTorch 2.x, CUDA 13.0）。兼顾兼容性与灵活性，采用**轻量级 PyTorch 原生层**实现四大模型（如 `nn.GRU`、`nn.TransformerEncoder` / `F.scaled_dot_product_attention` 配合 `is_causal`，以及纯 PyTorch 批量张量矩阵乘法实现 SR-GNN 的 GGNN 层，不引入重型 C++ 图扩展库）。
- **评测指标库**：使用 `torchmetrics`（结合向量化张量计算）完成推荐检索排序指标统计。
- **命令行、终端交互与进度条**：使用 `tyro`（基于 Type Hints / `dataclass` 自动生成 CLI）、`rich`（结构化日志与终端表格展示）和 `tqdm`（训练与评测进度条）。
- **序列化与实验日志**：默认使用 `orjson` 读写实验配置与评测指标记录。
- **可视化图表与论文排版**：默认使用 `vega-altair` + `vl-convert-python` 生成并导出 `.svg` / `.pdf` 矢量图，最终报告基于 `reports/main.typ`（Typst CVPR 模板）编写。

---

## 4. 项目目录与架构约定

```text
.
├── AGENTS.md                   # 项目框架说明与 Agent 协作规范（本文件）
├── PLAN.md                     # 项目总体规划与实时进度追踪（含顶部进度面板）
├── 任务介绍.md                  # 课程任务书原始说明
├── pyproject.toml              # uv 项目配置与依赖声明（name = "seqrec"）
├── data/
│   └── processed/              # Polars 处理后的标准化 Parquet 文件与词表元数据（原始数据由 HF datasets 自动缓存）
├── src/
│   └── seqrec/                 # 核心 Python 包
│       ├── __init__.py
│       ├── cli.py              # 基于 tyro 的统一命令行入口 (uv run seqrec ...)
│       ├── config.py           # 实验与模型超参数 dataclass 定义（带完整 Type Hints）
│       ├── data/
│       │   ├── __init__.py
│       │   ├── preprocess.py   # HF datasets 加载、5-core 校验、ID 重映射、LOO 及全局时间切分
│       │   └── dataset.py      # PyTorch Dataset (支持 Causal / Cloze / Session-Graph)
│       ├── models/
│       │   ├── __init__.py
│       │   ├── base.py         # 统一序列推荐模型抽象基类 (SequentialRecommender)
│       │   ├── baselines.py    # Pop 流行度基线
│       │   ├── gru4rec.py      # GRU4Rec 模型实现
│       │   ├── sasrec.py       # SASRec 模型实现
│       │   ├── bert4rec.py     # BERT4Rec 模型实现
│       │   └── srgnn.py        # SR-GNN 模型实现
│       ├── engine/
│       │   ├── __init__.py
│       │   ├── trainer.py      # 统一训练器（支持混合精度、梯度裁剪、Early Stopping）
│       │   └── evaluator.py    # 统一评测器（torchmetrics、全库/采样排序、长度/热度分桶、ARP/Coverage）
│       └── analysis/
│           ├── __init__.py
│           ├── leakage_exp.py  # 专题实验：未来信息泄露与 Train-Test Mismatch 验证
│           └── plot_results.py # Vega-Altair + vl-convert 导出全套矢量图表与三维对照表
├── notebooks/                  # Marimo 交互式分析与临时验证笔记本
└── reports/                    # 实验产出与报告目录
    ├── main.typ                # Typst (CVPR 模板) 最终研究报告源码 (含 main.bib, logo.typ)
    ├── metrics/                # orjson 实验配置与评测指标持久化文件
    └── figures/                # Vega-Altair 导出的 SVG/PDF 图表
```

---

## 5. 代码实现与实验准则

1. **ID 编码规范**：
   - `0` 严格预留给 `<PAD>`（填充 Token）；
   - 真实物品 ID 映射区间为 `[1, num_items]`；
   - BERT4Rec 的 `<MASK>` Token 固定使用 `num_items + 1`。
2. **统一评测接口**：
   - 所有模型继承 `src/seqrec/models/base.py`，实现统一的打分接口 `score_items(seq_tensors, ...)`，返回形状为 `[batch_size, num_items + 1]` 的 logits（并将 `<PAD>` 及历史已交互物品根据评测协议屏蔽为 `-inf`）。
   - 主评测协议采用 **全物品库排序（Full-Ranking）**，辅以 **1正+100负采样排序（Sampled-Ranking）** 复现原论文设定。
3. **可复现性**：
   - 所有实验必须固定随机种子（`seed=42`），并在 `reports/metrics/` 下用 `orjson` 保存完整超参数配置、训练耗时、验证集最优曲线与测试集各项分桶指标。
