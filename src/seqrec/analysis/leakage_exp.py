"""专题三（未来信息泄露与评测陷阱）自动化实验脚本。

Runs and persists three controlled comparisons to ``reports/metrics/``:

1. **陷阱 1 — Train-Test Mismatch（BERT4Rec 未来上下文依赖）**
   ``pure_cloze``（``last_item_mask_ratio=0.0``）、``last_mask_0.1`` / ``last_mask_0.2``
   （混入末尾掩码训练样本），以及 ``loo_val`` 上的三组配对评测：

   - **标准末尾 [MASK]**：``[h_1, ..., h_L, MASK]``；
   - **Matched Control**：与 Oracle 完全相同的截断历史与 [MASK] 绝对位置，但右侧为
     ``<PAD>``（``[h_shift, ..., h_L, MASK, PAD]``），用于剔除历史长度 / 掩码绝对
     位置带来的混杂；
   - **Oracle Middle Mask**：``[h_shift, ..., h_L, MASK, future_id]``。

   ``Oracle − Control`` 即纯“未来 1 项上下文”收益，``Control − Standard`` 即截断与
   掩码绝对位置带来的混杂偏差。

2. **陷阱 2 — LOO 跨序列时间穿越 vs Global Temporal Split**
   在 ``loo`` 与 ``temporal`` 两种划分下运行 ``Pop`` / ``GRU4Rec`` / ``SASRec`` /
   ``BERT4Rec``，观察全局时间切分下各模型相对流行度基线的强弱变化。

3. **陷阱 3 — 负采样评测偏差**
   汇总所有实验的 Full-Ranking 与 100-Negative Sampled-Ranking 指标，量化
   ``sampled / full`` 指标夸大倍率与模型间排名反转。

所有训练均复用 :mod:`seqrec.engine.trainer.run_experiment`，超参与 Task 1 基准完全一致
（``d=64, L=2, heads=2, dropout=0.2, AdamW(lr=1e-3), epochs=200, patience=10``）。
"""

from __future__ import annotations

import dataclasses
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, cast

import orjson
import torch
import tyro
from rich.console import Console
from rich.table import Table
from torch.utils.data import DataLoader

from seqrec.config import (
    DataConfig,
    DatasetName,
    EvalConfig,
    ExperimentConfig,
    ModelConfig,
    ModelName,
    SplitStrategy,
    TrainConfig,
)
from seqrec.data.dataset import DataBundle, EvalBatch, build_databundle
from seqrec.engine.evaluator import Evaluator
from seqrec.engine.trainer import run_experiment
from seqrec.models import build_model
from seqrec.models.base import SequentialRecommender

console = Console()

DEFAULT_METRICS_DIR = Path("reports/metrics")
DEFAULT_CHECKPOINTS_DIR = Path("data/processed/checkpoints")

#: Default truncation window used for the leakage controlled comparisons.
DEFAULT_WINDOW: dict[DatasetName, int] = {"beauty": 50, "ml-1m": 200}
#: Full window-length sensitivity grid for the BERT4Rec main experiments.
MAIN_WINDOWS: dict[DatasetName, tuple[int, ...]] = {
    "beauty": (5, 10, 20, 50),
    "ml-1m": (10, 20, 50, 100, 200),
}

HEADLINE_KEYS: tuple[str, ...] = (
    "HR@5",
    "NDCG@5",
    "HR@10",
    "NDCG@10",
    "HR@20",
    "NDCG@20",
    "MRR",
    "ARP@10",
    "Coverage@10",
)


# ----------------------------------------------------------------------------------- helpers


def make_config(
    dataset: DatasetName,
    split_strategy: SplitStrategy,
    seq_len: int,
    model: ModelName,
    *,
    last_item_mask_ratio: float = 0.1,
    epochs: int = 200,
    device: str = "cuda",
    exp_name: str | None = None,
) -> ExperimentConfig:
    """Build an experiment config identical to the Task 1 baseline sweep."""
    data = DataConfig(
        dataset=dataset,
        split_strategy=split_strategy,
        max_seq_len=seq_len,
        batch_size=256,
        eval_batch_size=256,
        num_workers=0,
        mask_prob=0.2,
        last_item_mask_ratio=last_item_mask_ratio,
    )
    model_cfg = ModelConfig(
        name=model,
        hidden_size=64,
        num_layers=2,
        num_heads=2,
        dropout=0.2,
        loss_type="ce",
        share_item_embeddings=True,
    )
    train = TrainConfig(epochs=epochs, device=device)
    return ExperimentConfig(
        data=data,
        model=model_cfg,
        train=train,
        eval=EvalConfig(),
        exp_name=exp_name,
    )


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    """Persist a JSON artifact (orjson, indented) and return its path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(
        orjson.dumps(payload, option=orjson.OPT_INDENT_2, default=_json_default)
    )
    return path


def _json_default(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (set, frozenset)):
        return sorted(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serialisable")


def _config_signature(config: ExperimentConfig) -> dict[str, Any]:
    """Hyper-parameters that must match before an existing result may be reused."""
    data = config.data
    model = config.model
    train = config.train
    return {
        "dataset": data.dataset,
        "split_strategy": data.split_strategy,
        "max_seq_len": data.resolved_max_seq_len,
        "mask_prob": data.mask_prob,
        "last_item_mask_ratio": data.last_item_mask_ratio,
        "sliding_window": data.sliding_window,
        "model": model.name,
        "hidden_size": model.hidden_size,
        "num_layers": model.num_layers,
        "num_heads": model.num_heads,
        "dropout": model.dropout,
        "loss_type": model.loss_type,
        "share_item_embeddings": model.share_item_embeddings,
        "epochs": train.epochs,
        "lr": train.lr,
        "seed": train.seed,
        "patience": train.patience,
    }


def _existing_signature(raw_cfg: dict[str, Any]) -> dict[str, Any]:
    """Extract the same signature from the ``config`` block of a saved result JSON."""
    data = raw_cfg["data"]
    model = raw_cfg["model"]
    train = raw_cfg["train"]
    return {
        "dataset": data.get("dataset"),
        "split_strategy": data.get("split_strategy"),
        "max_seq_len": data.get("max_seq_len"),
        "mask_prob": data.get("mask_prob", 0.2),
        "last_item_mask_ratio": data.get("last_item_mask_ratio", 0.1),
        "sliding_window": data.get("sliding_window", False),
        "model": model.get("name"),
        "hidden_size": model.get("hidden_size"),
        "num_layers": model.get("num_layers"),
        "num_heads": model.get("num_heads"),
        "dropout": model.get("dropout"),
        "loss_type": model.get("loss_type"),
        "share_item_embeddings": model.get("share_item_embeddings", True),
        "epochs": train.get("epochs"),
        "lr": train.get("lr"),
        "seed": train.get("seed"),
        "patience": train.get("patience"),
    }


def _run_if_missing(
    config: ExperimentConfig,
    *,
    resume: bool = True,
    show_progress: bool = True,
    require_checkpoint: bool = False,
) -> Path:
    """Run :func:`run_experiment` unless a reusable result already exists (idempotent).

    A result is reused only when its metrics JSON exists **and** its recorded
    hyper-parameters match ``config``; experiments that need checkpoint re-evaluation
    (Oracle / matched control) additionally require the ``.pt`` checkpoint, which is *not*
    version controlled. Missing or mismatched artifacts trigger a fresh run, so the script
    never silently mixes results from different configurations.
    """
    target = config.metrics_dir / f"{config.exp_name}.json"
    checkpoint = config.checkpoints_dir / f"{config.exp_name}.pt"
    if resume and target.exists():
        existing = _existing_signature(orjson.loads(target.read_bytes())["config"])
        signature_ok = existing == _config_signature(config)
        if signature_ok and (not require_checkpoint or checkpoint.exists()):
            console.print(f"[yellow]skip existing[/yellow] {config.exp_name}")
            return target
        if not signature_ok:
            console.print(
                f"[yellow]config mismatch, re-running[/yellow] {config.exp_name}"
            )
        else:
            console.print(
                f"[yellow]checkpoint missing, re-running[/yellow] {config.exp_name}"
            )
    console.print(f"[bold green]running[/bold green] {config.exp_name}")
    run_experiment(config, show_progress=show_progress)
    return target


def _load_metrics(path: Path, field: str) -> dict[str, float]:
    """Load one metrics dictionary (e.g. ``best_val_metrics``) from a result JSON."""
    raw = orjson.loads(path.read_bytes())
    return dict(raw.get(field, {}))


def _headline(metrics: dict[str, float]) -> dict[str, float]:
    """Extract the headline ranking metrics for compact summary artifacts."""
    return {key: float(metrics[key]) for key in HEADLINE_KEYS if key in metrics}


def _config_from_json(path: Path) -> ExperimentConfig:
    """Reconstruct an :class:`ExperimentConfig` from a saved result JSON."""
    raw = orjson.loads(path.read_bytes())
    cfg = raw["config"]
    data_raw = dict(cfg["data"])
    data_raw["data_dir"] = Path(data_raw["data_dir"])
    eval_raw = dict(cfg["eval"])
    eval_raw["ks"] = tuple(eval_raw["ks"])
    return ExperimentConfig(
        data=DataConfig(**data_raw),
        model=ModelConfig(**cfg["model"]),
        train=TrainConfig(**cfg["train"]),
        eval=EvalConfig(**eval_raw),
        exp_name=cfg.get("exp_name"),
        metrics_dir=Path(cfg.get("metrics_dir", "reports/metrics")),
        checkpoints_dir=Path(cfg.get("checkpoints_dir", "data/processed/checkpoints")),
    )


def _load_checkpoint_model(
    exp_name: str,
    *,
    device: str,
    metrics_dir: Path,
    checkpoints_dir: Path,
) -> tuple[SequentialRecommender, ExperimentConfig, DataBundle]:
    """Rebuild a trained model and its LOO data bundle from the metrics JSON + checkpoint."""
    metrics_path = metrics_dir / f"{exp_name}.json"
    checkpoint_path = checkpoints_dir / f"{exp_name}.pt"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Metrics JSON not found: {metrics_path}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}. Checkpoints are not version "
            "controlled; re-run the experiment so the checkpoint gets recreated."
        )

    config = _config_from_json(metrics_path)
    config.data.split_strategy = "loo"
    bundle = build_databundle(config.data, model_name=config.model.name)
    model = build_model(
        config.model,
        num_items=bundle.num_items,
        max_seq_len=bundle.max_seq_len,
        item_pop_counts=bundle.item_pop_counts,
    )
    state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state["model"])
    model.to(device)
    return model, config, bundle


def evaluate_oracle(
    exp_name: str,
    *,
    device: str = "cuda",
    metrics_dir: Path = DEFAULT_METRICS_DIR,
    checkpoints_dir: Path = DEFAULT_CHECKPOINTS_DIR,
) -> dict[str, float]:
    """Re-evaluate a trained checkpoint on ``loo_val`` with ``oracle_middle_mask=True``.

    The oracle protocol exposes the true next interaction as right-hand context
    (``[..., <MASK>, future_id]``), quantifying the upper bound a bidirectional model could
    reach if it truly saw future context at test time.
    """
    model, config, bundle = _load_checkpoint_model(
        exp_name,
        device=device,
        metrics_dir=metrics_dir,
        checkpoints_dir=checkpoints_dir,
    )
    oracle_cfg = dataclasses.replace(config.eval, oracle_middle_mask=True)
    evaluator = Evaluator(
        oracle_cfg,
        num_items=bundle.num_items,
        item_pop_counts=bundle.item_pop_counts,
        device=torch.device(device),
    )
    return evaluator.evaluate(
        model, bundle.val_loader, "val_oracle", show_progress=False
    )


def _truncate_for_oracle_match(batch: EvalBatch, window: int) -> EvalBatch:
    """Drop the oldest history tokens exactly like the oracle protocol.

    The standard scorer then appends ``<MASK>`` at the same absolute position as the
    oracle input (the kept length is always ``<= window - 2``), so the resulting sequence
    ``[h_shift, ..., h_L, MASK, PAD]`` differs from the oracle protocol only by the missing
    future item — history content, history length and ``<MASK>`` position are identical.
    """
    input_ids = batch["input_ids"]
    seq_len = batch["seq_len"]
    batch_size = input_ids.size(0)

    shift = (seq_len + 2 - window).clamp(min=0)
    kept = seq_len - shift
    columns = (
        torch.arange(window, device=input_ids.device)
        .unsqueeze(0)
        .expand(batch_size, -1)
    )
    source = (columns + shift.unsqueeze(1)).clamp(max=window - 1)
    gathered = input_ids.gather(1, source)
    keep_mask = columns < kept.unsqueeze(1)

    matched = dict(batch)
    matched["input_ids"] = torch.where(keep_mask, gathered, torch.zeros_like(gathered))
    matched["seq_len"] = kept
    return cast(EvalBatch, matched)


def _matched_control_batches(
    loader: DataLoader[Any], window: int
) -> Iterator[EvalBatch]:
    """Stream val batches with the oracle-matched history truncation applied."""
    for batch in loader:
        yield _truncate_for_oracle_match(batch, window)


def evaluate_matched_control(
    exp_name: str,
    *,
    device: str = "cuda",
    metrics_dir: Path = DEFAULT_METRICS_DIR,
    checkpoints_dir: Path = DEFAULT_CHECKPOINTS_DIR,
) -> dict[str, float]:
    """Matched control for the oracle protocol.

    Same truncated history, same ``<MASK>`` absolute position as the oracle input, but the
    slot right of the mask stays ``<PAD>`` (no future context). Comparing this control to
    the oracle isolates the pure "one future item" gain; comparing it to the standard
    protocol quantifies the truncation / mask-position confound.
    """
    model, config, bundle = _load_checkpoint_model(
        exp_name,
        device=device,
        metrics_dir=metrics_dir,
        checkpoints_dir=checkpoints_dir,
    )
    evaluator = Evaluator(
        config.eval,
        num_items=bundle.num_items,
        item_pop_counts=bundle.item_pop_counts,
        device=torch.device(device),
    )
    control_loader = cast(
        Any, _matched_control_batches(bundle.val_loader, bundle.max_seq_len)
    )
    return evaluator.evaluate(model, control_loader, "val_control", show_progress=False)


# --------------------------------------------------------------------------- 陷阱 1: mismatch


def run_main_n_sweep(
    datasets: tuple[DatasetName, ...],
    *,
    device: str = "cuda",
    epochs: int = 200,
    resume: bool = True,
) -> list[Path]:
    """BERT4Rec 主实验 + 窗口长度 ``N`` 敏感性扫描（``last_item_mask_ratio=0.1``）。"""
    outputs: list[Path] = []
    for dataset in datasets:
        for seq_len in MAIN_WINDOWS[dataset]:
            exp_name = f"bert4rec_{dataset}_loo_n{seq_len}"
            config = make_config(
                dataset,
                "loo",
                seq_len,
                "bert4rec",
                last_item_mask_ratio=0.1,
                epochs=epochs,
                device=device,
                exp_name=exp_name,
            )
            outputs.append(_run_if_missing(config, resume=resume))
    return outputs


def run_mismatch_experiments(
    datasets: tuple[DatasetName, ...],
    *,
    device: str = "cuda",
    epochs: int = 200,
    resume: bool = True,
) -> list[Path]:
    """陷阱 1：BERT4Rec 掩码策略消融 + Oracle Middle Mask 与 Matched Control 对照。"""
    variants: tuple[tuple[str, float], ...] = (
        ("pure_cloze", 0.0),
        ("last_mask_0.1", 0.1),
        ("last_mask_0.2", 0.2),
    )
    outputs: list[Path] = []
    for dataset in datasets:
        window = DEFAULT_WINDOW[dataset]
        standard: dict[str, dict[str, float]] = {}
        control: dict[str, dict[str, float]] = {}
        oracle: dict[str, dict[str, float]] = {}
        for name, last_item_mask_ratio in variants:
            if last_item_mask_ratio == 0.1:
                exp_name = f"bert4rec_{dataset}_loo_n{window}"
            else:
                exp_name = f"bert4rec_{dataset}_loo_n{window}_lmr{last_item_mask_ratio}"
            config = make_config(
                dataset,
                "loo",
                window,
                "bert4rec",
                last_item_mask_ratio=last_item_mask_ratio,
                epochs=epochs,
                device=device,
                exp_name=exp_name,
            )
            # Oracle / matched-control re-evaluation needs the checkpoint, which is not
            # version controlled; retrain automatically when it is missing.
            path = _run_if_missing(config, resume=resume, require_checkpoint=True)
            outputs.append(path)
            standard[name] = _headline(_load_metrics(path, "best_val_metrics"))
            control[name] = _headline(evaluate_matched_control(exp_name, device=device))
            oracle[name] = _headline(evaluate_oracle(exp_name, device=device))

        summary: dict[str, Any] = {
            "dataset": dataset,
            "window": window,
            "mask_prob": 0.2,
            "standard_end_mask_loo_val": standard,
            "matched_control_loo_val": control,
            "oracle_middle_mask_loo_val": oracle,
        }
        out = DEFAULT_METRICS_DIR / f"leakage_bert4rec_mismatch_{dataset}.json"
        outputs.append(_write_json(out, summary))
        _print_mismatch_summary(dataset, standard, control, oracle)
    return outputs


def _print_mismatch_summary(
    dataset: DatasetName,
    standard: dict[str, dict[str, float]],
    control: dict[str, dict[str, float]],
    oracle: dict[str, dict[str, float]],
) -> None:
    """Print the three-way protocol comparison with pure-future and confound deltas."""
    table = Table(title=f"陷阱1 · BERT4Rec Train-Test Mismatch — {dataset}")
    table.add_column("variant", style="cyan")
    table.add_column("std HR@10", justify="right")
    table.add_column("ctrl HR@10", justify="right")
    table.add_column("oracle HR@10", justify="right")
    table.add_column("Δ future", justify="right")
    table.add_column("Δ confound", justify="right")
    for name, s in standard.items():
        c = control[name]
        o = oracle[name]
        table.add_row(
            name,
            f"{s.get('HR@10', float('nan')):.4f}",
            f"{c.get('HR@10', float('nan')):.4f}",
            f"{o.get('HR@10', float('nan')):.4f}",
            f"{o.get('HR@10', float('nan')) - c.get('HR@10', float('nan')):+.4f}",
            f"{c.get('HR@10', float('nan')) - s.get('HR@10', float('nan')):+.4f}",
        )
    console.print(table)


# --------------------------------------------------------------------- 陷阱 2: temporal split


def run_split_comparison(
    datasets: tuple[DatasetName, ...],
    *,
    device: str = "cuda",
    epochs: int = 200,
    resume: bool = True,
) -> list[Path]:
    """陷阱 2：LOO vs Global Temporal Split 下四模型的强弱对照。"""
    models: tuple[ModelName, ...] = ("pop", "gru4rec", "sasrec", "bert4rec")
    outputs: list[Path] = []
    for dataset in datasets:
        window = DEFAULT_WINDOW[dataset]
        splits: dict[str, dict[str, dict[str, float]]] = {}
        for split in ("loo", "temporal"):
            for model in models:
                exp_name = f"{model}_{dataset}_{split}_n{window}"
                config = make_config(
                    dataset,
                    split,
                    window,
                    model,
                    epochs=epochs,
                    device=device,
                    exp_name=exp_name,
                )
                path = _run_if_missing(config, resume=resume)
                outputs.append(path)
                splits.setdefault(split, {})[model] = _headline(
                    _load_metrics(path, "test_metrics")
                )

        summary: dict[str, Any] = {
            "dataset": dataset,
            "window": window,
            "splits": splits,
        }
        out = DEFAULT_METRICS_DIR / f"leakage_split_comparison_{dataset}.json"
        outputs.append(_write_json(out, summary))
        _print_split_summary(dataset, splits)
    return outputs


def _print_split_summary(
    dataset: DatasetName,
    splits: dict[str, dict[str, dict[str, float]]],
) -> None:
    table = Table(title=f"陷阱2 · LOO vs Temporal — {dataset} (HR@10 / NDCG@10)")
    table.add_column("model", style="cyan")
    for split in ("loo", "temporal"):
        table.add_column(f"{split} HR@10", justify="right")
        table.add_column(f"{split} NDCG@10", justify="right")
    for model in ("pop", "gru4rec", "sasrec", "bert4rec"):
        row = [model]
        for split in ("loo", "temporal"):
            metrics = splits.get(split, {}).get(model, {})
            row.append(f"{metrics.get('HR@10', float('nan')):.4f}")
            row.append(f"{metrics.get('NDCG@10', float('nan')):.4f}")
        table.add_row(*row)
    console.print(table)


# ------------------------------------------------------------------- 陷阱 3: sampled bias


def summarize_sampled_bias(metrics_dir: Path = DEFAULT_METRICS_DIR) -> Path:
    """陷阱 3：汇总 Full-Ranking vs 100-Negative Sampled-Ranking 的夸大与排名反转。"""
    rows: list[dict[str, Any]] = []
    for path in sorted(metrics_dir.glob("*.json")):
        if path.name.startswith("leakage_"):
            continue
        raw = orjson.loads(path.read_bytes())
        test = raw.get("test_metrics", {})
        if "HR@10" not in test or "sampled_uniform/HR@10" not in test:
            continue
        config = raw["config"]
        rows.append(
            {
                "exp_name": raw.get("exp_name", path.stem),
                "dataset": config["data"]["dataset"],
                "model": config["model"]["name"],
                "split": config["data"]["split_strategy"],
                "window": config["data"]["max_seq_len"],
                "full_HR@10": float(test["HR@10"]),
                "full_NDCG@10": float(test["NDCG@10"]),
                "sampled_uniform_HR@10": float(test["sampled_uniform/HR@10"]),
                "sampled_uniform_NDCG@10": float(test["sampled_uniform/NDCG@10"]),
                "sampled_popularity_HR@10": float(test["sampled_popularity/HR@10"]),
                "sampled_popularity_NDCG@10": float(test["sampled_popularity/NDCG@10"]),
            }
        )

    for row in rows:
        row["inflation_uniform_HR@10"] = _ratio(
            row["sampled_uniform_HR@10"], row["full_HR@10"]
        )
        row["inflation_uniform_NDCG@10"] = _ratio(
            row["sampled_uniform_NDCG@10"], row["full_NDCG@10"]
        )
        row["inflation_popularity_HR@10"] = _ratio(
            row["sampled_popularity_HR@10"], row["full_HR@10"]
        )

    groups = _group_rank_reversals(rows)
    summary: dict[str, Any] = {
        "per_experiment": rows,
        "rank_reversals": groups,
    }
    out = metrics_dir / "leakage_sampled_bias_summary.json"
    _write_json(out, summary)
    console.print(
        f"[green]sampled-bias summary saved to[/green] {out} "
        f"({len(rows)} experiments, {len(groups)} comparison groups)"
    )
    return out


def _ratio(numerator: float, denominator: float) -> float | None:
    """``numerator / denominator`` inflation ratio (``None`` when undefined)."""
    if denominator <= 0:
        return None
    return numerator / denominator


def _group_rank_reversals(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Count pairwise ranking inversions between Full and Sampled rankings per group.

    Ablation variants (``*_lmr*`` BERT4Rec masking runs) are excluded from the per-model
    ranking groups: they share the ``bert4rec`` model name / window with the canonical run
    and would otherwise appear as duplicate entries of the same architecture.
    """
    groups: dict[tuple[str, str, int], list[dict[str, Any]]] = {}
    for row in rows:
        if "_lmr" in row["exp_name"]:
            continue
        key = (row["dataset"], row["split"], int(row["window"]))
        groups.setdefault(key, []).append(row)

    output: list[dict[str, Any]] = []
    for (dataset, split, window), group in sorted(groups.items()):
        full_order = [
            r["model"]
            for r in sorted(group, key=lambda r: r["full_HR@10"], reverse=True)
        ]
        sampled_order = [
            r["model"]
            for r in sorted(
                group, key=lambda r: r["sampled_uniform_HR@10"], reverse=True
            )
        ]
        output.append(
            {
                "dataset": dataset,
                "split": split,
                "window": window,
                "models": len(group),
                "full_HR@10_order": full_order,
                "sampled_uniform_HR@10_order": sampled_order,
                "pairwise_inversions": _count_inversions(full_order, sampled_order),
                "order_changed": full_order != sampled_order,
            }
        )
    return output


def _count_inversions(a: list[str], b: list[str]) -> int:
    """Number of pairwise inversions between two permutations of the same items."""
    position = {item: index for index, item in enumerate(b)}
    inversions = 0
    for i, left in enumerate(a):
        for right in a[i + 1 :]:
            if position[left] > position[right]:
                inversions += 1
    return inversions


# ------------------------------------------------------------------------------ entrypoint


def run_all(
    datasets: tuple[DatasetName, ...],
    *,
    device: str = "cuda",
    epochs: int = 200,
    resume: bool = True,
) -> None:
    """Run the BERT4Rec main sweep plus all three leakage experiments in sequence.

    ``resume`` is passed through to every phase; existing results are only reused when
    their recorded hyper-parameters match the current configuration (and, for the mismatch
    experiments, when the checkpoint is available).
    """
    run_main_n_sweep(datasets, device=device, epochs=epochs, resume=resume)
    run_mismatch_experiments(datasets, device=device, epochs=epochs, resume=resume)
    run_split_comparison(datasets, device=device, epochs=epochs, resume=resume)
    summarize_sampled_bias()


@dataclass(slots=True)
class MainCmd:
    """BERT4Rec 主实验 + 窗口长度 N 敏感性扫描."""

    datasets: tuple[DatasetName, ...] = ("beauty", "ml-1m")
    device: str = "cuda"
    epochs: int = 200
    resume: bool = True


@dataclass(slots=True)
class MaskingCmd:
    """陷阱1：BERT4Rec Train-Test Mismatch 与 Oracle Middle Mask."""

    datasets: tuple[DatasetName, ...] = ("beauty", "ml-1m")
    device: str = "cuda"
    epochs: int = 200
    resume: bool = True


@dataclass(slots=True)
class TemporalCmd:
    """陷阱2：LOO vs Global Temporal Split 对照."""

    datasets: tuple[DatasetName, ...] = ("beauty", "ml-1m")
    device: str = "cuda"
    epochs: int = 200
    resume: bool = True


@dataclass(slots=True)
class SampledCmd:
    """陷阱3：Full-Ranking vs Sampled-Ranking 偏差汇总."""

    metrics_dir: Path = DEFAULT_METRICS_DIR


@dataclass(slots=True)
class AllCmd:
    """依次运行 BERT4Rec 主实验与陷阱1/2/3 全部实验."""

    datasets: tuple[DatasetName, ...] = ("beauty", "ml-1m")
    device: str = "cuda"
    epochs: int = 200
    resume: bool = True


LeakageCommand = (
    Annotated[MainCmd, tyro.conf.subcommand(name="main")]
    | Annotated[MaskingCmd, tyro.conf.subcommand(name="masking")]
    | Annotated[TemporalCmd, tyro.conf.subcommand(name="temporal")]
    | Annotated[SampledCmd, tyro.conf.subcommand(name="sampled")]
    | Annotated[AllCmd, tyro.conf.subcommand(name="all")]
)


def main() -> None:
    """CLI entry point: ``uv run python -m seqrec.analysis.leakage_exp <subcommand>``."""
    command = tyro.cli(LeakageCommand)
    if isinstance(command, MainCmd):
        run_main_n_sweep(
            command.datasets,
            device=command.device,
            epochs=command.epochs,
            resume=command.resume,
        )
    elif isinstance(command, MaskingCmd):
        run_mismatch_experiments(
            command.datasets,
            device=command.device,
            epochs=command.epochs,
            resume=command.resume,
        )
    elif isinstance(command, TemporalCmd):
        run_split_comparison(
            command.datasets,
            device=command.device,
            epochs=command.epochs,
            resume=command.resume,
        )
    elif isinstance(command, SampledCmd):
        summarize_sampled_bias(command.metrics_dir)
    else:
        run_all(
            command.datasets,
            device=command.device,
            epochs=command.epochs,
            resume=command.resume,
        )


if __name__ == "__main__":
    main()
