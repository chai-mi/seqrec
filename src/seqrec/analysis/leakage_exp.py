"""Task 2 experiment planning and orchestration (not another training engine).

Usage::

    uv run python -m seqrec.analysis.leakage_exp --action plan
    uv run python -m seqrec.analysis.leakage_exp --action run
    uv run python -m seqrec.analysis.leakage_exp --action summary

The default backend uses Task 1's Trainer.fit and Evaluator directly. Optional
--backend module:callable adapters must return TrainingResult, not write to the
orchestrator's result path. The best STANDARD-validation checkpoint is restored
before all post-hoc comparisons; Oracle never selects a checkpoint.

Sampling follows the merged common Evaluator exactly: 100 draws WITH replacement
(excluding PAD, history and target). Duplicate negatives are counted separately.
Popularity sampling gives zero probability to training-unseen items; an empty
eligible distribution raises instead of silently falling back. Full MRR is
untruncated; sampled HR/NDCG are reported at primary_k only (currently 10).
The RNG seed and sample cohort are reset for each paired Oracle evaluation.
"""

from __future__ import annotations

import hashlib
import importlib
import inspect
import math
import random
import tempfile
from collections.abc import Callable, Sized
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Literal, Protocol, cast

import orjson
import torch
import tyro
from rich.console import Console
from torch import nn
from torch.utils.data import DataLoader, Subset

from seqrec.config import (
    LEN_BUCKETS,
    POP_BUCKETS,
    DataConfig,
    DatasetName,
    EvalConfig,
    ExperimentConfig,
    ModelConfig,
    ModelName,
    SplitStrategy,
    TrainConfig,
    default_max_seq_len,
)
from seqrec.data.dataset import DataBundle, EvalSample, EvalSeqDataset, build_databundle
from seqrec.engine.evaluator import Evaluator
from seqrec.engine.trainer import Trainer
from seqrec.models import build_model
from seqrec.models.base import SequentialRecommender

SCHEMA_VERSION = 2
WINDOWS: dict[DatasetName, tuple[int, ...]] = {
    "beauty": (5, 10, 20, 50),
    "ml-1m": (10, 20, 50, 100, 200),
}
MODELS: tuple[ModelName, ...] = ("pop", "gru4rec", "sasrec", "bert4rec")
type Suite = Literal["all", "windows", "masking", "temporal"]


@dataclass(slots=True)
class RankingMetrics:
    """Common evaluator adapter output. Empty buckets may be empty dictionaries."""

    num_examples: int
    overall: dict[str, float]
    length_buckets: dict[str, dict[str, float]]
    popularity_buckets: dict[str, dict[str, float]]


@dataclass(slots=True)
class Evaluation:
    """All three candidate protocols from one checkpoint and sample set."""

    full: RankingMetrics
    sampled_uniform: RankingMetrics
    sampled_popularity: RankingMetrics


class Evaluate(Protocol):
    """Adapter for Task 1's Evaluator; must not reset model's Oracle flags."""

    def __call__(
        self,
        model: nn.Module,
        loader: DataLoader[EvalSample],
        bundle: DataBundle,
        config: EvalConfig,
        seed: int,
    ) -> Evaluation: ...


@dataclass(slots=True)
class TrainingResult:
    """Task 1 backend result, with best STANDARD checkpoint already restored.

    ``protocol`` describes ties, target restoration, negative sampling, empty
    buckets and temporal cold-start eligibility. ``history`` must include the
    standard validation curve. Pop may have no checkpoint/history/best_epoch.
    """

    model: nn.Module
    bundle: DataBundle
    evaluate: Evaluate
    checkpoint: Path | None
    history: list[dict[str, float]]
    best_epoch: int | None
    train_seconds: float
    protocol: dict[str, str]
    engine_result: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class RunSpec:
    """One deduplicated training configuration and its experiment memberships."""

    config: ExperimentConfig
    purposes: list[str]


@dataclass(slots=True)
class LeakageConfig:
    """Default action only prints the plan; training is always explicit."""

    action: Literal["plan", "run", "summary"] = "plan"
    suite: Suite = "all"
    datasets: tuple[DatasetName, ...] = ("beauty", "ml-1m")
    backend: str | None = "seqrec.analysis.leakage_exp:train_with_engine"
    data_dir: Path = Path("data/processed")
    metrics_dir: Path = Path("reports/metrics")
    checkpoints_dir: Path = Path("data/processed/checkpoints")
    batch_size: int = 64
    eval_batch_size: int = 128
    num_workers: int = 0
    train: TrainConfig = field(default_factory=TrainConfig)
    model: ModelConfig = field(default_factory=lambda: ModelConfig(name="bert4rec"))
    skip_existing: bool = True
    cpu_threads: int = 2


def json_default(value: object) -> str:
    """Only allow the non-JSON type actually present in experiment dataclasses."""
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def json_bytes(value: object) -> bytes:
    """Stable serialization for configuration identities and persisted artifacts."""
    return orjson.dumps(value, default=json_default, option=orjson.OPT_SORT_KEYS)


def write_json(path: Path, value: object) -> None:
    """Atomically persist complete results; interrupted runs are never skippable."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent, prefix=path.stem, delete=False
    ) as stream:
        temporary = Path(stream.name)
        try:
            stream.write(json_bytes(value))
            stream.flush()
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def config_identity(config: ExperimentConfig) -> str:
    """Hash all hyperparameters, independent of the human-readable run name."""
    return hashlib.sha256(json_bytes(replace(config, exp_name=None))).hexdigest()[:12]


def build_plan(options: LeakageConfig) -> list[RunSpec]:
    """Build 27 unique runs for the full two-dataset suite (no redundant training)."""
    if not options.datasets or len(set(options.datasets)) != len(options.datasets):
        raise ValueError("datasets must be nonempty and unique")
    if options.batch_size < 1 or options.eval_batch_size < 1 or options.num_workers < 0:
        raise ValueError(
            "batch sizes must be positive; num_workers must be nonnegative"
        )
    if (
        options.train.epochs < 1
        or options.train.patience < 1
        or options.train.val_every < 1
    ):
        raise ValueError("epochs, patience and val_every must be positive")
    if options.model.loss_type != "ce":
        raise ValueError("The controlled comparison uses ce for all trainable models")
    runs: dict[str, RunSpec] = {}
    for dataset in options.datasets:
        default_window = default_max_seq_len(dataset)
        requests: list[tuple[ModelName, SplitStrategy, int, float, str]] = []
        if options.suite in ("all", "windows"):
            requests.extend(
                ("bert4rec", "loo", n, 0.1, "windows") for n in WINDOWS[dataset]
            )
        if options.suite in ("all", "masking"):
            requests.extend(
                ("bert4rec", "loo", default_window, ratio, "masking")
                for ratio in (0.0, 0.1, 0.2)
            )
        if options.suite in ("all", "temporal"):
            requests.extend(
                (model, split, default_window, 0.1, "temporal")
                for model in MODELS
                for split in ("loo", "temporal")
            )
        for model, split, window, ratio, purpose in requests:
            config = ExperimentConfig(
                data=DataConfig(
                    dataset=dataset,
                    split_strategy=split,
                    data_dir=options.data_dir,
                    max_seq_len=window,
                    batch_size=options.batch_size,
                    eval_batch_size=options.eval_batch_size,
                    num_workers=options.num_workers,
                    mask_prob=0.2,
                    last_item_mask_ratio=ratio,
                ),
                model=replace(options.model, name=model),
                train=replace(options.train),
                eval=EvalConfig(oracle_middle_mask=False),
                metrics_dir=options.metrics_dir,
                checkpoints_dir=options.checkpoints_dir,
            )
            identity = config_identity(config)
            prefix = "bert4rec" if model == "bert4rec" else f"leakage_{model}"
            config.exp_name = (
                f"{prefix}_{dataset}_{split}_n{window}_last{round(ratio * 100):02d}"
                f"_seed{config.train.seed}_{identity}"
            )
            if identity in runs:
                runs[identity].purposes.append(purpose)
            else:
                runs[identity] = RunSpec(config, [purpose])
    return list(runs.values())


def seed_everything(seed: int) -> None:
    """Seed before data/model construction as well as each evaluation."""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def validate_evaluation(result: Evaluation, config: EvalConfig, count: int) -> None:
    """Reject missing/nonfinite results rather than writing plausible partial metrics."""
    required = {"MRR"} | {f"{name}@{k}" for name in ("HR", "NDCG") for k in config.ks}
    for name in ("full", "sampled_uniform", "sampled_popularity"):
        metrics: RankingMetrics = getattr(result, name)
        if count < 1 or metrics.num_examples != count:
            raise ValueError(
                f"{name}: evaluator sample count differs from the requested cohort"
            )
        keys = (
            required | {"ARP@10", "Coverage@10"}
            if name == "full"
            else {f"HR@{config.primary_k}", f"NDCG@{config.primary_k}"}
        )
        if not keys <= metrics.overall.keys():
            raise ValueError(f"{name}: missing metrics {keys - metrics.overall.keys()}")
        if set(metrics.length_buckets) != set(LEN_BUCKETS):
            raise ValueError(f"{name}: expected all sequence-length buckets")
        if set(metrics.popularity_buckets) != set(POP_BUCKETS):
            raise ValueError(f"{name}: expected all popularity buckets")
        buckets = [
            *metrics.length_buckets.values(),
            *metrics.popularity_buckets.values(),
        ]
        for bucket in buckets:
            if bucket and not {"HR@10", "NDCG@10"} <= bucket.keys():
                raise ValueError(f"{name}: nonempty buckets need HR@10 and NDCG@10")
        for values in [metrics.overall, *buckets]:
            if any(not math.isfinite(value) for value in values.values()):
                raise ValueError(f"{name}: nonfinite metric")


def evaluate_once(
    trained: TrainingResult,
    loader: DataLoader[EvalSample],
    config: ExperimentConfig,
    *,
    oracle: bool = False,
) -> Evaluation:
    """Evaluate deterministically without gradients, preserving the model mode."""
    seed_everything(config.train.seed)
    was_training = trained.model.training
    try:
        trained.model.eval()
        with torch.inference_mode():
            result = trained.evaluate(
                trained.model,
                loader,
                trained.bundle,
                replace(config.eval, oracle_middle_mask=oracle),
                config.train.seed,
            )
        if not isinstance(loader.dataset, Sized):
            raise TypeError("Evaluation requires a finite, sized dataset")
        validate_evaluation(result, config.eval, len(loader.dataset))
        return result
    finally:
        trained.model.train(was_training)


def oracle_comparison(
    trained: TrainingResult, config: ExperimentConfig
) -> dict[str, object]:
    """Compare three modes on identical future-available LOO validation examples."""
    from seqrec.models.bert4rec import BERT4Rec

    if config.data.split_strategy != "loo" or not isinstance(trained.model, BERT4Rec):
        raise ValueError("Oracle is supported only for BERT4Rec on loo_val")
    dataset = trained.bundle.val_loader.dataset
    if not isinstance(dataset, EvalSeqDataset):
        raise TypeError("Oracle cohort selection requires the common EvalSeqDataset")
    indices = [i for i, sample in enumerate(dataset.samples) if sample.future_id > 0]
    if not indices:
        raise ValueError(
            "loo_val has no future-available examples; Oracle cannot be reported"
        )
    loader = DataLoader(
        Subset(dataset, indices),
        batch_size=config.data.eval_batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        collate_fn=EvalSeqDataset.collate_fn,
    )
    model = trained.model
    previous = model.oracle_middle_mask, model.oracle_history_control
    result: dict[str, object] = {
        "split": "loo_val",
        "diagnostic_only": True,
        "num_examples": len(indices),
        "total_validation_examples": len(dataset),
        "missing_future_examples": len(dataset) - len(indices),
        "same_checkpoint": True,
    }
    try:
        for name, oracle, control in (
            ("standard", False, False),
            ("matched_history", False, True),
            ("oracle", True, False),
        ):
            model.oracle_middle_mask, model.oracle_history_control = oracle, control
            result[name] = asdict(evaluate_once(trained, loader, config, oracle=oracle))
    finally:
        model.oracle_middle_mask, model.oracle_history_control = previous
    return result


def flatten_evaluation(result: Evaluation) -> dict[str, float]:
    """Keep the common flat JSON schema in sync with seeded post-hoc metrics."""
    flat = dict(result.full.overall)
    flat["num_queries"] = float(result.full.num_examples)
    for prefix, ranking in (
        ("", result.full),
        ("sampled_uniform/", result.sampled_uniform),
        ("sampled_popularity/", result.sampled_popularity),
    ):
        if prefix:
            flat.update(
                {f"{prefix}{key}": value for key, value in ranking.overall.items()}
            )
        for kind, buckets in (
            ("len", ranking.length_buckets),
            ("pop", ranking.popularity_buckets),
        ):
            for name, values in buckets.items():
                for key, value in values.items():
                    flat[f"{kind}:{name}/{'' if key == 'count' else prefix}{key}"] = (
                        value
                    )
    return flat


class ReportingTrainer(Trainer):
    """Use the unchanged shared loop, adding sparse progress to long-running logs."""

    def _train_epoch(
        self,
        loader: DataLoader[Any],
        optimizer: torch.optim.Optimizer,
        scaler: torch.amp.GradScaler,
        amp_enabled: bool,
        epoch: int,
    ) -> tuple[float, float]:
        loss, seconds = super()._train_epoch(
            loader, optimizer, scaler, amp_enabled, epoch
        )
        if epoch == 1 or epoch % 10 == 0:
            Console().print(
                f"{self.exp_name}: epoch={epoch}, loss={loss:.5f}, train={seconds:.1f}s"
            )
        return loss, seconds


def evaluate_with_engine(
    model: nn.Module,
    loader: DataLoader[EvalSample],
    bundle: DataBundle,
    config: EvalConfig,
    seed: int,
) -> Evaluation:
    """Adapt the common evaluator's flat metrics without reimplementing ranking.

    Sampled metrics are limited to HR/NDCG at primary_k, as in Task 1. Additional
    bucket aggregation uses Evaluator's existing bucket_metrics extension point.
    """
    if not isinstance(model, SequentialRecommender):
        raise TypeError("The common Evaluator requires SequentialRecommender")
    if not config.compute_sampled_metrics or config.primary_k != 10:
        raise ValueError(
            "The comparison suite requires sampled metrics and primary_k=10"
        )
    seed_everything(seed)
    device = next(iter(model.parameters()), bundle.item_pop_counts).device
    # Pop stores counts in a buffer, not a trainable Parameter.
    buffers = list(model.buffers())
    if not list(model.parameters()) and buffers:
        device = buffers[0].device
    bucket_keys = tuple(
        f"{prefix}{metric}@10"
        for prefix in ("", "sampled_uniform/", "sampled_popularity/")
        for metric in ("HR", "NDCG")
    )
    evaluator = Evaluator(
        config,
        bundle.num_items,
        bundle.item_pop_counts,
        device,
        bucket_metrics=bucket_keys,
    )
    flat = evaluator.evaluate(model, loader, show_progress=False)
    rankings: list[RankingMetrics] = []
    for prefix in ("", "sampled_uniform/", "sampled_popularity/"):
        overall = {
            key.removeprefix(prefix): value
            for key, value in flat.items()
            if (
                key.startswith(prefix)
                if prefix
                else "/" not in key and key != "num_queries"
            )
        }
        grouped: list[dict[str, dict[str, float]]] = []
        for kind, names in (("len", LEN_BUCKETS), ("pop", POP_BUCKETS)):
            buckets: dict[str, dict[str, float]] = {}
            for name in names:
                source_prefix = f"{kind}:{name}/{prefix}"
                values = {
                    metric: flat[f"{source_prefix}{metric}"]
                    for metric in ("HR@10", "NDCG@10")
                    if f"{source_prefix}{metric}" in flat
                }
                if values:
                    values["count"] = flat[f"{kind}:{name}/count"]
                buckets[name] = values
            grouped.append(buckets)
        rankings.append(RankingMetrics(int(flat["num_queries"]), overall, *grouped))
    return Evaluation(*rankings)


def train_with_engine(config: ExperimentConfig) -> TrainingResult:
    """Train with Task 1, restore its best checkpoint, and preserve its native schema.

    A completed training cache alongside the checkpoint permits retrying post-hoc
    evaluation after interruption without training twice. It is NOT a resume of
    a partially completed optimizer run. Caches are accepted only when config,
    source, processed data and checkpoint hashes all agree.
    """
    if config.eval.oracle_middle_mask:
        raise ValueError("Oracle must not be used for training or early stopping")
    seed_everything(config.train.seed)
    bundle = build_databundle(config.data, model_name=config.model.name)
    model = build_model(
        config.model, bundle.num_items, bundle.max_seq_len, bundle.item_pop_counts
    )
    trainer = ReportingTrainer(config, model, bundle, show_progress=False)
    cache_path = trainer.checkpoint_path.with_suffix(".training.json")
    signature = {
        "config": config_identity(config),
        "inputs": input_fingerprints(config),
        "source": source_fingerprints(),
        "torch": torch.__version__,
    }
    native: dict[str, Any]
    if cache_path.exists():
        cached = orjson.loads(cache_path.read_bytes())
        if cached["signature"] != signature:
            raise ValueError(
                f"Training cache differs; preserve/archive it before rerunning: {cache_path}"
            )
        native = cached["result"]
        if model.requires_training:
            if file_digest(trainer.checkpoint_path) != cached["checkpoint_sha256"]:
                raise ValueError(
                    "Training checkpoint hash does not match the completed cache"
                )
            state = torch.load(
                trainer.checkpoint_path, map_location=trainer.device, weights_only=True
            )
            model.load_state_dict(state["model"])
        model.to(trainer.device)
    else:
        Console().print(f"[cyan]Training:[/cyan] {config.exp_name}")
        native = trainer.fit().to_dict()
        write_json(
            cache_path,
            {
                "signature": signature,
                "checkpoint_sha256": file_digest(trainer.checkpoint_path)
                if model.requires_training
                else None,
                "result": native,
            },
        )
    history: list[dict[str, float]] = []
    for record in native["history"]:
        row = {
            key: float(record[key])
            for key in ("epoch", "train_loss", "lr", "train_seconds")
        }
        row.update(
            {f"val/{key}": float(value) for key, value in record["val_metrics"].items()}
        )
        history.append(row)
    return TrainingResult(
        model=model,
        bundle=bundle,
        evaluate=evaluate_with_engine,
        checkpoint=trainer.checkpoint_path if model.requires_training else None,
        history=history,
        best_epoch=native["best_epoch"],
        train_seconds=native["timing"]["train_seconds"],
        protocol={
            "ranking": "score descending, ties by item ID ascending (common Evaluator)",
            "history": "PAD and full seen_ids excluded; repeated target retained",
            "sampling": "100 draws WITH replacement; duplicate negatives count separately",
            "popularity": "train-count weights; zero counts excluded; empty eligible mass raises",
            "sampled_metrics": "HR@10 and NDCG@10 only; no sampled MRR available from shared engine",
            "empty_buckets": "empty dict; counts recorded for nonempty buckets",
            "temporal": "existing preprocessing: split-specific frequencies; full global catalogue",
            "oracle": "post-hoc loo_val future-available cohort; same checkpoint and seeded candidates",
            "selection": "standard Full-Ranking validation NDCG@10, never Oracle",
        },
        engine_result=native,
    )


def source_fingerprints() -> dict[str, str]:
    """Identify project code that determines training and evaluation behavior."""
    root = Path(__file__).parents[1]
    return {
        str(path.relative_to(root)): file_digest(path)
        for path in sorted(root.rglob("*.py"))
    }


def file_digest(path: Path) -> str:
    """Hash an artifact without reading the entire file into memory."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def input_fingerprints(config: ExperimentConfig) -> dict[str, str]:
    """Require processed inputs and identify exactly which split was evaluated."""
    directory = config.data.dataset_dir
    names = ["metadata.json"] + [
        f"{config.data.split_strategy}_{part}.parquet"
        for part in ("train", "val", "test", "item_pop")
    ]
    return {name: file_digest(directory / name) for name in names}


def load_backend(name: str | None) -> Callable[[ExperimentConfig], TrainingResult]:
    """Load an explicit adapter; never mistake the current CLI placeholder for training."""
    if not name or ":" not in name:
        raise RuntimeError(
            "Pass --backend module:callable implementing (ExperimentConfig) -> TrainingResult; "
            "the default backend uses Task 1's merged Trainer/Evaluator."
        )
    module, attribute = name.split(":", 1)
    backend = getattr(importlib.import_module(module), attribute)
    if not callable(backend):
        raise TypeError("backend must be callable")
    return cast(Callable[[ExperimentConfig], TrainingResult], backend)


def result_path(spec: RunSpec) -> Path:
    """Use only Task 2-owned metric prefixes, including for other-model comparisons."""
    return spec.config.metrics_dir / f"{spec.config.exp_name}.json"


def run_experiments(options: LeakageConfig, plan: list[RunSpec]) -> None:
    """Run/resume the plan through the shared engine adapter, with post-hoc Oracle."""
    backend = load_backend(options.backend)
    # Fail before any expensive training if any requested processed split is missing.
    fingerprints = {
        config_identity(spec.config): input_fingerprints(spec.config) for spec in plan
    }
    source = source_fingerprints()
    try:
        backend_source = inspect.getsourcefile(backend)
    except TypeError as error:
        raise TypeError(
            "backend must be a source-inspectable Python callable"
        ) from error
    if backend_source is None:
        raise TypeError("backend source is required for reproducible resume checks")
    source["backend"] = file_digest(Path(backend_source))
    console = Console()
    for spec in plan:
        config = spec.config
        path = result_path(spec)
        identity = config_identity(config)
        provenance = {
            "backend": options.backend,
            "inputs": fingerprints[identity],
            "source": source,
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
        }
        if path.exists():
            existing = orjson.loads(path.read_bytes())
            if (
                existing.get("status") != "complete"
                or existing.get("schema_version") != SCHEMA_VERSION
                or existing.get("config") != orjson.loads(json_bytes(config))
                or existing.get("provenance") != provenance
            ):
                raise ValueError(
                    f"Existing result differs from this run; preserve/archive it: {path}"
                )
            if options.skip_existing:
                console.print(f"[green]Already complete:[/green] {path}")
                continue
            raise FileExistsError(
                f"Refusing to overwrite an existing experiment: {path}"
            )
        seed_everything(config.train.seed)
        trained = backend(config)
        if not isinstance(trained, TrainingResult):
            raise TypeError("backend must return TrainingResult")
        if not math.isfinite(trained.train_seconds) or trained.train_seconds < 0:
            raise ValueError("invalid training duration")
        if not trained.protocol:
            raise ValueError("backend must document its evaluation protocol")
        if trained.bundle.max_seq_len != config.data.resolved_max_seq_len:
            raise ValueError("backend returned a different sequence window")
        if trained.bundle.metadata.dataset != config.data.dataset:
            raise ValueError("backend returned a different dataset")
        if any(
            not math.isfinite(value)
            for row in trained.history
            for value in row.values()
        ):
            raise ValueError("training/validation history contains nonfinite values")
        if config.model.name != "pop":
            if trained.checkpoint is None or not trained.checkpoint.is_file():
                raise ValueError(
                    "trainable models must provide a saved best checkpoint"
                )
            if (
                not trained.history
                or trained.best_epoch is None
                or trained.best_epoch < 1
            ):
                raise ValueError(
                    "trainable models must provide training/validation history"
                )
        if getattr(trained.model, "oracle_middle_mask", False) or getattr(
            trained.model, "oracle_history_control", False
        ):
            raise ValueError(
                "backend returned a model with diagnostic evaluation enabled"
            )
        standard_val = evaluate_once(trained, trained.bundle.val_loader, config)
        standard_test = evaluate_once(trained, trained.bundle.test_loader, config)
        # Always include Oracle for default-window LOO BERT runs, so resume does
        # not depend on whether this run originated from windows/masking/temporal.
        oracle = None
        if (
            config.model.name == "bert4rec"
            and config.data.split_strategy == "loo"
            and config.data.resolved_max_seq_len
            == default_max_seq_len(config.data.dataset)
        ):
            oracle = oracle_comparison(trained, config)
        checkpoint = trained.checkpoint
        write_json(
            path,
            {
                **trained.engine_result,
                "schema_version": SCHEMA_VERSION,
                "status": "complete",
                "config": config,
                "provenance": provenance,
                "protocol": trained.protocol,
                "checkpoint": checkpoint,
                "checkpoint_sha256": file_digest(checkpoint) if checkpoint else None,
                "best_epoch": trained.best_epoch,
                "train_seconds": trained.train_seconds,
                "history": trained.engine_result.get("history", trained.history),
                "best_val_metrics": flatten_evaluation(standard_val),
                "test_metrics": flatten_evaluation(standard_test),
                "validation": standard_val,
                "test": standard_test,
                "oracle_validation": oracle,
            },
        )
        console.print(
            f"[green]Saved:[/green] {path} "
            f"(best epoch={trained.best_epoch}, test NDCG@10={standard_test.full.overall['NDCG@10']:.6f})"
        )
        del trained
        save_summary(options, plan)


def summarize(plan: list[RunSpec]) -> dict[str, object]:
    """Summarize actual results only; preserve missing experiments as explicit gaps."""
    rows: list[dict[str, object]] = []
    missing: list[str] = []
    temporal: dict[tuple[str, str, int], dict[str, float]] = {}
    rankings: dict[tuple[str, str], dict[str, dict[str, float]]] = {}
    for spec in plan:
        path = result_path(spec)
        if not path.is_file():
            missing.append(str(path))
            continue
        record = orjson.loads(path.read_bytes())
        if (
            record.get("status") != "complete"
            or record.get("schema_version") != SCHEMA_VERSION
            or record.get("config") != orjson.loads(json_bytes(spec.config))
        ):
            raise ValueError(f"Incomplete or mismatched result: {path}")
        config = spec.config
        full = record["test"]["full"]["overall"]["NDCG@10"]
        uniform = record["test"]["sampled_uniform"]["overall"]["NDCG@10"]
        popularity = record["test"]["sampled_popularity"]["overall"]["NDCG@10"]
        row: dict[str, object] = {
            "path": str(path),
            "config": config,
            "purposes": spec.purposes,
            "full_hr10": record["test"]["full"]["overall"]["HR@10"],
            "full_ndcg10": full,
            "uniform_ndcg10": uniform,
            "popularity_ndcg10": popularity,
            "uniform_minus_full": uniform - full,
            "popularity_minus_full": popularity - full,
            "uniform_over_full": uniform / full if full > 0 else None,
            "popularity_over_full": popularity / full if full > 0 else None,
        }
        oracle = record["oracle_validation"]
        if oracle is not None:
            scores = {
                mode: oracle[mode]["full"]["overall"]["NDCG@10"]
                for mode in ("standard", "matched_history", "oracle")
            }
            row["oracle_comparison"] = {
                "num_examples": oracle["num_examples"],
                "ndcg10": scores,
                "oracle_minus_standard": scores["oracle"] - scores["standard"],
                "oracle_minus_matched_history": scores["oracle"]
                - scores["matched_history"],
            }
        rows.append(row)
        if "temporal" in spec.purposes:
            key = (config.data.dataset, config.model.name, config.train.seed)
            temporal.setdefault(key, {})[config.data.split_strategy] = full
            group = rankings.setdefault(
                (config.data.dataset, config.data.split_strategy), {}
            )
            group[config.model.name] = {
                "full": full,
                "uniform": uniform,
                "popularity": popularity,
            }
    deltas = [
        {
            "dataset": dataset,
            "model": model,
            "seed": seed,
            "ndcg10": scores,
            "temporal_minus_loo": scores["temporal"] - scores["loo"],
        }
        for (dataset, model, seed), scores in temporal.items()
        if {"loo", "temporal"} <= scores.keys()
    ]
    reversals: list[dict[str, str]] = []
    for (dataset, split), group in rankings.items():
        names = sorted(group)
        for i, left in enumerate(names):
            for right in names[i + 1 :]:
                for protocol in ("uniform", "popularity"):
                    if (group[left]["full"] - group[right]["full"]) * (
                        group[left][protocol] - group[right][protocol]
                    ) < 0:
                        reversals.append(
                            {
                                "dataset": dataset,
                                "split": split,
                                "left": left,
                                "right": right,
                                "sampled": protocol,
                            }
                        )
    return {
        "schema_version": SCHEMA_VERSION,
        "complete": not missing,
        "missing": missing,
        "experiments": rows,
        "temporal_deltas": deltas,
        "ranking_reversals": reversals,
        "caveat": "LOO/temporal populations differ; deltas alone do not establish leakage causality.",
    }


def save_summary(options: LeakageConfig, plan: list[RunSpec]) -> Path:
    """Update only the task-owned summary for this exact experiment grid."""
    identity = hashlib.sha256(json_bytes(plan)).hexdigest()[:12]
    path = options.metrics_dir / f"leakage_summary_{options.suite}_{identity}.json"
    write_json(path, summarize(plan))
    return path


def main() -> None:
    """Expose planning, execution and explicit partial-result summaries via tyro."""
    options = tyro.cli(LeakageConfig)
    if options.cpu_threads < 1:
        raise ValueError("cpu_threads must be positive")
    torch.set_num_threads(options.cpu_threads)
    plan = build_plan(options)
    if options.action == "plan":
        Console().print_json(json_bytes({"num_runs": len(plan), "runs": plan}).decode())
    elif options.action == "run":
        run_experiments(options, plan)
        Console().print(f"Summary: {save_summary(options, plan)}")
    else:
        path = save_summary(options, plan)
        Console().print(f"Summary (see complete/missing fields): {path}")


if __name__ == "__main__":
    # Keep dataclass identity canonical: external adapters import TrainingResult
    # from seqrec.analysis.leakage_exp, not from a second __main__ module.
    importlib.import_module("seqrec.analysis.leakage_exp").main()
