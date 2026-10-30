"""Unified trainer: optimisation loop, early stopping, checkpointing and metric persistence.

A single :class:`Trainer` drives every model through the
:class:`~seqrec.models.base.SequentialRecommender` contract, so AMP, gradient clipping,
early stopping on validation ``NDCG@10`` and ``orjson`` metric dumps are implemented exactly
once.
"""

from __future__ import annotations

import dataclasses
import random
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import orjson
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from seqrec.config import ExperimentConfig
from seqrec.data.dataset import DataBundle, build_databundle
from seqrec.engine.evaluator import Evaluator, move_batch_to_device
from seqrec.models import build_model
from seqrec.models.base import SequentialRecommender

__all__ = [
    "EpochRecord",
    "ExperimentResult",
    "Trainer",
    "default_exp_name",
    "resolve_device",
    "run_experiment",
    "seed_everything",
]


def resolve_device(name: str) -> torch.device:
    """Resolve a device string, falling back to CPU (with a warning) when CUDA is unavailable."""
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        warnings.warn(
            f"Requested device {name!r} but CUDA is not available; falling back to CPU.",
            stacklevel=2,
        )
        return torch.device("cpu")
    return device


def seed_everything(seed: int) -> None:
    """Seed Python and PyTorch RNGs for reproducible runs."""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def default_exp_name(config: ExperimentConfig) -> str:
    """``{model}_{dataset}_{split}_n{N}`` unless an explicit ``exp_name`` is given."""
    if config.exp_name:
        return config.exp_name
    return (
        f"{config.model.name}_{config.data.dataset}_"
        f"{config.data.split_strategy}_n{config.data.resolved_max_seq_len}"
    )


@dataclass(slots=True)
class EpochRecord:
    """One epoch of the training curve."""

    epoch: int
    train_loss: float
    lr: float
    train_seconds: float
    val_metrics: dict[str, float] = field(default_factory=dict)
    val_primary: float = float("nan")


@dataclass(slots=True)
class ExperimentResult:
    """Everything produced by a single training + evaluation run."""

    exp_name: str
    config: ExperimentConfig
    dataset_summary: dict[str, Any]
    device: str
    num_parameters: int
    history: list[EpochRecord]
    best_epoch: int | None
    best_val_metrics: dict[str, float]
    test_metrics: dict[str, float]
    train_seconds: float
    eval_seconds: float

    def to_dict(self) -> dict[str, Any]:
        """Serialise the result into a JSON-ready dictionary."""
        return {
            "exp_name": self.exp_name,
            "config": dataclasses.asdict(self.config),
            "dataset": self.dataset_summary,
            "device": self.device,
            "num_parameters": self.num_parameters,
            "best_epoch": self.best_epoch,
            "best_val_metrics": self.best_val_metrics,
            "test_metrics": self.test_metrics,
            "history": [dataclasses.asdict(record) for record in self.history],
            "timing": {
                "train_seconds": self.train_seconds,
                "eval_seconds": self.eval_seconds,
                "total_seconds": self.train_seconds + self.eval_seconds,
            },
        }

    def save(self, path: Path) -> Path:
        """Persist the result as indented ``orjson`` JSON."""
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = orjson.dumps(
            self.to_dict(), option=orjson.OPT_INDENT_2, default=_json_default
        )
        path.write_bytes(payload)
        return path


def _json_default(obj: Any) -> Any:
    """Fallback encoder for ``pathlib.Path`` and other non-native JSON types."""
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (set, frozenset)):
        return sorted(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serialisable")


class Trainer:
    """Train (and evaluate) a single model on a single dataset / window configuration.

    Args:
        config: Full experiment configuration.
        model: The model to train; must implement the ``SequentialRecommender`` contract.
        bundle: Data bundle holding the loaders, popularity vectors and metadata.
        exp_name: Optional experiment name override (defaults to ``default_exp_name``).
        show_progress: Whether to display ``tqdm`` progress bars.
    """

    def __init__(
        self,
        config: ExperimentConfig,
        model: SequentialRecommender,
        bundle: DataBundle,
        exp_name: str | None = None,
        show_progress: bool = True,
    ) -> None:
        self.config = config
        self.model = model
        self.bundle = bundle
        self.exp_name = exp_name or default_exp_name(config)
        self.show_progress = show_progress
        self.device = resolve_device(config.train.device)
        self.evaluator = Evaluator(
            config=config.eval,
            num_items=bundle.num_items,
            item_pop_counts=bundle.item_pop_counts,
            device=self.device,
        )
        self.checkpoint_path = config.checkpoints_dir / f"{self.exp_name}.pt"

    # ------------------------------------------------------------------ public API

    def fit(self) -> ExperimentResult:
        """Run the full training + validation + test pipeline."""
        seed_everything(self.config.train.seed)
        self.model.to(self.device)

        train_start = time.perf_counter()
        history: list[EpochRecord] = []
        best_epoch: int | None = None
        best_val_metrics: dict[str, float] = {}

        if self.model.requires_training:
            history, best_epoch, best_val_metrics = self._fit_supervised()
        else:
            best_val_metrics = self.evaluator.evaluate(
                self.model,
                self.bundle.val_loader,
                "val",
                show_progress=self.show_progress,
            )
        train_seconds = time.perf_counter() - train_start

        eval_start = time.perf_counter()
        test_metrics = self.evaluator.evaluate(
            self.model,
            self.bundle.test_loader,
            "test",
            show_progress=self.show_progress,
        )
        eval_seconds = time.perf_counter() - eval_start

        return ExperimentResult(
            exp_name=self.exp_name,
            config=self.config,
            dataset_summary=self._dataset_summary(),
            device=str(self.device),
            num_parameters=self.model.num_trainable_parameters(),
            history=history,
            best_epoch=best_epoch,
            best_val_metrics=best_val_metrics,
            test_metrics=test_metrics,
            train_seconds=train_seconds,
            eval_seconds=eval_seconds,
        )

    # ------------------------------------------------------------------ internals

    def _fit_supervised(
        self,
    ) -> tuple[list[EpochRecord], int | None, dict[str, float]]:
        """Run the optimisation loop with early stopping and best-checkpoint restoration."""
        train_loader = self.bundle.train_loader
        if train_loader is None:
            raise ValueError(
                f"Model {type(self.model).__name__} requires a training DataLoader, but none was built."
            )

        train_cfg = self.config.train
        primary_key = f"NDCG@{self.config.eval.primary_k}"
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=train_cfg.lr,
            weight_decay=train_cfg.weight_decay,
        )
        amp_enabled = train_cfg.use_amp and self.device.type == "cuda"
        scaler = torch.amp.GradScaler(self.device.type, enabled=amp_enabled)

        history: list[EpochRecord] = []
        best_primary = float("-inf")
        best_epoch: int | None = None
        best_val_metrics: dict[str, float] = {}
        epochs_without_improvement = 0

        for epoch in range(1, train_cfg.epochs + 1):
            train_loss, seconds = self._train_epoch(
                train_loader, optimizer, scaler, amp_enabled, epoch
            )
            record = EpochRecord(
                epoch=epoch,
                train_loss=train_loss,
                lr=float(optimizer.param_groups[0]["lr"]),
                train_seconds=seconds,
            )

            if epoch % max(1, train_cfg.val_every) == 0:
                val_metrics = self.evaluator.evaluate(
                    self.model,
                    self.bundle.val_loader,
                    "val",
                    show_progress=self.show_progress,
                )
                record.val_metrics = val_metrics
                record.val_primary = val_metrics.get(primary_key, float("nan"))
                if record.val_primary > best_primary:
                    best_primary = record.val_primary
                    best_epoch = epoch
                    best_val_metrics = val_metrics
                    epochs_without_improvement = 0
                    self._save_checkpoint(epoch, val_metrics)
                else:
                    epochs_without_improvement += 1
            history.append(record)

            if epochs_without_improvement >= train_cfg.patience:
                break

        self._load_best_checkpoint()
        return history, best_epoch, best_val_metrics

    def _train_epoch(
        self,
        loader: DataLoader[Any],
        optimizer: torch.optim.Optimizer,
        scaler: torch.amp.GradScaler,
        amp_enabled: bool,
        epoch: int,
    ) -> tuple[float, float]:
        """Run one optimisation epoch and return ``(mean_loss, seconds)``."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        start = time.perf_counter()
        progress = tqdm(
            loader,
            desc=f"train/epoch{epoch}",
            leave=False,
            disable=not self.show_progress,
            dynamic_ncols=True,
        )
        for raw_batch in progress:
            batch = move_batch_to_device(raw_batch, self.device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=self.device.type, enabled=amp_enabled):
                loss = self.model.compute_loss(batch)
            scaler.scale(loss).backward()
            if self.config.train.grad_clip > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.train.grad_clip
                )
            scaler.step(optimizer)
            scaler.update()

            total_loss += float(loss.detach().item())
            num_batches += 1
            progress.set_postfix(loss=f"{total_loss / max(num_batches, 1):.4f}")
        return total_loss / max(num_batches, 1), time.perf_counter() - start

    def _save_checkpoint(self, epoch: int, val_metrics: dict[str, float]) -> None:
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model": self.model.state_dict(),
                "epoch": epoch,
                "val_metrics": val_metrics,
                "exp_name": self.exp_name,
            },
            self.checkpoint_path,
        )

    def _load_best_checkpoint(self) -> None:
        if not self.checkpoint_path.exists():
            return
        state = torch.load(
            self.checkpoint_path, map_location=self.device, weights_only=True
        )
        self.model.load_state_dict(state["model"])

    def _dataset_summary(self) -> dict[str, Any]:
        metadata = self.bundle.metadata
        return {
            "dataset": metadata.dataset,
            "split_strategy": self.config.data.split_strategy,
            "num_users": metadata.num_users,
            "num_items": metadata.num_items,
            "num_interactions": metadata.num_interactions,
            "density_pct": metadata.density_pct,
            "avg_seq_len": metadata.avg_seq_len,
            "median_seq_len": metadata.median_seq_len,
            "p90_seq_len": metadata.p90_seq_len,
            "repeat_click_ratio": metadata.repeat_click_ratio,
            "max_seq_len": self.bundle.max_seq_len,
        }


def run_experiment(
    config: ExperimentConfig, show_progress: bool = True
) -> ExperimentResult:
    """Build the data bundle and model, run :class:`Trainer`, and persist the metrics JSON.

    Args:
        config: Full experiment configuration (dataset, model, optimisation, evaluation).
        show_progress: Whether to display ``tqdm`` progress bars.

    Returns:
        The completed :class:`ExperimentResult`.
    """
    bundle = build_databundle(config.data, model_name=config.model.name)
    model = build_model(
        config.model,
        num_items=bundle.num_items,
        max_seq_len=bundle.max_seq_len,
        item_pop_counts=bundle.item_pop_counts,
    )
    trainer = Trainer(config, model, bundle, show_progress=show_progress)
    result = trainer.fit()
    result.save(config.metrics_dir / f"{result.exp_name}.json")
    return result
