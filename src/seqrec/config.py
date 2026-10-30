from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

DatasetName = Literal["ml-1m", "beauty"]
SplitStrategy = Literal["loo", "temporal"]
ModelName = Literal["pop", "gru4rec", "sasrec", "bert4rec", "srgnn"]
LossType = Literal["ce", "bce", "bpr_max"]

LEN_BUCKETS: tuple[str, ...] = ("<=5", "6-15", "16-50", ">50")
POP_BUCKETS: tuple[str, ...] = ("Head", "Mid", "Tail")

LEN_BUCKET_TO_ID: dict[str, int] = {name: idx for idx, name in enumerate(LEN_BUCKETS)}
POP_BUCKET_TO_ID: dict[str, int] = {name: idx for idx, name in enumerate(POP_BUCKETS)}


def default_max_seq_len(dataset: DatasetName) -> int:
    """Return the default maximum sequence truncation length N for a dataset."""
    return 200 if dataset == "ml-1m" else 50


@dataclass(slots=True)
class PreprocessConfig:
    """Configuration for dataset downloading, 5-core filtering, and splitting."""

    dataset: DatasetName | Literal["all"] = "all"
    min_interactions: int = 5
    output_dir: Path = Path("data/processed")
    temporal_train_ratio: float = 0.8
    temporal_val_ratio: float = 0.1
    force: bool = False


@dataclass(slots=True)
class DataConfig:
    """Configuration for dataset loading, sequence padding, and DataLoader creation."""

    dataset: DatasetName = "beauty"
    split_strategy: SplitStrategy = "loo"
    data_dir: Path = Path("data/processed")
    max_seq_len: int | None = None
    batch_size: int = 256
    eval_batch_size: int = 256
    num_workers: int = 0
    # Causal dataset options
    sliding_window: bool = False
    sliding_step: int = 50
    # BERT4Rec Cloze dataset options
    mask_prob: float = 0.2
    last_item_mask_ratio: float = 0.1
    # SR-GNN Session-Graph dataset options
    max_prefixes_per_user: int = 50

    @property
    def resolved_max_seq_len(self) -> int:
        if self.max_seq_len is not None:
            return self.max_seq_len
        return default_max_seq_len(self.dataset)

    @property
    def dataset_dir(self) -> Path:
        return self.data_dir / self.dataset


@dataclass(slots=True)
class ModelConfig:
    """Configuration for sequential recommendation model architectures."""

    name: ModelName = "sasrec"
    hidden_size: int = 64
    num_layers: int = 2
    num_heads: int = 2
    dropout: float = 0.2
    loss_type: LossType = "ce"
    ggnn_steps: int = 1
    share_item_embeddings: bool = True


@dataclass(slots=True)
class TrainConfig:
    """Configuration for model training and optimization."""

    epochs: int = 100
    lr: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip: float = 5.0
    patience: int = 10
    val_every: int = 1
    use_amp: bool = True
    seed: int = 42
    device: str = "cuda"


@dataclass(slots=True)
class EvalConfig:
    """Configuration for ranking evaluation metrics and protocols."""

    ks: tuple[int, ...] = (5, 10, 20)
    primary_k: int = 10
    num_neg_samples: int = 100
    compute_sampled_metrics: bool = True
    oracle_middle_mask: bool = False


@dataclass(slots=True)
class ExperimentConfig:
    """Top-level configuration for a complete training and evaluation run."""

    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    exp_name: str | None = None
    metrics_dir: Path = Path("reports/metrics")
    checkpoints_dir: Path = Path("data/processed/checkpoints")
