"""Model registry: every sequential recommender plus a ``build_model`` factory.

``bert4rec`` and ``srgnn`` are imported lazily so that this module keeps working while the
other model files are still being developed in parallel branches.
"""

from __future__ import annotations

import importlib
import inspect
from typing import Any

import torch

from seqrec.config import ModelConfig, ModelName
from seqrec.models.base import SequentialRecommender
from seqrec.models.baselines import Pop
from seqrec.models.gru4rec import GRU4Rec
from seqrec.models.sasrec import SASRec

__all__ = [
    "GRU4Rec",
    "Pop",
    "SASRec",
    "SequentialRecommender",
    "build_model",
    "requires_training",
]

#: Models owned by other tasks, resolved lazily as ``model name -> (module, class name)``.
_LAZY_MODELS: dict[str, tuple[str, str]] = {
    "bert4rec": ("seqrec.models.bert4rec", "BERT4Rec"),
    "srgnn": ("seqrec.models.srgnn", "SRGNN"),
}


def build_model(
    config: ModelConfig,
    num_items: int,
    max_seq_len: int,
    item_pop_counts: torch.Tensor | None = None,
) -> SequentialRecommender:
    """Instantiate the model described by ``config``.

    Args:
        config: Model hyper-parameters (architecture name, sizes, loss type, ...).
        num_items: Number of real items ``V`` in the dataset.
        max_seq_len: Maximum truncation window ``N`` (position embedding size / graph width).
        item_pop_counts: Training-set popularity counts of shape ``[V + 1]``; required by the
            ``Pop`` baseline and ignored by every other model.

    Returns:
        A ready-to-train :class:`~seqrec.models.base.SequentialRecommender`.
    """
    name: ModelName = config.name

    if name == "pop":
        if item_pop_counts is None:
            raise ValueError(
                "The 'pop' baseline requires `item_pop_counts` from the DataBundle."
            )
        return Pop(num_items=num_items, item_pop_counts=item_pop_counts)

    if name == "gru4rec":
        return GRU4Rec(
            num_items=num_items,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            dropout=config.dropout,
            loss_type=config.loss_type,
            share_item_embeddings=config.share_item_embeddings,
        )

    if name == "sasrec":
        return SASRec(
            num_items=num_items,
            max_seq_len=max_seq_len,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            num_heads=config.num_heads,
            dropout=config.dropout,
            loss_type=config.loss_type,
            share_item_embeddings=config.share_item_embeddings,
        )

    return _build_lazy_model(name, config, num_items, max_seq_len, item_pop_counts)


def _build_lazy_model(
    name: str,
    config: ModelConfig,
    num_items: int,
    max_seq_len: int,
    item_pop_counts: torch.Tensor | None,
) -> SequentialRecommender:
    """Instantiate a model that lives in a file owned by another task (``bert4rec``/``srgnn``)."""
    if name not in _LAZY_MODELS:
        raise ValueError(f"Unknown model name: {name!r}")
    module_name, class_name = _LAZY_MODELS[name]
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:  # pragma: no cover - depends on parallel branch state
        raise ImportError(
            f"Model {name!r} is not available yet (missing {module_name}). "
            "It is implemented by another task; see PLAN.md."
        ) from exc

    model_cls: type[Any] = getattr(module, class_name)
    candidates: dict[str, Any] = {
        "config": config,
        "num_items": num_items,
        "max_seq_len": max_seq_len,
        "hidden_size": config.hidden_size,
        "num_layers": config.num_layers,
        "num_heads": config.num_heads,
        "dropout": config.dropout,
        "loss_type": config.loss_type,
        "share_item_embeddings": config.share_item_embeddings,
        "ggnn_steps": config.ggnn_steps,
        "item_pop_counts": item_pop_counts,
    }
    accepted = set(inspect.signature(model_cls.__init__).parameters)  # type: ignore[arg-type]
    kwargs = {key: value for key, value in candidates.items() if key in accepted}
    model = model_cls(**kwargs)
    if not isinstance(model, SequentialRecommender):
        raise TypeError(f"{class_name} must inherit from SequentialRecommender")
    return model


def requires_training(model: SequentialRecommender) -> bool:
    """Whether ``model`` has parameters that must be optimised."""
    return bool(model.requires_training)
