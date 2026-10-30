"""Popularity baseline.

``Pop`` is intentionally parameter free: it scores every item with its training-set
interaction frequency and therefore needs no gradient based optimisation. It serves as the
"how much can pure popularity get you" reference point for every dataset.
"""

from __future__ import annotations

from typing import Any, ClassVar

import torch
from torch import nn

from seqrec.data.dataset import EvalBatch
from seqrec.models.base import Paradigm, SequentialRecommender

__all__ = ["Pop"]


class Pop(SequentialRecommender):
    """Non-personalised popularity ranker driven by ``DataBundle.item_pop_counts``.

    Args:
        num_items: Number of real items ``V``.
        item_pop_counts: Training-set interaction counts of shape ``[V + 1]`` where index
            ``0`` is the ``<PAD>`` slot and must be ``0``.
    """

    paradigm: ClassVar[Paradigm] = "stateless"
    requires_training: ClassVar[bool] = False

    pop_scores: torch.Tensor

    def __init__(self, num_items: int, item_pop_counts: torch.Tensor) -> None:
        super().__init__(num_items)
        if item_pop_counts.numel() != num_items + 1:
            raise ValueError(
                f"item_pop_counts must have {num_items + 1} entries, got {item_pop_counts.numel()}"
            )
        # log1p keeps the ranking identical while avoiding huge magnitudes; the <PAD> row is
        # pinned to -inf so it can never be recommended.
        scores = torch.log1p(item_pop_counts.float().clone())
        scores[0] = float("-inf")
        self.register_buffer("pop_scores", scores)

    def compute_loss(self, batch: Any) -> torch.Tensor:
        """``Pop`` has no trainable parameters."""
        raise NotImplementedError(
            "Pop is a non-parametric baseline and cannot be trained."
        )

    def score_items(self, batch: EvalBatch) -> torch.Tensor:
        """Broadcast the global popularity vector to every row of the eval batch."""
        batch_size = batch["input_ids"].size(0)
        return self.pop_scores.unsqueeze(0).expand(batch_size, -1)

    def predict_topk(self, batch: EvalBatch, k: int) -> torch.Tensor:
        """Top-``k`` popular items (identical for every user)."""
        k = min(k, self.num_items)
        return self.pop_scores[1:].topk(k).indices + 1

    def extra_repr(self) -> str:
        return f"num_items={self.num_items}, requires_training=False"


def is_popularity_model(model: nn.Module) -> bool:
    """Whether ``model`` is the non-trainable popularity baseline."""
    return isinstance(model, Pop)
