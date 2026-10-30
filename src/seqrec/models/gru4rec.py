"""GRU4Rec: gated recurrent unit based next-item recommendation.

Reference: Hidasi et al., *Session-based Recommendations with Recurrent Neural Networks* (ICLR 2016).

The implementation follows the unified "causal next-item" contract: the model consumes
``CausalTrainBatch`` (``input_ids``/``target_ids`` of shape ``[B, N]`` with right padding and
``seq_len`` of shape ``[B]``) and scores items from the GRU hidden state at the last valid
position.
"""

from __future__ import annotations

from typing import ClassVar

import torch
from torch import nn

from seqrec.config import LossType
from seqrec.data.dataset import CausalTrainBatch, EvalBatch
from seqrec.models.base import (
    Paradigm,
    SequentialRecommender,
    full_bce_loss,
    full_softmax_ce_loss,
    gather_last_valid,
    init_item_embedding,
    prepend_pad_column,
    sampled_bpr_max_loss,
    select_supervised_positions,
)

__all__ = ["GRU4Rec"]


class GRU4Rec(SequentialRecommender):
    """Item-embedding + GRU encoder with a tied (or linear-projected) output layer.

    Args:
        num_items: Number of real items ``V`` (IDs ``1..V``; ``0`` is ``<PAD>``).
        hidden_size: GRU hidden state / item embedding dimension.
        num_layers: Number of stacked GRU layers.
        dropout: Dropout applied to embeddings, GRU outputs and the output projection.
        loss_type: Training objective, one of ``ce`` (full softmax), ``bce`` (full binary
            cross entropy with one-hot labels) or ``bpr_max`` (sampled pairwise BPR-max).
        share_item_embeddings: If ``True`` the output layer is tied to the item embedding
            matrix; otherwise a linear projection is applied before the dot product.
    """

    paradigm: ClassVar[Paradigm] = "causal"

    def __init__(
        self,
        num_items: int,
        hidden_size: int = 64,
        num_layers: int = 1,
        dropout: float = 0.2,
        loss_type: LossType = "ce",
        share_item_embeddings: bool = True,
    ) -> None:
        super().__init__(num_items)
        if loss_type not in ("ce", "bce", "bpr_max"):
            raise ValueError(f"Unsupported loss_type for GRU4Rec: {loss_type!r}")
        self.hidden_size = hidden_size
        self.loss_type = loss_type
        self.share_item_embeddings = share_item_embeddings

        self.item_embedding = init_item_embedding(
            nn.Embedding(num_items + 1, hidden_size, padding_idx=0)
        )
        self.emb_dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.output = (
            nn.Identity()
            if share_item_embeddings
            else nn.Linear(hidden_size, hidden_size)
        )
        self.output_dropout = nn.Dropout(dropout)

    def encode(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Run the GRU over a right-padded ID sequence.

        Args:
            input_ids: Item IDs of shape ``[B, N]`` (``0`` = ``<PAD>``).

        Returns:
            Hidden states of shape ``[B, N, H]``.
        """
        embeddings = self.emb_dropout(self.item_embedding(input_ids))
        hidden, _ = self.gru(embeddings)
        return hidden

    def project(self, hidden: torch.Tensor) -> torch.Tensor:
        """Apply the (optional) output projection and dropout used before item scoring."""
        return self.output_dropout(self.output(hidden))

    def _item_logits(self, hidden: torch.Tensor) -> torch.Tensor:
        """Dot product between projected hidden states and the item embedding matrix."""
        return hidden @ self.item_embedding.weight[1:].T

    def compute_loss(self, batch: CausalTrainBatch) -> torch.Tensor:
        """Next-item loss over all supervised positions of the batch."""
        hidden = self.encode(batch["input_ids"])
        flat_hidden, flat_targets = select_supervised_positions(
            hidden, batch["target_ids"]
        )
        projected = self.project(flat_hidden)
        embeddings = self.item_embedding.weight
        if self.loss_type == "ce":
            return full_softmax_ce_loss(projected, embeddings, flat_targets)
        if self.loss_type == "bce":
            return full_bce_loss(projected, embeddings, flat_targets)
        return sampled_bpr_max_loss(projected, embeddings, flat_targets)

    def score_items(self, batch: EvalBatch) -> torch.Tensor:
        """Score all items from the hidden state at the last valid sequence position."""
        hidden = self.encode(batch["input_ids"])
        last_hidden = self.project(gather_last_valid(hidden, batch["seq_len"]))
        return prepend_pad_column(self._item_logits(last_hidden))

    def extra_repr(self) -> str:
        return (
            f"num_items={self.num_items}, hidden_size={self.hidden_size}, "
            f"loss_type={self.loss_type}, share_item_embeddings={self.share_item_embeddings}"
        )
