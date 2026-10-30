"""SASRec: self-attentive sequential recommendation with a *causal* (unidirectional) encoder.

Reference: Kang & McAuley, *Self-Attentive Sequential Recommendation* (ICDM 2018).

This is the unidirectional counterpart of ``BERT4Rec``: position ``t`` may only attend to
positions ``<= t`` (``is_causal=True``), which keeps training and inference consistent
("predict the next item from the past only").
"""

from __future__ import annotations

from typing import ClassVar

import torch
import torch.nn.functional as F
from torch import nn

from seqrec.config import LossType
from seqrec.data.dataset import CausalTrainBatch, EvalBatch
from seqrec.models.base import (
    Paradigm,
    SequentialRecommender,
    full_bce_loss,
    full_softmax_ce_loss,
    gather_positions,
    init_item_embedding,
    prepend_pad_column,
    score_position,
    select_supervised_positions,
)

__all__ = ["CausalTransformerBlock", "SASRec"]


class CausalTransformerBlock(nn.Module):
    """Pre-LayerNorm Transformer block with causal self-attention and a point-wise FFN.

    Because the model consumes right-padded sequences and only ever reads the hidden state at
    the last valid position, padding keys can never influence a valid query (all keys
    ``<= t`` of a valid query ``t`` are valid as well). No explicit key-padding mask is needed.

    Args:
        hidden_size: Model dimension ``d``.
        num_heads: Number of attention heads; must divide ``hidden_size``.
        dropout: Dropout applied to attention weights and sub-layer outputs.
        ffn_multiplier: Hidden dimension multiplier of the point-wise feed-forward network.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        dropout: float,
        ffn_multiplier: int = 4,
    ) -> None:
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError(
                f"hidden_size={hidden_size} must be divisible by num_heads={num_heads}"
            )
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.attention_dropout = dropout

        self.attn_norm = nn.LayerNorm(hidden_size)
        self.qkv = nn.Linear(hidden_size, 3 * hidden_size)
        self.attn_out = nn.Linear(hidden_size, hidden_size)
        self.attn_dropout = nn.Dropout(dropout)

        self.ffn_norm = nn.LayerNorm(hidden_size)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, ffn_multiplier * hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_multiplier * hidden_size, hidden_size),
        )
        self.ffn_dropout = nn.Dropout(dropout)

    def _split_heads(self, tensor: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = tensor.shape
        return tensor.view(
            batch_size, seq_len, self.num_heads, self.head_dim
        ).transpose(1, 2)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:  # ty: ignore[invalid-mutable-override]
        """Apply causal self-attention followed by a feed-forward network.

        Args:
            hidden: Input representations of shape ``[B, N, H]``.

        Returns:
            Output representations of shape ``[B, N, H]``.
        """
        query, key, value = self.qkv(self.attn_norm(hidden)).chunk(3, dim=-1)
        attended = F.scaled_dot_product_attention(
            self._split_heads(query),
            self._split_heads(key),
            self._split_heads(value),
            dropout_p=self.attention_dropout if self.training else 0.0,
            is_causal=True,
        )
        attended = attended.transpose(1, 2).reshape(hidden.shape)
        hidden = hidden + self.attn_dropout(self.attn_out(attended))
        return hidden + self.ffn_dropout(self.ffn(self.ffn_norm(hidden)))


class SASRec(SequentialRecommender):
    """Item + learnable positional embeddings followed by ``L`` causal Transformer blocks.

    Args:
        num_items: Number of real items ``V`` (IDs ``1..V``; ``0`` is ``<PAD>``).
        max_seq_len: Maximum truncation window ``N``; also the size of the position table.
        hidden_size: Model dimension ``d``.
        num_layers: Number of stacked causal Transformer blocks.
        num_heads: Number of self-attention heads.
        dropout: Dropout rate.
        loss_type: ``ce`` (full softmax) or ``bce`` (full binary cross entropy).
        share_item_embeddings: Tie the output layer to the input item embedding matrix.
    """

    paradigm: ClassVar[Paradigm] = "causal"

    def __init__(
        self,
        num_items: int,
        max_seq_len: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        num_heads: int = 2,
        dropout: float = 0.2,
        loss_type: LossType = "ce",
        share_item_embeddings: bool = True,
    ) -> None:
        super().__init__(num_items)
        if loss_type not in ("ce", "bce"):
            raise ValueError(f"Unsupported loss_type for SASRec: {loss_type!r}")
        self.max_seq_len = max_seq_len
        self.hidden_size = hidden_size
        self.loss_type = loss_type
        self.share_item_embeddings = share_item_embeddings

        self.item_embedding = init_item_embedding(
            nn.Embedding(num_items + 1, hidden_size, padding_idx=0)
        )
        self.position_embedding = nn.Embedding(max_seq_len, hidden_size)
        nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)
        self.emb_dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            CausalTransformerBlock(hidden_size, num_heads, dropout)
            for _ in range(num_layers)
        )
        self.final_norm = nn.LayerNorm(hidden_size)
        self.output = (
            nn.Identity()
            if share_item_embeddings
            else nn.Linear(hidden_size, hidden_size)
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def encode(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Encode a right-padded ID sequence with causal self-attention.

        Args:
            input_ids: Item IDs of shape ``[B, N]`` (``0`` = ``<PAD>``).

        Returns:
            Contextual representations of shape ``[B, N, H]``.
        """
        seq_len = input_ids.size(1)
        positions = torch.arange(seq_len, device=input_ids.device)
        hidden = self.item_embedding(input_ids) * self.hidden_size**0.5
        hidden = hidden + self.position_embedding(positions).unsqueeze(0)
        hidden = self.emb_dropout(hidden)
        for block in self.blocks:
            hidden = block(hidden)
        return self.final_norm(hidden)

    def project(self, hidden: torch.Tensor) -> torch.Tensor:
        """Output projection shared by the training and the inference scoring path.

        With ``share_item_embeddings=True`` this is the identity (the item embedding matrix is
        tied); otherwise it is a trained linear map. Both :meth:`compute_loss` and
        :meth:`score_items` must go through it so that evaluation never uses an untrained
        projection.
        """
        return self.output(hidden)

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
        return full_bce_loss(projected, embeddings, flat_targets)

    def score_items(self, batch: EvalBatch) -> torch.Tensor:
        """Score all items from the representation at the scoring position (default: last)."""
        hidden = self.encode(batch["input_ids"])
        scored_hidden = self.project(gather_positions(hidden, score_position(batch)))
        return prepend_pad_column(self._item_logits(scored_hidden))

    def extra_repr(self) -> str:
        return (
            f"num_items={self.num_items}, max_seq_len={self.max_seq_len}, "
            f"hidden_size={self.hidden_size}, loss_type={self.loss_type}"
        )
