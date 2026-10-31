"""BERT4Rec: bidirectional Transformer for Cloze masked-item recommendation.

Reference: Sun et al., *BERT4Rec: Sequential Recommendation with Bidirectional Encoder
Representations from Transformer* (CIKM 2019).

Unlike :class:`~seqrec.models.sasrec.SASRec`, every position attends to **both** past and
future tokens (no causal mask); the only attention mask hides right-padding ``<PAD>``
positions. Training therefore follows the Cloze objective (predict masked items), while
inference appends a single ``<MASK>`` token at the end and scores items from its final
hidden state — the classic train-test mismatch studied in Task 2.
"""

from __future__ import annotations

from typing import ClassVar

import torch
import torch.nn.functional as F
from torch import nn

from seqrec.config import LossType
from seqrec.data.dataset import ClozeTrainBatch, EvalBatch
from seqrec.models.base import (
    Paradigm,
    SequentialRecommender,
    gather_positions,
    init_item_embedding,
    prepend_pad_column,
    select_supervised_positions,
)

__all__ = ["BERT4Rec", "BidirectionalTransformerBlock"]


class BidirectionalTransformerBlock(nn.Module):
    """Pre-LayerNorm Transformer block with *bidirectional* self-attention.

    The block accepts an explicit boolean attention mask (``True`` = attend, ``False`` =
    masked out) that hides ``<PAD>`` keys, so padding tokens can never pollute the
    bidirectional attention weights of a valid query.

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

    def forward(  # ty: ignore[invalid-mutable-override]
        self,
        hidden: torch.Tensor,
        attn_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Apply bidirectional self-attention followed by a feed-forward network.

        Args:
            hidden: Input representations of shape ``[B, N, H]``.
            attn_mask: Boolean mask broadcastable to ``[B, num_heads, N, N]`` where ``True``
                marks positions that may be attended to (i.e. non-padding keys).

        Returns:
            Output representations of shape ``[B, N, H]``.
        """
        query, key, value = self.qkv(self.attn_norm(hidden)).chunk(3, dim=-1)
        attended = F.scaled_dot_product_attention(
            self._split_heads(query),
            self._split_heads(key),
            self._split_heads(value),
            attn_mask=attn_mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
        )
        attended = attended.transpose(1, 2).reshape(hidden.shape)
        hidden = hidden + self.attn_dropout(self.attn_out(attended))
        return hidden + self.ffn_dropout(self.ffn(self.ffn_norm(hidden)))


class BERT4Rec(SequentialRecommender):
    """Bidirectional Transformer trained with the Cloze objective.

    Vocabulary layout (``num_items + 2`` rows in the item embedding table):

    - ``0``: ``<PAD>`` (``padding_idx=0``);
    - ``1 .. num_items``: real items;
    - ``num_items + 1``: ``<MASK>``.

    Args:
        num_items: Number of real items ``V``.
        max_seq_len: Maximum truncation window ``N``; also the position table size.
        hidden_size: Model dimension ``d``.
        num_layers: Number of stacked bidirectional Transformer blocks.
        num_heads: Number of self-attention heads.
        dropout: Dropout rate.
        loss_type: Training objective; only ``ce`` is supported.
        share_item_embeddings: Tie the scoring matrix to the input item embedding table.
    """

    paradigm: ClassVar[Paradigm] = "cloze"
    #: Chunk size for the full-vocabulary cross-entropy loss (bounds peak memory).
    ce_chunk_size: ClassVar[int] = 4096

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
        if loss_type != "ce":
            raise ValueError(
                f"Unsupported loss_type for BERT4Rec: {loss_type!r} (only 'ce')."
            )
        self.max_seq_len = max_seq_len
        self.hidden_size = hidden_size
        self.loss_type = loss_type
        self.share_item_embeddings = share_item_embeddings
        # Evaluator's oracle middle-mask protocol reads this token id.
        self.mask_token_id = num_items + 1

        self.item_embedding = init_item_embedding(
            nn.Embedding(num_items + 2, hidden_size, padding_idx=0)
        )
        self.position_embedding = nn.Embedding(max_seq_len, hidden_size)
        nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)
        self.emb_dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            BidirectionalTransformerBlock(hidden_size, num_heads, dropout)
            for _ in range(num_layers)
        )
        self.final_norm = nn.LayerNorm(hidden_size)

        # Output head: GELU + LayerNorm, then dot with the (tied) item embedding matrix.
        self.output = (
            nn.Identity()
            if share_item_embeddings
            else nn.Linear(hidden_size, hidden_size)
        )
        self.out_norm = nn.LayerNorm(hidden_size)
        self.out_bias = nn.Parameter(torch.zeros(num_items))
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
        """Encode a right-padded ID sequence with bidirectional self-attention.

        Args:
            input_ids: Item IDs of shape ``[B, N]`` (``0`` = ``<PAD>``; the ``<MASK>``
                token is a valid, attendable token).

        Returns:
            Contextual representations of shape ``[B, N, H]``.
        """
        seq_len = input_ids.size(1)
        positions = torch.arange(seq_len, device=input_ids.device)
        hidden = self.item_embedding(input_ids) * self.hidden_size**0.5
        hidden = hidden + self.position_embedding(positions).unsqueeze(0)
        hidden = self.emb_dropout(hidden)

        # ``scaled_dot_product_attention`` treats boolean ``True`` as "may attend"; we keep
        # only non-padding keys.
        attn_mask = (input_ids != 0).unsqueeze(1).unsqueeze(2)
        for block in self.blocks:
            hidden = block(hidden, attn_mask)
        return self.final_norm(hidden)

    def project(self, hidden: torch.Tensor) -> torch.Tensor:
        """Output head shared by training and inference (GELU + LayerNorm + optional proj)."""
        return self.out_norm(F.gelu(self.output(hidden)))

    def _item_logits(self, hidden: torch.Tensor) -> torch.Tensor:
        """Dot product against the shared item embeddings plus the output bias term."""
        item_weights = self.item_embedding.weight[1 : self.num_items + 1]
        return hidden @ item_weights.T + self.out_bias

    def _cross_entropy(
        self,
        hidden: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Full-vocabulary CE over supervised (masked) positions, chunked to bound memory."""
        total = targets.numel()
        if total == 0:
            return hidden.new_zeros(())
        item_weights = self.item_embedding.weight[1 : self.num_items + 1]
        loss_sum = torch.zeros((), dtype=torch.float32, device=hidden.device)
        for start in range(0, total, self.ce_chunk_size):
            chunk_hidden = hidden[start : start + self.ce_chunk_size]
            chunk_targets = targets[start : start + self.ce_chunk_size]
            logits = chunk_hidden @ item_weights.T + self.out_bias
            loss_sum = loss_sum + F.cross_entropy(
                logits.float(),
                chunk_targets - 1,
                reduction="sum",
            )
        return loss_sum / total

    def compute_loss(self, batch: ClozeTrainBatch) -> torch.Tensor:
        """Cloze loss computed only at masked positions (``labels > 0``)."""
        hidden = self.encode(batch["input_ids"])
        flat_hidden, flat_targets = select_supervised_positions(hidden, batch["labels"])
        return self._cross_entropy(self.project(flat_hidden), flat_targets)

    def _append_mask_token(
        self,
        input_ids: torch.Tensor,
        seq_len: torch.Tensor,
    ) -> torch.Tensor:
        """Rewrite an eval sequence into ``[history..., <MASK>]``.

        When ``seq_len < max_seq_len`` the mask is simply written into the first padding slot
        (position ``seq_len``). When the window is full the sequence is shifted left by one so
        the mask can occupy the final position ``max_seq_len - 1``.
        """
        batch_size, window = input_ids.shape
        out = input_ids.clone()
        full = seq_len >= window
        if bool(full.any()):
            shifted = torch.zeros_like(input_ids)
            if window > 1:
                shifted[:, :-1] = input_ids[:, 1:]
            out = torch.where(full.unsqueeze(1), shifted, out)

        mask_token_id = self.mask_token_id
        assert mask_token_id is not None
        mask_pos = seq_len.clamp(max=window - 1)
        rows = torch.arange(batch_size, device=input_ids.device)
        out[rows, mask_pos] = mask_token_id
        return out

    def score_items(self, batch: EvalBatch) -> torch.Tensor:
        """Score all items from the ``<MASK>`` position (standard or oracle protocol).

        When the evaluator publishes ``score_pos`` (oracle middle-mask protocol) the input
        already contains ``[..., <MASK>, future_id]`` and we score that exact position;
        otherwise we append a ``<MASK>`` at the end of the history.
        """
        score_pos = batch.get("score_pos")  # type: ignore[call-overload]
        if score_pos is None:
            input_ids = self._append_mask_token(batch["input_ids"], batch["seq_len"])
            hidden = self.encode(input_ids)
            mask_pos = batch["seq_len"].clamp(max=self.max_seq_len - 1)
            scored_hidden = self.project(gather_positions(hidden, mask_pos))
        else:
            hidden = self.encode(batch["input_ids"])
            scored_hidden = self.project(gather_positions(hidden, score_pos))
        return prepend_pad_column(self._item_logits(scored_hidden))

    def extra_repr(self) -> str:
        return (
            f"num_items={self.num_items}, max_seq_len={self.max_seq_len}, "
            f"hidden_size={self.hidden_size}, loss_type={self.loss_type}, "
            f"mask_token_id={self.mask_token_id}"
        )
