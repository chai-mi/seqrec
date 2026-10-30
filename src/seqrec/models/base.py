"""Unified abstraction for every sequential recommender in this project.

All models (``Pop``, ``GRU4Rec``, ``SASRec``, ``BERT4Rec``, ``SR-GNN``) share a single
contract so that the :class:`seqrec.engine.trainer.Trainer` and
:class:`seqrec.engine.evaluator.Evaluator` can drive them without special casing.

ID convention (see ``AGENTS.md``):

- ``0`` is strictly reserved for ``<PAD>``;
- real item IDs live in ``[1, num_items]``;
- ``num_items + 1`` is reserved for ``BERT4Rec``'s ``<MASK>`` token.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, ClassVar, Literal

import torch
import torch.nn.functional as F
from torch import nn

from seqrec.data.dataset import EvalBatch

Paradigm = Literal["causal", "cloze", "graph", "stateless"]

__all__ = [
    "Paradigm",
    "SequentialRecommender",
    "full_bce_loss",
    "full_softmax_ce_loss",
    "gather_last_valid",
    "gather_positions",
    "init_item_embedding",
    "prepend_pad_column",
    "sampled_bpr_max_loss",
    "score_position",
    "select_supervised_positions",
]


class SequentialRecommender(nn.Module, ABC):
    """Abstract base class for all sequential recommendation models.

    Subclasses declare their training paradigm via class attributes and implement the two
    abstract methods used by the engine:

    - :meth:`compute_loss` consumes a paradigm specific training batch;
    - :meth:`score_items` consumes the unified :class:`~seqrec.data.dataset.EvalBatch` and
      returns raw logits of shape ``[batch_size, num_items + 1]`` where column ``0``
      (``<PAD>``) must be ``-inf``.

    Scoring position contract: models must read the hidden state at
    :func:`score_position`, i.e. ``batch["score_pos"]`` when the evaluator provides it
    (Cloze / oracle middle-mask protocol, where the item to predict sits at a mask position
    inside the sequence) and the last valid position ``seq_len - 1`` otherwise.

    Attributes:
        paradigm: Name of the training paradigm the model consumes.
        requires_training: ``False`` for parameter free baselines such as ``Pop``.
        mask_token_id: ``<MASK>`` token id (only set by ``BERT4Rec``), used by the
            evaluator's oracle middle-mask protocol.
        num_items: Number of real items (``V``), excluding ``<PAD>`` and ``<MASK>``.
    """

    paradigm: ClassVar[Paradigm] = "causal"
    requires_training: ClassVar[bool] = True

    def __init__(self, num_items: int) -> None:
        super().__init__()
        self.num_items = int(num_items)
        # Set by `BERT4Rec` (to `num_items + 1`); consumed by the evaluator's oracle protocol.
        self.mask_token_id: int | None = None

    @property
    def num_logits(self) -> int:
        """Number of output columns of :meth:`score_items` (``num_items + 1``)."""
        return self.num_items + 1

    @abstractmethod
    def compute_loss(self, batch: Any) -> torch.Tensor:
        """Compute the scalar training loss for one mini-batch."""

    @abstractmethod
    def score_items(self, batch: EvalBatch) -> torch.Tensor:
        """Return ``[B, num_items + 1]`` logits with column ``0`` (``<PAD>``) set to ``-inf``."""

    def predict_topk(self, batch: EvalBatch, k: int) -> torch.Tensor:
        """Return the top-``k`` item IDs (``<PAD>`` excluded) for every row of an eval batch."""
        logits = self.score_items(batch)
        k = min(k, self.num_items)
        return logits[:, 1:].topk(k, dim=1).indices + 1

    def num_trainable_parameters(self) -> int:
        """Number of trainable parameters (``0`` for frozen baselines)."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def extra_repr(self) -> str:
        return f"num_items={self.num_items}"


def gather_positions(hidden: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    """Gather one hidden state per sequence at the given positions.

    Args:
        hidden: Sequence representations of shape ``[B, N, H]``.
        positions: 0-based positions of shape ``[B]``.

    Returns:
        Tensor of shape ``[B, H]``.
    """
    index = positions.clamp(min=0).view(-1, 1, 1).expand(-1, 1, hidden.size(-1))
    return hidden.gather(1, index).squeeze(1)


def gather_last_valid(hidden: torch.Tensor, seq_len: torch.Tensor) -> torch.Tensor:
    """Gather the hidden state at the last valid (non-padding) position of each sequence.

    Args:
        hidden: Sequence representations of shape ``[B, N, H]`` (right padded).
        seq_len: Valid lengths of shape ``[B]`` with ``1 <= seq_len <= N``.

    Returns:
        Tensor of shape ``[B, H]``.
    """
    return gather_positions(hidden, seq_len.clamp(min=1) - 1)


def score_position(batch: EvalBatch) -> torch.Tensor:
    """Position whose hidden state is used to score items.

    The evaluator publishes ``batch["score_pos"]`` when the item to predict is not the last
    token of the input (e.g. the oracle middle-mask protocol, ``[..., <MASK>, future]``).
    Every other protocol defaults to the last valid position ``seq_len - 1``.
    """
    positions = batch.get("score_pos")  # type: ignore[call-overload]
    if positions is None:
        return (batch["seq_len"] - 1).clamp(min=0)
    return positions


def prepend_pad_column(logits: torch.Tensor) -> torch.Tensor:
    """Prepend a ``-inf`` column so that the ``<PAD>`` token can never be recommended.

    Args:
        logits: Item logits of shape ``[B, num_items]`` covering IDs ``1..num_items``.

    Returns:
        Tensor of shape ``[B, num_items + 1]``.
    """
    neg_inf = logits.new_full((logits.size(0), 1), float("-inf"))
    return torch.cat([neg_inf, logits], dim=1)


def init_item_embedding(embedding: nn.Embedding, std: float = 0.02) -> nn.Embedding:
    """Initialize an item embedding table with ``N(0, std)`` and a zeroed ``<PAD>`` row."""
    nn.init.normal_(embedding.weight, mean=0.0, std=std)
    with torch.no_grad():
        embedding.weight[0].zero_()
    return embedding


def select_supervised_positions(
    hidden: torch.Tensor,
    targets: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Flatten hidden states/targets and drop padding positions (``target == 0``).

    Args:
        hidden: Sequence representations of shape ``[B, N, H]``.
        targets: Target token ids of shape ``[B, N]`` where ``0`` marks non-supervised positions.

    Returns:
        Tuple ``(flat_hidden [M, H], flat_targets [M])`` containing only supervised positions.
    """
    flat_hidden = hidden.reshape(-1, hidden.size(-1))
    flat_targets = targets.reshape(-1)
    mask = flat_targets > 0
    return flat_hidden[mask], flat_targets[mask]


def full_softmax_ce_loss(
    hidden: torch.Tensor,
    item_embeddings: torch.Tensor,
    targets: torch.Tensor,
    chunk_size: int = 4096,
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    """Full-vocabulary softmax cross entropy, evaluated in chunks to bound memory.

    Args:
        hidden: Supervised hidden states of shape ``[M, H]``.
        item_embeddings: Full item embedding matrix of shape ``[num_items + 1, H]``
            (row ``0`` is ``<PAD>`` and is excluded from the softmax).
        targets: Real item IDs of shape ``[M]`` with values in ``[1, num_items]``.
        chunk_size: Number of positions scored at once.
        label_smoothing: Optional label smoothing factor.

    Returns:
        Scalar loss averaged over the ``M`` supervised positions.
    """
    total = targets.numel()
    if total == 0:
        return hidden.new_zeros(())
    loss_sum = torch.zeros((), dtype=torch.float32, device=hidden.device)
    for start in range(0, total, chunk_size):
        chunk_hidden = hidden[start : start + chunk_size]
        chunk_targets = targets[start : start + chunk_size]
        logits = chunk_hidden @ item_embeddings[1:].T
        loss_sum = loss_sum + F.cross_entropy(
            logits.float(),
            chunk_targets - 1,
            reduction="sum",
            label_smoothing=label_smoothing,
        )
    return loss_sum / total


def full_bce_loss(
    hidden: torch.Tensor,
    item_embeddings: torch.Tensor,
    targets: torch.Tensor,
    chunk_size: int = 1024,
) -> torch.Tensor:
    """Full-vocabulary binary cross entropy with one-hot labels (SASRec/GRU4Rec ``bce`` mode).

    Args:
        hidden: Supervised hidden states of shape ``[M, H]``.
        item_embeddings: Full item embedding matrix of shape ``[num_items + 1, H]``.
        targets: Real item IDs of shape ``[M]`` with values in ``[1, num_items]``.
        chunk_size: Number of positions scored at once.

    Returns:
        Scalar loss averaged over all ``M * num_items`` binary decisions.
    """
    total = targets.numel()
    if total == 0:
        return hidden.new_zeros(())
    num_items = item_embeddings.size(0) - 1
    loss_sum = torch.zeros((), dtype=torch.float32, device=hidden.device)
    for start in range(0, total, chunk_size):
        chunk_hidden = hidden[start : start + chunk_size].float()
        chunk_targets = targets[start : start + chunk_size]
        logits = chunk_hidden @ item_embeddings[1:].float().T
        labels = F.one_hot(chunk_targets - 1, num_classes=num_items).float()
        loss_sum = loss_sum + F.binary_cross_entropy_with_logits(
            logits, labels, reduction="sum"
        )
    return loss_sum / (total * num_items)


def sampled_bpr_max_loss(
    hidden: torch.Tensor,
    item_embeddings: torch.Tensor,
    targets: torch.Tensor,
    num_negatives: int = 100,
) -> torch.Tensor:
    """BPR-max pairwise loss with uniformly sampled negatives (GRU4Rec's best performing variant).

    ``L = -log( sum_j softmax_j(s_neg) * sigmoid(s_pos - s_neg_j) )``

    Args:
        hidden: Supervised hidden states of shape ``[M, H]``.
        item_embeddings: Full item embedding matrix of shape ``[num_items + 1, H]``.
        targets: Real item IDs of shape ``[M]`` with values in ``[1, num_items]``.
        num_negatives: Number of negatives sampled per positive position.

    Returns:
        Scalar loss averaged over the ``M`` supervised positions.
    """
    total = targets.numel()
    if total == 0:
        return hidden.new_zeros(())
    num_items = item_embeddings.size(0) - 1
    hidden = hidden.float()
    pos_scores = (hidden * item_embeddings[targets].float()).sum(dim=-1, keepdim=True)
    neg_ids = torch.randint(
        1, num_items + 1, (total, num_negatives), device=hidden.device
    )
    neg_scores = torch.bmm(
        item_embeddings[neg_ids].float(), hidden.unsqueeze(-1)
    ).squeeze(-1)
    weights = F.softmax(neg_scores, dim=-1)
    pair = torch.sigmoid(pos_scores - neg_scores)
    return -torch.log((weights * pair).sum(dim=-1).clamp_min(1e-12)).mean()
