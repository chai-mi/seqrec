"""Bidirectional Cloze recommendation, with explicitly opt-in Oracle evaluation.

The unified factory passes ModelConfig and the base receives num_items.
Evaluator-prepared Oracle batches carry score_pos and are scored unchanged;
raw histories receive a terminal MASK here. Diagnostic flags must remain False
during training, checkpoint selection and standard tests.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import ClassVar, cast

import torch
from torch import nn
from torch.nn import functional as F

from seqrec.config import ModelConfig
from seqrec.data.dataset import ClozeTrainBatch, EvalBatch
from seqrec.models.base import Paradigm, SequentialRecommender


class BERT4Rec(SequentialRecommender):
    """BERT encoder with a real-item-only, optionally tied prediction head.

    ``oracle_history_control`` reserves the same history budget as Oracle but
    hides the future item. This separates future context from truncation effects.
    Neither evaluation flag is a learned parameter or a training augmentation.
    """

    paradigm: ClassVar[Paradigm] = "cloze"
    requires_training: ClassVar[bool] = True

    def __init__(
        self,
        num_items: int,
        max_seq_len: int,
        config: ModelConfig | None = None,
        *,
        oracle_middle_mask: bool = False,
    ) -> None:
        super().__init__(num_items)
        config = config if config is not None else ModelConfig(name="bert4rec")
        if num_items < 1 or max_seq_len < 1:
            raise ValueError("num_items and max_seq_len must be positive")
        if config.hidden_size < 1 or config.num_heads < 1 or config.num_layers < 1:
            raise ValueError("hidden_size, num_heads and num_layers must be positive")
        if config.hidden_size % config.num_heads:
            raise ValueError("hidden_size must be divisible by num_heads")
        if config.loss_type != "ce":
            raise ValueError("BERT4Rec supports masked-item cross-entropy (ce) only")
        if not 0 <= config.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")

        self.num_items = num_items
        self.max_seq_len = max_seq_len
        self.mask_token_id = num_items + 1
        self.oracle_middle_mask = oracle_middle_mask
        self.oracle_history_control = False
        self.item_embeddings = nn.Embedding(
            num_items + 2, config.hidden_size, padding_idx=0
        )
        self.position_embeddings = nn.Embedding(max_seq_len, config.hidden_size)
        self.input_norm = nn.LayerNorm(config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)
        layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_size,
            nhead=config.num_heads,
            dim_feedforward=4 * config.hidden_size,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer, num_layers=config.num_layers, enable_nested_tensor=False
        )
        self.prediction_transform = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.GELU(),
            nn.LayerNorm(config.hidden_size),
        )
        self.output_bias = nn.Parameter(torch.zeros(num_items))
        self.output_projection = (
            None
            if config.share_item_embeddings
            else nn.Linear(config.hidden_size, num_items, bias=False)
        )
        self.apply(self._init_weights)
        # MultiheadAttention's packed projection is not an nn.Linear submodule.
        for module in self.modules():
            if isinstance(module, nn.MultiheadAttention):
                nn.init.normal_(module.in_proj_weight, std=0.02)
                if module.in_proj_bias is not None:
                    nn.init.zeros_(module.in_proj_bias)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        """Independently initialize cloned encoder layers and zero the PAD row."""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)
            if isinstance(module, nn.Embedding) and module.padding_idx is not None:
                with torch.no_grad():
                    module.weight[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:  # ty: ignore[invalid-mutable-override]
        """Encode [B, N] IDs to [B, N, H], using bidirectional PAD-masked attention."""
        if input_ids.ndim != 2 or not 1 <= input_ids.shape[1] <= self.max_seq_len:
            raise ValueError("input_ids must have shape [B, N], 1 <= N <= max_seq_len")
        if input_ids.dtype != torch.long:
            raise ValueError("input_ids must be torch.long")
        if torch.any((input_ids < 0) | (input_ids > self.num_items + 1)):
            raise ValueError("input_ids contain an invalid token ID")
        padding = input_ids.eq(0)
        # An all-PAD row has no attention keys; unmask one zero input to avoid NaNs.
        safe_padding = padding.clone()
        safe_padding[:, 0] &= ~padding.all(dim=1)
        positions = torch.arange(input_ids.shape[1], device=input_ids.device)
        hidden = self.item_embeddings(input_ids) + self.position_embeddings(positions)
        hidden = self.dropout(self.input_norm(hidden))
        hidden = hidden.masked_fill(padding.unsqueeze(-1), 0.0)
        hidden = self.encoder(
            hidden, src_key_padding_mask=safe_padding, is_causal=False
        )
        return hidden.masked_fill(padding.unsqueeze(-1), 0.0)

    def _item_logits(self, hidden: torch.Tensor) -> torch.Tensor:
        """Project only selected positions; never materialize [B, N, vocabulary]."""
        hidden = self.prediction_transform(hidden)
        weight = (
            self.item_embeddings.weight[1 : self.num_items + 1]
            if self.output_projection is None
            else self.output_projection.weight
        )
        return F.linear(hidden, weight, self.output_bias)

    def compute_loss(self, batch: ClozeTrainBatch) -> torch.Tensor:
        """Compute CE only at labelled positions; PAD/MASK are not output classes."""
        labels = batch["labels"]
        if labels.shape != batch["input_ids"].shape or labels.dtype != torch.long:
            raise ValueError(
                "labels must be torch.long with the same shape as input_ids"
            )
        if torch.any((labels < 0) | (labels > self.num_items)):
            raise ValueError("labels must be 0 (ignored) or a real item ID")
        selected = labels.gt(0)
        if torch.any(selected & batch["input_ids"].eq(0)):
            raise ValueError("PAD positions cannot have supervised labels")
        if not selected.any():
            # A graph-connected zero, unlike mean CE on an empty target (NaN).
            return self.output_bias.sum() * 0.0
        hidden = self(batch["input_ids"])
        return F.cross_entropy(
            self._item_logits(hidden[selected]), labels[selected] - 1
        )

    def prepare_eval_inputs(
        self, batch: EvalBatch
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return new right-padded IDs and MASK positions without mutating the batch.

        Missing future IDs fall back to ordinary next-item prediction. With a
        one-token window Oracle is undefined and explicitly rejected. Empty
        histories are supported (their prediction input consists of MASK alone).
        """
        history, lengths = batch["input_ids"], batch["seq_len"]
        if history.ndim != 2 or history.shape[1] < 1 or history.dtype != torch.long:
            raise ValueError("history must be a nonzero-width torch.long matrix")
        if lengths.shape != (history.shape[0],) or lengths.dtype != torch.long:
            raise ValueError("seq_len must be a torch.long vector of batch size")
        if torch.any((lengths < 0) | (lengths > history.shape[1])):
            raise ValueError("seq_len must be within the input width")
        positions_in = torch.arange(history.shape[1], device=history.device)
        valid = positions_in.unsqueeze(0) < lengths.unsqueeze(1)
        if torch.any(valid & ((history < 1) | (history > self.num_items))):
            raise ValueError("valid history positions must contain real items only")
        if torch.any(~valid & history.ne(0)):
            raise ValueError("history must be right-padded with zeros")

        use_future = torch.zeros_like(lengths, dtype=torch.bool)
        if self.oracle_middle_mask or self.oracle_history_control:
            future = batch["future_id"]
            if future.shape != lengths.shape or future.dtype != torch.long:
                raise ValueError("future_id must be a torch.long vector of batch size")
            if torch.any((future < 0) | (future > self.num_items)):
                raise ValueError("future_id must be 0 or a real item ID")
            use_future = future.gt(0)
            if self.max_seq_len < 2 and use_future.any():
                raise ValueError("Oracle evaluation requires max_seq_len >= 2")

        budget = self.max_seq_len - 1 - use_future.long()
        kept = torch.minimum(lengths, budget)
        start = lengths - kept
        positions = torch.arange(self.max_seq_len, device=history.device).unsqueeze(0)
        source = (start.unsqueeze(1) + positions).clamp(max=history.shape[1] - 1)
        gathered = history.gather(1, source)
        prepared = torch.where(positions < kept.unsqueeze(1), gathered, 0)
        prepared.scatter_(1, kept.unsqueeze(1), self.num_items + 1)
        if self.oracle_middle_mask:
            rows = use_future.nonzero(as_tuple=True)[0]
            prepared[rows, kept[rows] + 1] = batch["future_id"][rows]
        return prepared, kept

    def score_items(self, batch: EvalBatch) -> torch.Tensor:
        """Return [B, num_items + 1]; Evaluator owns seen-item filtering."""
        supplied_positions = cast(Mapping[str, torch.Tensor], batch).get("score_pos")
        if supplied_positions is None:
            inputs, mask_positions = self.prepare_eval_inputs(batch)
        else:
            # The common Evaluator has already inserted MASK + future. Inserting
            # again would shift the target and could score the wrong item.
            inputs, mask_positions = batch["input_ids"], supplied_positions
            if (
                mask_positions.shape != (inputs.shape[0],)
                or mask_positions.dtype != torch.long
            ):
                raise ValueError("score_pos must be a torch.long vector of batch size")
            if torch.any((mask_positions < 0) | (mask_positions >= inputs.shape[1])):
                raise ValueError("score_pos is outside the input window")
            if torch.any(
                inputs.gather(1, mask_positions[:, None]) != self.num_items + 1
            ):
                raise ValueError("BERT4Rec score_pos must point to a MASK token")
        hidden = self(inputs)
        rows = torch.arange(inputs.shape[0], device=inputs.device)
        logits = self._item_logits(hidden[rows, mask_positions])
        return torch.cat(
            (logits.new_full((inputs.shape[0], 1), -torch.inf), logits), dim=1
        )
