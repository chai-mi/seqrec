from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypedDict

import polars as pl
import torch
from torch.utils.data import DataLoader, Dataset

from seqrec.config import (
    LEN_BUCKET_TO_ID,
    POP_BUCKET_TO_ID,
    DataConfig,
    ModelName,
)
from seqrec.data.preprocess import DatasetMetadata, load_dataset_metadata


class CausalTrainBatch(TypedDict):
    """Batch dictionary for Causal next-item prediction (GRU4Rec, SASRec)."""

    user_id: torch.Tensor
    input_ids: torch.Tensor
    target_ids: torch.Tensor
    seq_len: torch.Tensor


class ClozeTrainBatch(TypedDict):
    """Batch dictionary for Cloze masked-item prediction (BERT4Rec)."""

    user_id: torch.Tensor
    input_ids: torch.Tensor
    labels: torch.Tensor
    seq_len: torch.Tensor


class SessionGraphTrainBatch(TypedDict):
    """Batch dictionary for Session-Graph next-item prediction (SR-GNN)."""

    user_id: torch.Tensor
    input_ids: torch.Tensor
    target_id: torch.Tensor
    seq_len: torch.Tensor
    graph_items: torch.Tensor
    alias_inputs: torch.Tensor
    adj_in: torch.Tensor
    adj_out: torch.Tensor


class EvalBatch(TypedDict):
    """Unified evaluation batch dictionary across all models."""

    user_id: torch.Tensor
    input_ids: torch.Tensor
    seq_len: torch.Tensor
    raw_seq_len: torch.Tensor
    len_bucket_id: torch.Tensor
    target_id: torch.Tensor
    target_pop_bucket_id: torch.Tensor
    future_id: torch.Tensor
    seen_ids: torch.Tensor


def build_session_graph_batch(
    input_ids: torch.Tensor,
    seq_len: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Construct SR-GNN session graphs in 100% vectorized PyTorch without Python loops.

    Args:
        input_ids: Right-padded sequence tensor of shape `[B, N]` (`0` = `<PAD>`).
        seq_len: Valid sequence lengths of shape `[B]` (`1 <= seq_len <= N`).

    Returns:
        Tuple of:
        - `graph_items`: `[B, N]` unique item IDs placed at their first-occurrence index (`0` elsewhere).
        - `alias_inputs`: `[B, N]` mapping each sequence position `t` to its first-occurrence node index in `0..N-1`.
        - `adj_in`: `[B, N, N]` row-normalized incoming adjacency matrix ($A^{\\text{in}}$).
        - `adj_out`: `[B, N, N]` row-normalized outgoing adjacency matrix ($A^{\\text{out}}$).
    """
    batch_size, max_len = input_ids.shape
    device = input_ids.device

    arange_n = torch.arange(max_len, device=device).unsqueeze(0)  # [1, N]
    valid_mask = arange_n < seq_len.unsqueeze(1)  # [B, N]

    # Identify first occurrence index for each token in the sequence
    same = (
        (input_ids.unsqueeze(2) == input_ids.unsqueeze(1))
        & valid_mask.unsqueeze(1)
        & valid_mask.unsqueeze(2)
    )
    alias_inputs = same.int().argmax(dim=2).long()  # [B, N]
    is_unique_node = (alias_inputs == arange_n) & valid_mask
    graph_items = torch.where(is_unique_node, input_ids, torch.zeros_like(input_ids))

    # Build directed transitions (u -> v) for t in 0 .. seq_len - 2
    adj = torch.zeros((batch_size, max_len, max_len), dtype=torch.float32, device=device)
    if max_len > 1:
        u = alias_inputs[:, :-1]  # [B, N-1]
        v = alias_inputs[:, 1:]  # [B, N-1]
        edge_mask = (
            torch.arange(max_len - 1, device=device).unsqueeze(0)
            < (seq_len - 1).clamp(min=0).unsqueeze(1)
        ).float()
        flat_idx = u * max_len + v
        adj.view(batch_size, max_len * max_len).scatter_add_(1, flat_idx, edge_mask)

    # Normalize incoming and outgoing degree matrices
    in_deg = adj.sum(dim=1, keepdim=True).clamp(min=1.0)
    adj_in = (adj / in_deg).transpose(1, 2).contiguous()

    out_deg = adj.sum(dim=2, keepdim=True).clamp(min=1.0)
    adj_out = (adj / out_deg).contiguous()

    return graph_items, alias_inputs, adj_in, adj_out


def apply_cloze_mask_batch(
    clean_ids: torch.Tensor,
    seq_len: torch.Tensor,
    num_items: int,
    mask_prob: float = 0.2,
    last_item_mask_ratio: float = 0.1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply BERT4Rec Cloze masking + Last-Item Masking augmentation to a batch of sequences.

    Args:
        clean_ids: Right-padded clean item IDs of shape `[B, N]`.
        seq_len: Valid lengths of shape `[B]`.
        num_items: Total number of real items (`mask_token_id = num_items + 1`).
        mask_prob: Probability of masking each valid token in Cloze mode.
        last_item_mask_ratio: Fraction of sequences in the batch where only the last item is masked.

    Returns:
        Tuple `(masked_input_ids, labels)` both of shape `[B, N]`, where `labels == 0` on unmasked positions.
    """
    batch_size, max_len = clean_ids.shape
    device = clean_ids.device
    mask_token_id = num_items + 1

    arange_n = torch.arange(max_len, device=device).unsqueeze(0)
    valid_mask = arange_n < seq_len.unsqueeze(1)  # [B, N]

    # Standard random Cloze mask
    masked = (torch.rand((batch_size, max_len), device=device) < mask_prob) & valid_mask

    # Ensure every sequence has at least 1 masked position
    no_mask_rows = masked.sum(dim=1) == 0
    if no_mask_rows.any():
        rand_pos = (torch.rand(batch_size, device=device) * seq_len.float()).long()
        rand_pos = rand_pos.clamp(min=0, max=max_len - 1)
        fallback_mask = arange_n == rand_pos.unsqueeze(1)
        masked = torch.where(no_mask_rows.unsqueeze(1), fallback_mask, masked)

    # Last-item masking augmentation (to bridge Train-Test Mismatch)
    if last_item_mask_ratio > 0.0:
        use_last_only = torch.rand(batch_size, device=device) < last_item_mask_ratio
        last_pos_mask = arange_n == (seq_len - 1).clamp(min=0).unsqueeze(1)
        masked = torch.where(use_last_only.unsqueeze(1), last_pos_mask, masked)

    # 80% [MASK], 10% random item, 10% unchanged
    mode_draw = torch.rand((batch_size, max_len), device=device)
    random_items = torch.randint(1, num_items + 1, (batch_size, max_len), device=device)

    replaced = torch.where(
        mode_draw < 0.8,
        torch.full_like(clean_ids, mask_token_id),
        torch.where(mode_draw < 0.9, random_items, clean_ids),
    )

    input_ids = torch.where(masked, replaced, clean_ids)
    labels = torch.where(masked, clean_ids, torch.zeros_like(clean_ids))
    return input_ids, labels


class CausalSeqDataset(Dataset[CausalTrainBatch]):
    """Dataset for step-wise causal next-item training (GRU4Rec, SASRec)."""

    def __init__(
        self,
        train_df: pl.DataFrame,
        max_seq_len: int,
        sliding_window: bool = False,
        sliding_step: int = 50,
    ) -> None:
        super().__init__()
        self.max_seq_len = max_seq_len

        user_ids_list: list[int] = []
        inputs_list: list[list[int]] = []
        targets_list: list[list[int]] = []
        lengths_list: list[int] = []

        window_span = max_seq_len + 1
        step = max(1, sliding_step)

        for row in train_df.iter_rows(named=True):
            uid = int(row["user_id"])
            seq: list[int] = row["item_seq"]
            if len(seq) < 2:
                continue

            if not sliding_window or len(seq) <= window_span:
                windows = [seq[-window_span:]]
            else:
                windows = []
                end = len(seq)
                while end >= 2:
                    start = max(0, end - window_span)
                    windows.append(seq[start:end])
                    if start == 0:
                        break
                    end -= step

            for w in windows:
                eff_len = len(w) - 1
                inp = w[:-1]
                tgt = w[1:]
                pad_len = max_seq_len - eff_len
                user_ids_list.append(uid)
                inputs_list.append(inp + [0] * pad_len)
                targets_list.append(tgt + [0] * pad_len)
                lengths_list.append(eff_len)

        self.user_ids = torch.tensor(user_ids_list, dtype=torch.long)
        self.input_ids = torch.tensor(inputs_list, dtype=torch.long)
        self.target_ids = torch.tensor(targets_list, dtype=torch.long)
        self.seq_lens = torch.tensor(lengths_list, dtype=torch.long)

    def __len__(self) -> int:
        return int(self.user_ids.size(0))

    def __getitem__(self, index: int) -> CausalTrainBatch:
        return {
            "user_id": self.user_ids[index],
            "input_ids": self.input_ids[index],
            "target_ids": self.target_ids[index],
            "seq_len": self.seq_lens[index],
        }


class ClozeSeqDataset(Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
    """Dataset for bidirectional Cloze masked training (BERT4Rec)."""

    def __init__(
        self,
        train_df: pl.DataFrame,
        max_seq_len: int,
        num_items: int,
        mask_prob: float = 0.2,
        last_item_mask_ratio: float = 0.1,
        sliding_window: bool = False,
        sliding_step: int = 50,
    ) -> None:
        super().__init__()
        self.max_seq_len = max_seq_len
        self.num_items = num_items
        self.mask_prob = mask_prob
        self.last_item_mask_ratio = last_item_mask_ratio

        user_ids_list: list[int] = []
        clean_list: list[list[int]] = []
        lengths_list: list[int] = []

        step = max(1, sliding_step)
        for row in train_df.iter_rows(named=True):
            uid = int(row["user_id"])
            seq: list[int] = row["item_seq"]
            if len(seq) < 2:
                continue

            if not sliding_window or len(seq) <= max_seq_len:
                windows = [seq[-max_seq_len:]]
            else:
                windows = []
                end = len(seq)
                while end >= 2:
                    start = max(0, end - max_seq_len)
                    windows.append(seq[start:end])
                    if start == 0:
                        break
                    end -= step

            for w in windows:
                eff_len = len(w)
                pad_len = max_seq_len - eff_len
                user_ids_list.append(uid)
                clean_list.append(w + [0] * pad_len)
                lengths_list.append(eff_len)

        self.user_ids = torch.tensor(user_ids_list, dtype=torch.long)
        self.clean_ids = torch.tensor(clean_list, dtype=torch.long)
        self.seq_lens = torch.tensor(lengths_list, dtype=torch.long)

    def __len__(self) -> int:
        return int(self.user_ids.size(0))

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.user_ids[index], self.clean_ids[index], self.seq_lens[index]

    def collate_fn(
        self,
        batch: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    ) -> ClozeTrainBatch:
        user_id = torch.stack([b[0] for b in batch], dim=0)
        clean_ids = torch.stack([b[1] for b in batch], dim=0)
        seq_len = torch.stack([b[2] for b in batch], dim=0)
        input_ids, labels = apply_cloze_mask_batch(
            clean_ids=clean_ids,
            seq_len=seq_len,
            num_items=self.num_items,
            mask_prob=self.mask_prob,
            last_item_mask_ratio=self.last_item_mask_ratio,
        )
        return {
            "user_id": user_id,
            "input_ids": input_ids,
            "labels": labels,
            "seq_len": seq_len,
        }


class SessionGraphDataset(Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]):
    """Prefix-expanded dataset for session-graph training (SR-GNN)."""

    def __init__(
        self,
        train_df: pl.DataFrame,
        max_seq_len: int,
        max_prefixes_per_user: int = 50,
    ) -> None:
        super().__init__()
        self.max_seq_len = max_seq_len

        user_ids_list: list[int] = []
        inputs_list: list[list[int]] = []
        targets_list: list[int] = []
        lengths_list: list[int] = []

        for row in train_df.iter_rows(named=True):
            uid = int(row["user_id"])
            seq: list[int] = row["item_seq"]
            if len(seq) < 2:
                continue

            # Expand into (prefix, next_item) pairs, keeping up to `max_prefixes_per_user` recent prefixes
            start_target_idx = (
                max(1, len(seq) - max_prefixes_per_user) if max_prefixes_per_user > 0 else 1
            )
            for tgt_idx in range(start_target_idx, len(seq)):
                prefix = seq[max(0, tgt_idx - max_seq_len) : tgt_idx]
                eff_len = len(prefix)
                pad_len = max_seq_len - eff_len
                user_ids_list.append(uid)
                inputs_list.append(prefix + [0] * pad_len)
                targets_list.append(int(seq[tgt_idx]))
                lengths_list.append(eff_len)

        self.user_ids = torch.tensor(user_ids_list, dtype=torch.long)
        self.input_ids = torch.tensor(inputs_list, dtype=torch.long)
        self.target_ids = torch.tensor(targets_list, dtype=torch.long)
        self.seq_lens = torch.tensor(lengths_list, dtype=torch.long)

    def __len__(self) -> int:
        return int(self.user_ids.size(0))

    def __getitem__(
        self, index: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            self.user_ids[index],
            self.input_ids[index],
            self.target_ids[index],
            self.seq_lens[index],
        )

    @staticmethod
    def collate_fn(
        batch: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]],
    ) -> SessionGraphTrainBatch:
        user_id = torch.stack([b[0] for b in batch], dim=0)
        input_ids = torch.stack([b[1] for b in batch], dim=0)
        target_id = torch.stack([b[2] for b in batch], dim=0)
        seq_len = torch.stack([b[3] for b in batch], dim=0)
        graph_items, alias_inputs, adj_in, adj_out = build_session_graph_batch(input_ids, seq_len)
        return {
            "user_id": user_id,
            "input_ids": input_ids,
            "target_id": target_id,
            "seq_len": seq_len,
            "graph_items": graph_items,
            "alias_inputs": alias_inputs,
            "adj_in": adj_in,
            "adj_out": adj_out,
        }


@dataclass(slots=True)
class EvalSample:
    """Single evaluation row prior to batch collation."""

    user_id: int
    input_ids: list[int]
    seq_len: int
    raw_seq_len: int
    len_bucket_id: int
    target_id: int
    target_pop_bucket_id: int
    future_id: int
    seen_ids: list[int]


class EvalSeqDataset(Dataset[EvalSample]):
    """Unified evaluation dataset for Validation and Test splits across all models."""

    def __init__(
        self,
        eval_df: pl.DataFrame,
        max_seq_len: int,
    ) -> None:
        super().__init__()
        self.max_seq_len = max_seq_len
        self.samples: list[EvalSample] = []

        for row in eval_df.iter_rows(named=True):
            history: list[int] = row["history"]
            raw_len = len(history)
            trunc = history[-max_seq_len:]
            eff_len = len(trunc)
            pad_len = max_seq_len - eff_len

            self.samples.append(
                EvalSample(
                    user_id=int(row["user_id"]),
                    input_ids=trunc + [0] * pad_len,
                    seq_len=eff_len,
                    raw_seq_len=raw_len,
                    len_bucket_id=LEN_BUCKET_TO_ID[str(row["len_bucket"])],
                    target_id=int(row["target_id"]),
                    target_pop_bucket_id=POP_BUCKET_TO_ID[str(row["target_pop_bucket"])],
                    future_id=int(row["future_id"]),
                    seen_ids=history,
                )
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> EvalSample:
        return self.samples[index]

    @staticmethod
    def collate_fn(batch: list[EvalSample]) -> EvalBatch:
        max_seen = max(len(s.seen_ids) for s in batch)
        padded_seen = [s.seen_ids + [0] * (max_seen - len(s.seen_ids)) for s in batch]

        return {
            "user_id": torch.tensor([s.user_id for s in batch], dtype=torch.long),
            "input_ids": torch.tensor([s.input_ids for s in batch], dtype=torch.long),
            "seq_len": torch.tensor([s.seq_len for s in batch], dtype=torch.long),
            "raw_seq_len": torch.tensor([s.raw_seq_len for s in batch], dtype=torch.long),
            "len_bucket_id": torch.tensor([s.len_bucket_id for s in batch], dtype=torch.long),
            "target_id": torch.tensor([s.target_id for s in batch], dtype=torch.long),
            "target_pop_bucket_id": torch.tensor(
                [s.target_pop_bucket_id for s in batch], dtype=torch.long
            ),
            "future_id": torch.tensor([s.future_id for s in batch], dtype=torch.long),
            "seen_ids": torch.tensor(padded_seen, dtype=torch.long),
        }


@dataclass(slots=True)
class DataBundle:
    """Container holding metadata, item popularity tensors, and train/val/test DataLoaders."""

    metadata: DatasetMetadata
    max_seq_len: int
    item_pop_counts: torch.Tensor  # shape [num_items + 1], index 0 is 0
    item_pop_buckets: torch.Tensor  # shape [num_items + 1], index 0 is -1, 1..V in {0,1,2}
    train_loader: DataLoader[Any] | None
    val_loader: DataLoader[EvalSample]
    test_loader: DataLoader[EvalSample]

    @property
    def num_users(self) -> int:
        return self.metadata.num_users

    @property
    def num_items(self) -> int:
        return self.metadata.num_items


def load_split_dataframes(
    dataset_dir: Path,
    split_strategy: str,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Load train, val, test, and item_pop DataFrames for a given split strategy."""
    train_df = pl.read_parquet(dataset_dir / f"{split_strategy}_train.parquet")
    val_df = pl.read_parquet(dataset_dir / f"{split_strategy}_val.parquet")
    test_df = pl.read_parquet(dataset_dir / f"{split_strategy}_test.parquet")
    pop_df = pl.read_parquet(dataset_dir / f"{split_strategy}_item_pop.parquet")
    return train_df, val_df, test_df, pop_df


def build_databundle(
    config: DataConfig,
    model_name: ModelName = "sasrec",
) -> DataBundle:
    """Build the complete DataBundle (metadata, popularity vectors, and DataLoaders)."""
    metadata = load_dataset_metadata(config.data_dir, config.dataset)
    max_seq_len = config.resolved_max_seq_len
    train_df, val_df, test_df, pop_df = load_split_dataframes(
        config.dataset_dir, config.split_strategy
    )

    # Build item popularity count and bucket tensors indexed by item_id in [0, num_items]
    pop_sorted = pop_df.sort("item_id")
    counts_list = [0] + [int(x) for x in pop_sorted["train_count"].to_list()]
    buckets_list = [-1] + [POP_BUCKET_TO_ID[str(b)] for b in pop_sorted["pop_bucket"].to_list()]
    item_pop_counts = torch.tensor(counts_list, dtype=torch.float32)
    item_pop_buckets = torch.tensor(buckets_list, dtype=torch.long)

    train_loader: DataLoader[Any] | None = None

    if model_name in ("gru4rec", "sasrec"):
        causal_ds = CausalSeqDataset(
            train_df=train_df,
            max_seq_len=max_seq_len,
            sliding_window=config.sliding_window,
            sliding_step=config.sliding_step,
        )
        train_loader = DataLoader(
            causal_ds,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
        )
    elif model_name == "bert4rec":
        cloze_ds = ClozeSeqDataset(
            train_df=train_df,
            max_seq_len=max_seq_len,
            num_items=metadata.num_items,
            mask_prob=config.mask_prob,
            last_item_mask_ratio=config.last_item_mask_ratio,
            sliding_window=config.sliding_window,
            sliding_step=config.sliding_step,
        )
        train_loader = DataLoader(
            cloze_ds,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            collate_fn=cloze_ds.collate_fn,
        )
    elif model_name == "srgnn":
        graph_ds = SessionGraphDataset(
            train_df=train_df,
            max_seq_len=max_seq_len,
            max_prefixes_per_user=config.max_prefixes_per_user,
        )
        train_loader = DataLoader(
            graph_ds,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            collate_fn=SessionGraphDataset.collate_fn,
        )

    val_ds = EvalSeqDataset(eval_df=val_df, max_seq_len=max_seq_len)
    test_ds = EvalSeqDataset(eval_df=test_df, max_seq_len=max_seq_len)

    val_loader = DataLoader(
        val_ds,
        batch_size=config.eval_batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        collate_fn=EvalSeqDataset.collate_fn,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=config.eval_batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        collate_fn=EvalSeqDataset.collate_fn,
    )

    return DataBundle(
        metadata=metadata,
        max_seq_len=max_seq_len,
        item_pop_counts=item_pop_counts,
        item_pop_buckets=item_pop_buckets,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
    )
