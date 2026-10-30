"""Unified evaluator for every sequential recommender.

The evaluator consumes the model agnostic :class:`~seqrec.data.dataset.EvalBatch`, masks
``<PAD>`` plus the user's interaction history, and reports:

- **Full-Ranking** (main protocol): ``HR@K``, ``NDCG@K`` for ``K in {5, 10, 20}``, ``MRR``,
  ``ARP@10`` (Average Recommendation Popularity) and ``Coverage@10``;
- **Sampled-Ranking** (1 positive + 100 negatives): ``HR@10`` / ``NDCG@10`` under both
  uniform random and popularity-proportional negative sampling;
- **Bucket aggregations**: ``HR@10`` / ``NDCG@10`` sliced by user history-length bucket
  (``<=5``, ``6-15``, ``16-50``, ``>50``) and by target-popularity bucket
  (``Head`` / ``Mid`` / ``Tail``).

Implementation note: all ranking metrics are computed with exact rank arithmetic on the full
logit matrix (``rank = #{items scored strictly higher than the target}``). The
``torchmetrics.functional.retrieval_*`` helpers are *not* used because they only accept 1-D
per-query inputs and additionally filter candidates with a ``preds > 0`` relevance rule,
which is unsound for raw (signed) logits produced by these models.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from seqrec.config import LEN_BUCKETS, POP_BUCKETS, EvalConfig
from seqrec.data.dataset import EvalBatch
from seqrec.models.base import SequentialRecommender

__all__ = ["Evaluator", "move_batch_to_device"]


def move_batch_to_device[BatchT: Mapping[str, torch.Tensor]](
    batch: BatchT, device: torch.device
) -> BatchT:
    """Move every tensor of a batch dictionary to ``device`` (keys and types are preserved)."""
    moved = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
    return cast(BatchT, moved)


class Evaluator:
    """Ranking evaluator shared by all models and datasets.

    Args:
        config: Metric / protocol configuration.
        num_items: Number of real items ``V``.
        item_pop_counts: Training-set popularity counts of shape ``[V + 1]`` (index ``0``
            is the ``<PAD>`` slot), used for ``ARP@10`` and popularity-based sampling.
        device: Torch device used for all metric computations.
        bucket_metrics: Metric names that are additionally aggregated per bucket.
    """

    def __init__(
        self,
        config: EvalConfig,
        num_items: int,
        item_pop_counts: torch.Tensor,
        device: torch.device,
        bucket_metrics: tuple[str, ...] = ("HR@10", "NDCG@10"),
    ) -> None:
        self.config = config
        self.num_items = int(num_items)
        self.device = device
        self.bucket_metrics = bucket_metrics
        self.ks = tuple(sorted({*config.ks, config.primary_k}))
        self.max_k = max(self.ks)
        self.item_pop_counts = item_pop_counts.float().to(device)
        self._pop_probs = self._build_pop_sampling_probs(self.item_pop_counts)

    # ------------------------------------------------------------------ evaluation

    def evaluate(
        self,
        model: SequentialRecommender,
        loader: DataLoader[Any],
        split_name: str = "eval",
        show_progress: bool = True,
    ) -> dict[str, float]:
        """Evaluate ``model`` over ``loader`` and return a flat ``metric -> value`` mapping."""
        model.eval()
        sums: dict[str, float] = {}
        counts: dict[str, int] = {}
        bucket_totals: dict[str, int] = {}
        coverage_hits = torch.zeros(
            self.num_items + 1, dtype=torch.long, device=self.device
        )
        num_queries = 0

        progress = tqdm(
            loader,
            desc=f"eval/{split_name}",
            leave=False,
            disable=not show_progress,
            dynamic_ncols=True,
        )
        for raw_batch in progress:
            batch = move_batch_to_device(raw_batch, self.device)
            if self.config.oracle_middle_mask:
                batch = self.apply_oracle_middle_mask(model, batch)

            with torch.inference_mode():
                logits = model.score_items(batch)
            scores = self.mask_candidates(logits, batch)
            batch_metrics, top_ids = self._batch_metrics(scores, batch)

            num_queries += scores.size(0)
            for key, values in batch_metrics.items():
                sums[key] = sums.get(key, 0.0) + float(values.sum().item())
                counts[key] = counts.get(key, 0) + int(values.numel())

            for kind, bucket_ids, bucket_names in (
                ("len", batch["len_bucket_id"], LEN_BUCKETS),
                ("pop", batch["target_pop_bucket_id"], POP_BUCKETS),
            ):
                bucket_counts = torch.bincount(bucket_ids, minlength=len(bucket_names))
                for index, name in enumerate(bucket_names):
                    total = int(bucket_counts[index].item())
                    if total == 0:
                        continue
                    bucket_key = f"{kind}:{name}"
                    bucket_totals[bucket_key] = bucket_totals.get(bucket_key, 0) + total
                for key in self.bucket_metrics:
                    values = batch_metrics.get(key)
                    if values is None:
                        continue
                    bucket_sums = torch.zeros(
                        len(bucket_names), dtype=torch.float64, device=self.device
                    ).index_add_(0, bucket_ids, values.double())
                    for index, name in enumerate(bucket_names):
                        total = int(bucket_counts[index].item())
                        if total == 0:
                            continue
                        stat_key = f"{kind}:{name}/{key}"
                        sums[stat_key] = sums.get(stat_key, 0.0) + float(
                            bucket_sums[index].item()
                        )
                        counts[stat_key] = counts.get(stat_key, 0) + total

            top_bias = top_ids[:, : self._bias_k].reshape(-1)
            coverage_hits.index_add_(
                0,
                top_bias,
                torch.ones(top_bias.numel(), dtype=torch.long, device=self.device),
            )
            progress.set_postfix(
                **{
                    key: f"{float(values.mean().item()):.4f}"
                    for key, values in batch_metrics.items()
                    if key in self.bucket_metrics
                }
            )

        results = {
            key: total / counts[key]
            for key, total in sums.items()
            if counts.get(key, 0) > 0
        }
        results.update(
            {f"{key}/count": float(total) for key, total in bucket_totals.items()}
        )
        covered = int((coverage_hits > 0).sum().item())
        results[f"Coverage@{self._bias_k}"] = covered / max(self.num_items, 1)
        results["num_queries"] = float(num_queries)
        return results

    # ------------------------------------------------------------------ masking

    def mask_candidates(self, logits: torch.Tensor, batch: EvalBatch) -> torch.Tensor:
        """Mask ``<PAD>`` and every already-interacted item to ``-inf``.

        The ground-truth target is *not* masked even if it occurs in the history (repeated
        clicks), otherwise users who re-consume an item would be scored as automatic misses.
        """
        if logits.size(1) != self.num_items + 1:
            raise ValueError(
                f"score_items must return [B, {self.num_items + 1}] logits, got {tuple(logits.shape)}"
            )
        scores = logits.clone()
        scores[:, 0] = float("-inf")
        seen = batch["seen_ids"].masked_fill(
            batch["seen_ids"] == batch["target_id"].unsqueeze(1), 0
        )
        rows = (
            torch.arange(scores.size(0), device=scores.device)
            .unsqueeze(1)
            .expand_as(seen)
        )
        scores[rows.reshape(-1), seen.reshape(-1)] = float("-inf")
        return scores

    def apply_oracle_middle_mask(
        self,
        model: SequentialRecommender,
        batch: EvalBatch,
    ) -> EvalBatch:
        """Build the BERT4Rec "oracle" middle-mask input used for the train-test mismatch study.

        The last history item is replaced by ``<MASK>`` and the *future* (next) item is appended
        right after it, so the bidirectional encoder can exploit right-hand context exactly as it
        did during Cloze training.
        """
        mask_token_id = getattr(model, "mask_token_id", None)
        if mask_token_id is None:
            raise ValueError(
                "EvalConfig.oracle_middle_mask requires a model exposing `mask_token_id` "
                "(e.g. BERT4Rec)."
            )
        input_ids = batch["input_ids"].clone()
        seq_len = batch["seq_len"].clone()
        batch_size, window = input_ids.shape
        rows = torch.arange(batch_size, device=input_ids.device)

        overflow = seq_len >= window
        if bool(overflow.any()):
            shifted = torch.zeros_like(input_ids)
            shifted[:, :-1] = input_ids[:, 1:]
            input_ids = torch.where(overflow.unsqueeze(1), shifted, input_ids)
            seq_len = torch.where(overflow, seq_len - 1, seq_len)

        mask_pos = (seq_len - 1).clamp(min=0)
        future_pos = (mask_pos + 1).clamp(max=window - 1)
        input_ids[rows, mask_pos] = mask_token_id
        input_ids[rows, future_pos] = batch["future_id"]

        masked_batch = dict(batch)
        masked_batch["input_ids"] = input_ids
        masked_batch["seq_len"] = (seq_len + 1).clamp(max=window)
        return cast(EvalBatch, masked_batch)

    # ------------------------------------------------------------------ metrics

    @property
    def _bias_k(self) -> int:
        """Rank cutoff used for the popularity-bias metrics (``ARP@10`` / ``Coverage@10``)."""
        return min(10, self.num_items)

    def _batch_metrics(
        self,
        scores: torch.Tensor,
        batch: EvalBatch,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        """Compute per-query metrics for one batch.

        Args:
            scores: Masked logits of shape ``[B, V + 1]``.
            batch: The evaluation batch (targets and popularity buckets).

        Returns:
            Tuple ``(metrics, top_ids)`` where ``metrics`` maps a metric name to a
            ``[B]`` tensor of per-query values and ``top_ids`` holds the top-``10`` item IDs.
        """
        target = batch["target_id"].unsqueeze(1)
        target_scores = scores.gather(1, target)
        rank = (scores > target_scores).sum(dim=1)  # 0-based rank of the target
        rank_f = rank.double()

        metrics: dict[str, torch.Tensor] = {}
        for k in self.ks:
            hit = rank < k
            metrics[f"HR@{k}"] = hit.double()
            metrics[f"NDCG@{k}"] = torch.where(
                hit, 1.0 / torch.log2(rank_f + 2.0), torch.zeros_like(rank_f)
            )
        metrics["MRR"] = 1.0 / (rank_f + 1.0)

        top_ids = scores[:, 1:].topk(self._bias_k, dim=1).indices + 1
        metrics[f"ARP@{self._bias_k}"] = (
            self.item_pop_counts[top_ids].mean(dim=1).double()
        )

        if self.config.compute_sampled_metrics:
            for strategy in ("uniform", "popularity"):
                sampled = self._sampled_metrics(scores, batch, strategy)
                for key, values in sampled.items():
                    metrics[f"sampled_{strategy}/{key}"] = values

        return metrics, top_ids

    def _sampled_metrics(
        self,
        scores: torch.Tensor,
        batch: EvalBatch,
        strategy: str,
    ) -> dict[str, torch.Tensor]:
        """``HR@k`` / ``NDCG@k`` over ``1`` positive + ``num_neg_samples`` sampled negatives."""
        k = self.config.primary_k
        num_neg = self.config.num_neg_samples
        negatives = self._sample_negatives(batch, num_neg, strategy)  # [B, num_neg]
        pos_scores = scores.gather(1, batch["target_id"].unsqueeze(1))
        neg_scores = scores.gather(1, negatives)
        rank = (
            (neg_scores > pos_scores).sum(dim=1).double()
        )  # rank within the candidate set
        hit = rank < k
        return {
            f"HR@{k}": hit.double(),
            f"NDCG@{k}": torch.where(
                hit, 1.0 / torch.log2(rank + 2.0), torch.zeros_like(rank)
            ),
        }

    def _sample_negatives(
        self,
        batch: EvalBatch,
        num_neg: int,
        strategy: str,
        max_retries: int = 5,
    ) -> torch.Tensor:
        """Sample negatives that avoid the user's history and the positive target."""
        batch_size = batch["input_ids"].size(0)
        seen = batch["seen_ids"]
        target = batch["target_id"].unsqueeze(1)

        def draw() -> torch.Tensor:
            if strategy == "uniform":
                return torch.randint(
                    1, self.num_items + 1, (batch_size, num_neg), device=self.device
                )
            flat = torch.multinomial(
                self._pop_probs, batch_size * num_neg, replacement=True
            )
            return flat.view(batch_size, num_neg) + 1

        negatives = draw()
        for _ in range(max_retries):
            invalid = (negatives == target) | (
                negatives.unsqueeze(2) == seen.unsqueeze(1)
            ).any(dim=2)
            if not bool(invalid.any()):
                break
            negatives = torch.where(invalid, draw(), negatives)
        return negatives

    def _build_pop_sampling_probs(self, item_pop_counts: torch.Tensor) -> torch.Tensor:
        """Popularity-proportional sampling distribution over real item IDs."""
        probs = item_pop_counts[1:].clamp(min=0).double()
        if float(probs.sum().item()) <= 0.0:
            probs = torch.ones_like(probs)
        return probs / probs.sum()
