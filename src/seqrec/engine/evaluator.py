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

Ranking rule
------------
Every ranking number (full-ranking rank, sampled-ranking rank, and the Top-K list used for
``ARP`` / ``Coverage``) is derived from **one single total order**: descending score with
ascending item id as the tie-breaker. It is materialised with
``torch.argsort(scores, descending=True, stable=True)``, so tied items are ranked by id and
the accuracy metrics can never disagree with the recommendation list. A target is therefore
"hit at K" exactly when it appears in the first ``K`` entries of that order.

Implementation note: the ``torchmetrics.functional.retrieval_*`` helpers are *not* used
because they only accept 1-D per-query inputs and additionally filter candidates with a
``preds > 0`` relevance rule, which is unsound for raw (signed) logits.
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
        self._pop_weights = self._build_pop_sampling_weights(self.item_pop_counts)

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
        """Build the BERT4Rec "oracle" input used for the train-test mismatch study.

        With ``history=[1, 2, 3]``, ``target_id=4`` and ``future_id=5`` the batch is rewritten
        to ``input_ids=[1, 2, 3, <MASK>, 5]``: the ``<MASK>`` sits exactly at the *target*
        position (the item to be predicted) and the known next interaction is kept as
        right-hand context. The position the model must score is published through the
        optional ``score_pos`` batch entry (see :meth:`SequentialRecommender.score_position`),
        and the oldest history tokens are dropped when the window would overflow.

        The protocol is only meaningful on a split that has a known future item (the LOO
        validation split); samples with ``future_id == 0`` are rejected explicitly instead of
        silently scoring a masked position without right-hand context.

        Raises:
            ValueError: If the model exposes no ``mask_token_id``, or if any sample has
                ``future_id == 0``.
        """
        mask_token_id = getattr(model, "mask_token_id", None)
        if mask_token_id is None:
            raise ValueError(
                "EvalConfig.oracle_middle_mask requires a model exposing `mask_token_id` "
                "(e.g. BERT4Rec)."
            )
        missing_future = batch["future_id"] <= 0
        if bool(missing_future.any()):
            raise ValueError(
                f"{int(missing_future.sum().item())} sample(s) in this split have future_id=0, "
                "so the oracle middle-mask protocol has no right-hand context. Use the LOO "
                "validation split (which carries future_id) for oracle evaluation."
            )

        input_ids = batch["input_ids"]
        seq_len = batch["seq_len"]
        batch_size, window = input_ids.shape
        if window < 3:
            raise ValueError(f"oracle_middle_mask needs max_seq_len >= 3, got {window}")

        # Reserve two slots (MASK + future); drop the oldest history tokens if necessary.
        shift = (seq_len + 2 - window).clamp(min=0)
        kept = seq_len - shift
        columns = (
            torch.arange(window, device=input_ids.device)
            .unsqueeze(0)
            .expand(batch_size, -1)
        )
        source = (columns + shift.unsqueeze(1)).clamp(max=window - 1)
        gathered = input_ids.gather(1, source)
        keep_mask = columns < kept.unsqueeze(1)
        new_ids = torch.where(keep_mask, gathered, torch.zeros_like(gathered))

        rows = torch.arange(batch_size, device=input_ids.device)
        mask_pos = kept
        future_pos = kept + 1
        new_ids[rows, mask_pos] = mask_token_id
        new_ids[rows, future_pos] = batch["future_id"]

        masked_batch = dict(batch)
        masked_batch["input_ids"] = new_ids
        masked_batch["seq_len"] = future_pos + 1
        masked_batch["score_pos"] = mask_pos
        return cast(EvalBatch, masked_batch)

    # ------------------------------------------------------------------ metrics

    @property
    def _bias_k(self) -> int:
        """Rank cutoff used for the popularity-bias metrics (``ARP@10`` / ``Coverage@10``)."""
        return min(10, self.num_items)

    def _total_order(self, scores: torch.Tensor) -> torch.Tensor:
        """Return item indices sorted by descending score, ties broken by ascending id."""
        return torch.argsort(scores, dim=1, descending=True, stable=True)

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
        order = self._total_order(scores)
        target = batch["target_id"].unsqueeze(1)
        # 0-based rank of the target inside the single total order shared with `top_ids`.
        rank = (order == target).int().argmax(dim=1)
        rank_f = rank.double()

        metrics: dict[str, torch.Tensor] = {}
        for k in self.ks:
            hit = rank < k
            metrics[f"HR@{k}"] = hit.double()
            metrics[f"NDCG@{k}"] = torch.where(
                hit, 1.0 / torch.log2(rank_f + 2.0), torch.zeros_like(rank_f)
            )
        metrics["MRR"] = 1.0 / (rank_f + 1.0)

        top_ids = order[:, : self._bias_k]
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
        """``HR@k`` / ``NDCG@k`` over ``1`` positive + ``num_neg_samples`` sampled negatives.

        The positive candidate is placed at column ``0`` and ranked with the same total order
        rule as the full-ranking metrics (score descending, then candidate index ascending).
        """
        k = self.config.primary_k
        negatives = self._sample_negatives(batch, self.config.num_neg_samples, strategy)
        positives = batch["target_id"].unsqueeze(1)
        candidates = torch.cat(
            [scores.gather(1, positives), scores.gather(1, negatives)], dim=1
        )
        order = self._total_order(candidates)
        rank = (order == 0).int().argmax(dim=1).double()
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
    ) -> torch.Tensor:
        """Sample negatives from the *valid* candidate distribution.

        Candidates are restricted to items that are neither in the user's history nor the
        positive target, so every returned negative is legal by construction (no rejection
        loop that can silently give up for long-history users). When fewer valid candidates
        than ``num_neg`` remain, sampling is performed with replacement and duplicates are
        returned; a user whose history covers the whole catalogue raises a clear error.
        """
        batch_size = batch["input_ids"].size(0)
        allowed = torch.ones(
            (batch_size, self.num_items + 1), dtype=torch.bool, device=self.device
        )
        allowed[:, 0] = False  # <PAD> is never a valid candidate
        allowed.scatter_(1, batch["seen_ids"], False)
        allowed.scatter_(1, batch["target_id"].unsqueeze(1), False)

        weights = allowed.to(torch.float32)
        if strategy == "popularity":
            weights = weights * self._pop_weights.unsqueeze(0)
        totals = weights.sum(dim=1, keepdim=True)
        empty = totals.squeeze(1) <= 0
        if bool(empty.any()):
            raise ValueError(
                f"{int(empty.sum().item())} sample(s) have no valid negative candidate left "
                "(history + target cover the whole catalogue)."
            )
        return torch.multinomial(weights / totals, num_neg, replacement=True)

    def _build_pop_sampling_weights(
        self, item_pop_counts: torch.Tensor
    ) -> torch.Tensor:
        """Popularity-proportional sampling weights over item IDs ``0..V`` (``<PAD>`` = 0)."""
        weights = item_pop_counts.clamp(min=0).float()
        weights[0] = 0.0
        return weights
