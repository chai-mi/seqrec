"""Task 2 synthetic regression checks; never produces benchmark metrics.

Run: uv run python notebooks/bert4rec_leakage_analysis.py
Open: uv run marimo edit notebooks/bert4rec_leakage_analysis.py

Uses the real Task 1 base, model registry, Trainer and Evaluator. Synthetic
orchestration fixtures are explicit and only write inside TemporaryDirectory.
Real dataset results are produced separately by leakage_exp --action run.
"""

import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")

with app.setup:
    import io
    import sys
    from collections.abc import Sized
    from dataclasses import replace
    from pathlib import Path
    from tempfile import TemporaryDirectory
    from types import ModuleType
    from typing import Any, cast
    from unittest.mock import MagicMock, patch

    import marimo as mo
    import orjson
    import polars as pl
    import torch
    from torch import nn
    from torch.nn import functional as F
    from torch.utils.data import DataLoader

    from seqrec.analysis.leakage_exp import (
        SCHEMA_VERSION,
        Evaluation,
        LeakageConfig,
        RankingMetrics,
        TrainingResult,
        build_plan,
        config_identity,
        evaluate_with_engine,
        file_digest,
        flatten_evaluation,
        json_bytes,
        load_backend,
        oracle_comparison,
        result_path,
        run_experiments,
        source_fingerprints,
        summarize,
        validate_evaluation,
        write_json,
    )
    from seqrec.config import (
        LEN_BUCKETS,
        POP_BUCKETS,
        EvalConfig,
        ExperimentConfig,
        ModelConfig,
        TrainConfig,
    )
    from seqrec.data.dataset import (
        ClozeSeqDataset,
        ClozeTrainBatch,
        DataBundle,
        EvalBatch,
        EvalSample,
        EvalSeqDataset,
    )
    from seqrec.data.preprocess import DatasetMetadata
    from seqrec.engine.evaluator import Evaluator
    from seqrec.engine.trainer import Trainer
    from seqrec.models import build_model
    from seqrec.models.base import SequentialRecommender
    from seqrec.models.bert4rec import BERT4Rec


@app.function
def load_model_for_checks() -> type[BERT4Rec]:
    """Use the actual merged base; no production or test fallback remains."""
    assert issubclass(BERT4Rec, SequentialRecommender)
    return BERT4Rec


@app.function
def sample_batch() -> EvalBatch:
    """Cover full, short, empty and future-unavailable histories in one batch."""
    histories = [[1, 2, 3, 4], [2, 3], [], [4, 5]]
    futures = [5, 6, 0, 0]
    samples = [
        EvalSample(
            user_id=i + 1,
            input_ids=history + [0] * (4 - len(history)),
            seq_len=len(history),
            raw_seq_len=len(history),
            len_bucket_id=0,
            target_id=7,
            target_pop_bucket_id=0,
            future_id=futures[i],
            seen_ids=history,
        )
        for i, history in enumerate(histories)
    ]
    return EvalSeqDataset.collate_fn(samples)


@app.function
def check_eval_inputs(model_type: Any) -> str:
    """Verify exact insertion/truncation, no target use and no input mutation."""
    model = model_type(8, 4, ModelConfig(name="bert4rec", hidden_size=8, dropout=0.0))
    model.eval()
    batch = sample_batch()
    before = {
        key: value.clone()
        for key, value in cast(dict[str, torch.Tensor], batch).items()
    }
    standard, positions = model.prepare_eval_inputs(batch)
    assert standard.tolist() == [[2, 3, 4, 9], [2, 3, 9, 0], [9, 0, 0, 0], [4, 5, 9, 0]]
    assert positions.tolist() == [3, 2, 0, 2]
    scores = model.score_items(batch)
    assert scores.shape == (4, 9)
    assert torch.isneginf(scores[:, 0]).all() and torch.isfinite(scores[:, 1:]).all()
    changed = batch.copy()
    changed["target_id"] = torch.ones(4, dtype=torch.long)
    changed["future_id"] = torch.full((4,), 8, dtype=torch.long)
    torch.testing.assert_close(scores, model.score_items(changed))
    model.oracle_middle_mask = True
    oracle, oracle_positions = model.prepare_eval_inputs(batch)
    assert oracle.tolist() == [[3, 4, 9, 5], [2, 3, 9, 6], [9, 0, 0, 0], [4, 5, 9, 0]]
    assert oracle_positions.tolist() == [2, 2, 0, 2]
    oracle_scores = model.score_items(batch)
    torch.testing.assert_close(scores[2:], oracle_scores[2:])
    model.oracle_middle_mask, model.oracle_history_control = False, True
    control, control_positions = model.prepare_eval_inputs(batch)
    assert control.tolist() == [[3, 4, 9, 0], [2, 3, 9, 0], [9, 0, 0, 0], [4, 5, 9, 0]]
    torch.testing.assert_close(control_positions, oracle_positions)
    for key, value in batch.items():
        torch.testing.assert_close(value, before[key])
    return "standard / Oracle / matched-history insertion, immutability: passed"


@app.function
def check_attention_and_loss(model_type: Any) -> str:
    """Check bidirectionality, PAD isolation, tied/untied CE and zero-label safety."""
    for tied in (True, False):
        model = model_type(
            8,
            4,
            ModelConfig(
                name="bert4rec",
                hidden_size=8,
                num_layers=2,
                dropout=0.0,
                share_item_embeddings=tied,
            ),
        )
        model.eval()
        prefix = torch.tensor([[1, 2]])
        padded = torch.tensor([[1, 2, 0, 0]])
        torch.testing.assert_close(
            model(prefix), model(padded)[:, :2], atol=1e-6, rtol=1e-5
        )
        left = model(torch.tensor([[1, 2, 3]]))[:, 0]
        right = model(torch.tensor([[1, 2, 4]]))[:, 0]
        assert not torch.allclose(left, right, atol=1e-8, rtol=1e-8)
        assert torch.isfinite(model(torch.zeros((2, 4), dtype=torch.long))).all()
        batch: ClozeTrainBatch = {
            "user_id": torch.tensor([1, 2]),
            "input_ids": torch.tensor([[1, 9, 0, 0], [9, 2, 3, 0]]),
            "labels": torch.tensor([[0, 3, 0, 0], [4, 0, 2, 0]]),
            "seq_len": torch.tensor([2, 3]),
        }
        labels = batch["labels"]
        selected = labels > 0
        hidden = model(batch["input_ids"])[selected]
        weight = (
            model.item_embeddings.weight[1:9]
            if tied
            else model.output_projection.weight
        )
        manual = F.cross_entropy(
            F.linear(model.prediction_transform(hidden), weight, model.output_bias),
            labels[selected] - 1,
        )
        loss = model.compute_loss(batch)
        torch.testing.assert_close(loss, manual)
        assert loss.ndim == 0 and torch.isfinite(loss)
        loss.backward()
        assert model.item_embeddings.weight.grad is not None
        assert model.item_embeddings.weight.grad[0].eq(0).all()
        assert model.item_embeddings.weight.grad[9].abs().sum() > 0
        for parameter in model.parameters():
            if parameter.grad is not None:
                assert torch.isfinite(parameter.grad).all()
        empty = batch.copy()
        empty["labels"] = torch.zeros_like(labels)
        model.zero_grad(set_to_none=True)
        zero_loss = model.compute_loss(empty)
        assert zero_loss.item() == 0
        zero_loss.backward()
        invalid = batch.copy()
        invalid["labels"] = labels.clone()
        invalid["labels"][0, 2] = 1
        try:
            model.compute_loss(invalid)
        except ValueError:
            pass
        else:
            raise AssertionError("Labelled PAD was accepted")
        # Encoder clones should not start with identical packed attention weights.
        assert not torch.equal(
            model.encoder.layers[0].self_attn.in_proj_weight,
            model.encoder.layers[1].self_attn.in_proj_weight,
        )
    return "bidirectional attention / PAD isolation / CE / gradients / tied+untied head: passed"


@app.function
def check_small_window_and_training(model_type: Any) -> str:
    """Exercise N=1, overfit a tiny batch and round-trip a state_dict in memory."""
    config = ModelConfig(name="bert4rec", hidden_size=8, num_layers=1, dropout=0.0)
    tiny = model_type(8, 1, config)
    assert tiny.prepare_eval_inputs(sample_batch())[0].tolist() == [[9], [9], [9], [9]]
    tiny.oracle_middle_mask = True
    try:
        tiny.prepare_eval_inputs(sample_batch())
    except ValueError:
        pass
    else:
        raise AssertionError("N=1 Oracle should be rejected")
    model = model_type(8, 4, config)
    batch: ClozeTrainBatch = {
        "user_id": torch.tensor([1, 2]),
        "input_ids": torch.tensor([[1, 9, 0, 0], [2, 9, 0, 0]]),
        "labels": torch.tensor([[0, 3, 0, 0], [0, 4, 0, 0]]),
        "seq_len": torch.tensor([2, 2]),
    }
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.02)
    initial = float(model.compute_loss(batch).detach())
    for _ in range(40):
        optimizer.zero_grad(set_to_none=True)
        loss = model.compute_loss(batch)
        loss.backward()
        optimizer.step()
    final = float(model.compute_loss(batch).detach())
    assert final < initial * 0.6, (initial, final)
    assert model.item_embeddings.weight[0].eq(0).all()
    model.eval()
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    buffer.seek(0)
    restored = model_type(8, 4, config).eval()
    restored.load_state_dict(torch.load(buffer, weights_only=True))
    torch.testing.assert_close(
        model.score_items(sample_batch()), restored.score_items(sample_batch())
    )
    return f"N=1 / synthetic overfit ({initial:.3f} -> {final:.3f}) / checkpoint round-trip: passed"


@app.function
def check_device(model_type: Any) -> str:
    """Smoke-test actual CUDA/autocast if available, without claiming a dataset run."""
    if not torch.cuda.is_available():
        return "CUDA check skipped: no available CUDA runtime"
    model = model_type(
        8, 4, ModelConfig(name="bert4rec", hidden_size=8, num_layers=1)
    ).cuda()
    batch: ClozeTrainBatch = {
        "user_id": torch.tensor([1], device="cuda"),
        "input_ids": torch.tensor([[1, 9, 0, 0]], device="cuda"),
        "labels": torch.tensor([[0, 2, 0, 0]], device="cuda"),
        "seq_len": torch.tensor([2], device="cuda"),
    }
    with torch.autocast("cuda", dtype=torch.float16):
        loss = model.compute_loss(batch)
    assert torch.isfinite(loss)
    loss.backward()
    assert all(
        torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None
    )
    return "CUDA float16 autocast forward/backward: passed"


@app.function
def synthetic_metrics() -> Evaluation:
    """TEST FIXTURE ONLY: validates artifact handling, never saved to reports/."""
    values = {f"{name}@{k}": 0.25 for name in ("HR", "NDCG") for k in (5, 10, 20)}
    values.update({"MRR": 0.25, "ARP@10": 2.0, "Coverage@10": 0.5})
    metrics = RankingMetrics(
        4, values, {key: {} for key in LEN_BUCKETS}, {key: {} for key in POP_BUCKETS}
    )
    return Evaluation(metrics, metrics, metrics)


@app.function
def synthetic_evaluate(
    model: nn.Module,
    loader: DataLoader[EvalSample],
    bundle: DataBundle,
    config: EvalConfig,
    seed: int,
) -> Evaluation:
    """Adapter TEST DOUBLE, not a ranking implementation or benchmark result."""
    assert not model.training and not torch.is_grad_enabled()
    assert config.oracle_middle_mask == getattr(model, "oracle_middle_mask", False)
    scorer = getattr(model, "score_items", None)
    if callable(scorer):
        for batch in loader:
            scores = scorer(batch)
            assert torch.isfinite(scores[:, 1:]).all()
    metrics = synthetic_metrics()
    count = len(cast(Sized, loader.dataset))
    return Evaluation(
        *(
            replace(getattr(metrics, name), num_examples=count)
            for name in ("full", "sampled_uniform", "sampled_popularity")
        )
    )


@app.function
def synthetic_backend(config: ExperimentConfig) -> TrainingResult:
    """TEST ONLY: exercise orchestration without implementing Task 1's engine."""
    frame = pl.DataFrame(
        {
            "user_id": [1, 2, 3, 4],
            "history": [[1, 2, 3, 4], [2, 3], [1], [4, 5]],
            "len_bucket": ["<=5"] * 4,
            "target_id": [7] * 4,
            "target_pop_bucket": ["Head"] * 4,
            "future_id": [5, 6, 0, 0],
        }
    )
    dataset = EvalSeqDataset(frame, config.data.resolved_max_seq_len)
    loader = DataLoader(dataset, batch_size=2, collate_fn=EvalSeqDataset.collate_fn)
    bundle = cast(DataBundle, MagicMock(spec=DataBundle))
    bundle.val_loader = loader
    bundle.test_loader = loader
    bundle.metadata = MagicMock(dataset=config.data.dataset)
    bundle.max_seq_len = config.data.resolved_max_seq_len
    return TrainingResult(
        nn.Linear(1, 1),
        bundle,
        synthetic_evaluate,
        None,
        [],
        None,
        0.0,
        {"test_only": "synthetic metrics, not real experiments"},
    )


@app.function
def check_orchestration(model_type: Any) -> str:
    """Test cohort selection, mode restoration, persistence/resume and provenance guards."""
    config = ExperimentConfig()
    config.data.max_seq_len = 4
    trained = synthetic_backend(config)
    trained.model = model_type(8, 4, ModelConfig(name="bert4rec", hidden_size=8))
    private_module: Any = ModuleType("seqrec.models.bert4rec")
    private_module.BERT4Rec = model_type
    with patch.dict(sys.modules, {"seqrec.models.bert4rec": private_module}):
        comparison = oracle_comparison(trained, config)
        assert comparison["num_examples"] == 2
        assert comparison["missing_future_examples"] == 2
        assert trained.model.training
        assert not trained.model.oracle_middle_mask
        assert not trained.model.oracle_history_control
        # Errors in an evaluator must not leave diagnostic flags enabled.
        trained.evaluate = MagicMock(side_effect=RuntimeError("test failure"))
        trained.model.oracle_history_control = True
        try:
            oracle_comparison(trained, config)
        except RuntimeError as error:
            assert str(error) == "test failure"
        else:
            raise AssertionError("Evaluator error must propagate")
        assert not trained.model.oracle_middle_mask
        assert trained.model.oracle_history_control
        assert trained.model.training
    with TemporaryDirectory() as directory:
        options = LeakageConfig(
            suite="temporal",
            datasets=("beauty",),
            metrics_dir=Path(directory),
            backend="test_only:synthetic",
        )
        # Pop has no checkpoint, so no fake checkpoint file is necessary here.
        plan = [spec for spec in build_plan(options) if spec.config.model.name == "pop"]
        with (
            patch(
                "seqrec.analysis.leakage_exp.load_backend",
                return_value=synthetic_backend,
            ),
            patch(
                "seqrec.analysis.leakage_exp.input_fingerprints",
                return_value={"test": "only"},
            ),
        ):
            run_experiments(options, plan)
            before = {
                result_path(spec): result_path(spec).stat().st_mtime_ns for spec in plan
            }
            run_experiments(options, plan)
            assert all(
                path.stat().st_mtime_ns == stamp for path, stamp in before.items()
            )
            assert summarize(plan)["complete"] is True
            record = orjson.loads(result_path(plan[0]).read_bytes())
            record["provenance"]["inputs"]["test"] = "changed"
            write_json(result_path(plan[0]), record)
            try:
                run_experiments(options, plan)
            except ValueError as error:
                assert "differs" in str(error)
            else:
                raise AssertionError("Resume accepted changed input provenance")
    return "TEST DOUBLES: Oracle cohort / exception restoration / atomic save / resume guards: passed"


@app.function
def small_bundle(config: ExperimentConfig) -> DataBundle:
    """Real Dataset/DataLoader objects over a tiny deterministic in-memory table."""
    frame = pl.DataFrame(
        {
            "user_id": [1, 2, 3, 4],
            "history": [[1, 2, 3, 4], [2, 3], [1], [4, 5]],
            "len_bucket": ["<=5"] * 4,
            "target_id": [7] * 4,
            "target_pop_bucket": ["Head"] * 4,
            "future_id": [5, 6, 2, 3],
        }
    )
    dataset = EvalSeqDataset(frame, config.data.resolved_max_seq_len)
    loader = DataLoader(dataset, batch_size=2, collate_fn=EvalSeqDataset.collate_fn)
    train = ClozeSeqDataset(
        pl.DataFrame(
            {
                "user_id": [1, 2, 3, 4],
                "item_seq": [[1, 2, 3, 7], [2, 3, 4, 7], [3, 4, 5, 7], [1, 4, 6, 7]],
            }
        ),
        max_seq_len=config.data.resolved_max_seq_len,
        num_items=128,
    )
    metadata = DatasetMetadata(
        dataset="beauty",
        hf_repo="synthetic-test-only",
        min_interactions=1,
        num_users=4,
        num_items=128,
        num_interactions=16,
        density_pct=0.0,
        avg_seq_len=4.0,
        median_seq_len=4.0,
        p90_seq_len=4.0,
        min_seq_len=4,
        max_seq_len=4,
        repeat_click_ratio=0.0,
        temporal_t80=0,
        temporal_t90=0,
        loo_counts={},
        temporal_counts={},
        loo_test_len_buckets={},
        loo_test_pop_buckets={},
    )
    counts = torch.ones(129)
    counts[0] = 0
    return DataBundle(
        metadata,
        config.data.resolved_max_seq_len,
        counts,
        torch.zeros(129, dtype=torch.long),
        DataLoader(train, batch_size=2, collate_fn=train.collate_fn),
        loader,
        loader,
    )


@app.function
def check_shared_engine() -> str:
    """Real factory, Trainer checkpoint selection and Evaluator Oracle integration."""
    torch.manual_seed(42)
    config = ExperimentConfig()
    config.model = ModelConfig(
        name="bert4rec",
        hidden_size=12,
        num_heads=3,
        num_layers=1,
        dropout=0.0,
        share_item_embeddings=False,
    )
    config.data.max_seq_len = 4
    config.train.device = "cpu"
    config.train.epochs = 2
    bundle = small_bundle(config)
    model = build_model(config.model, bundle.num_items, 4)
    assert isinstance(model, BERT4Rec) and model.paradigm == "cloze"
    assert model.item_embeddings.embedding_dim == 12
    assert model.encoder.layers[0].self_attn.num_heads == 3
    assert model.output_projection is not None
    model.eval()
    batch = next(iter(bundle.val_loader))
    evaluator = Evaluator(
        EvalConfig(oracle_middle_mask=True),
        bundle.num_items,
        bundle.item_pop_counts,
        torch.device("cpu"),
    )
    rewritten = evaluator.apply_oracle_middle_mask(model, batch)
    with torch.inference_mode():
        model.oracle_middle_mask = True
        direct = model.score_items(batch)
        prepared = model.score_items(rewritten)
        torch.testing.assert_close(direct, prepared)
        model.oracle_middle_mask = False
        torch.testing.assert_close(prepared, model.score_items(rewritten))
        hidden = model(rewritten["input_ids"])
        pos = cast(dict[str, torch.Tensor], rewritten)["score_pos"]
        manual = model._item_logits(hidden[torch.arange(len(pos)), pos])
        torch.testing.assert_close(prepared[:, 1:], manual)
    with TemporaryDirectory() as directory:
        config.checkpoints_dir = Path(directory)
        trainer = Trainer(config, model, bundle, show_progress=False)
        result = trainer.fit()
        assert trainer.checkpoint_path.is_file()
        assert result.best_epoch in (1, 2) and len(result.history) == 2
        state = torch.load(trainer.checkpoint_path, weights_only=True)
        for name, value in model.state_dict().items():
            torch.testing.assert_close(value, state["model"][name])
        metrics = evaluate_with_engine(
            model, bundle.val_loader, bundle, config.eval, 42
        )
        validate_evaluation(metrics, config.eval, 4)
        native = Trainer(config, model, bundle, show_progress=False).evaluator
        torch.manual_seed(42)
        flat = native.evaluate(model, bundle.val_loader, show_progress=False)
        assert metrics.full.overall["NDCG@10"] == flat["NDCG@10"]
        assert (
            metrics.sampled_uniform.overall["NDCG@10"]
            == flat["sampled_uniform/NDCG@10"]
        )
        trained = TrainingResult(
            model,
            bundle,
            evaluate_with_engine,
            trainer.checkpoint_path,
            [],
            result.best_epoch,
            result.train_seconds,
            {"test": "only"},
        )
        comparison = oracle_comparison(trained, config)
        assert comparison["num_examples"] == 4
        assert not model.oracle_middle_mask and not model.oracle_history_control
    return "REAL ENGINE: config propagation / score_pos / 2-epoch training / checkpoint / paired Oracle: passed"


@app.function
def check_experiment_plan() -> str:
    """Check deduplication, owned filenames, explicit dependencies and missing results."""
    options = LeakageConfig()
    plan = build_plan(options)
    assert len(plan) == 27
    assert len({config_identity(run.config) for run in plan}) == 27
    assert len(build_plan(replace(options, suite="windows"))) == 9
    assert len(build_plan(replace(options, suite="masking"))) == 6
    assert len(build_plan(replace(options, suite="temporal"))) == 16
    assert all(not spec.config.eval.oracle_middle_mask for spec in plan)
    assert all(
        result_path(spec).name.startswith(("bert4rec_", "leakage_")) for spec in plan
    )
    assert config_identity(plan[0].config) != config_identity(
        replace(plan[0].config, train=replace(plan[0].config.train, lr=0.005))
    )
    try:
        load_backend(None)
    except RuntimeError as error:
        assert "Task 1" in str(error)
    else:
        raise AssertionError("Missing backend must not appear to train successfully")
    metrics = synthetic_metrics()
    validate_evaluation(metrics, EvalConfig(), 4)
    try:
        validate_evaluation(metrics, EvalConfig(), 3)
    except ValueError:
        pass
    else:
        raise AssertionError("Mismatched sample count was accepted")
    with TemporaryDirectory() as directory:
        subset = build_plan(
            replace(
                options,
                suite="masking",
                datasets=("beauty",),
                metrics_dir=Path(directory),
            )
        )
        summary = summarize(subset)
        assert (
            summary["complete"] is False
            and len(cast(list[str], summary["missing"])) == 3
        )
        spec = subset[0]
        write_json(
            result_path(spec),
            {
                "schema_version": SCHEMA_VERSION,
                "status": "complete",
                "config": spec.config,
                "test": metrics,
                "oracle_validation": None,
            },
        )
        partial = summarize(subset)
        assert (
            partial["complete"] is False
            and len(cast(list[dict[str, object]], partial["experiments"])) == 1
        )
        assert len(cast(list[str], partial["missing"])) == 2
        assert b"uniform_minus_full" in json_bytes(partial)
    return "27-run plan / deduplication / explicit blockers / partial summary / metric schema: passed"


@app.function
def load_real_results(metrics_dir: Path) -> pl.DataFrame:
    """Read only the official 27-run grid; exclude temporary smoke/pilot outputs."""
    options = LeakageConfig(
        metrics_dir=metrics_dir,
        train=TrainConfig(epochs=200),
        batch_size=256,
        eval_batch_size=256,
    )
    rows: list[dict[str, object]] = []
    for spec in build_plan(options):
        path = result_path(spec)
        if not path.is_file():
            continue
        record = orjson.loads(path.read_bytes())
        if record.get("status") != "complete" or record["config"] != orjson.loads(
            json_bytes(spec.config)
        ):
            raise ValueError(f"Incomplete/mismatched artifact: {path}")
        config = spec.config
        oracle = record["oracle_validation"]
        rows.append(
            {
                "dataset": config.data.dataset,
                "model": config.model.name,
                "split": config.data.split_strategy,
                "window": config.data.resolved_max_seq_len,
                "last_mask": config.data.last_item_mask_ratio,
                "best_epoch": record["best_epoch"],
                "epochs_run": len(record["history"]),
                "test_hr10": record["test"]["full"]["overall"]["HR@10"],
                "test_ndcg10": record["test"]["full"]["overall"]["NDCG@10"],
                "uniform_ndcg10": record["test"]["sampled_uniform"]["overall"][
                    "NDCG@10"
                ],
                "popularity_ndcg10": record["test"]["sampled_popularity"]["overall"][
                    "NDCG@10"
                ],
                "val_standard": oracle["standard"]["full"]["overall"]["NDCG@10"]
                if oracle
                else None,
                "val_matched": oracle["matched_history"]["full"]["overall"]["NDCG@10"]
                if oracle
                else None,
                "val_oracle": oracle["oracle"]["full"]["overall"]["NDCG@10"]
                if oracle
                else None,
                "path": str(path),
            }
        )
    schema = {
        "dataset": pl.String,
        "model": pl.String,
        "split": pl.String,
        "window": pl.Int64,
        "last_mask": pl.Float64,
        "best_epoch": pl.Int64,
        "epochs_run": pl.Int64,
        "test_hr10": pl.Float64,
        "test_ndcg10": pl.Float64,
        "uniform_ndcg10": pl.Float64,
        "popularity_ndcg10": pl.Float64,
        "val_standard": pl.Float64,
        "val_matched": pl.Float64,
        "val_oracle": pl.Float64,
        "path": pl.String,
    }
    return (
        pl.DataFrame(rows, schema=schema)
        .lazy()
        .sort("dataset", "model", "split", "window", "last_mask")
        .collect()
    )


@app.function
def comparison_tables(results: pl.DataFrame) -> dict[str, pl.DataFrame]:
    """Lazy projections for the windows, masking, temporal and sampling studies."""
    lf = results.lazy()
    default_window = ((pl.col("dataset") == "beauty") & (pl.col("window") == 50)) | (
        (pl.col("dataset") == "ml-1m") & (pl.col("window") == 200)
    )
    primary = lf.filter(default_window & (pl.col("last_mask") == 0.1))
    windows = (
        lf.filter(
            (pl.col("model") == "bert4rec")
            & (pl.col("split") == "loo")
            & (pl.col("last_mask") == 0.1)
        )
        .select("dataset", "window", "test_hr10", "test_ndcg10", "best_epoch")
        .collect()
    )
    masking = (
        lf.filter(pl.col("val_oracle").is_not_null())
        .with_columns(
            (pl.col("val_oracle") - pl.col("val_matched")).alias("future_gain")
        )
        .select(
            "dataset",
            "last_mask",
            "test_ndcg10",
            "val_standard",
            "val_matched",
            "val_oracle",
            "future_gain",
        )
        .collect()
    )
    temporal = (
        primary.filter(pl.col("split") == "loo")
        .select("dataset", "model", pl.col("test_ndcg10").alias("loo_ndcg10"))
        .join(
            primary.filter(pl.col("split") == "temporal").select(
                "dataset", "model", pl.col("test_ndcg10").alias("temporal_ndcg10")
            ),
            on=["dataset", "model"],
            how="full",
            coalesce=True,
            validate="1:1",
        )
        .with_columns(
            (pl.col("temporal_ndcg10") - pl.col("loo_ndcg10")).alias(
                "temporal_minus_loo"
            )
        )
        .sort("dataset", "model")
        .collect()
    )
    sampling = primary.select(
        "dataset",
        "model",
        "split",
        "test_ndcg10",
        "uniform_ndcg10",
        "popularity_ndcg10",
    ).collect()
    return {
        "窗口敏感性": windows,
        "掩码/Oracle 对照（val）": masking,
        "LOO / temporal": temporal,
        "Full / sampled": sampling,
    }


@app.function
def audit_completed_experiments(metrics_dir: Path) -> dict[str, int]:
    """Check real artifacts, checkpoint identity, selection curves and bucket consistency."""
    plan = build_plan(
        LeakageConfig(
            metrics_dir=metrics_dir,
            train=TrainConfig(epochs=200),
            batch_size=256,
            eval_batch_size=256,
        )
    )
    completed = 0
    source = source_fingerprints()
    for spec in plan:
        path = result_path(spec)
        if not path.exists():
            continue
        record = orjson.loads(path.read_bytes())
        assert (
            record["status"] == "complete"
            and record["schema_version"] == SCHEMA_VERSION
        )
        assert record["config"] == orjson.loads(json_bytes(spec.config))
        assert all(
            record["provenance"]["source"][key] == value
            for key, value in source.items()
        )
        assert "WITH replacement" in record["protocol"]["sampling"]
        if spec.config.model.name != "pop":
            checkpoint = Path(record["checkpoint"])
            assert file_digest(checkpoint) == record["checkpoint_sha256"]
            state = torch.load(checkpoint, map_location="cpu", weights_only=True)
            assert state["epoch"] == record["best_epoch"]
            assert state["exp_name"] == spec.config.exp_name
            selected = max(
                (row for row in record["history"] if row["val_metrics"]),
                key=lambda row: row["val_primary"],
            )
            assert selected["epoch"] == record["best_epoch"]
            assert (
                abs(
                    selected["val_primary"]
                    - record["validation"]["full"]["overall"]["NDCG@10"]
                )
                < 1e-10
            )
        evaluations = [record["validation"], record["test"]]
        oracle = record["oracle_validation"]
        if oracle is not None:
            assert (
                spec.config.data.split_strategy == "loo" and oracle["diagnostic_only"]
            )
            assert (
                oracle["num_examples"] + oracle["missing_future_examples"]
                == oracle["total_validation_examples"]
            )
            evaluations.extend(
                oracle[name] for name in ("standard", "matched_history", "oracle")
            )
        for evaluation in evaluations:
            for name in ("full", "sampled_uniform", "sampled_popularity"):
                metrics = evaluation[name]
                count = metrics["num_examples"]
                for kind in ("length_buckets", "popularity_buckets"):
                    populated = [bucket for bucket in metrics[kind].values() if bucket]
                    assert sum(bucket["count"] for bucket in populated) == count
                    for metric in ("HR@10", "NDCG@10"):
                        weighted = (
                            sum(
                                bucket["count"] * bucket[metric] for bucket in populated
                            )
                            / count
                        )
                        assert abs(weighted - metrics["overall"][metric]) < 1e-10
                for metric, value in metrics["overall"].items():
                    if metric != "ARP@10":
                        assert 0.0 <= value <= 1.0, (path, metric, value)
        test = Evaluation(
            *(
                RankingMetrics(**record["test"][name])
                for name in ("full", "sampled_uniform", "sampled_popularity")
            )
        )
        assert record["test_metrics"] == flatten_evaluation(test)
        completed += 1
    return {"completed_and_audited": completed, "expected": len(plan)}


@app.cell
def _():
    real_results = load_real_results(Path("reports/metrics"))
    real_tables = comparison_tables(real_results)
    real_audit = audit_completed_experiments(Path("reports/metrics"))
    print(
        f"Official experiments audited: {real_audit['completed_and_audited']}/{real_audit['expected']}"
    )
    mo.vstack(
        [
            mo.md(
                f"## 正式实验：已完成 {real_results.height}/27 个配置\n"
                "缺失结果保持空缺，Oracle 仅为验证集诊断，不属于正常测试指标。"
            ),
            mo.ui.tabs(
                {name: mo.ui.table(frame) for name, frame in real_tables.items()}
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # BERT4Rec / 专题三：实现与验证

    **已对接 Task 1 的真实基类、模型工厂、Trainer 和 Evaluator。**
    本笔记本的合成检查不代表正式指标；真实实验的完成情况见下方结果表及
    `reports/metrics/leakage_summary_*.json` 的 `complete` / `missing` 字段。
    测试替身仅用于异常/恢复路径测试，其产物只写入临时目录。

    - 模型：`src/seqrec/models/bert4rec.py`，使用真实公共基类，工厂完整传递 ModelConfig。
    - 编排：`src/seqrec/analysis/leakage_exp.py`，提供 27 个去重配置、配对 Oracle
      对照、结果持久化/恢复、时间切分差值和采样排名反转汇总。
    - 默认后端已实现：直接复用 `Trainer.fit()`，恢复普通验证集选出的最佳 checkpoint。
      Oracle 通过 `score_pos` 协议评分，不会重复插入 MASK，不用于训练或早停。
    - 保持公共评测协议：负样本为有放回 100 次抽取，重复项分别计数；不是去重的
      100 个负样本。采样排序仅提供 HR@10 / NDCG@10，不伪造未实现的 sampled MRR。

    ```bash
    uv run python notebooks/bert4rec_leakage_analysis.py
    uv run python -m seqrec.analysis.leakage_exp --action plan
    uv run python -m seqrec.analysis.leakage_exp --action plan --suite masking --datasets beauty
    uv run python -m seqrec.analysis.leakage_exp --action run --train.epochs 200 --batch-size 256 --eval-batch-size 256
    uv run python -m seqrec.analysis.leakage_exp --action summary --train.epochs 200 --batch-size 256 --eval-batch-size 256
    ```

    `--action run` 使用内置真实后端，无需另外指定 backend。
    `--action summary` 只汇总已有真实结果，缺失实验会保留在 `missing` 中。
    本机 4GB GPU，显存验证后采用与 Task 1 一致的 batch size 256 / eval batch size 256、
    seed=42、上限 200 epochs、
    patience=10，默认模型 d=64 / 2 层 / 2 heads。为保持同配置对照，专题三的
    Pop/GRU4Rec/SASRec 也在本机重跑（不会覆写 Task 1 已提交的结果）。
    相同配置和来源的已完成实验会跳过；训练完成后的 checkpoint 与训练曲线缓存可用于
    失败后重试后处理。正在训练的优化器状态不支持中途续训，停止后会从头训练该配置。
    原始 Trainer 指标字段保留给 Task 3，并另外保存配对 Oracle 和采样分桶结果。

    本轮新增文件的 Ruff / 类型检查 / Marimo 检查通过。全仓格式检查仍有已有文件
    不合规，完整 `ty check` 仍有旧探索笔记本的两处 Altair 属性诊断；未越界修改。
    """)
    return


@app.cell
def _():
    torch.manual_seed(42)
    torch.set_num_threads(2)
    model_type = load_model_for_checks()
    check_results = [
        check_eval_inputs(model_type),
        check_attention_and_loss(model_type),
        check_small_window_and_training(model_type),
        check_device(model_type),
        check_experiment_plan(),
        check_orchestration(model_type),
        check_shared_engine(),
    ]
    for _result in check_results:
        print(_result)
    mo.md(
        "## Task 2 回归检查\n\n"
        + "\n\n".join(check_results)
        + "\n\n**这些是合成数据验证，不是 Beauty/ML-1M 实验指标。**"
        + "\n\n已使用真实公共基类、注册工厂及评测器。"
    )
    return


if __name__ == "__main__":
    app.run()
