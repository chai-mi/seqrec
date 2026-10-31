"""SR-GNN correctness checks, real-data sweeps, and vector figure regeneration.

uv run python notebooks/srgnn_and_plots.py
SEQREC_TASK3_TRAIN=1 uv run python notebooks/srgnn_and_plots.py

Training is opt-in; it calls the shared run_experiment entrypoint, seed 42, 200 epochs
maximum, patience 10 and the prescribed nine windows. Completed runs with matching
configurations are retained. Checkpoints and processed data remain untracked.
"""

import marimo

__generated_with = "0.25.2"
app = marimo.App(width="full")

with app.setup:
    import dataclasses
    import os
    import tempfile
    from pathlib import Path
    from typing import cast

    import orjson
    import polars as pl
    import torch
    from torch.nn import functional as F
    from torch.utils.data import DataLoader, Subset

    from seqrec.analysis.plot_results import (
        PlotConfig,
        build_charts,
        format_table,
        missing_experiments,
        read_metrics,
        table_rows,
    )
    from seqrec.analysis.plot_results import (
        main as plot_all,
    )
    from seqrec.config import (
        DataConfig,
        EvalConfig,
        ExperimentConfig,
        LossType,
        ModelConfig,
        TrainConfig,
    )
    from seqrec.data.dataset import (
        EvalBatch,
        SessionGraphDataset,
        SessionGraphTrainBatch,
        build_databundle,
        build_session_graph_batch,
    )
    from seqrec.engine.evaluator import Evaluator, move_batch_to_device
    from seqrec.engine.trainer import (
        Trainer,
        run_experiment,
        seed_everything,
    )
    from seqrec.models import build_model
    from seqrec.models.srgnn import SRGNN


@app.function
def example_eval(ids: torch.Tensor, lengths: torch.Tensor) -> EvalBatch:
    """Make a small typed evaluation batch including the evaluator's metadata."""
    size = ids.size(0)
    zeros = torch.zeros(size, dtype=torch.long, device=ids.device)
    return {
        "input_ids": ids,
        "seq_len": lengths,
        "user_id": zeros + 1,
        "raw_seq_len": lengths,
        "len_bucket_id": zeros,
        "target_id": zeros + 7,
        "target_pop_bucket_id": zeros,
        "future_id": zeros + 8,
        "seen_ids": ids,
    }


@app.function
def check_model() -> dict[str, str]:
    """Verify graph direction, repeated aliases, padding, gradients and CUDA AMP."""
    seed_everything(42)
    ids = torch.tensor([[2, 3, 2, 4, 0], [5, 0, 0, 0, 0]])
    lengths = torch.tensor([4, 1])
    items, aliases, adj_in, adj_out = build_session_graph_batch(ids, lengths)
    assert items.tolist() == [[2, 3, 0, 4, 0], [5, 0, 0, 0, 0]]
    assert aliases[0, :4].tolist() == [0, 1, 0, 3]
    assert adj_out[0, 0].tolist() == [0, 0.5, 0, 0.5, 0]
    assert adj_in[0, 0, 1] == adj_in[0, 1, 0] == adj_in[0, 3, 0] == 1
    assert adj_in[1].count_nonzero() == adj_out[1].count_nonzero() == 0
    model = build_model(ModelConfig(name="srgnn", dropout=0), 10, 5)
    assert isinstance(model, SRGNN) and model.paradigm == "graph"
    model.eval()
    batch = example_eval(ids, lengths)
    scores = model.score_items(batch)
    assert scores.shape == (2, 11) and scores[:, 0].isneginf().all()
    assert scores[:, 1:].isfinite().all()
    graphs = model.encode_graph(items, aliases, adj_in, adj_out, lengths)
    torch.testing.assert_close(scores[:, 1:], model.project(graphs))
    torch.testing.assert_close(
        scores[:1], model.score_items(example_eval(ids[:1, :4], lengths[:1]))
    )
    torch.testing.assert_close(
        scores[1:], model.score_items(example_eval(ids[1:, :1], lengths[1:]))
    )
    # Relabel graph nodes (the item identities and resulting session stay the same).
    permutation = torch.tensor([3, 1, 4, 0, 2])
    inverse = permutation.argsort()
    permuted = model.encode_graph(
        items[:, permutation],
        inverse[aliases],
        adj_in[:, permutation][:, :, permutation],
        adj_out[:, permutation][:, :, permutation],
        lengths,
    )
    torch.testing.assert_close(graphs, permuted)
    reversed_edges = model.encode_graph(items, aliases, adj_out, adj_in, lengths)
    assert not torch.allclose(graphs[0], reversed_edges[0])
    changed_metadata: EvalBatch = {
        **batch,
        "target_id": torch.tensor([9, 9]),
        "future_id": torch.tensor([1, 1]),
    }
    torch.testing.assert_close(scores, model.score_items(changed_metadata))
    train_batch = SessionGraphDataset.collate_fn(
        [
            (torch.tensor(1), ids[0], torch.tensor(7), lengths[0]),
            (torch.tensor(2), ids[1], torch.tensor(7), lengths[1]),
        ]
    )
    loss = model.compute_loss(train_batch)
    torch.testing.assert_close(
        loss, F.cross_entropy(scores[:, 1:], torch.tensor([6, 6]))
    )
    loss.backward()
    assert model.item_embedding.weight.grad is not None
    assert model.item_embedding.weight.grad[0].count_nonzero() == 0
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None and parameter.grad.isfinite().all(), name
        assert parameter.grad.abs().sum() > 0, name
    before = loss.item()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    for _ in range(20):
        optimizer.zero_grad()
        current = model.compute_loss(train_batch)
        current.backward()
        optimizer.step()
    assert model.compute_loss(train_batch).item() < before / 2
    invalid_configs: tuple[tuple[int, LossType, bool], ...] = (
        (0, "ce", True),
        (1, "bce", True),
        (1, "ce", False),
    )
    for steps, loss_type, shared in invalid_configs:
        try:
            SRGNN(
                10, ggnn_steps=steps, loss_type=loss_type, share_item_embeddings=shared
            )
        except ValueError:
            pass
        else:
            raise AssertionError("Unsupported SR-GNN configuration accepted")
    status = {"cpu": "passed", "cuda_amp": "unavailable"}
    if torch.cuda.is_available():
        gpu = SRGNN(10, dropout=0).cuda()
        gpu_batch = cast(
            SessionGraphTrainBatch,
            move_batch_to_device(
                cast(dict[str, torch.Tensor], train_batch),
                torch.device("cuda"),
            ),
        )
        with torch.amp.autocast("cuda"):
            gpu_loss = gpu.compute_loss(gpu_batch)
        gpu_loss.backward()
        assert gpu_loss.isfinite()
        assert all(
            p.grad is not None and p.grad.isfinite().all() for p in gpu.parameters()
        )
        status["cuda_amp"] = "passed"
    return status


@app.function
def check_plots(metrics_dir: Path = Path("reports/metrics")) -> dict[str, int]:
    """Verify real schemas, empty buckets, duplicate rejection and vector specs."""
    frame = read_metrics(metrics_dir)
    assert set(frame["model"].to_list()).issubset(
        {"Pop", "GRU4Rec", "SASRec", "BERT4Rec", "SR-GNN"}
    )
    charts = build_charts(frame, metrics_dir)
    for chart in charts.values():
        chart.to_dict(validate=True)
    rows = table_rows(frame)
    assert len(rows) == 17 and all(len(row) == 10 for row in rows)
    assert "#table(columns: 10" in format_table(rows, "typst")
    assert "| Dataset |" in format_table(rows, "markdown")
    with tempfile.TemporaryDirectory() as temporary:
        directory = Path(temporary)
        source = metrics_dir / "pop_beauty_loo_n50.json"
        record = orjson.loads(source.read_bytes())
        for partition in ("test_metrics", "best_val_metrics"):
            metrics = record[partition]
            removed = metrics.get("len:>50/count", 0)
            metrics["len:>50/count"] = 0
            metrics["len:<=5/count"] += removed
            # Deliberate stale nonzero number: an empty bucket must remain N/A.
            metrics["len:>50/HR@10"] = 1.0
        (directory / source.name).write_bytes(orjson.dumps(record))
        empty = read_metrics(directory)
        values = (
            empty.lazy()
            .filter((pl.col("group") == "len") & (pl.col("bucket") == ">50"))
            .collect()
        )
        assert values["value"].null_count() == values.height
        (directory / "leakage_summary.json").write_bytes(
            orjson.dumps({"complete": True})
        )
        assert read_metrics(directory).height == empty.height
        (directory / "duplicate.json").write_bytes(orjson.dumps(record))
        try:
            read_metrics(directory)
        except ValueError as exc:
            assert "Duplicate" in str(exc)
        else:
            raise AssertionError("Duplicate experiment accepted")
    return {
        "normalized_rows": frame.height,
        "figures": len(charts),
        "missing_cells": len(missing_experiments(frame)),
    }


@app.function
def check_integration() -> str:
    """Exercise the actual Trainer/Evaluator, AMP and checkpoint restoration."""
    if not Path("data/processed/beauty/metadata.json").exists():
        return "pending preprocessing"
    with tempfile.TemporaryDirectory() as temporary:
        config = ExperimentConfig(
            data=DataConfig(
                max_seq_len=5, batch_size=8, eval_batch_size=8, max_prefixes_per_user=1
            ),
            model=ModelConfig(name="srgnn"),
            train=TrainConfig(epochs=1),
            eval=EvalConfig(),
            checkpoints_dir=Path(temporary),
        )
        seed_everything(42)
        bundle = build_databundle(config.data, "srgnn")
        assert bundle.train_loader is not None
        bundle.train_loader = DataLoader(
            Subset(bundle.train_loader.dataset, list(range(16))),
            batch_size=8,
            collate_fn=SessionGraphDataset.collate_fn,
        )
        bundle.val_loader = DataLoader(
            Subset(bundle.val_loader.dataset, list(range(16))),
            batch_size=8,
            collate_fn=bundle.val_loader.collate_fn,
        )
        bundle.test_loader = DataLoader(
            Subset(bundle.test_loader.dataset, list(range(16))),
            batch_size=8,
            collate_fn=bundle.test_loader.collate_fn,
        )
        model = build_model(config.model, bundle.num_items, bundle.max_seq_len)
        trainer = Trainer(config, model, bundle, show_progress=False)
        result = trainer.fit()
        assert result.best_epoch == 1 and len(result.history) == 1
        assert result.test_metrics["num_queries"] == 16
        checkpoint = torch.load(
            trainer.checkpoint_path, weights_only=True, map_location=trainer.device
        )
        for name, value in model.state_dict().items():
            torch.testing.assert_close(value, checkpoint["model"][name])
        evaluator = Evaluator(
            config.eval, bundle.num_items, bundle.item_pop_counts, trainer.device
        )
        batch = move_batch_to_device(next(iter(bundle.test_loader)), trainer.device)
        scores = evaluator.mask_candidates(model.score_items(batch), batch)
        assert scores[:, 0].isneginf().all()
        assert scores.gather(1, batch["seen_ids"]).isneginf().all()
        artifact = result.save(Path(temporary) / "integration.json")
        saved = orjson.loads(artifact.read_bytes())
        assert saved["test_metrics"] == result.test_metrics
    return "passed (real data, Trainer, Evaluator, checkpoint and serialization)"


@app.function
def run_srgnn_sweeps() -> list[str]:
    """Run the prescribed grid through the shared training and serialization entrypoint."""
    completed: list[str] = []
    torch.set_num_threads(4)
    for dataset, windows in (
        ("beauty", (5, 10, 20, 50)),
        ("ml-1m", (10, 20, 50, 100, 200)),
    ):
        for window in windows:
            name = f"srgnn_{dataset}_loo_n{window}"
            cfg = ExperimentConfig(
                data=DataConfig(dataset=dataset, max_seq_len=window),
                model=ModelConfig(name="srgnn"),
                train=TrainConfig(epochs=200),
                exp_name=name,
            )
            path = cfg.metrics_dir / f"{name}.json"
            if path.exists():
                saved = orjson.loads(path.read_bytes())
                expected = orjson.loads(
                    orjson.dumps(dataclasses.asdict(cfg), default=str)
                )
                # Path separators are host-specific; they do not change the experiment.
                for config in (saved["config"], expected):
                    config["data"]["data_dir"] = config["data"]["data_dir"].replace(
                        "\\", "/"
                    )
                    for key in ("metrics_dir", "checkpoints_dir"):
                        config[key] = config[key].replace("\\", "/")
                if saved["config"] != expected or not saved.get("history"):
                    raise ValueError(f"Existing incompatible result: {path}")
                print(f"Reusing {name}", flush=True)
            else:
                print(f"Training {name}", flush=True)
                result = run_experiment(cfg, show_progress=False)
                print(
                    f"Completed {name}: best_epoch={result.best_epoch}, "
                    f"test={result.test_metrics['NDCG@10']:.6f}",
                    flush=True,
                )
            completed.append(name)
    return completed


@app.cell
def _() -> tuple[dict[str, str]]:
    model_checks = check_model()
    print("Model checks:", model_checks, flush=True)
    return (model_checks,)


@app.cell
def _() -> tuple[dict[str, int]]:
    plot_checks = check_plots()
    print("Plot checks:", plot_checks, flush=True)
    return (plot_checks,)


@app.cell
def _() -> tuple[str]:
    integration_check = check_integration()
    print("Integration:", integration_check, flush=True)
    return (integration_check,)


@app.cell
def _() -> None:
    if os.environ.get("SEQREC_TASK3_TRAIN") == "1":
        training_results = run_srgnn_sweeps()
        print(training_results, flush=True)
        plot_all(PlotConfig(strict=True))
    return


if __name__ == "__main__":
    app.run()
