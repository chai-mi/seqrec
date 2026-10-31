"""SR-GNN correctness checks, real-data sweeps, and vector figure regeneration.

uv run python notebooks/srgnn_and_plots.py
SEQREC_TASK3_TRAIN=1 uv run python notebooks/srgnn_and_plots.py
SEQREC_TASK3_AUDIT=1 uv run python notebooks/srgnn_and_plots.py

Training is opt-in; it uses the public Trainer unchanged, seed 42, 200 epochs
maximum, patience 10 and the prescribed nine windows. Completed compatible
runs are retained. Checkpoints and processed data remain untracked.
"""

import marimo

__generated_with = "0.25.2"
app = marimo.App(width="full")

with app.setup:
    import dataclasses
    import hashlib
    import math
    import os
    import platform
    import subprocess
    import tempfile
    from functools import partial
    from pathlib import Path
    from time import perf_counter
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
        ExperimentResult,
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
    for steps, loss_type, shared in (
        (0, "ce", True),
        (1, "bce", True),
        (1, "ce", False),
    ):
        try:
            SRGNN(
                10, ggnn_steps=steps, loss_type=loss_type, share_item_embeddings=shared
            )
        except ValueError:
            pass
        else:
            raise AssertionError("Unsupported SR-GNN configuration accepted")
    try:
        model.score_items(
            example_eval(torch.zeros(1, 5, dtype=torch.long), torch.tensor([0]))
        )
    except ValueError:
        pass
    else:
        raise AssertionError("Empty history was accepted")
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
    with tempfile.TemporaryDirectory(dir=Path.cwd().parent) as temporary:
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
    with tempfile.TemporaryDirectory(dir=Path.cwd().parent) as temporary:
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
def collate_cuda_graphs(
    samples: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]],
    device: str = "cuda",
) -> SessionGraphTrainBatch:
    """Call the existing graph builder on CUDA before the unchanged Trainer.

    This collator requires num_workers=0. It transfers O(BN) IDs rather than
    two O(BN²) dense adjacency tensors; no graph algorithm is replaced.
    """
    user_id = torch.stack([row[0] for row in samples]).to(device)
    input_ids = torch.stack([row[1] for row in samples]).to(device)
    target_id = torch.stack([row[2] for row in samples]).to(device)
    seq_len = torch.stack([row[3] for row in samples]).to(device)
    items, aliases, adj_in, adj_out = build_session_graph_batch(input_ids, seq_len)
    return {
        "user_id": user_id,
        "input_ids": input_ids,
        "target_id": target_id,
        "seq_len": seq_len,
        "graph_items": items,
        "alias_inputs": aliases,
        "adj_in": adj_in,
        "adj_out": adj_out,
    }


@app.function
def check_cuda_collation() -> dict[str, str | float]:
    """Verify exact batches for every unique-history length 1..200 and unchanged RNG."""
    if not torch.cuda.is_available():
        return {"status": "CUDA unavailable"}
    torch.set_num_threads(4)
    seed_everything(42)
    width = 200
    ids = torch.arange(1, width + 1).expand(width, -1).clone()
    lengths = torch.arange(1, width + 1)
    ids[torch.arange(width).unsqueeze(0) >= lengths.unsqueeze(1)] = 0
    samples = [
        (torch.tensor(index + 1), ids[index], torch.tensor(201), lengths[index])
        for index in range(width)
    ]
    expected = SessionGraphDataset.collate_fn(samples)
    cpu_rng, cuda_rng = torch.get_rng_state(), torch.cuda.get_rng_state()
    actual = collate_cuda_graphs(samples)
    for key, value in cast(dict[str, torch.Tensor], expected).items():
        assert torch.equal(value.cuda(), cast(dict[str, torch.Tensor], actual)[key]), (
            key
        )
    assert torch.equal(cpu_rng, torch.get_rng_state())
    assert torch.equal(cuda_rng, torch.cuda.get_rng_state())
    # This includes adjacency transfer, which the GPU collator avoids.
    torch.cuda.synchronize()
    start = perf_counter()
    for _ in range(8):
        move_batch_to_device(
            cast(dict[str, torch.Tensor], SessionGraphDataset.collate_fn(samples)),
            torch.device("cuda"),
        )
    torch.cuda.synchronize()
    cpu_ms = (perf_counter() - start) * 1000 / 8
    start = perf_counter()
    for _ in range(8):
        collate_cuda_graphs(samples)
    torch.cuda.synchronize()
    gpu_ms = (perf_counter() - start) * 1000 / 8
    return {
        "status": "exact batches; unchanged CPU/CUDA RNG",
        "cpu_graph_and_transfer_ms": cpu_ms,
        "cuda_graph_ms": gpu_ms,
    }


@app.function
def run_cuda_graph_experiment(config: ExperimentConfig) -> ExperimentResult:
    """Use identical public data/model/Trainer logic with device-local graph collation."""
    if not (
        config.train.device.startswith("cuda")
        and torch.cuda.is_available()
        and config.data.num_workers == 0
    ):
        return run_experiment(config, show_progress=False)
    seed_everything(config.train.seed)
    bundle = build_databundle(config.data, model_name=config.model.name)
    assert bundle.train_loader is not None
    rng = torch.get_rng_state()
    bundle.train_loader = DataLoader(
        bundle.train_loader.dataset,
        batch_size=config.data.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=partial(collate_cuda_graphs, device=config.train.device),
    )
    assert torch.equal(rng, torch.get_rng_state())
    model = build_model(
        config.model, bundle.num_items, bundle.max_seq_len, bundle.item_pop_counts
    )
    result = Trainer(config, model, bundle, show_progress=False).fit()
    result.save(config.metrics_dir / f"{result.exp_name}.json")
    return result


@app.function
def training_sources() -> tuple[Path, ...]:
    """Enumerate every source file fingerprinted by the SR-GNN runs."""
    return tuple(
        Path(name)
        for name in (
            "src/seqrec/models/srgnn.py",
            "src/seqrec/models/__init__.py",
            "src/seqrec/config.py",
            "src/seqrec/engine/trainer.py",
            "src/seqrec/engine/evaluator.py",
            "src/seqrec/data/dataset.py",
            "src/seqrec/data/preprocess.py",
            "uv.lock",
        )
    )


@app.function
def newline_source_hashes(raw: bytes) -> set[str]:
    """Recognize original, LF and CRLF bytes without accepting code changes."""
    normalized = raw.replace(b"\r\n", b"\n")
    return {
        hashlib.sha256(value).hexdigest()
        for value in (
            raw,
            normalized,
            normalized.replace(b"\n", b"\r\n"),
        )
    }


@app.function
def write_training_snapshot(metrics_dir: Path = Path("reports/metrics")) -> Path:
    """Preserve original source bytes so CRLF/LF checkouts remain auditable.

    The historical SHA-256 is never changed. Only newline differences are
    accepted when replaying on a host whose Git checkout uses LF instead.
    """
    path = metrics_dir / "srgnn_training_source_snapshot.json"
    workflow_path = Path("notebooks/srgnn_and_plots.py")
    workflow_raw = workflow_path.read_bytes()
    workflow_sha = hashlib.sha256(workflow_raw).hexdigest()
    if path.exists():
        snapshot = orjson.loads(path.read_bytes())
        assert set(snapshot["sources"]) == {p.as_posix() for p in training_sources()}
        for source in training_sources():
            saved = snapshot["sources"][source.as_posix()]
            original = saved["text"].encode("utf-8")
            assert hashlib.sha256(original).hexdigest() == saved["sha256"]
            assert original.replace(b"\r\n", b"\n") == source.read_bytes().replace(
                b"\r\n", b"\n"
            ), source
        workflows = snapshot.setdefault("workflows", {})
        for sha, entry in workflows.items():
            assert hashlib.sha256(entry["text"].encode("utf-8")).hexdigest() == sha
        if workflow_sha not in workflows:
            workflows[workflow_sha] = {
                "path": workflow_path.as_posix(),
                "text": workflow_raw.decode("utf-8"),
            }
            path.write_bytes(orjson.dumps(snapshot, option=orjson.OPT_INDENT_2))
        return path
    snapshot = {
        "kind": "training_source_snapshot",
        "sources": {},
        "workflows": {
            workflow_sha: {
                "path": workflow_path.as_posix(),
                "text": workflow_raw.decode("utf-8"),
            },
        },
    }
    for source in training_sources():
        raw = source.read_bytes()
        snapshot["sources"][source.as_posix()] = {
            "sha256": hashlib.sha256(raw).hexdigest(),
            "text": raw.decode("utf-8"),
        }
    metrics_dir.mkdir(parents=True, exist_ok=True)
    path.write_bytes(orjson.dumps(snapshot, option=orjson.OPT_INDENT_2))
    return path


@app.function
def check_source_snapshot() -> dict[str, int]:
    """Prove newline portability and rejection of tampered source snapshots."""
    with tempfile.TemporaryDirectory(dir=Path.cwd().parent) as temporary:
        directory = Path(temporary)
        path = write_training_snapshot(directory)
        snapshot = orjson.loads(path.read_bytes())
        assert len(snapshot["sources"]) == 8
        first = next(iter(snapshot["sources"].values()))
        original = first["text"]
        for line_ending in ("\n", "\r\n"):
            first["text"] = original.replace("\r\n", "\n").replace("\n", line_ending)
            first["sha256"] = hashlib.sha256(first["text"].encode()).hexdigest()
            assert first["sha256"] in newline_source_hashes(original.encode())
            path.write_bytes(orjson.dumps(snapshot))
            write_training_snapshot(directory)
        for alter_hash in (False, True):
            first["text"] = original + "\n# deliberately changed source\n"
            first["sha256"] = (
                hashlib.sha256(first["text"].encode()).hexdigest()
                if alter_hash
                else "0" * 64
            )
            path.write_bytes(orjson.dumps(snapshot))
            try:
                write_training_snapshot(directory)
            except AssertionError:
                pass
            else:
                raise AssertionError("Tampered source snapshot accepted")
    return {"sources": 8, "newline_variants": 2, "tamper_cases": 2}


@app.function
def path_json_default(value: object) -> str:
    """Persist relative paths portably across Windows and POSIX hosts."""
    if isinstance(value, Path):
        return value.as_posix()
    raise TypeError(f"Unsupported JSON value: {type(value).__name__}")


@app.function
def run_srgnn_sweeps() -> list[str]:
    """Run all nine real experiments, preserving source and runtime provenance."""
    completed: list[str] = []
    torch.set_num_threads(4)
    snapshot = orjson.loads(write_training_snapshot().read_bytes())
    workflow_raw = Path("notebooks/srgnn_and_plots.py").read_bytes()
    workflow_sha = hashlib.sha256(workflow_raw).hexdigest()
    for dataset, windows in (
        ("beauty", (5, 10, 20, 50)),
        ("ml-1m", (10, 20, 50, 100, 200)),
    ):
        for window in windows:
            cfg = ExperimentConfig(
                data=DataConfig(dataset=dataset, max_seq_len=window),
                model=ModelConfig(name="srgnn"),
                train=TrainConfig(epochs=200),
            )
            name = f"srgnn_{dataset}_loo_n{window}"
            path = cfg.metrics_dir / f"{name}.json"
            sources = training_sources()
            source_hashes = {
                p.as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in sources
            }
            data_sources = [
                cfg.data.dataset_dir / name
                for name in (
                    "metadata.json",
                    "loo_train.parquet",
                    "loo_val.parquet",
                    "loo_test.parquet",
                    "loo_item_pop.parquet",
                )
            ]
            data_hashes = {
                p.as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in data_sources
            }
            if path.exists():
                saved = orjson.loads(path.read_bytes())
                expected = orjson.loads(
                    orjson.dumps(dataclasses.asdict(cfg), default=path_json_default)
                )
                provenance = saved.get("provenance", {})
                cached_sources = provenance.get("source_sha256", {})
                source_compatible = set(cached_sources) == set(source_hashes) and all(
                    cached_sources[source]
                    in newline_source_hashes(
                        snapshot["sources"][source]["text"].encode("utf-8")
                    )
                    for source in source_hashes
                )
                if (
                    saved.get("config") != expected
                    or not saved.get("history")
                    or not source_compatible
                    or provenance.get("data_sha256") != data_hashes
                    or (
                        provenance.get("workflow_sha256") is not None
                        and provenance["workflow_sha256"]
                        not in newline_source_hashes(workflow_raw)
                    )
                ):
                    raise ValueError(f"Existing incompatible result: {path}")
                completed.append(name)
                continue
            print(f"Training {name}", flush=True)
            result = run_cuda_graph_experiment(cfg)
            payload = result.to_dict()
            try:
                revision = subprocess.run(
                    ["git", "rev-parse", "HEAD"],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                base_commit = (
                    revision.stdout.strip() if revision.returncode == 0 else None
                )
            except FileNotFoundError:
                base_commit = None
            payload["provenance"] = {
                "base_commit": base_commit,
                "source_sha256": source_hashes,
                "data_sha256": data_hashes,
                "workflow_sha256": workflow_sha,
                "training_graph_device": result.device
                if result.device.startswith("cuda") and cfg.data.num_workers == 0
                else "cpu",
                "python": platform.python_version(),
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "gpu": torch.cuda.get_device_name()
                if torch.cuda.is_available()
                else None,
                "tie_break": "score descending, item ID ascending",
                "negative_sampling": "100 draws with replacement; duplicates count separately",
                "selection": "validation Full-Ranking NDCG@10; standard Trainer early stopping",
            }
            path.write_bytes(
                orjson.dumps(
                    payload, option=orjson.OPT_INDENT_2, default=path_json_default
                )
            )
            print(
                f"Completed {name}: best_epoch={result.best_epoch}, test={result.test_metrics['NDCG@10']:.6f}",
                flush=True,
            )
            completed.append(name)
    return completed


@app.function
def audit_srgnn_results() -> dict[str, int | float]:
    """Re-evaluate every saved best checkpoint and audit artifact provenance."""
    torch.set_num_threads(4)
    paths = sorted(Path("reports/metrics").glob("srgnn_*_loo_n*.json"))
    assert len(paths) == 9, f"Expected nine sweep artifacts, found {len(paths)}"
    snapshot = orjson.loads(
        Path("reports/metrics/srgnn_training_source_snapshot.json").read_bytes()
    )
    largest_drift = 0.0
    for path in paths:
        record = orjson.loads(path.read_bytes())
        workflow_sha = record["provenance"].get("workflow_sha256")
        if workflow_sha is not None:
            workflow = snapshot["workflows"][workflow_sha]["text"].encode("utf-8")
            assert hashlib.sha256(workflow).hexdigest() == workflow_sha
        data = record["config"]["data"]
        data["data_dir"] = Path(data["data_dir"].replace("\\", "/"))
        config = ExperimentConfig(
            data=DataConfig(**data),
            model=ModelConfig(**record["config"]["model"]),
            train=TrainConfig(**record["config"]["train"]),
            eval=EvalConfig(**record["config"]["eval"]),
            checkpoints_dir=Path(
                record["config"]["checkpoints_dir"].replace("\\", "/")
            ),
        )
        for kind in ("source_sha256", "data_sha256"):
            for source, expected in record["provenance"][kind].items():
                raw = Path(source).read_bytes()
                if kind == "source_sha256":
                    original = snapshot["sources"][source]["text"].encode("utf-8")
                    assert expected in newline_source_hashes(original), source
                    assert original.replace(b"\r\n", b"\n") == raw.replace(
                        b"\r\n", b"\n"
                    ), source
                else:
                    assert hashlib.sha256(raw).hexdigest() == expected, source
        history = record["history"]
        assert all(math.isfinite(row["train_loss"]) for row in history)
        best = max(history, key=lambda row: row["val_primary"])
        assert best["epoch"] == record["best_epoch"]
        assert best["val_metrics"] == record["best_val_metrics"]
        assert len(history) == 200 or len(history) - record["best_epoch"] == 10
        checkpoint = torch.load(
            config.checkpoints_dir / (record["exp_name"] + ".pt"),
            map_location="cpu",
            weights_only=True,
        )
        assert checkpoint["epoch"] == record["best_epoch"]
        assert checkpoint["val_metrics"] == record["best_val_metrics"]
        # Evaluation-only bundle: no need to allocate expanded training prefixes.
        bundle = build_databundle(config.data, model_name="pop")
        model = build_model(config.model, bundle.num_items, bundle.max_seq_len).to(
            config.train.device
        )
        model.load_state_dict(checkpoint["model"])
        evaluator = Evaluator(
            config.eval,
            bundle.num_items,
            bundle.item_pop_counts,
            torch.device(config.train.device),
        )
        for partition, loader in (
            ("test_metrics", bundle.test_loader),
            ("best_val_metrics", bundle.val_loader),
        ):
            actual = evaluator.evaluate(model, loader, partition, show_progress=False)
            for key, expected in record[partition].items():
                # Sampled metrics consume training RNG state; full-ranking replay
                # is deterministic and is the protocol used for model selection.
                if key.startswith("sampled_"):
                    continue
                drift = abs(actual[key] - expected)
                largest_drift = max(largest_drift, drift)
                assert math.isclose(
                    actual[key], expected, rel_tol=1e-7, abs_tol=1e-8
                ), (path, key, actual[key], expected)
        print(f"Audited {path.name}", flush=True)
    return {"experiments": len(paths), "largest_full_ranking_drift": largest_drift}


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
def _() -> tuple[dict[str, int]]:
    source_checks = check_source_snapshot()
    print("Source snapshot checks:", source_checks, flush=True)
    return (source_checks,)


@app.cell
def _() -> tuple[dict[str, str | float]]:
    cuda_graph_checks = check_cuda_collation()
    print("CUDA graph collation:", cuda_graph_checks, flush=True)
    return (cuda_graph_checks,)


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


@app.cell
def _() -> None:
    if os.environ.get("SEQREC_TASK3_AUDIT") == "1":
        write_training_snapshot()
        artifact_audit = audit_srgnn_results()
        print("Artifact audit:", artifact_audit, flush=True)
    return


if __name__ == "__main__":
    app.run()
