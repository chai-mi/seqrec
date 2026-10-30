from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Annotated

import tyro
from rich.console import Console
from rich.table import Table

from seqrec.config import DataConfig, ExperimentConfig, ModelName, PreprocessConfig
from seqrec.data.dataset import build_databundle
from seqrec.data.preprocess import print_metadata_summary, run_preprocess
from seqrec.engine.trainer import ExperimentResult, run_experiment

console = Console()


@dataclass(slots=True)
class PreprocessCmd:
    """Download, 5-core filter, and split MovieLens-1M and Amazon-Beauty datasets."""

    config: tyro.conf.OmitArgPrefixes[PreprocessConfig] = field(
        default_factory=PreprocessConfig
    )


@dataclass(slots=True)
class InspectDataCmd:
    """Verify DataLoaders and inspect batch tensor shapes for any model paradigm."""

    data: tyro.conf.OmitArgPrefixes[DataConfig] = field(default_factory=DataConfig)
    model: ModelName = "sasrec"


@dataclass(slots=True)
class TrainCmd:
    """Train and evaluate a sequential recommendation model."""

    exp: tyro.conf.OmitArgPrefixes[ExperimentConfig] = field(
        default_factory=ExperimentConfig
    )


@dataclass(slots=True)
class SweepCmd:
    """Run the truncation-window N (and seed) sensitivity grid for one model / dataset."""

    exp: tyro.conf.OmitArgPrefixes[ExperimentConfig] = field(
        default_factory=ExperimentConfig
    )
    seq_lens: tuple[int, ...] = (10, 20, 50)
    seeds: tuple[int, ...] = (42,)
    resume: bool = False


CliCommand = (
    Annotated[PreprocessCmd, tyro.conf.subcommand(name="preprocess")]
    | Annotated[InspectDataCmd, tyro.conf.subcommand(name="inspect-data")]
    | Annotated[TrainCmd, tyro.conf.subcommand(name="train")]
    | Annotated[SweepCmd, tyro.conf.subcommand(name="sweep")]
)


def run_inspect_data(cmd: InspectDataCmd) -> None:
    """Build DataBundle and print tensor diagnostics for train, val, and test loaders."""
    bundle = build_databundle(cmd.data, model_name=cmd.model)
    print_metadata_summary([bundle.metadata])

    table = Table(
        title=f"DataLoader Inspection — dataset={cmd.data.dataset}, "
        f"split={cmd.data.split_strategy}, model={cmd.model}, N={bundle.max_seq_len}",
        show_lines=True,
    )
    table.add_column("Split", style="bold green")
    table.add_column("#Batches", justify="right")
    table.add_column("Key", style="cyan")
    table.add_column("Shape", justify="right")
    table.add_column("Dtype", justify="left")
    table.add_column("Min / Max", justify="right")

    if bundle.train_loader is not None:
        train_batch = next(iter(bundle.train_loader))
        for idx, (k, tensor) in enumerate(train_batch.items()):
            table.add_row(
                "train" if idx == 0 else "",
                str(len(bundle.train_loader)) if idx == 0 else "",
                str(k),
                str(tuple(tensor.shape)),
                str(tensor.dtype).replace("torch.", ""),
                f"{tensor.min().item():.2f} / {tensor.max().item():.2f}"
                if tensor.is_floating_point()
                else f"{int(tensor.min().item())} / {int(tensor.max().item())}",
            )

    for split_name, loader in [
        ("val", bundle.val_loader),
        ("test", bundle.test_loader),
    ]:
        batch = next(iter(loader))
        for idx, (k, tensor) in enumerate(batch.items()):
            table.add_row(
                split_name if idx == 0 else "",
                str(len(loader)) if idx == 0 else "",
                str(k),
                str(tuple(tensor.shape)),
                str(tensor.dtype).replace("torch.", ""),
                f"{int(tensor.min().item())} / {int(tensor.max().item())}",
            )

    console.print(table)


def print_experiment_summary(result: ExperimentResult) -> None:
    """Print the headline test metrics of a finished experiment as a Rich table."""
    table = Table(
        title=f"{result.exp_name} — {result.device}, {result.num_parameters:,} params, "
        f"best epoch {result.best_epoch}",
        show_lines=False,
    )
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right")
    for key, value in result.test_metrics.items():
        if key.startswith(("len:", "pop:", "sampled_")):
            continue
        table.add_row(key, f"{value:.4f}" if key != "num_queries" else f"{int(value)}")
    console.print(table)


def run_train(cmd: TrainCmd) -> ExperimentResult:
    """Train + evaluate one configuration and persist its metrics JSON."""
    result = run_experiment(cmd.exp)
    print_experiment_summary(result)
    console.print(
        f"[green]metrics saved to[/green] {result.config.metrics_dir / (result.exp_name + '.json')}"
    )
    return result


def sweep_exp_name(
    base: ExperimentConfig,
    seq_len: int,
    seed: int,
    multi_seed: bool,
) -> str:
    """Unique experiment name for one sweep cell: ``<base>_n{N}[_s{seed}]``.

    A user supplied ``--exp-name`` is treated as a *prefix* so that every window (and every
    seed) gets its own metrics JSON and checkpoint instead of overwriting/skipping each other.
    """
    prefix = base.exp_name or (
        f"{base.model.name}_{base.data.dataset}_{base.data.split_strategy}"
    )
    name = f"{prefix}_n{seq_len}"
    return f"{name}_s{seed}" if multi_seed else name


def run_sweep(cmd: SweepCmd) -> list[ExperimentResult]:
    """Run the ``N`` (truncation window) / seed sensitivity grid sequentially."""
    results: list[ExperimentResult] = []
    for seq_len in cmd.seq_lens:
        for seed in cmd.seeds:
            exp = copy.deepcopy(cmd.exp)
            exp.data.max_seq_len = seq_len
            exp.train.seed = seed
            exp.exp_name = sweep_exp_name(
                cmd.exp, seq_len, seed, multi_seed=len(cmd.seeds) > 1
            )
            target = exp.metrics_dir / f"{exp.exp_name}.json"
            if cmd.resume and target.exists():
                console.print(f"[yellow]skip existing[/yellow] {exp.exp_name}")
                continue
            console.print(f"[bold green]running[/bold green] {exp.exp_name}")
            results.append(run_experiment(exp))

    if results:
        table = Table(
            title=f"Sweep summary — {cmd.exp.model.name} / {cmd.exp.data.dataset}"
        )
        table.add_column("exp_name", style="cyan")
        table.add_column("N", justify="right")
        table.add_column("HR@10", justify="right")
        table.add_column("NDCG@10", justify="right")
        table.add_column("MRR", justify="right")
        for result in results:
            table.add_row(
                result.exp_name,
                str(result.config.data.resolved_max_seq_len),
                f"{result.test_metrics.get('HR@10', float('nan')):.4f}",
                f"{result.test_metrics.get('NDCG@10', float('nan')):.4f}",
                f"{result.test_metrics.get('MRR', float('nan')):.4f}",
            )
        console.print(table)
    return results


def main() -> None:
    """Main CLI entry point via tyro."""
    cmd = tyro.cli(CliCommand)
    if isinstance(cmd, PreprocessCmd):
        run_preprocess(cmd.config)
    elif isinstance(cmd, InspectDataCmd):
        run_inspect_data(cmd)
    elif isinstance(cmd, TrainCmd):
        run_train(cmd)
    elif isinstance(cmd, SweepCmd):
        run_sweep(cmd)


if __name__ == "__main__":
    main()
