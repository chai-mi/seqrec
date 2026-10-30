from __future__ import annotations

from dataclasses import dataclass, field
from typing import Annotated

import tyro
from rich.console import Console
from rich.table import Table

from seqrec.config import DataConfig, ExperimentConfig, ModelName, PreprocessConfig
from seqrec.data.dataset import build_databundle
from seqrec.data.preprocess import print_metadata_summary, run_preprocess

console = Console()


@dataclass(slots=True)
class PreprocessCmd:
    """Download, 5-core filter, and split MovieLens-1M and Amazon-Beauty datasets."""

    config: tyro.conf.OmitArgPrefixes[PreprocessConfig] = field(default_factory=PreprocessConfig)


@dataclass(slots=True)
class InspectDataCmd:
    """Verify DataLoaders and inspect batch tensor shapes for any model paradigm."""

    data: tyro.conf.OmitArgPrefixes[DataConfig] = field(default_factory=DataConfig)
    model: ModelName = "sasrec"


@dataclass(slots=True)
class TrainCmd:
    """Train and evaluate a sequential recommendation model (Phase 2+)."""

    exp: tyro.conf.OmitArgPrefixes[ExperimentConfig] = field(default_factory=ExperimentConfig)


CliCommand = (
    Annotated[PreprocessCmd, tyro.conf.subcommand(name="preprocess")]
    | Annotated[InspectDataCmd, tyro.conf.subcommand(name="inspect-data")]
    | Annotated[TrainCmd, tyro.conf.subcommand(name="train")]
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

    for split_name, loader in [("val", bundle.val_loader), ("test", bundle.test_loader)]:
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


def main() -> None:
    """Main CLI entry point via tyro."""
    cmd = tyro.cli(CliCommand)
    if isinstance(cmd, PreprocessCmd):
        run_preprocess(cmd.config)
    elif isinstance(cmd, InspectDataCmd):
        run_inspect_data(cmd)
    elif isinstance(cmd, TrainCmd):
        console.print(
            "[yellow]Training engine will be unlocked in Phase 2.[/yellow]",
            cmd.exp,
        )


if __name__ == "__main__":
    main()
