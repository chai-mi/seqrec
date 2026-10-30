"""Training and evaluation engine: unified ``Trainer`` and ``Evaluator``."""

from __future__ import annotations

from seqrec.engine.evaluator import Evaluator, move_batch_to_device
from seqrec.engine.trainer import (
    EpochRecord,
    ExperimentResult,
    Trainer,
    default_exp_name,
    resolve_device,
    run_experiment,
    seed_everything,
)

__all__ = [
    "EpochRecord",
    "Evaluator",
    "ExperimentResult",
    "Trainer",
    "default_exp_name",
    "move_batch_to_device",
    "resolve_device",
    "run_experiment",
    "seed_everything",
]
