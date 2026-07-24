"""Training callbacks: the extension point the ``Trainer`` core stays free of.

One module per concern, mirroring ``losses/`` and ``datasets/``: ``base`` is the
``Callback`` contract + ``EpochLogs``; ``console`` logs progress; ``mlflow`` tracks
to an MLflow run; ``async_validation`` scores checkpoints off the loop. The trainer
imports only from here, never a concrete callback module, and never mlflow.
"""

from stemify.training.callbacks.async_validation import AsyncValidationCallback
from stemify.training.callbacks.base import Callback, EpochLogs, Metrics, StateDict
from stemify.training.callbacks.console import ConsoleCallback
from stemify.training.callbacks.mlflow import MLflowCallback

__all__ = [
    "AsyncValidationCallback",
    "Callback",
    "ConsoleCallback",
    "EpochLogs",
    "MLflowCallback",
    "Metrics",
    "StateDict",
]
