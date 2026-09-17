"""Reusable training utilities for the CIFAR-10 backbone education project."""

from .checkpoints import load_model_weights
from .config import DataConfig, ExperimentConfig, TrainConfig
from .data import (
    CIFAR10_CLASSES,
    CIFAR10_MEAN,
    CIFAR10_STD,
    download_cifar10,
    load_cifar10_data,
)
from .engine import evaluate_model, evaluate_model_detailed, train_model
from .environment import (
    RuntimeEnvironment,
    default_num_workers,
    detect_environment,
    select_device,
)
from .paths import LabPaths, get_lab_paths, resolve_checkpoint_dir, resolve_data_dir
from .registry import MODEL_REGISTRY, create_model, format_model_catalog, list_models

__version__ = "0.4.0"

__all__ = [
    "CIFAR10_CLASSES",
    "CIFAR10_MEAN",
    "CIFAR10_STD",
    "DataConfig",
    "ExperimentConfig",
    "LabPaths",
    "MODEL_REGISTRY",
    "RuntimeEnvironment",
    "TrainConfig",
    "create_model",
    "default_num_workers",
    "detect_environment",
    "download_cifar10",
    "evaluate_model",
    "evaluate_model_detailed",
    "format_model_catalog",
    "get_lab_paths",
    "list_models",
    "load_cifar10_data",
    "load_model_weights",
    "resolve_checkpoint_dir",
    "resolve_data_dir",
    "select_device",
    "train_model",
]
