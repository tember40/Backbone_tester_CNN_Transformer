"""Backward-compatible imports for older notebooks.

New code should import from ``cifar10_lab`` modules directly.
"""

from cifar10_lab.checkpoints import load_model_weights
from cifar10_lab.data import load_cifar10_data
from cifar10_lab.engine import evaluate_model, evaluate_model_detailed, train_model
from cifar10_lab.visualization import (
    plot_test_results,
    plot_training_history,
    visualize_data_overview,
)

__all__ = [
    "evaluate_model",
    "evaluate_model_detailed",
    "load_cifar10_data",
    "load_model_weights",
    "plot_test_results",
    "plot_training_history",
    "train_model",
    "visualize_data_overview",
]
