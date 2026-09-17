import os
import re

import torch

from .config import ExperimentConfig
from .experiments import experiment_fingerprint
from .paths import resolve_checkpoint_dir


def _validate_identifier(value, name):
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", value):
        raise ValueError(f"{name} may contain only letters, numbers, '.', '_' and '-'.")


def checkpoint_path(model_id, weight_dir=None, experiment_id=None):
    _validate_identifier(model_id, "model_id")
    identifier = experiment_id or model_id
    _validate_identifier(identifier, "experiment_id")
    return resolve_checkpoint_dir(weight_dir) / f"Cifar-10_{identifier}.pth"


def latest_checkpoint_path(model_id, weight_dir=None):
    _validate_identifier(model_id, "model_id")
    directory = resolve_checkpoint_dir(weight_dir)
    candidates = list(directory.glob(f"Cifar-10_{model_id}-*.pth"))
    legacy_path = checkpoint_path(model_id, weight_dir)
    if legacy_path.exists():
        candidates.append(legacy_path)
    return max(candidates, key=lambda path: path.stat().st_mtime) if candidates else None


def torch_load_compatible(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        # PyTorch 1.x compatibility: weights_only was added in newer versions.
        return torch.load(path, map_location=device)


def load_model_weights(
    model,
    model_id,
    device="cpu",
    weight_dir=None,
    experiment_id=None,
    checkpoint_file=None,
    expected_config=None,
):
    """model_id가 일치하는 checkpoint를 안전하게 불러온다."""
    weight_path = (
        checkpoint_path(model_id, weight_dir, experiment_id)
        if checkpoint_file is None
        else checkpoint_file
    )
    if not os.path.exists(weight_path):
        print(f"No checkpoint found for {model_id}: {weight_path}")
        return None

    checkpoint = torch_load_compatible(weight_path, device)
    if isinstance(checkpoint, dict) and "model_state" in checkpoint:
        saved_model_id = checkpoint.get("model_id")
        if saved_model_id != model_id:
            raise ValueError(
                f"Checkpoint model_id mismatch: expected {model_id!r}, "
                f"found {saved_model_id!r}."
            )
        saved_config = checkpoint.get("experiment_config")
        if expected_config is not None and saved_config is not None:
            restored_config = ExperimentConfig.from_dict(saved_config)
            expected_hash = experiment_fingerprint(expected_config)
            restored_hash = experiment_fingerprint(restored_config)
            if expected_hash != restored_hash:
                raise ValueError(
                    "Checkpoint configuration mismatch: "
                    f"expected {expected_hash}, found {restored_hash}."
                )
        model.load_state_dict(checkpoint["model_state"])
        checkpoint["checkpoint_path"] = str(weight_path)
        print(
            f"Loaded {model_id} checkpoint from {weight_path} "
            f"(best validation accuracy: {checkpoint.get('best_val_accuracy', 0.0):.2f}%)."
        )
        return checkpoint

    model.load_state_dict(checkpoint)
    print(f"Loaded legacy weights from {weight_path}; no training metadata is available.")
    return {
        "model_id": model_id,
        "history": None,
        "legacy": True,
        "checkpoint_path": str(weight_path),
    }
