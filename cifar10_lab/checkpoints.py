import os
import re

import torch


def checkpoint_path(model_id, weight_dir="weight"):
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", model_id):
        raise ValueError("model_id may contain only letters, numbers, '.', '_' and '-'.")
    return os.path.join(weight_dir, f"Cifar-10_{model_id}.pth")


def torch_load_compatible(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        # PyTorch 1.x compatibility: weights_only was added in newer versions.
        return torch.load(path, map_location=device)


def load_model_weights(model, model_id, device="cpu", weight_dir="weight"):
    """model_id가 일치하는 checkpoint를 안전하게 불러온다."""
    weight_path = checkpoint_path(model_id, weight_dir)
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
        model.load_state_dict(checkpoint["model_state"])
        print(
            f"Loaded {model_id} checkpoint from {weight_path} "
            f"(best validation accuracy: {checkpoint.get('best_val_accuracy', 0.0):.2f}%)."
        )
        return checkpoint

    model.load_state_dict(checkpoint)
    print(f"Loaded legacy weights from {weight_path}; no training metadata is available.")
    return {"model_id": model_id, "history": None, "legacy": True}

