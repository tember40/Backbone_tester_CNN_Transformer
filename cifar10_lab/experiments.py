"""Experiment identity and reproducibility helpers."""

import hashlib
import json
import random
from copy import deepcopy

import torch


def set_global_seed(seed, deterministic=False):
    """Seed model initialization and training, optionally preferring deterministic kernels."""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.use_deterministic_algorithms(deterministic, warn_only=True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = not deterministic
        torch.backends.cudnn.deterministic = deterministic


def experiment_payload(config):
    """Return the settings that define checkpoint compatibility.

    The target epoch count is intentionally excluded so the same run can be resumed
    from a shorter training horizon to a longer one.
    """
    payload = (
        config.to_dict()
        if hasattr(config, "to_dict")
        else deepcopy(dict(config))
    )
    payload.pop("device", None)
    payload.pop("weight_dir", None)
    data = payload.get("data", {})
    data.pop("data_root", None)
    data.pop("num_workers", None)
    train = payload.get("train", {})
    train.pop("epochs", None)
    return payload


def experiment_fingerprint(config, length=10):
    serialized = json.dumps(
        experiment_payload(config),
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:length]


def experiment_id(config):
    return f"{config.model_id}-{experiment_fingerprint(config)}"
