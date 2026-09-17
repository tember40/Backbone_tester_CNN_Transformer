import os
import platform
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class RuntimeEnvironment:
    device: torch.device
    num_workers: int
    pin_memory: bool

    def __str__(self):
        return (
            f"device={self.device}, num_workers={self.num_workers}, "
            f"pin_memory={self.pin_memory}"
        )


def select_device(preference="auto"):
    """사용 가능한 가속기를 CUDA, Apple MPS, CPU 순서로 선택한다."""
    preference = preference.lower()
    supported = {"auto", "cuda", "mps", "cpu"}
    if preference not in supported:
        raise ValueError(f"device must be one of {sorted(supported)}, got {preference!r}.")

    cuda_available = torch.cuda.is_available()
    mps_available = (
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_built()
        and torch.backends.mps.is_available()
    )

    if preference == "auto":
        if cuda_available:
            return torch.device("cuda")
        if mps_available:
            return torch.device("mps")
        return torch.device("cpu")

    if preference == "cuda" and not cuda_available:
        raise RuntimeError("CUDA was requested, but it is not available in this environment.")
    if preference == "mps" and not mps_available:
        raise RuntimeError("MPS was requested, but it is not available in this environment.")
    return torch.device(preference)


def default_num_workers(requested=None):
    """플랫폼에 맞는 안전한 DataLoader worker 수를 반환한다.

    Windows의 Jupyter 환경은 multiprocessing worker로 멈출 수 있어 기본값을 0으로
    둔다. 다른 환경에서는 CPU 수를 기준으로 최대 4개를 사용한다. 명시적인 값은
    플랫폼과 관계없이 그대로 존중한다.
    """
    if requested is not None:
        if requested < 0:
            raise ValueError("num_workers must be zero or greater.")
        return requested
    if platform.system() == "Windows":
        return 0
    return min(4, os.cpu_count() or 1)


def detect_environment(device="auto", num_workers=None):
    selected_device = select_device(device)
    return RuntimeEnvironment(
        device=selected_device,
        num_workers=default_num_workers(num_workers),
        pin_memory=selected_device.type == "cuda",
    )

