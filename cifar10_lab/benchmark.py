"""Small benchmarking helpers shared by training and comparison commands."""

import time

import torch


def count_parameters(model):
    return sum(parameter.numel() for parameter in model.parameters())


def checkpoint_size_mb(path):
    return path.stat().st_size / (1024 * 1024) if path and path.exists() else 0.0


def measure_inference_ms_per_image(model, dataloader, device, max_batches=5):
    model.eval()
    elapsed = 0.0
    image_count = 0
    with torch.no_grad():
        for batch_index, (images, _) in enumerate(dataloader):
            if batch_index > max_batches:
                break
            images = images.to(device, non_blocking=True)
            if batch_index == 0:
                model(images)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                continue
            if device.type == "cuda":
                torch.cuda.synchronize()
            started_at = time.perf_counter()
            model(images)
            if device.type == "cuda":
                torch.cuda.synchronize()
            elapsed += time.perf_counter() - started_at
            image_count += len(images)
    return elapsed * 1000.0 / image_count if image_count else 0.0
