"""Forward-pass validation for registered models."""

import gc

import torch

from .benchmark import count_parameters
from .registry import MODEL_REGISTRY, create_model


DEFAULT_VALIDATION_MODELS = (
    "perceptron",
    "mlp",
    "alexnet",
    "vgg11",
    "resnet18",
    "mobilenet_v2",
    "efficientnet_b0",
    "convnext_tiny",
    "se_resnet18",
    "vit_tiny",
    "pvt_tiny",
)


def validate_registered_models(model_ids=None, device="cpu", fail_fast=False):
    selected_ids = model_ids or DEFAULT_VALIDATION_MODELS
    selected_device = torch.device(device)
    results = []

    for model_id in selected_ids:
        spec = None
        model = None
        try:
            spec = MODEL_REGISTRY.get(model_id)
            model = create_model(
                model_id,
                num_classes=10,
                image_size=spec.default_image_size,
            ).to(selected_device).eval()
            sample = torch.randn(1, 3, spec.default_image_size, spec.default_image_size)
            sample = sample.to(selected_device)
            with torch.no_grad():
                output = model(sample)
                if isinstance(output, tuple):
                    output = output[0]
            output_shape = tuple(output.shape)
            if output_shape != (1, 10):
                raise ValueError(f"Expected output shape (1, 10), got {output_shape}.")
            results.append(
                {
                    "model_id": model_id,
                    "family": spec.family,
                    "status": "ok",
                    "output_shape": output_shape,
                    "parameter_count": count_parameters(model),
                    "error": None,
                }
            )
        except Exception as error:
            results.append(
                {
                    "model_id": model_id,
                    "family": spec.family if spec is not None else "unknown",
                    "status": "failed",
                    "output_shape": None,
                    "parameter_count": None,
                    "error": str(error),
                }
            )
            if fail_fast:
                break
        finally:
            if model is not None:
                del model
            gc.collect()
            if selected_device.type == "cuda":
                torch.cuda.empty_cache()
    return results
