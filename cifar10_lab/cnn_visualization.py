"""Visual explanations of convolution, pooling, and receptive fields."""

from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as functional

from .data import CIFAR10_MEAN, CIFAR10_STD


def trace_single_channel_convolution(image, kernel, stride=1, padding=0):
    """Apply one 2D filter and retain every patch and element-wise product."""
    image_tensor = torch.as_tensor(image, dtype=torch.float32)
    kernel_tensor = torch.as_tensor(kernel, dtype=torch.float32)
    if image_tensor.ndim != 2 or kernel_tensor.ndim != 2:
        raise ValueError("image and kernel must both be two-dimensional.")
    if stride < 1:
        raise ValueError("stride must be at least 1.")
    if padding < 0:
        raise ValueError("padding cannot be negative.")

    padded = functional.pad(
        image_tensor,
        (padding, padding, padding, padding),
    )
    kernel_height, kernel_width = kernel_tensor.shape
    output_height = (padded.shape[0] - kernel_height) // stride + 1
    output_width = (padded.shape[1] - kernel_width) // stride + 1
    if output_height < 1 or output_width < 1:
        raise ValueError("kernel cannot be larger than the padded image.")

    output = torch.empty((output_height, output_width), dtype=torch.float32)
    steps = []
    for output_row in range(output_height):
        for output_column in range(output_width):
            image_row = output_row * stride
            image_column = output_column * stride
            patch = padded[
                image_row : image_row + kernel_height,
                image_column : image_column + kernel_width,
            ]
            products = patch * kernel_tensor
            value = products.sum()
            output[output_row, output_column] = value
            steps.append(
                {
                    "output_position": (output_row, output_column),
                    "image_position": (image_row, image_column),
                    "patch": patch.clone(),
                    "products": products.clone(),
                    "sum": float(value.item()),
                }
            )
    return output, steps


def summarize_activations(features):
    """Return compact statistics for a feature-map tensor before or after activation."""
    feature_tensor = torch.as_tensor(features, dtype=torch.float32)
    if feature_tensor.ndim not in (3, 4):
        raise ValueError("features must have shape [C, H, W] or [N, C, H, W].")
    return {
        "shape": tuple(feature_tensor.shape),
        "minimum": float(feature_tensor.min().item()),
        "maximum": float(feature_tensor.max().item()),
        "mean": float(feature_tensor.mean().item()),
        "positive_ratio": float((feature_tensor > 0).float().mean().item()),
        "zero_ratio": float((feature_tensor == 0).float().mean().item()),
    }


def trace_pooling(feature_map, kernel_size=2, stride=None, mode="max"):
    """Pool one 2D feature map and retain the window used for every output value."""
    feature_tensor = torch.as_tensor(feature_map, dtype=torch.float32)
    if feature_tensor.ndim != 2:
        raise ValueError("feature_map must be two-dimensional.")
    if kernel_size < 1:
        raise ValueError("kernel_size must be at least 1.")
    stride = kernel_size if stride is None else stride
    if stride < 1:
        raise ValueError("stride must be at least 1.")
    if mode not in {"max", "average"}:
        raise ValueError("mode must be 'max' or 'average'.")

    output_height = (feature_tensor.shape[0] - kernel_size) // stride + 1
    output_width = (feature_tensor.shape[1] - kernel_size) // stride + 1
    if output_height < 1 or output_width < 1:
        raise ValueError("kernel_size cannot be larger than the feature map.")

    output = torch.empty((output_height, output_width), dtype=torch.float32)
    steps = []
    for output_row in range(output_height):
        for output_column in range(output_width):
            input_row = output_row * stride
            input_column = output_column * stride
            window = feature_tensor[
                input_row : input_row + kernel_size,
                input_column : input_column + kernel_size,
            ]
            value = window.max() if mode == "max" else window.mean()
            output[output_row, output_column] = value
            steps.append(
                {
                    "output_position": (output_row, output_column),
                    "input_position": (input_row, input_column),
                    "window": window.clone(),
                    "value": float(value.item()),
                }
            )
    return output, steps


def collect_feature_maps(model, image, max_layers=6):
    """Capture early Conv2d and pooling outputs from one forward pass."""
    records = []
    handles = []

    def capture(name, module):
        def hook(_, __, output):
            if len(records) >= max_layers or not isinstance(output, torch.Tensor):
                return
            records.append(
                {
                    "name": name,
                    "type": module.__class__.__name__,
                    "shape": tuple(output.shape),
                    "features": output[:1].detach().cpu(),
                }
            )

        return hook

    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.MaxPool2d, nn.AvgPool2d)):
            handles.append(module.register_forward_hook(capture(name, module)))

    try:
        model.eval()
        with torch.no_grad():
            model(image)
    finally:
        for handle in handles:
            handle.remove()

    if not records:
        raise ValueError("No convolution or pooling layers were found in this model.")
    return records


def plot_feature_map_progression(records, max_channels=4, show=True):
    """Display representative channels from each captured CNN stage."""
    row_count = len(records)
    figure, axes = plt.subplots(
        row_count,
        max_channels,
        figsize=(2.7 * max_channels, 2.35 * row_count),
        squeeze=False,
    )
    for row, record in enumerate(records):
        features = record["features"][0]
        channel_count = min(max_channels, features.shape[0])
        for column in range(max_channels):
            axis = axes[row, column]
            axis.axis("off")
            if column >= channel_count:
                continue
            feature = features[column]
            axis.imshow(feature.numpy(), cmap="viridis")
            axis.set_title(f"channel {column}", fontsize=9)
        axes[row, 0].text(
            -0.10,
            0.5,
            f"{record['name']}\n{record['type']}\n{record['shape'][2:]} ",
            transform=axes[row, 0].transAxes,
            ha="right",
            va="center",
            fontsize=9,
            clip_on=False,
        )
    figure.suptitle("How early CNN layers transform one image into feature maps")
    figure.tight_layout(rect=(0.08, 0, 1, 0.97))
    if show:
        plt.show()
    return figure


def _denormalize(image):
    mean = torch.tensor(CIFAR10_MEAN).view(3, 1, 1)
    std = torch.tensor(CIFAR10_STD).view(3, 1, 1)
    return (image.detach().cpu() * std + mean).clamp(0, 1)


def plot_pooling_comparison(image, show=True):
    """Compare max and average pooling while keeping the display size readable."""
    image = _denormalize(image[0] if image.ndim == 4 else image)
    batch = image.unsqueeze(0)
    max_pooled = functional.max_pool2d(batch, kernel_size=2, stride=2)
    average_pooled = functional.avg_pool2d(batch, kernel_size=2, stride=2)
    panels = (
        (image, "Original 32×32"),
        (max_pooled[0], "Max pooling 16×16"),
        (average_pooled[0], "Average pooling 16×16"),
    )
    figure, axes = plt.subplots(1, 3, figsize=(10, 3.4))
    for axis, (panel, title) in zip(axes, panels):
        axis.imshow(panel.permute(1, 2, 0).numpy(), interpolation="nearest")
        axis.set_title(title)
        axis.axis("off")
    figure.suptitle("Pooling reduces spatial resolution using different summaries")
    figure.tight_layout()
    if show:
        plt.show()
    return figure


def receptive_field_stages(model):
    """Calculate theoretical receptive field and sampling jump for sequential CNN stages."""
    receptive_field = 1
    jump = 1
    stages = []
    for name, module in model.named_modules():
        if not isinstance(module, (nn.Conv2d, nn.MaxPool2d, nn.AvgPool2d)):
            continue
        kernel = module.kernel_size
        stride = module.stride or module.kernel_size
        dilation = getattr(module, "dilation", 1)
        kernel = kernel[0] if isinstance(kernel, tuple) else kernel
        stride = stride[0] if isinstance(stride, tuple) else stride
        dilation = dilation[0] if isinstance(dilation, tuple) else dilation
        effective_kernel = dilation * (kernel - 1) + 1
        receptive_field += (effective_kernel - 1) * jump
        jump *= stride
        stages.append(
            {
                "name": name,
                "type": module.__class__.__name__,
                "receptive_field": receptive_field,
                "jump": jump,
            }
        )
    return stages


def plot_receptive_field_progression(model, show=True):
    stages = receptive_field_stages(model)
    if not stages:
        raise ValueError("No convolution or pooling layers were found in this model.")
    labels = [f"{stage['name']}\n{stage['type']}" for stage in stages]
    values = [stage["receptive_field"] for stage in stages]
    figure, axis = plt.subplots(figsize=(max(8, len(stages) * 1.15), 4.2))
    axis.plot(range(len(stages)), values, marker="o", linewidth=2)
    for index, value in enumerate(values):
        axis.annotate(
            str(value),
            (index, value),
            xytext=(0, 7),
            textcoords="offset points",
            ha="center",
        )
    axis.set_xticks(range(len(stages)), labels, rotation=30, ha="right")
    axis.set_ylabel("theoretical receptive field (input pixels)")
    axis.set_title("Deeper features combine information from a wider input region")
    axis.grid(alpha=0.25)
    figure.tight_layout()
    if show:
        plt.show()
    return figure


def save_cnn_visualizations(model, image, output_dir, model_id, max_layers=6, max_channels=4):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    records = collect_feature_maps(model, image, max_layers=max_layers)
    figures = {
        "feature-maps": plot_feature_map_progression(records, max_channels, show=False),
        "pooling": plot_pooling_comparison(image, show=False),
        "receptive-field": plot_receptive_field_progression(model, show=False),
    }
    paths = []
    for suffix, figure in figures.items():
        path = output_dir / f"{model_id}_{suffix}.png"
        figure.savefig(path, dpi=160, bbox_inches="tight")
        plt.close(figure)
        paths.append(path)
    return paths
