"""Foundational fully connected classifiers for 32x32 image experiments."""

import torch.nn as nn


class Perceptron(nn.Module):
    """A multiclass linear classifier: flatten pixels, then compute one affine map."""

    def __init__(self, num_classes=10, image_size=32):
        super().__init__()
        self.image_size = image_size
        self.flatten = nn.Flatten()
        self.classifier = nn.Linear(3 * image_size * image_size, num_classes)

    def forward(self, x):
        return self.classifier(self.flatten(x))


class MLP(nn.Module):
    """A two-hidden-layer perceptron baseline for CIFAR-10."""

    def __init__(
        self,
        num_classes=10,
        image_size=32,
        hidden_sizes=(512, 256),
        dropout=0.2,
    ):
        super().__init__()
        input_features = 3 * image_size * image_size
        layers = [nn.Flatten()]
        previous_features = input_features
        for hidden_features in hidden_sizes:
            layers.extend(
                [
                    nn.Linear(previous_features, hidden_features),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
            previous_features = hidden_features
        layers.append(nn.Linear(previous_features, num_classes))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


def perceptron(num_classes=10, image_size=32, **kwargs):
    """Create the linear CIFAR-10 baseline used as a multiclass perceptron."""
    return Perceptron(num_classes=num_classes, image_size=image_size, **kwargs)


def mlp(num_classes=10, image_size=32, **kwargs):
    """Create the default two-hidden-layer MLP."""
    return MLP(num_classes=num_classes, image_size=image_size, **kwargs)


def mlp_deep(num_classes=10, image_size=32, **kwargs):
    """Create a deeper MLP for studying depth without convolution."""
    kwargs.setdefault("hidden_sizes", (1024, 512, 256))
    return MLP(num_classes=num_classes, image_size=image_size, **kwargs)
