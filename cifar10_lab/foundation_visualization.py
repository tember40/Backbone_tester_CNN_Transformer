"""Textbook-style plots for perceptron and MLP learning concepts."""

import math

import matplotlib.pyplot as plt
import torch


CLASS_COLORS = ("#4C78A8", "#E45756")


def _as_numpy(tensor):
    return tensor.detach().cpu().numpy()


def plot_perceptron_anatomy(sample=(1.0, 0.5), weights=(0.8, -0.4), bias=0.1):
    """Draw inputs, weighted sum, step activation, and prediction as one computation graph."""
    sample_tensor = torch.tensor(sample, dtype=torch.float32)
    weight_tensor = torch.tensor(weights, dtype=torch.float32)
    contributions = sample_tensor * weight_tensor
    score = float(contributions.sum().item() + bias)
    prediction = int(score >= 0)

    figure, axis = plt.subplots(figsize=(12, 4.8))
    axis.set_xlim(-0.5, 10.5)
    axis.set_ylim(-2.0, 2.0)
    axis.axis("off")

    nodes = {
        "x1": (0.7, 1.0),
        "x2": (0.7, -1.0),
        "sum": (4.2, 0.0),
        "activation": (7.0, 0.0),
        "output": (9.7, 0.0),
    }
    for label, (x, y) in nodes.items():
        axis.scatter(x, y, s=1800, color="#F2F2F2", edgecolor="#333333", zorder=3)
        axis.text(x, y, label, ha="center", va="center", fontsize=12, zorder=4)

    for input_name, weight, contribution in zip(("x1", "x2"), weights, contributions):
        start = nodes[input_name]
        end = nodes["sum"]
        axis.annotate(
            "",
            xy=end,
            xytext=start,
            arrowprops={"arrowstyle": "->", "lw": 1.8, "shrinkA": 28, "shrinkB": 28},
        )
        midpoint_y = (start[1] + end[1]) / 2
        axis.text(
            2.3,
            midpoint_y,
            f"w={weight:+.2f}\nx·w={float(contribution):+.2f}",
            ha="center",
            va="center",
            fontsize=10,
        )

    for start_name, end_name in (("sum", "activation"), ("activation", "output")):
        axis.annotate(
            "",
            xy=nodes[end_name],
            xytext=nodes[start_name],
            arrowprops={"arrowstyle": "->", "lw": 1.8, "shrinkA": 30, "shrinkB": 30},
        )

    axis.text(4.2, -1.35, f"z = Σxw + b = {score:+.2f}\nb = {bias:+.2f}", ha="center")
    axis.text(7.0, -1.35, "step(z)\n1 if z ≥ 0 else 0", ha="center")
    axis.text(9.7, -1.35, f"prediction = {prediction}", ha="center")
    axis.set_title("A perceptron transforms inputs into one binary decision")
    figure.tight_layout()
    return figure


def plot_activation_functions():
    """Compare the most common activation functions on identical axes."""
    x = torch.linspace(-5, 5, 500)
    functions = (
        ("Step", (x >= 0).float()),
        ("Sigmoid", torch.sigmoid(x)),
        ("Tanh", torch.tanh(x)),
        ("ReLU", torch.relu(x)),
    )
    figure, axes = plt.subplots(1, 4, figsize=(15, 3.6), sharex=True)
    for axis, (name, values) in zip(axes, functions):
        axis.plot(_as_numpy(x), _as_numpy(values), linewidth=2)
        axis.axhline(0, color="gray", linewidth=0.8)
        axis.axvline(0, color="gray", linewidth=0.8)
        axis.set_title(name)
        axis.set_xlabel("weighted sum z")
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("activation output")
    figure.suptitle("Activation functions: from a hard decision to trainable nonlinearity")
    figure.tight_layout()
    return figure


def _draw_boundary(axis, weights, bias, x_limits=(-0.35, 1.35)):
    weights = torch.as_tensor(weights)
    if abs(float(weights[1])) > 1e-8:
        x_values = torch.tensor(x_limits)
        y_values = -(weights[0] * x_values + bias) / weights[1]
        axis.plot(_as_numpy(x_values), _as_numpy(y_values), color="#333333", linewidth=2)
    elif abs(float(weights[0])) > 1e-8:
        axis.axvline(-bias / float(weights[0]), color="#333333", linewidth=2)


def _draw_points(axis, features, targets):
    for target in (0, 1):
        mask = targets == target
        points = features[mask]
        axis.scatter(
            _as_numpy(points[:, 0]),
            _as_numpy(points[:, 1]),
            s=110,
            color=CLASS_COLORS[target],
            edgecolor="white",
            linewidth=1.2,
            label=f"class {target}",
            zorder=3,
        )
    axis.set_xlim(-0.35, 1.35)
    axis.set_ylim(-0.35, 1.35)
    axis.set_xlabel("x1")
    axis.set_ylabel("x2")
    axis.set_aspect("equal")
    axis.grid(alpha=0.2)


def plot_perceptron_learning(features, targets, snapshots, max_panels=6):
    """Show how individual perceptron updates move the linear decision boundary."""
    if not snapshots:
        raise ValueError("snapshots must not be empty.")
    update_indices = [index for index, snapshot in enumerate(snapshots) if snapshot.error != 0]
    if not update_indices:
        update_indices = [len(snapshots) - 1]
    if len(update_indices) > max_panels:
        positions = torch.linspace(0, len(update_indices) - 1, max_panels).round().long()
        update_indices = [update_indices[index] for index in positions]

    columns = min(3, len(update_indices))
    rows = math.ceil(len(update_indices) / columns)
    figure, axes = plt.subplots(rows, columns, figsize=(4.4 * columns, 4.2 * rows), squeeze=False)
    for axis in axes.flat:
        axis.set_visible(False)
    for axis, snapshot_index in zip(axes.flat, update_indices):
        axis.set_visible(True)
        snapshot = snapshots[snapshot_index]
        _draw_points(axis, features, targets)
        _draw_boundary(axis, snapshot.weights, snapshot.bias)
        axis.set_title(f"Update {snapshot.step + 1} · sample {snapshot.sample_index}")
        weight_text = ", ".join(f"{value:.1f}" for value in snapshot.weights.tolist())
        axis.text(
            0.03,
            0.04,
            f"error={snapshot.error:+d}\nw=[{weight_text}], b={snapshot.bias:+.1f}",
            transform=axis.transAxes,
            fontsize=9,
            va="bottom",
            bbox={"facecolor": "white", "alpha": 0.82, "edgecolor": "none"},
        )
    handles, labels = axes.flat[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=2)
    figure.suptitle("Each mistake shifts the decision boundary")
    figure.tight_layout(rect=(0, 0.07, 1, 0.94))
    return figure


def plot_xor_comparison(perceptron, mlp, features, targets):
    """Compare a straight perceptron boundary with an MLP nonlinear decision region."""
    grid_axis = torch.linspace(-0.35, 1.35, 180)
    grid_x, grid_y = torch.meshgrid(grid_axis, grid_axis, indexing="xy")
    grid = torch.stack((grid_x.flatten(), grid_y.flatten()), dim=1)

    perceptron_scores = perceptron.score(grid).reshape(grid_x.shape)
    with torch.no_grad():
        mlp_probabilities = torch.sigmoid(mlp(grid)).reshape(grid_x.shape)

    figure, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    surfaces = (
        (perceptron_scores, "Single perceptron: one straight boundary"),
        (mlp_probabilities - 0.5, "MLP: hidden units combine several boundaries"),
    )
    for axis, (surface, title) in zip(axes, surfaces):
        axis.contourf(
            _as_numpy(grid_x),
            _as_numpy(grid_y),
            _as_numpy(surface),
            levels=(-10, 0, 10),
            colors=CLASS_COLORS,
            alpha=0.18,
        )
        axis.contour(
            _as_numpy(grid_x),
            _as_numpy(grid_y),
            _as_numpy(surface),
            levels=(0,),
            colors=("#333333",),
            linewidths=2,
        )
        _draw_points(axis, features, targets)
        axis.set_title(title)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=2)
    figure.suptitle("Why XOR requires a hidden layer")
    figure.tight_layout(rect=(0, 0.08, 1, 0.94))
    return figure


def plot_mlp_learning_history(history):
    """Show how XOR loss and accuracy change during gradient-based learning."""
    epochs = torch.arange(1, len(history["loss"]) + 1)
    figure, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    axes[0].plot(_as_numpy(epochs), history["loss"], linewidth=2)
    axes[0].set_title("Binary cross-entropy loss")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("loss")
    axes[0].grid(alpha=0.25)

    axes[1].plot(_as_numpy(epochs), history["accuracy"], linewidth=2)
    axes[1].set_title("Training accuracy")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("accuracy (%)")
    axes[1].set_ylim(-2, 102)
    axes[1].grid(alpha=0.25)
    figure.suptitle("The MLP learns XOR through repeated gradient updates")
    figure.tight_layout()
    return figure


def plot_hidden_representation(model, features, targets):
    """Show the hidden activation pattern learned for each XOR input."""
    with torch.no_grad():
        hidden = model.hidden_representation(features)
    figure, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    _draw_points(axes[0], features, targets)
    axes[0].set_title("Input space: XOR is not linearly separable")
    image = axes[1].imshow(
        _as_numpy(hidden),
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        aspect="auto",
    )
    hidden_labels = [f"h{index + 1}" for index in range(hidden.shape[1])]
    axes[1].set_xticks(range(hidden.shape[1]), hidden_labels)
    row_labels = [
        f"({int(sample[0])}, {int(sample[1])}) → class {int(target)}"
        for sample, target in zip(features, targets)
    ]
    axes[1].set_yticks(range(len(features)), row_labels)
    axes[1].set_xlabel("hidden neuron")
    axes[1].set_ylabel("input and target")
    axes[1].set_title("Hidden activations: each input gets a learned code")
    for row in range(hidden.shape[0]):
        for column in range(hidden.shape[1]):
            value = float(hidden[row, column])
            axes[1].text(
                column,
                row,
                f"{value:+.2f}",
                ha="center",
                va="center",
                color="white" if abs(value) > 0.55 else "black",
                fontsize=9,
            )
    figure.colorbar(image, ax=axes[1], label="Tanh activation", fraction=0.046, pad=0.04)
    figure.suptitle("A hidden layer learns a more useful representation")
    figure.tight_layout()
    return figure
