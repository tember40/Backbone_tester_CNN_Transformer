import matplotlib.pyplot as plt
import torch

from .data import CIFAR10_MEAN, CIFAR10_STD


def visualize_data_overview(
    trainloader,
    valloader,
    testloader,
    classes,
    max_images=10,
):
    split_names = ["Train", "Validation", "Test"]
    split_sizes = [
        len(trainloader.dataset),
        len(valloader.dataset),
        len(testloader.dataset),
    ]

    images, labels = next(iter(trainloader))
    image_count = min(max_images, len(images))
    mean = torch.tensor(CIFAR10_MEAN).view(3, 1, 1)
    std = torch.tensor(CIFAR10_STD).view(3, 1, 1)
    images = (images[:image_count].cpu() * std + mean).clamp(0, 1)

    figure = plt.figure(figsize=(15, 6))
    grid = figure.add_gridspec(2, max(5, image_count))

    split_axis = figure.add_subplot(grid[:, :3])
    bars = split_axis.bar(
        split_names,
        split_sizes,
        color=["#4C78A8", "#F2CF5B", "#E45756"],
    )
    split_axis.set_title("CIFAR-10 data split")
    split_axis.set_ylabel("Number of images")
    split_axis.bar_label(bars, padding=3)
    split_axis.set_ylim(0, max(split_sizes) * 1.12)

    sample_columns = max(1, grid.ncols - 3)
    for index in range(image_count):
        row = index // sample_columns
        column = index % sample_columns + 3
        if row >= 2:
            break
        axis = figure.add_subplot(grid[row, column])
        axis.imshow(images[index].permute(1, 2, 0))
        axis.set_title(classes[labels[index].item()], fontsize=9)
        axis.axis("off")

    figure.suptitle("Step 1: split sizes and augmented train samples", fontsize=14)
    figure.tight_layout()
    plt.show()


def plot_training_history(history):
    if not history or not history.get("train_loss"):
        print("No training history is available.")
        return

    epochs = range(1, len(history["train_loss"]) + 1)
    figure, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    axes[0].plot(epochs, history["train_loss"], marker="o", label="Train")
    axes[0].plot(epochs, history["val_loss"], marker="o", label="Validation")
    axes[0].set_title("Loss by epoch")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-entropy loss")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].plot(epochs, history["train_accuracy"], marker="o", label="Train")
    axes[1].plot(epochs, history["val_accuracy"], marker="o", label="Validation")
    best_index = max(range(len(history["val_accuracy"])), key=history["val_accuracy"].__getitem__)
    axes[1].scatter(
        best_index + 1,
        history["val_accuracy"][best_index],
        color="red",
        zorder=3,
        label="Best validation",
    )
    axes[1].set_title("Accuracy by epoch")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    figure.suptitle("Step 2: training and validation progress", fontsize=14)
    figure.tight_layout()
    plt.show()


def plot_test_results(result, classes, normalize_confusion=True):
    matrix = result["confusion_matrix"].float()
    if normalize_confusion:
        row_sums = matrix.sum(dim=1, keepdim=True).clamp_min(1)
        matrix = matrix / row_sums

    figure, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    image = axes[0].imshow(
        matrix.numpy(),
        cmap="Blues",
        vmin=0,
        vmax=1 if normalize_confusion else None,
    )
    axes[0].set_title(
        "Normalized confusion matrix" if normalize_confusion else "Confusion matrix"
    )
    axes[0].set_xlabel("Predicted class")
    axes[0].set_ylabel("True class")
    axes[0].set_xticks(range(len(classes)), classes, rotation=45, ha="right")
    axes[0].set_yticks(range(len(classes)), classes)
    figure.colorbar(image, ax=axes[0], fraction=0.046, pad=0.04)

    class_accuracy = result["per_class_accuracy"].numpy()
    bars = axes[1].bar(classes, class_accuracy, color="#4C78A8")
    axes[1].bar_label(
        bars,
        labels=[f"{value:.1f}" for value in class_accuracy],
        padding=2,
        fontsize=8,
    )
    axes[1].set_title("Accuracy by class")
    axes[1].set_xlabel("Class")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].set_ylim(0, 105)
    axes[1].tick_params(axis="x", rotation=45)

    figure.suptitle(
        f"Step 3: independent test evaluation — overall accuracy {result['accuracy']:.2f}%",
        fontsize=14,
    )
    figure.tight_layout()
    plt.show()
