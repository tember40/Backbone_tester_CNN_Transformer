import torch
import torch.nn as nn
import torch.optim as optim

from .checkpoints import checkpoint_path, torch_load_compatible


def _main_logits(outputs):
    return outputs[0] if isinstance(outputs, tuple) else outputs


def run_evaluation(model, dataloader, device, criterion=None, collect_predictions=False):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    labels_all = []
    predictions_all = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            outputs = _main_logits(model(images))

            if criterion is not None:
                total_loss += criterion(outputs, labels).item() * labels.size(0)

            predictions = outputs.argmax(dim=1)
            total += labels.size(0)
            correct += (predictions == labels).sum().item()

            if collect_predictions:
                labels_all.append(labels.cpu())
                predictions_all.append(predictions.cpu())

    result = {
        "loss": total_loss / total if criterion is not None and total else None,
        "accuracy": 100.0 * correct / total if total else 0.0,
        "total": total,
    }
    if collect_predictions:
        result["labels"] = torch.cat(labels_all) if labels_all else torch.empty(0, dtype=torch.long)
        result["predictions"] = (
            torch.cat(predictions_all) if predictions_all else torch.empty(0, dtype=torch.long)
        )
    return result


def train_model(
    model,
    trainloader,
    valloader,
    device,
    epochs=10,
    learning_rate=0.001,
    model_id="model",
    weight_dir=None,
    experiment_config=None,
):
    """Train으로 학습하고 validation 정확도로 최적 checkpoint를 선택한다."""
    if epochs < 1:
        raise ValueError("epochs must be at least 1.")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    history = {
        "train_loss": [],
        "val_loss": [],
        "train_accuracy": [],
        "val_accuracy": [],
    }

    best_val_accuracy = float("-inf")
    best_epoch = 0
    weight_path = checkpoint_path(model_id, weight_dir)
    weight_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in trainloader:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad()
            outputs = _main_logits(model(inputs))
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * labels.size(0)
            predictions = outputs.argmax(dim=1)
            total += labels.size(0)
            correct += (predictions == labels).sum().item()

        train_loss = running_loss / total
        train_accuracy = 100.0 * correct / total
        val_result = run_evaluation(model, valloader, device, criterion=criterion)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_result["loss"])
        history["train_accuracy"].append(train_accuracy)
        history["val_accuracy"].append(val_result["accuracy"])

        print(
            f"Epoch {epoch + 1:02d}/{epochs} | "
            f"train loss {train_loss:.4f}, accuracy {train_accuracy:.2f}% | "
            f"validation loss {val_result['loss']:.4f}, "
            f"accuracy {val_result['accuracy']:.2f}%"
        )

        if val_result["accuracy"] > best_val_accuracy:
            best_val_accuracy = val_result["accuracy"]
            best_epoch = epoch + 1
            checkpoint = {
                "format_version": 1,
                "model_id": model_id,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "epoch": best_epoch,
                "best_val_accuracy": best_val_accuracy,
                "history": history,
                "num_classes": 10,
                "experiment_config": (
                    experiment_config.to_dict()
                    if hasattr(experiment_config, "to_dict")
                    else experiment_config
                ),
            }
            torch.save(checkpoint, weight_path)
            print(f"  Saved best checkpoint: {weight_path}")

    checkpoint = torch_load_compatible(weight_path, device)
    checkpoint["history"] = history
    checkpoint["completed_epochs"] = epochs
    torch.save(checkpoint, weight_path)
    model.load_state_dict(checkpoint["model_state"])

    print(
        f"Finished training. Best validation accuracy: "
        f"{best_val_accuracy:.2f}% at epoch {best_epoch}."
    )
    return history


def evaluate_model(model, dataloader, device):
    return run_evaluation(model, dataloader, device)["accuracy"]


def evaluate_model_detailed(model, dataloader, device, num_classes=10):
    result = run_evaluation(model, dataloader, device, collect_predictions=True)
    confusion_matrix = torch.zeros(num_classes, num_classes, dtype=torch.long)
    for label, prediction in zip(result["labels"], result["predictions"]):
        confusion_matrix[label.long(), prediction.long()] += 1

    class_totals = confusion_matrix.sum(dim=1)
    class_correct = confusion_matrix.diag()
    result["confusion_matrix"] = confusion_matrix
    result["per_class_accuracy"] = torch.where(
        class_totals > 0,
        class_correct.float() / class_totals.float() * 100.0,
        torch.zeros_like(class_totals, dtype=torch.float),
    )
    return result
