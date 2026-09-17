"""Write machine-readable and visual comparison reports."""

import csv
import json
from datetime import datetime, timezone

from .paths import resolve_results_dir


COMPARISON_COLUMNS = (
    "model_id",
    "run_id",
    "test_accuracy",
    "parameter_count",
    "training_seconds",
    "inference_ms_per_image",
    "checkpoint_size_mb",
    "completed_epochs",
    "trained_this_run",
)


def save_comparison_report(rows, results_dir=None):
    output_dir = resolve_results_dir(results_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    stem = f"comparison_{timestamp}"
    json_path = output_dir / f"{stem}.json"
    csv_path = output_dir / f"{stem}.csv"

    json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=COMPARISON_COLUMNS)
        writer.writeheader()
        writer.writerows({column: row.get(column) for column in COMPARISON_COLUMNS} for row in rows)

    plot_path = _save_comparison_plot(rows, output_dir / f"{stem}.png")
    return {"json": json_path, "csv": csv_path, "plot": plot_path}


def _save_comparison_plot(rows, output_path):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    labels = [row["model_id"] for row in rows]
    accuracies = [row["test_accuracy"] for row in rows]
    parameters = [row["parameter_count"] / 1_000_000 for row in rows]
    inference = [row["inference_ms_per_image"] for row in rows]

    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    bars = axes[0].bar(labels, accuracies, color="#4C78A8")
    axes[0].bar_label(bars, labels=[f"{value:.1f}" for value in accuracies], padding=2)
    axes[0].set_ylabel("test accuracy (%)")
    axes[0].set_title("Accuracy under the same experiment settings")
    axes[0].tick_params(axis="x", rotation=30)
    axes[0].set_ylim(0, max(accuracies, default=0) * 1.2 + 1)

    maximum_parameters = max(parameters, default=0)
    for label, parameter_count, latency in zip(labels, parameters, inference):
        axes[1].scatter(parameter_count, latency, s=90)
        rightmost = parameter_count == maximum_parameters and maximum_parameters > 0
        axes[1].annotate(
            label,
            (parameter_count, latency),
            xytext=(-5 if rightmost else 5, 5),
            textcoords="offset points",
            ha="right" if rightmost else "left",
        )
    axes[1].set_xlabel("parameters (millions)")
    axes[1].set_ylabel("inference time (ms/image)")
    axes[1].set_title("Model size and inference cost")
    axes[1].grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)
    return output_path
