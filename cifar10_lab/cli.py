import argparse
import json
import platform
import sys
import time
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import torch

from . import __version__
from .benchmark import checkpoint_size_mb, count_parameters, measure_inference_ms_per_image
from .checkpoints import (
    checkpoint_path,
    latest_checkpoint_path,
    load_model_weights,
    torch_load_compatible,
)
from .config import DataConfig, ExperimentConfig, TrainConfig
from .data import download_cifar10, load_cifar10_data
from .engine import evaluate_model_detailed, train_model
from .environment import detect_environment
from .experiments import experiment_id, set_global_seed
from .paths import get_lab_paths, resolve_data_dir, resolve_results_dir
from .reporting import save_comparison_report
from .registry import create_model, format_model_catalog, list_models


def _add_experiment_arguments(parser, include_training=False, include_model=True):
    if include_model:
        parser.add_argument("--model", default="resnet18", help="Registry model ID")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cuda", "mps", "cpu"), default="auto")
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--data-dir")
    parser.add_argument("--checkpoint-dir")
    parser.add_argument("--quick", action="store_true", help="Use a small dataset subset")
    parser.add_argument("--plots", action="store_true", help="Show result plots")
    parser.add_argument("--save-plots", action="store_true", help="Save plots as PNG files")
    parser.add_argument("--results-dir")
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Prefer deterministic PyTorch operations when available",
    )
    if include_training:
        parser.add_argument("--epochs", type=int)
        parser.add_argument("--learning-rate", type=float, default=0.001)
        parser.add_argument("--val-ratio", type=float, default=0.1)
        mode = parser.add_mutually_exclusive_group()
        mode.add_argument(
            "--retrain",
            action="store_true",
            help="Train again even if a checkpoint already exists",
        )
        mode.add_argument(
            "--resume",
            action="store_true",
            help="Continue an existing run up to --epochs",
        )


def build_parser():
    parser = argparse.ArgumentParser(
        prog="cifar10-lab",
        description="Train and evaluate CIFAR-10 backbone models.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("list-models", help="List registered CIFAR-10-ready models")
    subparsers.add_parser("doctor", help="Check the Python, PyTorch and device environment")

    download_parser = subparsers.add_parser(
        "download-data",
        help="Download the CIFAR-10 train and test splits",
    )
    download_parser.add_argument("--data-dir")

    train_parser = subparsers.add_parser("train", help="Train and test a model")
    _add_experiment_arguments(train_parser, include_training=True)

    evaluate_parser = subparsers.add_parser("evaluate", help="Evaluate a saved checkpoint")
    _add_experiment_arguments(evaluate_parser)
    evaluate_parser.add_argument("--run-id", help="Exact run ID; defaults to latest for model")

    compare_parser = subparsers.add_parser(
        "compare",
        help="Train and compare multiple models under the same settings",
    )
    compare_parser.add_argument("--models", nargs="+", required=True)
    _add_experiment_arguments(compare_parser, include_training=True, include_model=False)

    validate_parser = subparsers.add_parser(
        "validate-models",
        help="Run a forward-pass check for registered models",
    )
    validation_scope = validate_parser.add_mutually_exclusive_group()
    validation_scope.add_argument("--models", nargs="+")
    validation_scope.add_argument(
        "--all",
        action="store_true",
        help="Validate every CIFAR-10-ready model; this can require substantial memory",
    )
    validate_parser.add_argument("--device", choices=("auto", "cuda", "mps", "cpu"), default="auto")
    validate_parser.add_argument("--fail-fast", action="store_true")

    cnn_parser = subparsers.add_parser(
        "visualize-cnn",
        help="Save feature-map, pooling, and receptive-field explanations",
    )
    cnn_parser.add_argument("--model", default="alexnet")
    cnn_parser.add_argument("--device", choices=("auto", "cuda", "mps", "cpu"), default="auto")
    cnn_parser.add_argument("--data-dir")
    cnn_parser.add_argument("--checkpoint-dir")
    cnn_parser.add_argument("--output-dir")
    cnn_parser.add_argument("--max-layers", type=int, default=6)
    cnn_parser.add_argument("--max-channels", type=int, default=4)
    return parser


def _experiment_paths(args):
    defaults = get_lab_paths()
    run_kind = "quick" if args.quick else "full"
    checkpoint_dir = (
        Path(args.checkpoint_dir).expanduser().resolve()
        if args.checkpoint_dir
        else defaults.checkpoints_dir / run_kind
    )
    return resolve_data_dir(args.data_dir), checkpoint_dir


def _data_config(args, data_dir):
    return DataConfig(
        batch_size=args.batch_size,
        val_ratio=getattr(args, "val_ratio", 0.1),
        seed=args.seed,
        num_workers=args.num_workers,
        data_root=str(data_dir),
        max_train_samples=2048 if args.quick else None,
        max_val_samples=512 if args.quick else None,
        max_test_samples=512 if args.quick else None,
    )


def _build_loaders(config, runtime):
    return load_cifar10_data(
        batch_size=config.data.batch_size,
        val_ratio=config.data.val_ratio,
        seed=config.data.seed,
        num_workers=runtime.num_workers,
        pin_memory=runtime.pin_memory,
        image_size=config.image_size,
        data_root=config.data.data_root,
        max_train_samples=config.data.max_train_samples,
        max_val_samples=config.data.max_val_samples,
        max_test_samples=config.data.max_test_samples,
    )


def _create_experiment(args, include_training=False):
    data_dir, checkpoint_dir = _experiment_paths(args)
    epochs = (1 if args.quick else 10) if getattr(args, "epochs", None) is None else args.epochs
    train_config = TrainConfig(
        epochs=epochs,
        learning_rate=getattr(args, "learning_rate", 0.001),
    )
    return ExperimentConfig(
        model_id=args.model,
        num_classes=10,
        image_size=32,
        device=args.device,
        weight_dir=str(checkpoint_dir),
        data=_data_config(args, data_dir),
        train=train_config,
    )


def _print_result(result, classes):
    print(f"Test accuracy: {result['accuracy']:.2f}%")
    print("Per-class accuracy:")
    for class_name, accuracy in zip(classes, result["per_class_accuracy"].tolist()):
        print(f"  {class_name:>5}: {accuracy:6.2f}%")


def _save_result(result, config, runtime, run_id, results_dir=None, metrics=None):
    output_dir = resolve_results_dir(results_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_path = output_dir / f"{run_id}_{timestamp}.json"
    payload = {
        "created_at": timestamp,
        "package_version": __version__,
        "runtime": str(runtime),
        "run_id": run_id,
        "config": config.to_dict(),
        "test_accuracy": result["accuracy"],
        "per_class_accuracy": result["per_class_accuracy"].tolist(),
        "confusion_matrix": result["confusion_matrix"].tolist(),
    }
    if metrics:
        payload.update(metrics)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Result saved to: {output_path}")
    return output_path


def _save_result_plots(history, result, classes, run_id, results_dir, show=False):
    from .visualization import plot_test_results, plot_training_history

    output_dir = resolve_results_dir(results_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    history_figure = plot_training_history(history, show=show)
    result_figure = plot_test_results(result, classes, show=show)
    paths = []
    if history_figure is not None:
        history_path = output_dir / f"{run_id}_training.png"
        history_figure.savefig(history_path, dpi=160, bbox_inches="tight")
        paths.append(history_path)
    result_path = output_dir / f"{run_id}_test.png"
    result_figure.savefig(result_path, dpi=160, bbox_inches="tight")
    paths.append(result_path)
    for path in paths:
        print(f"Plot saved to: {path}")


def _execute_train(args):
    config = _create_experiment(args, include_training=True)
    run_id = experiment_id(config)
    set_global_seed(config.data.seed, deterministic=args.deterministic)
    runtime = detect_environment(config.device, config.data.num_workers)
    print(f"Run ID: {run_id}")
    print(f"Runtime: {runtime}")
    print(f"Data directory: {config.data.data_root}")
    print(f"Checkpoint directory: {config.weight_dir}")

    model = create_model(
        config.model_id,
        num_classes=config.num_classes,
        image_size=config.image_size,
    ).to(runtime.device)
    trainloader, valloader, testloader, classes = _build_loaders(config, runtime)

    checkpoint = None
    if not args.retrain:
        checkpoint = load_model_weights(
            model,
            config.model_id,
            device=runtime.device,
            weight_dir=config.weight_dir,
            experiment_id=run_id,
            expected_config=config,
        )
    if args.resume and checkpoint is None:
        raise FileNotFoundError(f"No checkpoint exists for run {run_id}; start without --resume.")

    completed_epochs = (
        int(checkpoint.get("completed_epochs", 0)) if checkpoint else 0
    )
    performed_training = checkpoint is None or (
        args.resume and completed_epochs < config.train.epochs
    )
    previous_training_seconds = (
        float(checkpoint.get("training_seconds", 0.0)) if checkpoint else 0.0
    )
    training_started_at = time.perf_counter()
    if performed_training:
        history = train_model(
            model,
            trainloader,
            valloader,
            runtime.device,
            epochs=config.train.epochs,
            learning_rate=config.train.learning_rate,
            model_id=config.model_id,
            weight_dir=config.weight_dir,
            experiment_config=config,
            experiment_id=run_id,
            resume_checkpoint=checkpoint if args.resume else None,
        )
    else:
        history = checkpoint.get("history")
        print(
            "Existing checkpoint loaded; training was skipped. "
            "Use --resume or --retrain to continue."
        )
    elapsed_this_run = time.perf_counter() - training_started_at

    result = evaluate_model_detailed(
        model, testloader, runtime.device, num_classes=config.num_classes
    )
    _print_result(result, classes)
    weight_path = checkpoint_path(config.model_id, config.weight_dir, run_id)
    if performed_training:
        stored_checkpoint = torch_load_compatible(weight_path, runtime.device)
        training_seconds = previous_training_seconds + elapsed_this_run
        stored_checkpoint["training_seconds"] = training_seconds
        torch.save(stored_checkpoint, weight_path)
    else:
        training_seconds = previous_training_seconds
    metrics = {
        "parameter_count": count_parameters(model),
        "training_seconds": training_seconds,
        "inference_ms_per_image": measure_inference_ms_per_image(
            model, testloader, runtime.device
        ),
        "checkpoint_size_mb": checkpoint_size_mb(weight_path),
        "completed_epochs": len(history.get("train_loss", [])) if history else 0,
        "trained_this_run": performed_training,
    }
    _save_result(result, config, runtime, run_id, args.results_dir, metrics)

    if args.plots or args.save_plots:
        _save_result_plots(
            history,
            result,
            classes,
            run_id,
            args.results_dir,
            show=args.plots,
        )
    return {
        "model_id": config.model_id,
        "run_id": run_id,
        "test_accuracy": result["accuracy"],
        **metrics,
    }


def run_train(args):
    try:
        _execute_train(args)
    except FileNotFoundError as error:
        print(error, file=sys.stderr)
        return 2
    return 0


def run_evaluate(args):
    data_dir, checkpoint_dir = _experiment_paths(args)
    selected_path = (
        checkpoint_path(args.model, checkpoint_dir, args.run_id)
        if args.run_id
        else latest_checkpoint_path(args.model, checkpoint_dir)
    )
    if selected_path is None or not selected_path.exists():
        print("No checkpoint to evaluate. Run the train command first.", file=sys.stderr)
        return 2

    raw_checkpoint = torch_load_compatible(selected_path, "cpu")
    saved_config = (
        raw_checkpoint.get("experiment_config")
        if isinstance(raw_checkpoint, dict)
        else None
    )
    if saved_config:
        config = ExperimentConfig.from_dict(saved_config)
        current_data = replace(
            config.data,
            data_root=str(data_dir),
            num_workers=args.num_workers,
        )
        config = replace(
            config,
            device=args.device,
            weight_dir=str(checkpoint_dir),
            data=current_data,
        )
    else:
        config = _create_experiment(args)

    set_global_seed(config.data.seed, deterministic=args.deterministic)
    runtime = detect_environment(config.device, config.data.num_workers)
    model = create_model(
        config.model_id,
        num_classes=config.num_classes,
        image_size=config.image_size,
    ).to(runtime.device)
    checkpoint = load_model_weights(
        model,
        config.model_id,
        device=runtime.device,
        weight_dir=config.weight_dir,
        checkpoint_file=selected_path,
    )
    _, _, testloader, classes = _build_loaders(config, runtime)
    result = evaluate_model_detailed(
        model, testloader, runtime.device, num_classes=config.num_classes
    )
    _print_result(result, classes)
    run_id = checkpoint.get("experiment_id") or selected_path.stem.removeprefix("Cifar-10_")
    _save_result(result, config, runtime, run_id, args.results_dir)
    if args.plots or args.save_plots:
        _save_result_plots(
            checkpoint.get("history"),
            result,
            classes,
            run_id,
            args.results_dir,
            show=args.plots,
        )
    return 0


def run_compare(args):
    rows = []
    failures = []
    for model_id in args.models:
        print(f"\n=== Comparing {model_id} ===")
        model_args = argparse.Namespace(**vars(args))
        model_args.model = model_id
        model_args.plots = False
        model_args.save_plots = False
        try:
            rows.append(_execute_train(model_args))
        except (FileNotFoundError, KeyError, RuntimeError, ValueError) as error:
            failures.append(model_id)
            print(f"{model_id} failed: {error}", file=sys.stderr)

    if not rows:
        print("No model completed successfully.", file=sys.stderr)
        return 2
    report_paths = save_comparison_report(rows, args.results_dir)
    print("\nComparison summary:")
    for row in sorted(rows, key=lambda item: item["test_accuracy"], reverse=True):
        print(
            f"  {row['model_id']:18s} accuracy={row['test_accuracy']:6.2f}% | "
            f"parameters={row['parameter_count']:,} | "
            f"inference={row['inference_ms_per_image']:.3f} ms/image"
        )
    for label, path in report_paths.items():
        if path is not None:
            print(f"Comparison {label}: {path}")
    return 1 if failures else 0


def run_validate_models(args):
    from .model_validation import validate_registered_models

    runtime = detect_environment(args.device)
    model_ids = args.models
    if args.all:
        model_ids = [
            spec.model_id for spec in list_models(cifar10_ready_only=True)
        ]
    results = validate_registered_models(
        model_ids=model_ids,
        device=runtime.device,
        fail_fast=args.fail_fast,
    )
    failed = []
    for result in results:
        if result["status"] == "ok":
            print(
                f"[ok] {result['model_id']:20s} "
                f"output={result['output_shape']} parameters={result['parameter_count']:,}"
            )
        else:
            failed.append(result)
            print(f"[failed] {result['model_id']:16s} {result['error']}")
    print(f"Validated {len(results)} model(s); {len(failed)} failure(s).")
    return 1 if failed else 0


def run_visualize_cnn(args):
    from .cnn_visualization import save_cnn_visualizations

    if args.max_layers < 1 or args.max_channels < 1:
        raise ValueError("max-layers and max-channels must be at least 1.")
    set_global_seed(42)
    runtime = detect_environment(args.device)
    data_dir = resolve_data_dir(args.data_dir)
    _, _, testloader, _ = load_cifar10_data(
        batch_size=1,
        seed=42,
        num_workers=runtime.num_workers,
        pin_memory=runtime.pin_memory,
        data_root=data_dir,
        max_train_samples=1,
        max_val_samples=1,
        max_test_samples=1,
    )
    model = create_model(args.model, num_classes=10, image_size=32).to(runtime.device)
    checkpoint_dir = args.checkpoint_dir
    if checkpoint_dir:
        saved_path = latest_checkpoint_path(args.model, checkpoint_dir)
        if saved_path is not None:
            load_model_weights(
                model,
                args.model,
                device=runtime.device,
                checkpoint_file=saved_path,
            )
        else:
            print(
                "No matching checkpoint was found; visualizing random initial features."
            )
    else:
        print("No checkpoint directory was provided; visualizing random initial features.")
    images, _ = next(iter(testloader))
    images = images.to(runtime.device)
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else resolve_results_dir() / "cnn-visualizations"
    )
    paths = save_cnn_visualizations(
        model,
        images,
        output_dir,
        args.model,
        max_layers=args.max_layers,
        max_channels=args.max_channels,
    )
    for path in paths:
        print(f"Visualization saved to: {path}")
    return 0


def run_doctor():
    runtime = detect_environment()
    paths = get_lab_paths()
    print(f"cifar10-backbone-lab: {__version__}")
    print(f"Python: {platform.python_version()}")
    print(f"PyTorch: {torch.__version__}")
    print(f"OS: {platform.platform()}")
    print(f"Runtime: {runtime}")
    print(f"Storage home: {paths.home}")

    model = create_model("resnet18", num_classes=10, image_size=32).eval()
    with torch.no_grad():
        output = model(torch.randn(1, 3, 32, 32))
    print(f"ResNet18 smoke output: {tuple(output.shape)}")
    print("Environment check passed.")
    return 0


def run_download_data(args):
    data_dir = download_cifar10(args.data_dir)
    print(f"CIFAR-10 is ready at: {data_dir}")
    return 0


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        if args.command == "list-models":
            print(format_model_catalog(cifar10_ready_only=True))
            return 0
        if args.command == "doctor":
            return run_doctor()
        if args.command == "download-data":
            return run_download_data(args)
        if args.command == "train":
            return run_train(args)
        if args.command == "evaluate":
            return run_evaluate(args)
        if args.command == "compare":
            return run_compare(args)
        if args.command == "validate-models":
            return run_validate_models(args)
        if args.command == "visualize-cnn":
            return run_visualize_cnn(args)
        raise RuntimeError(f"Unhandled command: {args.command}")
    except (FileNotFoundError, KeyError, ValueError) as error:
        print(error, file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
