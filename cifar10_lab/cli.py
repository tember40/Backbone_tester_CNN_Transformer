import argparse
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

from . import __version__
from .checkpoints import load_model_weights
from .config import DataConfig, ExperimentConfig, TrainConfig
from .data import download_cifar10, load_cifar10_data
from .engine import evaluate_model_detailed, train_model
from .environment import detect_environment
from .paths import get_lab_paths, resolve_data_dir, resolve_results_dir
from .registry import create_model, format_model_catalog


def _add_experiment_arguments(parser, include_training=False):
    parser.add_argument("--model", default="resnet18", help="Registry model ID")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cuda", "mps", "cpu"), default="auto")
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--data-dir")
    parser.add_argument("--checkpoint-dir")
    parser.add_argument("--quick", action="store_true", help="Use a small dataset subset")
    parser.add_argument("--plots", action="store_true", help="Show result plots")
    if include_training:
        parser.add_argument("--epochs", type=int)
        parser.add_argument("--learning-rate", type=float, default=0.001)
        parser.add_argument("--val-ratio", type=float, default=0.1)
        parser.add_argument("--results-dir")
        parser.add_argument(
            "--retrain",
            action="store_true",
            help="Train again even if a checkpoint already exists",
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


def _save_result(result, config, runtime, results_dir=None):
    output_dir = resolve_results_dir(results_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_path = output_dir / f"{config.model_id}_{timestamp}.json"
    payload = {
        "created_at": timestamp,
        "package_version": __version__,
        "runtime": str(runtime),
        "config": config.to_dict(),
        "test_accuracy": result["accuracy"],
        "per_class_accuracy": result["per_class_accuracy"].tolist(),
        "confusion_matrix": result["confusion_matrix"].tolist(),
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Result saved to: {output_path}")
    return output_path


def run_train(args):
    config = _create_experiment(args, include_training=True)
    runtime = detect_environment(config.device, config.data.num_workers)
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
        )
    if checkpoint is None:
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
        )
    else:
        history = checkpoint.get("history")
        print("Existing checkpoint loaded; training was skipped. Use --retrain to train again.")

    result = evaluate_model_detailed(
        model, testloader, runtime.device, num_classes=config.num_classes
    )
    _print_result(result, classes)
    _save_result(result, config, runtime, args.results_dir)

    if args.plots:
        from .visualization import plot_test_results, plot_training_history

        plot_training_history(history)
        plot_test_results(result, classes)
    return 0


def run_evaluate(args):
    config = _create_experiment(args)
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
    )
    if checkpoint is None:
        print("No checkpoint to evaluate. Run the train command first.", file=sys.stderr)
        return 2

    _, _, testloader, classes = _build_loaders(config, runtime)
    result = evaluate_model_detailed(
        model, testloader, runtime.device, num_classes=config.num_classes
    )
    _print_result(result, classes)
    if args.plots:
        from .visualization import plot_test_results

        plot_test_results(result, classes)
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
    raise RuntimeError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
