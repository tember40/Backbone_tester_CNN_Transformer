import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from cifar10_lab.cli import build_parser
from cifar10_lab.data import download_cifar10
from cifar10_lab.paths import get_lab_paths


class PathTests(unittest.TestCase):
    def test_environment_override_controls_all_default_paths(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            with patch.dict(
                os.environ,
                {"CIFAR10_LAB_HOME": temporary_directory},
                clear=False,
            ):
                paths = get_lab_paths(create=True)

            self.assertEqual(paths.home, Path(temporary_directory).resolve())
            self.assertTrue(paths.data_dir.is_dir())
            self.assertTrue(paths.checkpoints_dir.is_dir())
            self.assertTrue(paths.results_dir.is_dir())

    @patch("cifar10_lab.data.torchvision.datasets.CIFAR10")
    def test_download_fetches_train_and_test_splits(self, dataset):
        with tempfile.TemporaryDirectory() as temporary_directory:
            data_dir = download_cifar10(temporary_directory)

        self.assertEqual(data_dir, Path(temporary_directory).resolve())
        self.assertEqual(dataset.call_count, 2)
        self.assertEqual(
            [call.kwargs["train"] for call in dataset.call_args_list],
            [True, False],
        )
        self.assertTrue(all(call.kwargs["download"] for call in dataset.call_args_list))


class CliTests(unittest.TestCase):
    def test_quick_train_arguments(self):
        args = build_parser().parse_args([
            "train",
            "--model",
            "resnet18",
            "--quick",
            "--epochs",
            "1",
        ])
        self.assertEqual(args.command, "train")
        self.assertEqual(args.model, "resnet18")
        self.assertTrue(args.quick)
        self.assertEqual(args.epochs, 1)

    def test_evaluate_arguments(self):
        args = build_parser().parse_args(["evaluate", "--model", "mobilenet_v2"])
        self.assertEqual(args.command, "evaluate")
        self.assertEqual(args.model, "mobilenet_v2")

    def test_download_data_arguments(self):
        args = build_parser().parse_args(["download-data", "--data-dir", "datasets"])
        self.assertEqual(args.command, "download-data")
        self.assertEqual(args.data_dir, "datasets")


if __name__ == "__main__":
    unittest.main()
