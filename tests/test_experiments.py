import tempfile
import unittest
from dataclasses import replace

import torch
from torch.utils.data import DataLoader, TensorDataset

from cifar10_lab.checkpoints import checkpoint_path, load_model_weights, torch_load_compatible
from cifar10_lab.config import DataConfig, ExperimentConfig, TrainConfig
from cifar10_lab.engine import train_model
from cifar10_lab.experiments import experiment_fingerprint, experiment_id, set_global_seed


class ExperimentTests(unittest.TestCase):
    def _config(self, directory, epochs=1, seed=42):
        return ExperimentConfig(
            model_id="tiny",
            num_classes=2,
            image_size=4,
            device="cpu",
            weight_dir=directory,
            data=DataConfig(batch_size=4, seed=seed, data_root=directory),
            train=TrainConfig(epochs=epochs, learning_rate=0.01),
        )

    def _loaders(self):
        generator = torch.Generator().manual_seed(7)
        images = torch.randn(16, 3, 4, 4, generator=generator)
        labels = torch.tensor([0, 1] * 8)
        loader = DataLoader(TensorDataset(images, labels), batch_size=4, shuffle=False)
        return loader, loader

    def _model(self):
        return torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(48, 2))

    def test_fingerprint_ignores_epoch_target_but_tracks_seed(self):
        with tempfile.TemporaryDirectory() as directory:
            first = self._config(directory, epochs=1, seed=42)
            longer = self._config(directory, epochs=5, seed=42)
            different_seed = self._config(directory, epochs=1, seed=7)

        self.assertEqual(experiment_fingerprint(first), experiment_fingerprint(longer))
        self.assertNotEqual(
            experiment_fingerprint(first),
            experiment_fingerprint(different_seed),
        )

    def test_training_resumes_and_rejects_incompatible_config(self):
        with tempfile.TemporaryDirectory() as directory:
            first_config = self._config(directory, epochs=1)
            run_id = experiment_id(first_config)
            trainloader, valloader = self._loaders()
            set_global_seed(42, deterministic=True)
            first_model = self._model()
            first_history = train_model(
                first_model,
                trainloader,
                valloader,
                "cpu",
                epochs=1,
                learning_rate=0.01,
                model_id="tiny",
                weight_dir=directory,
                experiment_config=first_config,
                experiment_id=run_id,
            )
            path = checkpoint_path("tiny", directory, run_id)
            checkpoint = torch_load_compatible(path, "cpu")

            resumed_config = replace(first_config, train=replace(first_config.train, epochs=2))
            resumed_model = self._model()
            resumed_history = train_model(
                resumed_model,
                trainloader,
                valloader,
                "cpu",
                epochs=2,
                learning_rate=0.01,
                model_id="tiny",
                weight_dir=directory,
                experiment_config=resumed_config,
                experiment_id=run_id,
                resume_checkpoint=checkpoint,
            )

            saved = torch_load_compatible(path, "cpu")
            self.assertEqual(len(first_history["train_loss"]), 1)
            self.assertEqual(len(resumed_history["train_loss"]), 2)
            self.assertEqual(saved["completed_epochs"], 2)

            incompatible = replace(first_config, data=replace(first_config.data, seed=9))
            with self.assertRaises(ValueError):
                load_model_weights(
                    self._model(),
                    "tiny",
                    weight_dir=directory,
                    experiment_id=run_id,
                    expected_config=incompatible,
                )


if __name__ == "__main__":
    unittest.main()
