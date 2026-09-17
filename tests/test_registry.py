import unittest

import torch

from cifar10_lab import (
    DataConfig,
    ExperimentConfig,
    MODEL_REGISTRY,
    TrainConfig,
    create_model,
)


class RegistryTests(unittest.TestCase):
    def test_every_registered_factory_exists(self):
        self.assertGreaterEqual(len(MODEL_REGISTRY.model_ids()), 1)
        self.assertIn("resnet18", MODEL_REGISTRY.model_ids(cifar10_ready_only=True))

    def test_registered_model_has_expected_output_shape(self):
        model = create_model("resnet18", num_classes=10, image_size=32).eval()
        with torch.no_grad():
            output = model(torch.randn(1, 3, 32, 32))
        self.assertEqual(tuple(output.shape), (1, 10))

    def test_unsupported_cifar_model_is_guarded(self):
        with self.assertRaises(ValueError):
            create_model("inception_v3", num_classes=10, image_size=32)

    def test_unknown_model_has_clear_error(self):
        with self.assertRaises(KeyError):
            create_model("not_a_model")


class ConfigTests(unittest.TestCase):
    def test_nested_config_serializes(self):
        config = ExperimentConfig(
            model_id="resnet18",
            data=DataConfig(batch_size=32, val_ratio=0.2, seed=7),
            train=TrainConfig(epochs=2, learning_rate=0.01),
        )
        serialized = config.to_dict()
        self.assertEqual(serialized["model_id"], "resnet18")
        self.assertEqual(serialized["data"]["batch_size"], 32)
        self.assertEqual(serialized["train"]["epochs"], 2)

    def test_invalid_config_is_rejected(self):
        with self.assertRaises(ValueError):
            DataConfig(val_ratio=1.0)
        with self.assertRaises(ValueError):
            TrainConfig(epochs=0)


if __name__ == "__main__":
    unittest.main()
