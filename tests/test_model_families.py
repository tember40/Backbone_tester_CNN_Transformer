import unittest

import torch

from cifar10_lab.cnn_visualization import (
    collect_feature_maps,
    receptive_field_stages,
    summarize_activations,
    trace_pooling,
    trace_single_channel_convolution,
)
from cifar10_lab.model_validation import validate_registered_models
from cifar10_lab.registry import create_model


class ModelFamilyTests(unittest.TestCase):
    def test_unknown_model_validation_is_reported_without_crashing(self):
        result = validate_registered_models(["does-not-exist"])[0]

        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["family"], "unknown")
        self.assertIn("Unknown model_id", result["error"])

    def test_representative_from_each_family_runs_forward(self):
        representatives = (
            "perceptron",
            "mlp",
            "alexnet",
            "vgg11",
            "resnet18",
            "mobilenet_v2",
            "efficientnet_b0",
            "convnext_tiny",
            "se_resnet18",
            "vit_tiny",
            "pvt_tiny",
        )
        results = validate_registered_models(representatives)
        failures = [result for result in results if result["status"] != "ok"]
        self.assertEqual(failures, [])

    def test_alexnet_exposes_feature_maps_and_receptive_fields(self):
        model = create_model("alexnet", num_classes=10, image_size=32).eval()
        image = torch.randn(1, 3, 32, 32)
        records = collect_feature_maps(model, image, max_layers=4)
        stages = receptive_field_stages(model)

        self.assertEqual(len(records), 4)
        self.assertGreater(len(stages), 4)
        self.assertGreater(stages[-1]["receptive_field"], stages[0]["receptive_field"])

    def test_convolution_trace_matches_expected_edge_map(self):
        image = torch.tensor(
            [
                [0, 0, 1, 1],
                [0, 0, 1, 1],
                [0, 0, 1, 1],
                [0, 0, 1, 1],
            ],
            dtype=torch.float32,
        )
        kernel = torch.tensor(
            [[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=torch.float32
        )

        output, steps = trace_single_channel_convolution(image, kernel)

        self.assertEqual(tuple(output.shape), (2, 2))
        self.assertTrue(torch.equal(output, torch.tensor([[3.0, 3.0], [3.0, 3.0]])))
        self.assertEqual(len(steps), 4)
        self.assertEqual(steps[0]["output_position"], (0, 0))
        self.assertEqual(steps[0]["sum"], 3.0)

    def test_activation_summary_reports_relu_sparsity(self):
        features = torch.tensor([[[-2.0, 0.0], [1.0, 3.0]]])

        before = summarize_activations(features)
        after = summarize_activations(torch.relu(features))

        self.assertEqual(before["shape"], (1, 2, 2))
        self.assertEqual(before["minimum"], -2.0)
        self.assertEqual(before["positive_ratio"], 0.5)
        self.assertEqual(after["minimum"], 0.0)
        self.assertEqual(after["zero_ratio"], 0.5)

    def test_pooling_trace_keeps_each_window_and_summary(self):
        features = torch.tensor(
            [[1.0, 3.0, 2.0, 4.0], [5.0, 0.0, 7.0, 1.0],
             [2.0, 6.0, 8.0, 3.0], [4.0, 1.0, 5.0, 9.0]]
        )

        maximums, max_steps = trace_pooling(features, mode="max")
        averages, average_steps = trace_pooling(features, mode="average")

        self.assertTrue(torch.equal(maximums, torch.tensor([[5.0, 7.0], [6.0, 9.0]])))
        self.assertTrue(torch.equal(averages, torch.tensor([[2.25, 3.5], [3.25, 6.25]])))
        self.assertEqual(max_steps[2]["input_position"], (2, 0))
        self.assertEqual(max_steps[2]["value"], 6.0)
        self.assertEqual(average_steps[3]["output_position"], (1, 1))

    def test_alexnet_is_adapted_for_cifar10_geometry(self):
        model = create_model("alexnet", num_classes=10, image_size=32)
        first_convolution = model.features[0]
        pooling_layers = [
            module for module in model.features if isinstance(module, torch.nn.MaxPool2d)
        ]

        self.assertEqual(first_convolution.kernel_size, (3, 3))
        self.assertEqual(first_convolution.stride, (1, 1))
        self.assertEqual(first_convolution.padding, (1, 1))
        self.assertEqual(len(pooling_layers), 3)
        self.assertEqual(model.avgpool.output_size, (4, 4))
        self.assertEqual(model.classifier[0].in_features, 256 * 4 * 4)
        self.assertEqual(model.classifier[-1].out_features, 10)


if __name__ == "__main__":
    unittest.main()
