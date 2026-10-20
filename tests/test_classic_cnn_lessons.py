"""Verify lesson numbers against model topology without allocating large weights."""

import re
import unittest
from pathlib import Path

import torch
from torch import nn

from backbone import AlexNet, ResNet, VGG
from cifar10_lab.cnn_visualization import receptive_field_stages


ROOT = Path(__file__).resolve().parents[1]


class ClassicCNNLessonTests(unittest.TestCase):
    def test_alexnet_shapes_parameter_groups_and_final_receptive_extent(self):
        with torch.device('meta'):
            model = AlexNet.alexnet(init_weights=False).eval()
            inputs = torch.empty(2, 3, 32, 32)
            features = model.features(inputs)
            self.assertEqual(tuple(features.shape), (2, 256, 4, 4))
            self.assertEqual(tuple(model(inputs).shape), (2, 10))
        source = (ROOT / 'docs/assets/js/alexnet.js').read_text(encoding='utf-8')
        listed = [int(x) for x in re.findall(r'label: "(?:Conv\d|Linear \d)", value: (\d+)', source)]
        modules = [m for m in model.modules() if isinstance(m, (nn.Conv2d, nn.Linear))]
        self.assertEqual(listed, [sum(p.numel() for p in m.parameters()) for m in modules])
        self.assertEqual(sum(listed), 36051786)
        stages = receptive_field_stages(model.features)
        extent = stages[-1]['receptive_field'] + (features.shape[-1] - 1) * stages[-1]['jump']
        self.assertEqual(extent, 66)
        self.assertEqual(model.classifier[0].in_features, features[0].numel())
        self.assertRegex(source, rf'type: "classifier"[^\n]+receptive: {extent}')

    def test_all_four_vgg_variants_match_the_page_counts_and_spatial_trace(self):
        source = (ROOT / 'docs/assets/js/vgg.js').read_text(encoding='utf-8')
        for name in ('vgg11', 'vgg13', 'vgg16', 'vgg19'):
            with self.subTest(model=name), torch.device('meta'):
                model = getattr(VGG, name)(init_weights=False).eval()
                listed = int(re.search(rf'{name}:.*?parameters:\s*(\d+)', source).group(1))
                self.assertEqual(sum(p.numel() for p in model.parameters()), listed)
                inputs = torch.empty(2, 3, 32, 32)
                trace = []
                for layer in model.features:
                    inputs = layer(inputs)
                    if isinstance(layer, nn.MaxPool2d):
                        trace.append(tuple(inputs.shape[1:]))
                self.assertEqual(trace, [(64, 16, 16), (128, 8, 8), (256, 4, 4),
                                         (512, 2, 2), (512, 1, 1)])
                self.assertEqual(sum(p.numel() for p in model.classifier.parameters()), 18923530)
                self.assertEqual(tuple(model(torch.empty(2, 3, 32, 32)).shape), (2, 10))

    def test_resnet_variants_and_extension_counts_match_real_models(self):
        source = (ROOT / 'docs/assets/js/resnet.js').read_text(encoding='utf-8')
        for name in ('resnet18', 'resnet34', 'resnet50', 'resnext50_32x4d', 'wide_resnet50_2'):
            with self.subTest(model=name), torch.device('meta'):
                model = getattr(ResNet, name)().eval()
                listed = int(re.search(rf'{name}:.*?parameters:\s*(\d+)', source, re.S).group(1))
                self.assertEqual(sum(p.numel() for p in model.parameters()), listed)
                inputs = model.maxpool(model.relu(model.bn1(model.conv1(torch.empty(2, 3, 32, 32)))))
                self.assertEqual(tuple(inputs.shape), (2, 64, 8, 8))
                expansion = model.layer1[0].expansion
                for index in range(1, 5):
                    inputs = getattr(model, f'layer{index}')(inputs)
                    self.assertEqual(tuple(inputs.shape[1:]),
                                     (64 * 2 ** (index - 1) * expansion, 8 // 2 ** (index - 1),
                                      8 // 2 ** (index - 1)))
                self.assertEqual(tuple(model(torch.empty(2, 3, 32, 32)).shape), (2, 10))

    def test_projection_matches_both_paths_even_when_only_channels_change(self):
        with torch.device('meta'):
            basic = ResNet.resnet18().eval().layer2[0]
            inputs = torch.empty(2, 64, 8, 8)
            main = basic.bn2(basic.conv2(basic.relu(basic.bn1(basic.conv1(inputs)))))
            self.assertEqual(tuple(main.shape), (2, 128, 4, 4))
            self.assertEqual(main.shape, basic.downsample(inputs).shape)
            bottleneck = ResNet.resnet50().eval().layer1[0]
            self.assertIsNotNone(bottleneck.downsample)
            self.assertEqual(tuple(bottleneck.downsample(inputs).shape), (2, 256, 8, 8))
            self.assertEqual(tuple(bottleneck(inputs).shape), (2, 256, 8, 8))

    def test_residual_chain_rule_examples_match_autograd(self):
        for x, coefficient, expected_y, expected_gradient in [(1.5, 0.2, 1.8, 2.4), (1., -2., 0., 0.)]:
            with self.subTest(coefficient=coefficient):
                inputs = torch.tensor(x, dtype=torch.float64, requires_grad=True)
                output = torch.relu(inputs + coefficient * inputs)
                output.backward(torch.tensor(2., dtype=torch.float64))
                self.assertAlmostEqual(output.item(), expected_y)
                self.assertAlmostEqual(inputs.grad.item(), expected_gradient)
        plain = torch.tensor(1.5, requires_grad=True)
        torch.relu(0.2 * plain).backward(torch.tensor(2.))
        self.assertAlmostEqual(plain.grad.item(), 0.4)

    def test_dropout_train_scaling_eval_identity_and_no_grad_are_distinct(self):
        inputs = torch.arange(1., 5.).repeat(64)
        dropout = nn.Dropout(0.5).train()
        with torch.no_grad():
            output = dropout(inputs)
        self.assertTrue(torch.all((output == 0) | (output == inputs * 2)))
        self.assertTrue(torch.any(output == 0))
        self.assertTrue(torch.any(output != 0))
        dropout.eval()
        self.assertTrue(torch.equal(dropout(inputs), inputs))
        self.assertEqual((0.5 * 0) + (0.5 * 6), 3)


if __name__ == '__main__':
    unittest.main()
