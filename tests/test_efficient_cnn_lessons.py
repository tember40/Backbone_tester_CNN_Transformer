"""Numerical and topology checks for channel gating and efficient CNN lessons."""

import re
import unittest
from pathlib import Path

import torch
from torch import nn

from backbone import EfficientNet, MobileNet, SeNet


ROOT = Path(__file__).resolve().parents[1]


class EfficientCNNLessonTests(unittest.TestCase):
    def test_fixed_se_mlp_and_signed_scaling_match_actual_layer(self):
        layer = SeNet.SELayer(3, reduction=3).double()
        with torch.no_grad():
            layer.fc[0].weight.copy_(torch.tensor([[0.5, -0.5, 0.25]]))
            layer.fc[2].weight.copy_(torch.tensor([[1.], [0.5], [-1.]]))
        first = torch.tensor([[[1., 3.], [5., 7.]], [[2., 2.], [4., 4.]],
                              [[8., 4.], [0., 0.]]], dtype=torch.float64)
        inputs = torch.stack((first, torch.zeros_like(first)))
        summary = layer.avg_pool(inputs).flatten(1)
        self.assertTrue(torch.equal(summary[0], torch.tensor([4., 3., 3.], dtype=torch.float64)))
        hidden = layer.fc[1](layer.fc[0](summary))
        self.assertEqual(hidden[0].item(), 1.25)
        gates = layer.fc(summary)
        expected = torch.tensor([0.7772998611746911, 0.6513548646660542, 0.22270013882530884],
                                dtype=torch.float64)
        self.assertTrue(torch.allclose(gates[0], expected))
        self.assertTrue(torch.equal(gates[1], torch.full((3,), 0.5, dtype=torch.float64)))
        self.assertTrue(torch.allclose(layer(inputs), inputs * gates[:, :, None, None]))
        inputs[0, 0].fill_(-9)
        self.assertEqual(layer(inputs)[0, 0, 0, 0].item(), -4.5)

    def test_se_model_parameter_totals_and_added_parameters(self):
        source = (ROOT / 'docs/chapters/10-senet.html').read_text(encoding='utf-8')
        for depth in (18, 50):
            with self.subTest(depth=depth), torch.device('meta'):
                model = getattr(SeNet, f'se_resnet{depth}')().eval()
                row = re.search(rf'<th>{depth}층</th><td>([\d,]+)개</td><td>([\d,]+)개', source)
                base, total = (int(value.replace(',', '')) for value in row.groups())
                self.assertEqual(sum(p.numel() for p in model.parameters()), total)
                se_params = sum(p.numel() for m in model.modules() if isinstance(m, SeNet.SELayer)
                                for p in m.parameters())
                self.assertEqual(se_params, total - base)
                self.assertEqual(tuple(model(torch.empty(2, 3, 32, 32)).shape), (2, 10))

    def test_depthwise_is_channel_local_and_pointwise_combines_channels(self):
        spatial = nn.Conv2d(3, 3, 3, groups=3, bias=False)
        mixing = nn.Conv2d(3, 2, 1, bias=False)
        with torch.no_grad():
            spatial.weight.zero_()
            spatial.weight[:, 0, 1, 1].copy_(torch.tensor([1., 2., 3.]))
            mixing.weight.copy_(torch.tensor([[1., -2., 0.], [0.5, 0., 1.]])[:, :, None, None])
        inputs = torch.zeros(1, 3, 3, 3)
        inputs[0, :, 1, 1] = torch.tensor([2., 0.5, 1.])
        separated = spatial(inputs)
        self.assertTrue(torch.equal(separated.flatten(), torch.tensor([2., 1., 3.])))
        self.assertTrue(torch.equal(mixing(separated).flatten(), torch.tensor([0., 4.])))
        self.assertEqual(tuple(spatial.weight.shape), (3, 1, 3, 3))
        self.assertEqual(tuple(mixing.weight.shape), (2, 3, 1, 1))
        changed = inputs.clone()
        changed[:, 0] += 1
        self.assertTrue(torch.equal(spatial(changed)[:, 1:], separated[:, 1:]))

    def test_inverted_block_weights_strides_and_activation_order(self):
        with torch.device('meta'):
            for stride in (1, 2):
                block = MobileNet.InvertedResidual(24, 24, stride, 6).eval()
                inputs = torch.empty(2, 24, 8, 8)
                traces = []
                for part in block.conv:
                    inputs = part(inputs)
                    traces.append(tuple(inputs.shape[1:]))
                self.assertEqual(traces, [(144, 8, 8), (144, 8 // stride, 8 // stride),
                                          (24, 8 // stride, 8 // stride), (24, 8 // stride, 8 // stride)])
                self.assertEqual(sum(p.numel() for p in block.parameters()), 8832)
                self.assertEqual(sum(m.weight.numel() for m in block.modules() if isinstance(m, nn.Conv2d)), 8208)
                self.assertIsInstance(block.conv[-1], nn.BatchNorm2d)
                self.assertEqual(block.use_res_connect, stride == 1)
            block = MobileNet.InvertedResidual(16, 24, 1, 1)
            self.assertEqual(len(block.conv), 3)
            self.assertFalse(block.use_res_connect)

    def test_mobile_widths_match_page_totals_and_rounding_example(self):
        source = (ROOT / 'docs/assets/js/mobilenetv2.js').read_text(encoding='utf-8')
        totals = dict(re.findall(r'"(0\.5|0\.75|1)":(\d+)', source))
        for width, count in totals.items():
            with self.subTest(width=width), torch.device('meta'):
                model = MobileNet.mobilenet_v2(width_mult=float(width)).eval()
                self.assertEqual(sum(p.numel() for p in model.parameters()), int(count))
                self.assertEqual(model.last_channel, 1280)
                self.assertEqual(tuple(model.features(torch.empty(2, 3, 32, 32)).shape), (2, 1280, 1, 1))
        self.assertEqual(len(totals), 3)
        self.assertEqual(MobileNet._make_divisible(24 * 0.75, 8), 24)

    def test_efficientnet_se_shapes_and_bias_counts_match_worked_example(self):
        with torch.device('meta'):
            block = EfficientNet.InvertedResidual(16, 24, 2, 6, 3, True,
                                                  se_layer=EfficientNet.SqueezeExcitation).eval()
            inputs = torch.empty(2, 16, 16, 16)
            shapes = []
            for part in block.conv:
                inputs = part(inputs)
                shapes.append(tuple(inputs.shape[1:]))
            self.assertEqual(shapes, [(96, 16, 16), (96, 8, 8), (96, 8, 8),
                                      (24, 8, 8), (24, 8, 8)])
            gate = block.conv[2]
            self.assertEqual(gate.fc1.out_channels, 8)
            self.assertEqual(sum(p.numel() for p in gate.parameters()), 1640)
            self.assertFalse(block.use_res_connect)
        self.assertTrue(torch.equal(nn.Hardsigmoid()(torch.tensor([-4., -3., 0., 3., 4.])),
                                    torch.tensor([0., 0., 0.5, 1., 1.])))

    def test_efficientnet_presets_match_page_counts_and_integer_repeats(self):
        source = (ROOT / 'docs/assets/js/efficientnet.js').read_text(encoding='utf-8')
        for name in ('b0', 'b1', 'b2', 'b3'):
            with self.subTest(model=name), torch.device('meta'):
                model = getattr(EfficientNet, f'efficientnet_{name}')().eval()
                listed = int(re.search(rf'{name}:.*?parameters:\s*(\d+)', source).group(1))
                self.assertEqual(sum(p.numel() for p in model.parameters()), listed)
                self.assertEqual(tuple(model(torch.empty(2, 3, 32, 32)).shape), (2, 10))
                self.assertEqual(model.features(torch.empty(2, 3, 32, 32)).shape[-2:], (1, 1))
                if name == 'b1':
                    self.assertEqual(len(model.features) - 2, 23)
        self.assertEqual(EfficientNet._make_divisible(16 * 1.1, 8), 16)


if __name__ == '__main__':
    unittest.main()
