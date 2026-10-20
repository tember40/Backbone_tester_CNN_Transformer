import unittest

import torch
from torch import nn

from backbone.AlexNet import alexnet_cfg, make_layers
from cifar10_lab.cnn_visualization import collect_feature_maps, receptive_field_stages


class CNNLessonTests(unittest.TestCase):
    def test_rgb_worked_sum_and_padding_shapes_match_conv2d(self):
        patches = torch.tensor([[[[1., 2.], [3., 4.]],
                                 [[0., 1.], [2., 0.]],
                                 [[1., 1.], [1., 1.]]]])
        kernels = torch.tensor([[[[1., 0.], [0., -1.]],
                                 [[0., 1.], [1., 0.]],
                                 [[1., 0.], [0., 1.]]]])
        result = torch.nn.functional.conv2d(patches, kernels, torch.tensor([-0.5]))
        self.assertEqual(result.item(), 1.5)
        images = torch.zeros(2, 3, 32, 32)
        self.assertEqual(tuple(nn.Conv2d(3, 64, 3)(images).shape), (2, 64, 30, 30))
        self.assertEqual(tuple(nn.Conv2d(3, 64, 3, padding=1)(images).shape), (2, 64, 32, 32))

    def test_first_pooling_window_backward_matches_worked_example(self):
        for average, expected in [(False, [[0., 0.], [1., 0.]]),
                                  (True, [[0.25, 0.25], [0.25, 0.25]])]:
            with self.subTest(average=average):
                inputs = torch.tensor([[[[1., 3.], [5., 0.]]]], requires_grad=True)
                output = (nn.AvgPool2d(2) if average else nn.MaxPool2d(2))(inputs)
                output.backward()
                self.assertTrue(torch.equal(inputs.grad[0, 0], torch.tensor(expected)))
                self.assertEqual(output.item(), 2.25 if average else 5.)

    def test_alexnet_receptive_fields_match_the_actual_feature_layers(self):
        features = make_layers(alexnet_cfg)
        stages = receptive_field_stages(features)
        self.assertEqual([s['receptive_field'] for s in stages], [3, 4, 12, 14, 22, 30, 38, 42])
        self.assertEqual([s['jump'] for s in stages], [1, 2, 2, 4, 4, 4, 4, 8])

    def test_feature_capture_preserves_raw_values_and_mixed_training_states(self):
        model = nn.Sequential(nn.Conv2d(1, 1, 1, bias=False), nn.ReLU(inplace=True), nn.Dropout())
        with torch.no_grad():
            model[0].weight.fill_(1)
        model.train()
        model[2].eval()
        before = [module.training for module in model.modules()]
        records = collect_feature_maps(model, torch.tensor([[[[-2., 3.]]]]))
        self.assertTrue(torch.equal(records[0]['features'], torch.tensor([[[[-2., 3.]]]])))
        self.assertEqual([module.training for module in model.modules()], before)
        self.assertEqual(len(model[0]._forward_hooks), 0)

    def test_feature_capture_restores_state_and_hooks_after_forward_failure(self):
        class Failing(nn.Module):
            def forward(self, inputs):
                raise RuntimeError('intentional lesson test failure')
        model = nn.Sequential(nn.Conv2d(1, 1, 1), Failing()).train()
        with self.assertRaisesRegex(RuntimeError, 'intentional lesson'):
            collect_feature_maps(model, torch.ones(1, 1, 2, 2))
        self.assertTrue(all(module.training for module in model.modules()))
        self.assertEqual(len(model[0]._forward_hooks), 0)


if __name__ == '__main__':
    unittest.main()
