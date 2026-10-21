"""Numerical references and actual workflow checks for the final two lessons."""

import contextlib
import io
import math
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from backbone.ViT import Attention
from cifar10_lab import engine
from cifar10_lab.checkpoints import checkpoint_path, torch_load_compatible
from cifar10_lab.cli import build_parser, _create_experiment
from cifar10_lab.registry import list_models
from test_static_site import DOCS, ROOT, parse_page


class SynthesisLessonTests(unittest.TestCase):
    def test_remote_input_comparison_matches_grouped_conv_and_actual_attention(self):
        layer = Attention(2, heads=1, dim_head=2).double().eval()
        convolution = nn.Conv1d(2, 2, 3, groups=2, bias=False).double()
        with torch.no_grad():
            layer.to_qkv.weight.copy_(torch.eye(2).repeat(3, 1))
            convolution.weight.copy_(torch.tensor([.25, .5, .25]).reshape(1, 1, 3).expand(2, 1, 3))
        cases = [(-2, [.4251946838417307, .7663259087188861]),
                 (1, [.6169214179501338, .6169214179501338]),
                 (2, [1.053904690937801, .5024838289508026])]
        for far_value, expected in cases:
            x = torch.tensor([[[far_value, 0], [0, 1], [1, 1], [-1, 0], [0, -1]]], dtype=torch.float64)
            self.assertTrue(torch.allclose(layer(x)[0, 2], torch.tensor(expected, dtype=torch.float64), atol=1e-12))
            self.assertTrue(torch.equal(convolution(x.transpose(1, 2))[0, :, 1], torch.tensor([.25, .75], dtype=torch.float64)))

    def test_confusion_cases_match_real_engine_row_orientation_and_support(self):
        cases = [([[8, 2, 0], [1, 7, 2], [0, 3, 7]], 22 / 30 * 100, [80, 70, 70]),
                 ([[90, 0, 0], [5, 0, 0], [5, 0, 0]], 90, [100, 0, 0]),
                 ([[8, 2, 0], [1, 9, 0], [0, 0, 0]], 85, [80, 90, 0])]
        for matrix, accuracy, recalls in cases:
            labels, predictions = [], []
            for label, row in enumerate(matrix):
                for prediction, count in enumerate(row):
                    labels.extend([label] * count)
                    predictions.extend([prediction] * count)
            logits = nn.functional.one_hot(torch.tensor(predictions), 3).float()
            loader = DataLoader(TensorDataset(logits, torch.tensor(labels)), batch_size=13)
            result = engine.evaluate_model_detailed(nn.Identity(), loader, torch.device('cpu'), num_classes=3)
            self.assertEqual(result['confusion_matrix'].tolist(), matrix)
            self.assertAlmostEqual(result['accuracy'], accuracy)
            self.assertTrue(torch.allclose(result['per_class_accuracy'], torch.tensor(recalls).float()))

    def test_best_checkpoint_uses_accuracy_first_tie_not_minimum_loss(self):
        loader = DataLoader(TensorDataset(torch.tensor([[1., 2.], [2., 1.]]), torch.tensor([0, 1])), batch_size=2)
        model = nn.Linear(2, 2)
        evaluations = [{'accuracy':a, 'loss':loss} for a, loss in [(10, 2.), (20, 1.), (20, .5)]]
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            with patch.object(engine, 'run_evaluation', side_effect=evaluations):
                history = engine.train_model(model, loader, loader, torch.device('cpu'), epochs=3,
                                             model_id='lesson_fixture', weight_dir=directory)
            stored = torch_load_compatible(checkpoint_path('lesson_fixture', directory), 'cpu')
        self.assertEqual(stored['epoch'], 2)
        self.assertEqual(stored['completed_epochs'], 3)
        self.assertEqual(history['val_loss'], [2., 1., .5])
        self.assertTrue(all(torch.equal(model.state_dict()[key], value) for key, value in stored['model_state'].items()))

    def test_uniform_loss_partial_macs_and_patch_scaling_are_exact(self):
        loss = nn.CrossEntropyLoss()(torch.zeros(2, 10), torch.tensor([1, 9]))
        self.assertAlmostEqual(loss.item(), math.log(10), places=6)
        self.assertEqual(9 * 192 ** 2, 331776)
        self.assertEqual(8 ** 2 * 9 * 192 ** 2, 21233664)
        self.assertEqual(2 * 65 ** 2 * 192, 1622400)
        self.assertAlmostEqual(257 ** 2 / 65 ** 2, 15.632899408284023)

    def test_beginner_command_parses_to_documented_quick_cpu_settings(self):
        args = build_parser().parse_args(['train', '--model', 'perceptron', '--quick', '--device', 'cpu', '--num-workers', '0', '--save-plots'])
        config = _create_experiment(args, include_training=True)
        self.assertEqual((config.train.epochs, config.image_size, config.device), (1, 32, 'cpu'))
        self.assertEqual((config.data.max_train_samples, config.data.max_val_samples, config.data.max_test_samples), (2048, 512, 512))
        self.assertEqual(config.data.num_workers, 0)

    def test_all_eighteen_chapters_are_in_curriculum_and_redirects_with_pagination(self):
        pages = sorted((DOCS / 'chapters').glob('[0-9][0-9]-*.html'))
        self.assertEqual([int(p.name[:2]) for p in pages], list(range(1, 19)))
        home = parse_page(DOCS / 'index.html')
        for i, page in enumerate(pages):
            self.assertIn(f'chapters/{page.name}', home.links)
            if i:
                self.assertIn(pages[i - 1].name, parse_page(page).links)
            if i < 17:
                self.assertIn(pages[i + 1].name, parse_page(page).links)
            redirect = (ROOT / 'chapters' / page.name).read_text(encoding='utf-8')
            self.assertIn(f'../docs/chapters/{page.name}', redirect)
            self.assertIn('target.hash = window.location.hash', redirect)
        ready = list_models(cifar10_ready_only=True)
        home_text = (DOCS / 'index.html').read_text(encoding='utf-8')
        for count, label in ((len(ready), '실행 가능한 모델'), (len({m.family for m in ready}), '비교 모델 계열'), (18, '전체 단원')):
            self.assertIn(f'<dt>{count}</dt><dd>{label}</dd>', home_text)


if __name__ == '__main__':
    unittest.main()
