import unittest

import torch

from cifar10_lab.foundations import (
    ClassicPerceptron,
    evaluate_linear_boundary,
    make_logic_gate,
    train_tiny_mlp,
)


class FoundationsTests(unittest.TestCase):
    def test_single_hidden_neuron_chain_rule_matches_autograd(self):
        """Independently verify the worked backprop example in chapter 3."""
        weights = torch.tensor([0.5, -0.5], dtype=torch.float64, requires_grad=True)
        bias = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
        output_weight = torch.tensor(1.0, dtype=torch.float64, requires_grad=True)
        output_bias = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
        sample = torch.tensor([1.0, 0.0], dtype=torch.float64)
        target = torch.tensor(1.0, dtype=torch.float64)
        hidden = torch.tanh(sample @ weights + bias)
        logit = hidden * output_weight + output_bias
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logit, target)
        loss.backward()
        delta_output = torch.sigmoid(logit.detach()) - target
        delta_hidden = delta_output * output_weight.detach() * (1 - hidden.detach() ** 2)
        self.assertTrue(torch.allclose(weights.grad, delta_hidden * sample))
        self.assertAlmostEqual(bias.grad.item(), delta_hidden.item(), places=12)
        self.assertAlmostEqual(output_weight.grad.item(), (delta_output * hidden).item(), places=12)
        self.assertAlmostEqual(output_bias.grad.item(), delta_output.item(), places=12)
        self.assertAlmostEqual(hidden.item(), 0.4621171573, places=9)
        self.assertAlmostEqual(loss.item(), 0.4885484392, places=9)
        with torch.no_grad():
            for parameter in (weights, bias, output_weight, output_bias):
                parameter -= 0.1 * parameter.grad
            updated_logit = output_weight * torch.tanh(sample @ weights + bias) + output_bias
            updated_loss = torch.nn.functional.binary_cross_entropy_with_logits(updated_logit, target)
        self.assertAlmostEqual(updated_loss.item(), 0.4531941133, places=9)

    def test_and_gate_is_learned_by_one_perceptron(self):
        features, targets = make_logic_gate("AND")
        model = ClassicPerceptron(learning_rate=0.1)
        history = model.fit(features, targets, epochs=10)

        self.assertGreater(len(history), 0)
        self.assertTrue(torch.equal(model.predict(features), targets.long()))

    def test_xor_gate_is_learned_by_tiny_mlp(self):
        features, targets = make_logic_gate("XOR")
        model, history = train_tiny_mlp(features, targets, epochs=500, seed=42)

        with torch.no_grad():
            predictions = (torch.sigmoid(model(features)) >= 0.5).long()
        self.assertTrue(torch.equal(predictions, targets.long()))
        self.assertEqual(history["accuracy"][-1], 100.0)

        trace = model.forward_trace(features)
        self.assertEqual(tuple(trace["hidden_linear"].shape), (4, 4))
        self.assertEqual(tuple(trace["hidden"].shape), (4, 4))
        self.assertEqual(tuple(trace["logits"].shape), (4,))
        self.assertEqual(tuple(trace["probabilities"].shape), (4,))

    def test_one_linear_boundary_separates_and_but_not_xor(self):
        features, and_targets = make_logic_gate("AND")
        _, and_correct, and_accuracy = evaluate_linear_boundary(
            features, and_targets, weights=[1.0, 1.0], bias=-1.5
        )
        self.assertTrue(and_correct.all())
        self.assertEqual(and_accuracy, 1.0)

        _, xor_targets = make_logic_gate("XOR")
        _, _, xor_accuracy = evaluate_linear_boundary(
            features, xor_targets, weights=[1.0, 1.0], bias=-0.5
        )
        self.assertLess(xor_accuracy, 1.0)

    def test_unknown_logic_gate_has_clear_error(self):
        with self.assertRaises(ValueError):
            make_logic_gate("UNKNOWN")


if __name__ == "__main__":
    unittest.main()
