import unittest

import torch

from cifar10_lab.foundations import (
    ClassicPerceptron,
    evaluate_linear_boundary,
    make_logic_gate,
    train_tiny_mlp,
)


class FoundationsTests(unittest.TestCase):
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
