import unittest

import torch

from cifar10_lab.foundations import ClassicPerceptron, make_logic_gate, train_tiny_mlp


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

    def test_unknown_logic_gate_has_clear_error(self):
        with self.assertRaises(ValueError):
            make_logic_gate("UNKNOWN")


if __name__ == "__main__":
    unittest.main()
