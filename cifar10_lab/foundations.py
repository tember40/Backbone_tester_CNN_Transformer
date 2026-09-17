"""Small, inspectable learning algorithms used by the foundations notebook."""

from dataclasses import dataclass

import torch
import torch.nn as nn


LOGIC_GATES = {
    "AND": (0, 0, 0, 1),
    "OR": (0, 1, 1, 1),
    "NAND": (1, 1, 1, 0),
    "XOR": (0, 1, 1, 0),
}


def make_logic_gate(name="AND"):
    """Return the four binary inputs and targets for a named logic gate."""
    normalized_name = name.upper()
    if normalized_name not in LOGIC_GATES:
        available = ", ".join(LOGIC_GATES)
        raise ValueError(f"Unknown logic gate {name!r}. Choose one of: {available}.")
    features = torch.tensor(
        [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]],
        dtype=torch.float32,
    )
    targets = torch.tensor(LOGIC_GATES[normalized_name], dtype=torch.float32)
    return features, targets


@dataclass(frozen=True)
class PerceptronSnapshot:
    step: int
    epoch: int
    sample_index: int
    weights: torch.Tensor
    bias: float
    prediction: int
    target: int
    error: int


class ClassicPerceptron:
    """The textbook binary perceptron with a hard step activation."""

    def __init__(self, input_features=2, learning_rate=0.1):
        if input_features < 1:
            raise ValueError("input_features must be at least 1.")
        if learning_rate <= 0:
            raise ValueError("learning_rate must be greater than zero.")
        self.learning_rate = learning_rate
        self.weights = torch.zeros(input_features, dtype=torch.float32)
        self.bias = 0.0
        self.history = []

    def score(self, features):
        return features @ self.weights + self.bias

    def predict(self, features):
        return (self.score(features) >= 0).to(torch.long)

    def fit(self, features, targets, epochs=10):
        if features.ndim != 2 or features.shape[1] != len(self.weights):
            raise ValueError("features must have shape [samples, input_features].")
        if len(features) != len(targets):
            raise ValueError("features and targets must contain the same number of samples.")
        if epochs < 1:
            raise ValueError("epochs must be at least 1.")

        self.history = []
        step = 0
        for epoch in range(1, epochs + 1):
            for sample_index, (sample, target) in enumerate(zip(features, targets)):
                prediction = int(self.predict(sample).item())
                target_value = int(target.item())
                error = target_value - prediction
                self.weights += self.learning_rate * error * sample
                self.bias += self.learning_rate * error
                self.history.append(
                    PerceptronSnapshot(
                        step=step,
                        epoch=epoch,
                        sample_index=sample_index,
                        weights=self.weights.clone(),
                        bias=float(self.bias),
                        prediction=prediction,
                        target=target_value,
                        error=error,
                    )
                )
                step += 1
        return self.history


class TinyMLP(nn.Module):
    """A tiny nonlinear network whose two hidden units can solve XOR."""

    def __init__(self, hidden_features=4):
        super().__init__()
        self.hidden = nn.Linear(2, hidden_features)
        self.activation = nn.Tanh()
        self.output = nn.Linear(hidden_features, 1)

    def hidden_representation(self, features):
        return self.activation(self.hidden(features))

    def forward(self, features):
        hidden = self.hidden_representation(features)
        return self.output(hidden).squeeze(-1)


def train_tiny_mlp(features, targets, epochs=500, learning_rate=0.05, seed=42):
    """Train the XOR demonstration model and return the model and learning history."""
    torch.manual_seed(seed)
    model = TinyMLP()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.BCEWithLogitsLoss()
    history = {"loss": [], "accuracy": []}

    for _ in range(epochs):
        optimizer.zero_grad()
        logits = model(features)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()

        predictions = (torch.sigmoid(logits) >= 0.5).to(targets.dtype)
        accuracy = (predictions == targets).float().mean().item() * 100.0
        history["loss"].append(loss.item())
        history["accuracy"].append(accuracy)
    return model, history
