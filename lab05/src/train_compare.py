"""Compare a scalar-Value MLP with an equivalent PyTorch MLP on make_moons."""

from __future__ import annotations

from pathlib import Path
import sys
import time
from statistics import median

import numpy as np
import torch
from sklearn.datasets import make_moons
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from engine import Value

RUNS = 5
EPOCHS = 30
LEARNING_RATE = 0.1


class EngineMLP:
    def __init__(self, rng: np.random.Generator) -> None:
        scale1 = np.sqrt(2.0 / 2.0)
        scale2 = np.sqrt(2.0 / 8.0)
        self.w1 = [[Value(float(rng.normal(0.0, scale1))) for _ in range(2)] for _ in range(8)]
        self.b1 = [Value(0.0) for _ in range(8)]
        self.w2 = [Value(float(rng.normal(0.0, scale2))) for _ in range(8)]
        self.b2 = Value(0.0)
        self.parameters = [*sum(self.w1, []), *self.b1, *self.w2, self.b2]

    def __call__(self, features: np.ndarray) -> Value:
        hidden = []
        for weights, bias in zip(self.w1, self.b1):
            activation = bias
            for weight, feature in zip(weights, features):
                activation = activation + weight * float(feature)
            hidden.append(activation.relu())
        output = self.b2
        for weight, activation in zip(self.w2, hidden):
            output = output + weight * activation
        return 1.0 / (1.0 + (-output).exp())

    def zero_grad(self) -> None:
        for parameter in self.parameters:
            parameter.grad = 0.0

    def update(self) -> None:
        for parameter in self.parameters:
            parameter.data -= LEARNING_RATE * parameter.grad


def engine_bce(probability: Value, label: float) -> Value:
    epsilon = 1e-7
    return -(label * (probability + epsilon).log() + (1.0 - label) * (1.0 - probability + epsilon).log())


def train_engine(x: np.ndarray, y: np.ndarray, run: int) -> tuple[float, float]:
    rng = np.random.default_rng(42 + run)
    model = EngineMLP(rng)
    start = time.perf_counter()
    for _ in range(EPOCHS):
        order = rng.permutation(len(x))
        for index in order:
            model.zero_grad()
            loss = engine_bce(model(x[index]), float(y[index]))
            loss.backward()
            model.update()
    elapsed = time.perf_counter() - start
    final_loss = float(np.mean([engine_bce(model(row), float(label)).data for row, label in zip(x, y)]))
    return elapsed, final_loss


class TorchMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.network = nn.Sequential(nn.Linear(2, 8), nn.ReLU(), nn.Linear(8, 1), nn.Sigmoid())

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs)


def train_torch(x: np.ndarray, y: np.ndarray, run: int) -> tuple[float, float]:
    torch.manual_seed(42 + run)
    model = TorchMLP()
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.BCELoss()
    inputs = torch.from_numpy(x).to(dtype=torch.float32)
    labels = torch.from_numpy(y.reshape(-1, 1)).to(dtype=torch.float32)
    generator = torch.Generator().manual_seed(42 + run)
    start = time.perf_counter()
    for _ in range(EPOCHS):
        order = torch.randperm(len(inputs), generator=generator)
        for index in order:
            optimizer.zero_grad(set_to_none=True)
            probability = model(inputs[index : index + 1])
            loss = criterion(probability, labels[index : index + 1])
            loss.backward()
            optimizer.step()
    elapsed = time.perf_counter() - start
    with torch.no_grad():
        final_loss = float(criterion(model(inputs), labels).item())
    return elapsed, final_loss


def main() -> None:
    np.random.seed(42)
    torch.set_num_threads(1)
    x, y = make_moons(n_samples=200, noise=0.2, random_state=42)
    x = x.astype(np.float64)
    y = y.astype(np.float64)
    engine_runs = [train_engine(x, y, run) for run in range(RUNS)]
    torch_runs = [train_torch(x, y, run) for run in range(RUNS)]
    engine_times = [item[0] for item in engine_runs]
    torch_times = [item[0] for item in torch_runs]
    engine_median = float(median(engine_times))
    torch_median = float(median(torch_times))
    ratio = engine_median / torch_median
    lines = [
        "make_moons 2->8->1 training timing",
        "data: n_samples=200, noise=0.2, random_state=42",
        f"epochs={EPOCHS}, learning_rate={LEARNING_RATE}, runs={RUNS}, optimizer=plain SGD",
        "loss=BCE on sigmoid probabilities; each sample is one SGD update",
        f"engine run times (s)={[round(v, 6) for v in engine_times]}",
        f"PyTorch run times (s)={[round(v, 6) for v in torch_times]}",
        f"engine median (s)={engine_median:.9f}",
        f"PyTorch median (s)={torch_median:.9f}",
        f"engine/PyTorch median ratio={ratio:.3f}x",
        f"engine final losses={[round(item[1], 6) for item in engine_runs]}",
        f"PyTorch final losses={[round(item[1], 6) for item in torch_runs]}",
        f"engine median final loss={median([item[1] for item in engine_runs]):.9f}",
        f"PyTorch median final loss={median([item[1] for item in torch_runs]):.9f}",
    ]
    output = "\n".join(lines) + "\n"
    print(output, end="")
    result_path = Path(__file__).resolve().parents[1] / "results" / "training_timing.txt"
    result_path.write_text(output, encoding="utf-8")


if __name__ == "__main__":
    main()
