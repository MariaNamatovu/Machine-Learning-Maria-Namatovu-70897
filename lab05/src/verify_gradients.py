"""Verify the scalar engine against PyTorch and central finite differences."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from engine import Value

TRIPLES = [(-4.0, 2.0, 1.0), (1.5, -2.0, 4.0), (0.5, 1.2, -0.1)]


def engine_loss(values: list[tuple[Value, Value, Value]]) -> Value:
    total = Value(0.0)
    for a, b, c in values:
        total = total + ((a * b + c) ** 2).relu()
    return total


def scalar_loss(x: np.ndarray) -> float:
    total = 0.0
    for i in range(len(TRIPLES)):
        a, b, c = x[3 * i : 3 * i + 3]
        total += max((a * b + c) ** 2, 0.0)
    return total


def main() -> None:
    engine_values = [tuple(Value(v) for v in triple) for triple in TRIPLES]
    engine_output = engine_loss(engine_values)
    engine_output.backward()
    engine_grads = np.array([node.grad for triple in engine_values for node in triple])

    torch_values = [
        tuple(torch.tensor(v, dtype=torch.float64, requires_grad=True) for v in triple)
        for triple in TRIPLES
    ]
    torch_output = sum((a * b + c).pow(2).relu() for a, b, c in torch_values)
    torch_output.backward()
    torch_grads = np.array([node.grad.item() for triple in torch_values for node in triple])

    x = np.array([v for triple in TRIPLES for v in triple], dtype=float)
    numerical_grads = np.zeros_like(x)
    h = 1e-6
    for i in range(len(x)):
        plus = x.copy()
        minus = x.copy()
        plus[i] += h
        minus[i] -= h
        numerical_grads[i] = (scalar_loss(plus) - scalar_loss(minus)) / (2.0 * h)

    engine_matrix = engine_grads.reshape(-1, 3)
    torch_matrix = torch_grads.reshape(-1, 3)
    numerical_matrix = numerical_grads.reshape(-1, 3)
    engine_vs_torch = float(np.max(np.abs(engine_grads - torch_grads)))
    engine_vs_numerical = float(np.max(np.abs(engine_grads - numerical_grads)))

    lines = [
        "Scalar Value gradient verification",
        f"triples={TRIPLES}",
        f"engine loss={engine_output.data:.12f}",
        "columns: da, db, dc",
        f"engine gradients=\n{engine_matrix}",
        f"PyTorch gradients=\n{torch_matrix}",
        f"central finite-difference gradients (h={h})=\n{numerical_matrix}",
        f"max abs diff (engine vs PyTorch)={engine_vs_torch:.12e}",
        f"max abs diff (engine vs numerical)={engine_vs_numerical:.12e}",
        f"pass (<1e-6)={engine_vs_torch < 1e-6 and engine_vs_numerical < 1e-6}",
    ]
    output = "\n".join(lines) + "\n"
    print(output, end="")
    result_path = Path(__file__).resolve().parents[1] / "results" / "gradient_verification.txt"
    result_path.write_text(output, encoding="utf-8")


if __name__ == "__main__":
    main()

