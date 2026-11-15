"""Inspect PyTorch autograd, compilation, and tensor contiguity."""

from __future__ import annotations

from pathlib import Path
import time
from statistics import median

import torch
from torch import nn

WARMUPS = 5
TIMED_RUNS = 30


class SmallMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.network = nn.Sequential(nn.Linear(2, 8), nn.ReLU(), nn.Linear(8, 1))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs)


def graph_lines(function: object, depth: int = 0, maximum_depth: int = 3, seen: set[int] | None = None) -> list[str]:
    if seen is None:
        seen = set()
    if function is None or depth > maximum_depth:
        return []
    marker = id(function)
    indent = "  " * depth
    if marker in seen:
        return [f"{indent}<visited>"]
    seen.add(marker)
    name = type(function).__name__
    lines = [f"{indent}{name}"]
    next_functions = getattr(function, "next_functions", ())
    for child, _ in next_functions:
        lines.extend(graph_lines(child, depth + 1, maximum_depth, seen))
    return lines


def median_forward_time(model: nn.Module, inputs: torch.Tensor, runs: int = TIMED_RUNS) -> float:
    model.eval()
    with torch.inference_mode():
        for _ in range(WARMUPS):
            model(inputs)
        measurements = []
        for _ in range(runs):
            start = time.perf_counter()
            output = model(inputs)
            output.sum().item()
            measurements.append(time.perf_counter() - start)
    return float(median(measurements))


def median_matmul_time(left: torch.Tensor, right: torch.Tensor) -> float:
    measurements = []
    with torch.inference_mode():
        for _ in range(WARMUPS):
            (left @ right).sum().item()
        for _ in range(TIMED_RUNS):
            start = time.perf_counter()
            (left @ right).sum().item()
            measurements.append(time.perf_counter() - start)
    return float(median(measurements))


def main() -> None:
    torch.manual_seed(42)
    torch.set_num_threads(1)
    lines = ["PyTorch autograd/compile/contiguity inspection"]

    graph_model = SmallMLP()
    graph_input = torch.randn(4, 2, requires_grad=True)
    graph_loss = graph_model(graph_input).square().mean()
    graph_loss.backward()
    lines.append(f"loss={graph_loss.item():.9f}")
    lines.append("autograd graph (loss.grad_fn and three levels of next_functions):")
    lines.extend(graph_lines(graph_loss.grad_fn))

    eager_model = SmallMLP()
    eager_input = torch.randn(64, 2)
    eager_time = median_forward_time(eager_model, eager_input)
    lines.append(f"eager forward median ({TIMED_RUNS} timed runs, seconds)={eager_time:.12f}")

    compile_path = "torch.compile"
    compiled_time: float | None = None
    compile_note = ""
    try:
        compiled_model = torch.compile(eager_model)
        compiled_time = median_forward_time(compiled_model, eager_input)
        lines.append(f"compiled forward median ({TIMED_RUNS} timed runs, seconds)={compiled_time:.12f}")
        lines.append("compile path used: torch.compile")
    except Exception as exc:  # CPU/Windows backends may fail at compilation or first call.
        compile_path = "torch.jit.trace fallback"
        compile_note = f"torch.compile failed ({type(exc).__name__}: {str(exc)[:240]})"
        lines.append(compile_note)
        traced_model = torch.jit.trace(SmallMLP().eval(), eager_input)
        compiled_time = median_forward_time(traced_model, eager_input)
        lines.append(f"traced forward median ({TIMED_RUNS} timed runs, seconds)={compiled_time:.12f}")
        lines.append("compile path used: torch.jit.trace fallback")

    if compiled_time is not None:
        lines.append(f"eager/selected speed ratio={eager_time / compiled_time:.3f}x")
        lines.append(
            "steady-state note: selected path is faster than eager"
            if compiled_time < eager_time
            else "steady-state note: selected path is not faster than eager on this small CPU model"
        )
    lines.append("compile_path=" + compile_path)

    contiguous = torch.randn(512, 512)
    non_contiguous = contiguous.t()
    lines.append(
        f"base tensor: shape={tuple(contiguous.shape)}, dtype={contiguous.dtype}, "
        f"is_contiguous={contiguous.is_contiguous()}"
    )
    lines.append(
        f"non-contiguous transpose: shape={tuple(non_contiguous.shape)}, dtype={non_contiguous.dtype}, "
        f"is_contiguous={non_contiguous.is_contiguous()}"
    )
    rhs = torch.randn(512, 512)
    contiguous_time = median_matmul_time(contiguous, rhs)
    non_contiguous_time = median_matmul_time(non_contiguous, rhs)
    difference = non_contiguous_time - contiguous_time
    lines.append(f"contiguous matmul median (seconds)={contiguous_time:.12f}")
    lines.append(f"non-contiguous matmul median (seconds)={non_contiguous_time:.12f}")
    lines.append(f"non-contiguous minus contiguous (seconds)={difference:.12f}")
    lines.append(f"non-contiguous/contiguous ratio={non_contiguous_time / contiguous_time:.3f}x")

    output = "\n".join(lines) + "\n"
    print(output, end="")
    result_path = Path(__file__).resolve().parents[1] / "results" / "torch_inspection.txt"
    result_path.write_text(output, encoding="utf-8")


if __name__ == "__main__":
    main()
