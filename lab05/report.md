# Lab 05 — Scalar automatic differentiation and PyTorch inspection

## Value engine

`src/engine.py` implements a scalar `Value` node with `data`, accumulated
`grad`, parent nodes, an operation label, and a local `_backward` closure.
Addition, multiplication, powers, ReLU, exponentials, logarithms, negation,
subtraction, and division construct the computation graph and accumulate local
chain-rule contributions with `+=`. `Value.backward()` performs a DFS
post-order topological sort, seeds the output gradient with 1, and traverses
the graph in reverse order without clearing existing gradients.

## Gradient verification

The expression was evaluated for three scalar triples as the sum of
`((a*b+c)**2).relu()`. The engine was compared with PyTorch autograd and
central finite differences (`h=1e-6`). The maximum absolute differences were:

- engine vs PyTorch: **0.000000000000e+00**
- engine vs numerical: **3.913783075404e-09**

Both are below `1e-6`. Full gradients and the reproducible check are in
[gradient_verification.txt](results/gradient_verification.txt).

## make_moons training timing

A 200-sample noisy `make_moons` dataset (`noise=0.2`, seed 42) was used with
an identical 2->8(ReLU)->1(sigmoid) network, BCE loss, online plain SGD, learning
rate 0.1, and 30 epochs. Five fresh seeded runs were timed. The latest
verification run measured an engine median of **6.553910600 s** and a PyTorch
median of **4.921393000 s** (engine/PyTorch ratio **1.332x**); these wall-clock
values vary with CPU load, while the result file records both medians and five
runs. Median final losses were 0.093842991 (engine) and 0.089474134 (PyTorch).
See [training_timing.txt](results/training_timing.txt).

## PyTorch graph, compilation, and contiguity

The manual autograd walk showed `MeanBackward0 -> PowBackward0 ->
AddmmBackward0`, followed by parameter accumulation, ReLU, and transpose
backward nodes. On this Windows CPU setup, `torch.compile` failed because the
Inductor backend could not find a working C++ compiler, so the script used the
required `torch.jit.trace` fallback. The verification run measured eager forward
at **0.000248900033 s** and traced forward at **0.000102600083 s** (2.426x
speed ratio); tracing helped this run, but compilation/tracing can regress on
small models and must amortize one-time overhead on larger or repeatedly
executed workloads.

The 512x512 contiguous tensor was `float32` and contiguous; its transposed
view had the same shape and dtype but was non-contiguous. Median matmul times
were **0.018192199990 s** (contiguous) and **0.015943149920 s**
(non-contiguous), a **0.876x** non-contiguous/contiguous ratio in this run;
layout timing is workload- and system-dependent. All inspection output is in
[torch_inspection.txt](results/torch_inspection.txt).

## Reproducibility

Run from the repository root with the supplied environment:

```powershell
.\.venv\Scripts\python.exe lab05\src\verify_gradients.py
.\.venv\Scripts\python.exe lab05\src\train_compare.py
.\.venv\Scripts\python.exe lab05\src\torch_inspection.py
```
