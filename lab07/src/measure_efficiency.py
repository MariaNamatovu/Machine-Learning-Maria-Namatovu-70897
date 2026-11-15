"""Measure latency and dense versus sparse serialized storage."""
from pathlib import Path
import torch
from common import BaselineCNN, RESULTS_DIR, median_latency, save_csv


def load_model(path):
    model = BaselineCNN()
    model.load_state_dict(torch.load(path, map_location="cpu"))
    return model


def sparse_size_kb(state_dict, label):
    sparse_state = {}
    for key, tensor in state_dict.items():
        if key.endswith(".weight") and tensor.ndim >= 2:
            sparse_state[key] = tensor.to_sparse()
        else:
            sparse_state[key] = tensor
    path = RESULTS_DIR / f".sparse_{label}.pt"
    torch.save(sparse_state, path)
    size = path.stat().st_size / 1024.0
    path.unlink()
    return size


def row(name, value, checkpoint):
    state = torch.load(checkpoint, map_location="cpu")
    model = load_model(checkpoint)
    dense_size = checkpoint.stat().st_size / 1024.0
    sparse_size = sparse_size_kb(state, name)
    latency = median_latency(model, runs=100, batch_size=1)
    print(f"{name} latency_ms={latency:.6f} dense_size_kb={dense_size:.6f} sparse_size_kb={sparse_size:.6f}")
    return {
        "model": name,
        "sparsity_or_fraction": value,
        "latency_ms": f"{latency:.6f}",
        "dense_size_kb": f"{dense_size:.6f}",
        "sparse_size_kb_or_theoretical": f"{sparse_size:.6f}",
    }


def main():
    torch.set_num_threads(1)
    rows = [
        row("baseline", "0.00", RESULTS_DIR / "baseline.pt"),
        row("global_0.90", "0.90", RESULTS_DIR / "global_0.90.pt"),
        row("structured_0.50", "0.50", RESULTS_DIR / "structured_0.50.pt"),
    ]
    save_csv(
        RESULTS_DIR / "efficiency_metrics.csv",
        ["model", "sparsity_or_fraction", "latency_ms", "dense_size_kb", "sparse_size_kb_or_theoretical"],
        rows,
    )


if __name__ == "__main__":
    main()
