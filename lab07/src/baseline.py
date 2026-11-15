"""Train and measure the approximately one-million-parameter baseline CNN."""
import torch
from common import (
    BaselineCNN, RESULTS_DIR, evaluate, make_loaders, median_latency,
    parameter_count, save_csv, set_seed, train_epochs,
)

EPOCHS = 5


def main():
    torch.set_num_threads(1)
    set_seed()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    train_loader, _, test_loader = make_loaders()
    model = BaselineCNN()
    print(f"param_count={parameter_count(model)}")
    train_epochs(model, train_loader, EPOCHS)
    accuracy = evaluate(model, test_loader)
    checkpoint = RESULTS_DIR / "baseline.pt"
    torch.save(model.state_dict(), checkpoint)
    size_kb = checkpoint.stat().st_size / 1024.0
    latency_ms = median_latency(model, runs=100, batch_size=1)
    rows = [
        {"metric": "accuracy", "value": f"{accuracy:.8f}"},
        {"metric": "param_count", "value": str(parameter_count(model))},
        {"metric": "size_kb", "value": f"{size_kb:.6f}"},
        {"metric": "latency_ms", "value": f"{latency_ms:.6f}"},
    ]
    save_csv(RESULTS_DIR / "baseline_metrics.csv", ["metric", "value"], rows)
    print(f"accuracy={accuracy:.8f}")
    print(f"param_count={parameter_count(model)}")
    print(f"size_kb={size_kb:.6f}")
    print(f"latency_ms={latency_ms:.6f}")


if __name__ == "__main__":
    main()
