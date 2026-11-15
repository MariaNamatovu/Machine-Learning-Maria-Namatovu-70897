"""Train five reference CNNs with different batch sizes."""
import csv
import time
import psutil
import torch
from torch import nn
from common import LAB_DIR, CNN, evaluate, load_config, make_loaders, set_seed

def run_one(cfg, batch_size):
    set_seed(cfg["seed"])
    train_loader, val_loader, _ = make_loaders(cfg, batch_size=batch_size)
    model = CNN(); optimizer = torch.optim.Adam(model.parameters(), lr=cfg["training"]["learning_rate"]); criterion = nn.CrossEntropyLoss()
    process = psutil.Process(); base_rss = process.memory_info().rss; peak_rss = base_rss; total_samples = 0; start = time.perf_counter()
    for _ in range(cfg["training"]["epochs"]):
        model.train()
        for images, labels in train_loader:
            optimizer.zero_grad(); loss = criterion(model(images), labels); loss.backward(); optimizer.step()
            total_samples += labels.size(0); peak_rss = max(peak_rss, process.memory_info().rss)
    elapsed = time.perf_counter() - start
    _, accuracy = evaluate(model, val_loader)
    return {"batch_size": batch_size, "epochs": cfg["training"]["epochs"], "samples": total_samples,
            "elapsed_seconds": elapsed, "throughput_samples_per_second": total_samples / elapsed,
            "peak_memory_delta_mb": (peak_rss - base_rss) / (1024 ** 2), "final_validation_accuracy": accuracy}

def main():
    cfg = load_config(); rows = []
    for batch_size in cfg["study"]["batch_sizes"]:
        row = run_one(cfg, int(batch_size)); rows.append(row); print(row)
    path = LAB_DIR / "results" / "batch_size.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)

if __name__ == "__main__":
    main()
