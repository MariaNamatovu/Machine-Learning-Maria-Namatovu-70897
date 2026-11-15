"""Compare DataLoader worker counts on a fixed training pass."""
import csv
import time
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import LAB_DIR, load_config, make_loaders, set_seed

def main():
    cfg = load_config(); set_seed(cfg["seed"])
    rows = []
    max_batches = int(cfg["study"]["worker_max_batches"])
    for workers in cfg["study"]["num_workers_values"]:
        train_loader, _, _ = make_loaders(cfg, num_workers=int(workers))
        start = time.perf_counter(); samples = 0; batches = 0
        for images, _ in train_loader:
            samples += images.size(0); batches += 1
            if batches >= max_batches: break
        elapsed = time.perf_counter() - start
        rows.append({"num_workers": int(workers), "batches": batches, "samples": samples,
                     "elapsed_seconds": elapsed, "samples_per_second": samples / elapsed})
        print(rows[-1])
    path = LAB_DIR / "results" / "num_workers.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    plt.figure(figsize=(7, 4)); plt.plot([r["num_workers"] for r in rows], [r["samples_per_second"] for r in rows], marker="o")
    plt.xlabel("DataLoader num_workers"); plt.ylabel("Samples/second"); plt.title("Training input throughput by worker count")
    plt.grid(alpha=0.3); plt.tight_layout(); plt.savefig(LAB_DIR / "results" / "num_workers.png", dpi=150); plt.close()
    fastest = max(rows, key=lambda row: row["samples_per_second"])
    print(f"fastest_workers={fastest['num_workers']} throughput={fastest['samples_per_second']:.2f}")

if __name__ == "__main__":
    main()
