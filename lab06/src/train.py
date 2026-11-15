"""Train the Lab 06 reference CNN and log the run to local MLflow."""
import argparse
import csv
import mlflow
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch import nn
from common import CNN, LAB_DIR, load_config, make_loaders, evaluate, resolve_path, set_seed, train_epoch, config_params

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(LAB_DIR / "config.yaml"))
    args = parser.parse_args()
    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    train_loader, val_loader, _ = make_loaders(cfg)
    model = CNN()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["training"]["learning_rate"])
    criterion = nn.CrossEntropyLoss()
    rows = []
    tracking_uri = resolve_path(cfg["mlflow"]["tracking_uri"])
    tracking_uri.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(tracking_uri.as_uri())
    mlflow.set_experiment(cfg["mlflow"]["experiment_name"])
    with mlflow.start_run(run_name=cfg["mlflow"]["run_name"]):
        mlflow.log_params(config_params(cfg))
        for epoch in range(1, cfg["training"]["epochs"] + 1):
            train_loss, train_acc, _ = train_epoch(model, train_loader, optimizer, criterion)
            val_loss, val_acc = evaluate(model, val_loader, criterion)
            row = {"epoch": epoch, "train_loss": train_loss, "train_accuracy": train_acc,
                   "val_loss": val_loss, "val_accuracy": val_acc}
            rows.append(row)
            mlflow.log_metrics({k: float(v) for k, v in row.items() if k != "epoch"}, step=epoch)
            print(f"epoch={epoch} train_loss={train_loss:.5f} train_acc={train_acc:.4f} val_acc={val_acc:.4f}")
        checkpoint_dir = resolve_path(cfg["checkpoint_dir"])
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = checkpoint_dir / "reference_cnn.pt"
        torch.save({"model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": cfg["training"]["epochs"], "config": cfg}, checkpoint_path)
        curve_path = LAB_DIR / "results" / "training_curve.png"
        curve_path.parent.mkdir(parents=True, exist_ok=True)
        plt.figure(figsize=(7, 4))
        plt.plot([r["epoch"] for r in rows], [r["train_loss"] for r in rows], marker="o", label="train loss")
        plt.plot([r["epoch"] for r in rows], [r["val_accuracy"] for r in rows], marker="o", label="validation accuracy")
        plt.xlabel("Epoch"); plt.grid(alpha=0.3); plt.legend(); plt.tight_layout(); plt.savefig(curve_path, dpi=150); plt.close()
        mlflow.log_artifact(str(checkpoint_path)); mlflow.log_artifact(str(curve_path))
        csv_path = LAB_DIR / "results" / "training_history.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
        mlflow.log_artifact(str(csv_path))
    print(f"checkpoint={checkpoint_path}")
    print(f"mlruns={tracking_uri}")

if __name__ == "__main__":
    main()
