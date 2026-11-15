"""Compare learning-rate schedules and optimizers over ten real epochs."""
import csv
import json
import torch
from torch import nn
from common import LAB_DIR, CNN, evaluate, load_config, make_loaders, set_seed

def run_variant(cfg, name, kind):
    set_seed(cfg["seed"]); train_loader, val_loader, _ = make_loaders(cfg)
    model = CNN(); criterion = nn.CrossEntropyLoss()
    if kind.startswith("schedule"):
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg["training"]["learning_rate"])
        if kind == "schedule_step": scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=cfg["study"]["schedule_step_size"], gamma=cfg["study"]["schedule_gamma"])
        elif kind == "schedule_cosine": scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["study"]["cosine_t_max"])
        else: scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=cfg["study"]["plateau_patience"], factor=cfg["study"]["plateau_factor"])
    elif kind == "optimizer_adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg["training"]["learning_rate"]); scheduler = None
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9); scheduler = None
    losses, accuracies, lrs = [], [], []
    for epoch in range(cfg["study"]["schedule_epochs"]):
        model.train(); total_loss = 0.0; total = 0
        for images, labels in train_loader:
            optimizer.zero_grad(); outputs = model(images); loss = criterion(outputs, labels); loss.backward(); optimizer.step()
            total_loss += float(loss.item()) * labels.size(0); total += labels.size(0)
        epoch_loss = total_loss / total; _, val_acc = evaluate(model, val_loader); losses.append(epoch_loss); accuracies.append(val_acc); lrs.append(optimizer.param_groups[0]["lr"])
        if kind == "schedule_plateau": scheduler.step(epoch_loss)
        elif kind.startswith("schedule"): scheduler.step()
    return {"run": name, "category": "schedule" if kind.startswith("schedule") else "optimizer", "variant": kind,
            "epochs": cfg["study"]["schedule_epochs"], "final_loss": losses[-1], "final_validation_accuracy": accuracies[-1],
            "loss_curve": json.dumps(losses), "accuracy_curve": json.dumps(accuracies), "learning_rates": json.dumps(lrs)}

def main():
    cfg = load_config(); variants = [("StepLR", "schedule_step"), ("CosineAnnealingLR", "schedule_cosine"), ("ReduceLROnPlateau", "schedule_plateau"), ("Adam", "optimizer_adam"), ("SGD_momentum", "optimizer_sgd")]
    rows = []
    for name, kind in variants:
        row = run_variant(cfg, name, kind); rows.append(row); print(name, row["final_loss"], row["final_validation_accuracy"])
    path = LAB_DIR / "results" / "lr_optimizer_comparison.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)

if __name__ == "__main__":
    main()
