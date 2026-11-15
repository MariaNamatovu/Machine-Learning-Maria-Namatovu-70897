"""Compare gradient accumulation (64 x 4) with a true batch of 256."""
import csv
import torch
from torch import nn
from common import LAB_DIR, CNN, evaluate, load_config, make_loaders, set_seed

def run_accumulated(cfg):
    set_seed(cfg["seed"]); mini = int(cfg["study"]["accumulation_batch_size"]); steps = int(cfg["study"]["accumulation_steps"])
    train_loader, val_loader, _ = make_loaders(cfg, batch_size=mini); model = CNN(); optimizer = torch.optim.Adam(model.parameters(), lr=cfg["training"]["learning_rate"]); criterion = nn.CrossEntropyLoss(); rows=[]
    for epoch in range(1, cfg["training"]["accumulation_epochs"] + 1):
        model.train(); optimizer.zero_grad(); loss_sum=0.0; total=0; count=0; optimizer_steps=0
        for images, labels in train_loader:
            loss = criterion(model(images), labels); (loss / steps).backward(); loss_sum += float(loss.item()) * labels.size(0); total += labels.size(0); count += 1
            if count == steps:
                optimizer.step(); optimizer.zero_grad(); optimizer_steps += 1; count = 0
        if count:
            optimizer.step(); optimizer.zero_grad(); optimizer_steps += 1
        _, acc = evaluate(model, val_loader)
        rows.append({"run":"accumulate_64x4", "epoch":epoch, "train_loss":loss_sum/total, "validation_accuracy":acc, "optimizer_steps":optimizer_steps, "effective_batch_size":mini*steps})
    return rows

def run_true_batch(cfg):
    set_seed(cfg["seed"]); batch = int(cfg["study"]["accumulation_batch_size"]) * int(cfg["study"]["accumulation_steps"])
    train_loader, val_loader, _ = make_loaders(cfg, batch_size=batch); model=CNN(); optimizer=torch.optim.Adam(model.parameters(), lr=cfg["training"]["learning_rate"]); criterion=nn.CrossEntropyLoss(); rows=[]
    for epoch in range(1, cfg["training"]["accumulation_epochs"] + 1):
        model.train(); loss_sum=0.0; total=0; optimizer_steps=0
        for images, labels in train_loader:
            optimizer.zero_grad(); loss=criterion(model(images), labels); loss.backward(); optimizer.step(); loss_sum += float(loss.item())*labels.size(0); total += labels.size(0); optimizer_steps += 1
        _, acc=evaluate(model,val_loader)
        rows.append({"run":"true_batch_256", "epoch":epoch, "train_loss":loss_sum/total, "validation_accuracy":acc, "optimizer_steps":optimizer_steps, "effective_batch_size":batch})
    return rows

def main():
    cfg=load_config(); rows=run_accumulated(cfg)+run_true_batch(cfg)
    path=LAB_DIR/"results"/"gradient_accumulation.csv"
    with path.open("w",newline="",encoding="utf-8") as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    for row in rows: print(row)

if __name__ == "__main__": main()
