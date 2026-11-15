"""Simulate interruption at epoch 3 and resume the same CNN through epoch 6."""
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch import nn
from common import LAB_DIR, CNN, evaluate, load_config, make_loaders, resolve_path, set_seed

def run_epoch(model, loader, optimizer, criterion):
    model.train(); total_loss=0.0; total=0
    for images, labels in loader:
        optimizer.zero_grad(); loss=criterion(model(images),labels); loss.backward(); optimizer.step(); total_loss += float(loss.item())*labels.size(0); total += labels.size(0)
    return total_loss/total

def main():
    cfg=load_config(); set_seed(cfg["seed"]); train_loader,val_loader,_=make_loaders(cfg)
    model=CNN(); optimizer=torch.optim.Adam(model.parameters(),lr=cfg["training"]["learning_rate"]); criterion=nn.CrossEntropyLoss(); rows=[]
    stop_epoch=3; total_epochs=int(cfg["training"]["resume_total_epochs"])
    for epoch in range(1, stop_epoch+1):
        loss=run_epoch(model,train_loader,optimizer,criterion); _,acc=evaluate(model,val_loader); rows.append({"epoch":epoch,"phase":"initial","train_loss":loss,"validation_accuracy":acc}); print(rows[-1])
    checkpoint=resolve_path(cfg["checkpoint_dir"])/"resume_epoch3.pt"; checkpoint.parent.mkdir(parents=True,exist_ok=True)
    torch.save({"model_state_dict":model.state_dict(),"optimizer_state_dict":optimizer.state_dict(),"epoch":stop_epoch},checkpoint)
    resumed=CNN(); resumed_optimizer=torch.optim.Adam(resumed.parameters(),lr=cfg["training"]["learning_rate"]); payload=torch.load(checkpoint,map_location="cpu"); resumed.load_state_dict(payload["model_state_dict"]); resumed_optimizer.load_state_dict(payload["optimizer_state_dict"])
    for epoch in range(stop_epoch+1,total_epochs+1):
        loss=run_epoch(resumed,train_loader,resumed_optimizer,criterion); _,acc=evaluate(resumed,val_loader); rows.append({"epoch":epoch,"phase":"resumed","train_loss":loss,"validation_accuracy":acc}); print(rows[-1])
    csv_path=LAB_DIR/"results"/"resume_history.csv"
    with csv_path.open("w",newline="",encoding="utf-8") as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    fig,ax1=plt.subplots(figsize=(7,4)); epochs=[r["epoch"] for r in rows]; ax1.plot(epochs,[r["train_loss"] for r in rows],marker="o",label="train loss",color="tab:blue"); ax1.set_xlabel("Epoch"); ax1.set_ylabel("Train loss",color="tab:blue"); ax2=ax1.twinx(); ax2.plot(epochs,[r["validation_accuracy"] for r in rows],marker="s",label="validation accuracy",color="tab:orange"); ax2.set_ylabel("Validation accuracy",color="tab:orange"); ax1.axvline(stop_epoch+0.5,color="gray",linestyle="--",label="checkpoint/resume"); ax1.grid(alpha=0.3); fig.tight_layout(); fig.savefig(LAB_DIR/"results"/"resume_curves.png",dpi=150); plt.close(fig)
    print(f"checkpoint={checkpoint}")

if __name__ == "__main__": main()
