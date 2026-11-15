"""Shared, self-contained Lab 06 data/model/training helpers."""
from pathlib import Path
import random
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import datasets, transforms

ROOT = Path(__file__).resolve().parents[2]
LAB_DIR = ROOT / "lab06"
CONFIG_PATH = LAB_DIR / "config.yaml"

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.classifier = nn.Linear(32 * 7 * 7, 10)

    def forward(self, x):
        return self.classifier(torch.flatten(self.features(x), 1))

def load_config(path=CONFIG_PATH):
    import yaml
    with Path(path).open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)

def resolve_path(value):
    path = Path(value)
    return path if path.is_absolute() else ROOT / path

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def make_loaders(cfg, batch_size=None, num_workers=None, shuffle_train=True):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,)),
    ])
    root = resolve_path(cfg["data_root"])
    train_full = datasets.FashionMNIST(root=str(root), train=True, transform=transform, download=False)
    test_full = datasets.FashionMNIST(root=str(root), train=False, transform=transform, download=False)
    train_split, val_split = random_split(
        train_full, [55000, 5000], generator=torch.Generator().manual_seed(cfg["seed"])
    )
    subset_cfg = cfg["subset"]
    train_split = Subset(train_split, range(min(subset_cfg["train_samples"], len(train_split))))
    val_split = Subset(val_split, range(min(subset_cfg["validation_samples"], len(val_split))))
    test_split = Subset(test_full, range(min(subset_cfg["test_samples"], len(test_full))))
    bs = batch_size or cfg["training"]["batch_size"]
    workers = cfg["training"]["num_workers"] if num_workers is None else num_workers
    kwargs = {"batch_size": bs, "num_workers": workers, "pin_memory": False}
    return (
        DataLoader(train_split, shuffle=shuffle_train, **kwargs),
        DataLoader(val_split, shuffle=False, **kwargs),
        DataLoader(test_split, shuffle=False, **kwargs),
    )

def evaluate(model, loader, criterion=None):
    model.eval()
    total, correct, loss_sum = 0, 0, 0.0
    with torch.inference_mode():
        for images, labels in loader:
            outputs = model(images)
            if criterion is not None:
                loss_sum += float(criterion(outputs, labels).item()) * labels.size(0)
            correct += int((outputs.argmax(1) == labels).sum())
            total += labels.size(0)
    return (loss_sum / total if criterion is not None else None), correct / total

def train_epoch(model, loader, optimizer, criterion, profiler=None):
    model.train()
    total_loss, total, correct = 0.0, 0, 0
    iterator = iter(loader)
    while True:
        if profiler is not None:
            profiler.step()
        try:
            if profiler is not None:
                with torch.profiler.record_function("data_loading"):
                    images, labels = next(iterator)
            else:
                images, labels = next(iterator)
        except StopIteration:
            break
        if profiler is not None:
            with torch.profiler.record_function("forward"):
                outputs = model(images)
                loss = criterion(outputs, labels)
            with torch.profiler.record_function("backward"):
                optimizer.zero_grad()
                loss.backward()
            with torch.profiler.record_function("optimizer_step"):
                optimizer.step()
        else:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        batch_n = labels.size(0)
        total_loss += float(loss.item()) * batch_n
        total += batch_n
        correct += int((outputs.argmax(1) == labels).sum())
    return total_loss / total, correct / total, total

def config_params(cfg):
    params = {"seed": cfg["seed"], "data_root": cfg["data_root"]}
    for group_name in ("subset", "training", "study"):
        for key, value in cfg.get(group_name, {}).items():
            if isinstance(value, (str, int, float, bool)):
                params[f"{group_name}.{key}"] = value
            elif isinstance(value, (list, tuple)):
                params[f"{group_name}.{key}"] = str(value)
    return params
