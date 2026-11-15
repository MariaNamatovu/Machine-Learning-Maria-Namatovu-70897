"""Shared deterministic Fashion-MNIST helpers for Lab 07."""
from pathlib import Path
import csv
import random
import time
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import datasets, transforms

ROOT = Path(__file__).resolve().parents[2]
LAB_DIR = ROOT / "lab07"
RESULTS_DIR = LAB_DIR / "results"
DATA_DIR = ROOT / "data"
SEED = 42
BATCH_SIZE = 128
TRAIN_SAMPLES = 4000
VAL_SAMPLES = 1000
TEST_SAMPLES = 1000
LEARNING_RATE = 1e-3


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_loaders(batch_size=BATCH_SIZE):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,)),
    ])
    train_full = datasets.FashionMNIST(
        root=str(DATA_DIR), train=True, transform=transform, download=False
    )
    test_full = datasets.FashionMNIST(
        root=str(DATA_DIR), train=False, transform=transform, download=False
    )
    train_split, val_split = random_split(
        train_full, [55000, 5000], generator=torch.Generator().manual_seed(SEED)
    )
    train_split = Subset(train_split, range(TRAIN_SAMPLES))
    val_split = Subset(val_split, range(VAL_SAMPLES))
    test_split = Subset(test_full, range(TEST_SAMPLES))
    return (
        DataLoader(train_split, batch_size=batch_size, shuffle=True, num_workers=0),
        DataLoader(val_split, batch_size=batch_size, shuffle=False, num_workers=0),
        DataLoader(test_split, batch_size=batch_size, shuffle=False, num_workers=0),
    )


class BaselineCNN(nn.Module):
    """Four-layer CNN with approximately one million trainable parameters."""
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


class StudentCNN(nn.Module):
    """Compact CNN with approximately one hundred thousand parameters."""
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 64),
            nn.ReLU(),
            nn.Linear(64, 10),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def parameter_count(model):
    return sum(parameter.numel() for parameter in model.parameters())


def evaluate(model, loader):
    model.eval()
    correct = 0
    total = 0
    with torch.inference_mode():
        for images, labels in loader:
            correct += int((model(images).argmax(dim=1) == labels).sum())
            total += labels.numel()
    return correct / total


def train_epochs(model, loader, epochs, learning_rate=LEARNING_RATE, criterion=None):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = criterion or nn.CrossEntropyLoss()
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total = 0
        for images, labels in loader:
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * labels.size(0)
            total += labels.size(0)
        print(f"epoch={epoch} loss={total_loss / total:.6f}")
    return model


def median_latency(model, runs=50, batch_size=1):
    model.eval()
    dummy = torch.randn(batch_size, 1, 28, 28)
    with torch.inference_mode():
        for _ in range(5):
            model(dummy)
        timings = []
        for _ in range(runs):
            start = time.perf_counter()
            model(dummy)
            timings.append((time.perf_counter() - start) * 1000.0)
    timings.sort()
    middle = len(timings) // 2
    return timings[middle] if len(timings) % 2 else (timings[middle - 1] + timings[middle]) / 2.0


def state_dict_size_kb(state_dict):
    path = RESULTS_DIR / ".size_probe.pt"
    torch.save(state_dict, path)
    size = path.stat().st_size / 1024.0
    path.unlink()
    return size


def checkpoint_size_kb(path):
    return path.stat().st_size / 1024.0


def save_csv(path, fieldnames, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

