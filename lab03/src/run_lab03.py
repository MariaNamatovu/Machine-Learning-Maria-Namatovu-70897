"""End-to-end Fashion-MNIST architecture cost comparison for Lab 3."""
from pathlib import Path
import csv
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from torchinfo import summary
from thop import profile


ROOT = Path(__file__).resolve().parents[2]
LAB_DIR = ROOT / "lab03"
RESULTS_DIR = LAB_DIR / "results"
DATA_DIR = ROOT / "data"
BATCH_SIZE = 128
EPOCHS = 5
LEARNING_RATE = 1e-3


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        return self.network(x)


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Linear(32 * 7 * 7, 10)

    def forward(self, x):
        return self.classifier(torch.flatten(self.features(x), 1))


class Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch_projection = nn.Linear(49, 64)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=64,
            nhead=4,
            dim_feedforward=128,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.head = nn.Linear(64, 10)

    def forward(self, x):
        patches = x.unfold(2, 7, 7).unfold(3, 7, 7)
        patches = patches.permute(0, 2, 3, 1, 4, 5).reshape(x.shape[0], 16, 49)
        sequence = self.patch_projection(patches)
        encoded = self.encoder(sequence)
        return self.head(encoded.mean(dim=1))


def load_data():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,)),
    ])
    training_set = datasets.FashionMNIST(
        root=str(DATA_DIR), train=True, transform=transform, download=True
    )
    test_set = datasets.FashionMNIST(
        root=str(DATA_DIR), train=False, transform=transform, download=True
    )
    train_set, validation_set = random_split(
        training_set, [55000, 5000], generator=torch.Generator().manual_seed(42)
    )
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    validation_loader = DataLoader(
        validation_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=0
    )
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    return train_loader, validation_loader, test_loader


def count_parameters(model):
    model_summary = summary(model, input_size=(1, 1, 28, 28), verbose=0)
    return int(model_summary.total_params)


def hand_flops(model_name):
    if model_name == "MLP":
        return 2 * 784 * 256 + 2 * 256 * 128 + 2 * 128 * 10
    if model_name == "CNN":
        return (
            2 * 3 * 3 * 1 * 16 * 28 * 28
            + 2 * 3 * 3 * 16 * 32 * 14 * 14
            + 2 * (32 * 7 * 7) * 10
        )
    sequence_length = 16
    d_model = 64
    feedforward = 128
    patch = 2 * sequence_length * 49 * d_model
    qkv = 2 * sequence_length * d_model * 3 * d_model
    attention = 2 * (2 * sequence_length * sequence_length * d_model)
    output = 2 * sequence_length * d_model * d_model
    feedforward_flops = 2 * (2 * sequence_length * d_model * feedforward)
    encoder_layer = qkv + attention + output + feedforward_flops
    return patch + 2 * encoder_layer + 2 * d_model * 10


def automated_cost(model):
    dummy_input = torch.randn(1, 1, 28, 28)
    flops, params = profile(model, inputs=(dummy_input,), verbose=False)
    return int(flops), int(params)


def test_accuracy(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.inference_mode():
        for images, labels in test_loader:
            predictions = model(images).argmax(dim=1)
            correct += int((predictions == labels).sum())
            total += labels.numel()
    return correct / total


def train_model(model, train_loader, test_loader):
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        batches = 0
        for images, labels in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            running_loss += float(loss.item())
            batches += 1
        print(f"epoch={epoch + 1}, loss={running_loss / batches:.6f}")
    return test_accuracy(model, test_loader)


def median_latency(model, batch_size, runs):
    model.eval()
    dummy_input = torch.randn(batch_size, 1, 28, 28)
    with torch.inference_mode():
        for _ in range(5):
            model(dummy_input)
        timings = []
        for _ in range(runs):
            start = time.perf_counter()
            model(dummy_input)
            timings.append((time.perf_counter() - start) * 1000.0)
    timings.sort()
    middle = len(timings) // 2
    if len(timings) % 2:
        return timings[middle]
    return (timings[middle - 1] + timings[middle]) / 2.0


def save_csv(path, fieldnames, rows):
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def make_plots(cost_rows, training_rows, latency_rows):
    accuracies = {row["model"]: float(row["final_test_accuracy"]) for row in training_rows}
    parameter_counts = {row["model"]: int(row["total_parameters"]) for row in cost_rows}
    batch_one_latency = {
        row["model"]: float(row["median_latency_ms"])
        for row in latency_rows
        if int(row["batch_size"]) == 1
    }
    names = ["MLP", "CNN", "Attention"]

    plt.figure(figsize=(7, 5))
    for name in names:
        plt.scatter(parameter_counts[name], accuracies[name], label=name)
        plt.annotate(name, (parameter_counts[name], accuracies[name]), xytext=(5, 5), textcoords="offset points")
    plt.xlabel("Parameter count")
    plt.ylabel("Final test accuracy")
    plt.title("Fashion-MNIST accuracy vs parameters")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "accuracy_vs_parameters.png", dpi=150)
    plt.close()

    plt.figure(figsize=(7, 5))
    for name in names:
        plt.scatter(batch_one_latency[name], accuracies[name], label=name)
        plt.annotate(name, (batch_one_latency[name], accuracies[name]), xytext=(5, 5), textcoords="offset points")
    plt.xlabel("Batch-1 median inference latency (ms)")
    plt.ylabel("Final test accuracy")
    plt.title("Fashion-MNIST accuracy vs latency")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "accuracy_vs_latency.png", dpi=150)
    plt.close()


def main():
    torch.set_num_threads(1)
    torch.manual_seed(42)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    train_loader, validation_loader, test_loader = load_data()
    del validation_loader

    model_factories = [("MLP", MLP), ("CNN", CNN), ("Attention", Attention)]
    cost_rows = []
    training_rows = []
    latency_rows = []
    for name, factory in model_factories:
        print(f"\n{name}")
        model = factory()
        total_parameters = count_parameters(model)
        thop_flops, thop_parameters = automated_cost(model)
        cost_rows.append({
            "model": name,
            "total_parameters": total_parameters,
            "hand_flops": hand_flops(name),
            "thop_flops": thop_flops,
            "thop_parameters": thop_parameters,
        })
        print(f"parameters={total_parameters}, hand_flops={hand_flops(name)}, thop_flops={thop_flops}")
        accuracy = train_model(model, train_loader, test_loader)
        training_rows.append({"model": name, "final_test_accuracy": f"{accuracy:.8f}"})
        print(f"final_test_accuracy={accuracy:.8f}")
        for batch_size, runs in ((1, 100), (128, 20)):
            latency = median_latency(model, batch_size, runs)
            latency_rows.append({
                "model": name,
                "batch_size": batch_size,
                "num_runs": runs,
                "median_latency_ms": f"{latency:.6f}",
            })
            print(f"batch_size={batch_size}, median_latency_ms={latency:.6f}")

    save_csv(
        RESULTS_DIR / "model_costs.csv",
        ["model", "total_parameters", "hand_flops", "thop_flops", "thop_parameters"],
        cost_rows,
    )
    save_csv(
        RESULTS_DIR / "training_results.csv",
        ["model", "final_test_accuracy"],
        training_rows,
    )
    save_csv(
        RESULTS_DIR / "latency_results.csv",
        ["model", "batch_size", "num_runs", "median_latency_ms"],
        latency_rows,
    )
    make_plots(cost_rows, training_rows, latency_rows)
    print("\nLab 3 results written to lab03/results")


if __name__ == "__main__":
    main()


