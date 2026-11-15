"""Train plain and knowledge-distilled compact student CNNs."""
import csv
import torch
from torch import nn
import torch.nn.functional as F
from common import (
    BaselineCNN, RESULTS_DIR, StudentCNN, evaluate, make_loaders,
    median_latency, parameter_count, save_csv, set_seed,
)

EPOCHS = 5
TEMPERATURE = 4.0
ALPHA = 0.5


def load_baseline():
    teacher = BaselineCNN()
    teacher.load_state_dict(torch.load(RESULTS_DIR / "baseline.pt", map_location="cpu"))
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    return teacher


def train_plain(model, loader):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(1, EPOCHS + 1):
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
        print(f"plain epoch={epoch} loss={total_loss / total:.6f}")


def train_distilled(model, teacher, loader):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    ce = nn.CrossEntropyLoss()
    kl = nn.KLDivLoss(reduction="batchmean")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        total = 0
        for images, labels in loader:
            optimizer.zero_grad()
            student_logits = model(images)
            with torch.no_grad():
                teacher_logits = teacher(images)
            hard_loss = ce(student_logits, labels)
            soft_loss = kl(
                F.log_softmax(student_logits / TEMPERATURE, dim=1),
                F.softmax(teacher_logits / TEMPERATURE, dim=1),
            )
            loss = ALPHA * hard_loss + (1.0 - ALPHA) * (TEMPERATURE ** 2) * soft_loss
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * labels.size(0)
            total += labels.size(0)
        print(f"distilled epoch={epoch} loss={total_loss / total:.6f}")


def baseline_metrics():
    values = {}
    with (RESULTS_DIR / "baseline_metrics.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            values[row["metric"]] = float(row["value"])
    return values


def measure_student(name, model, path, test_loader):
    accuracy = evaluate(model, test_loader)
    torch.save(model.state_dict(), path)
    size_kb = path.stat().st_size / 1024.0
    latency_ms = median_latency(model, runs=100, batch_size=1)
    print(f"{name} accuracy={accuracy:.8f} size_kb={size_kb:.6f} latency_ms={latency_ms:.6f}")
    return {"model": name, "accuracy": f"{accuracy:.8f}", "size_kb": f"{size_kb:.6f}", "latency_ms": f"{latency_ms:.6f}"}


def main():
    torch.set_num_threads(1)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    train_loader, _, test_loader = make_loaders()
    teacher = load_baseline()
    base = baseline_metrics()
    rows = [{
        "model": "teacher_baseline",
        "accuracy": f"{base['accuracy']:.8f}",
        "size_kb": f"{base['size_kb']:.6f}",
        "latency_ms": f"{base['latency_ms']:.6f}",
    }]
    set_seed(3000)
    plain = StudentCNN()
    print(f"student_param_count={parameter_count(plain)}")
    train_plain(plain, train_loader)
    rows.append(measure_student("student_plain", plain, RESULTS_DIR / "student_plain.pt", test_loader))
    set_seed(3000)
    distilled = StudentCNN()
    train_distilled(distilled, teacher, train_loader)
    rows.append(measure_student("student_distilled", distilled, RESULTS_DIR / "student_distilled.pt", test_loader))
    save_csv(RESULTS_DIR / "distillation_metrics.csv", ["model", "accuracy", "size_kb", "latency_ms"], rows)


if __name__ == "__main__":
    main()
