"""Evaluate global, per-layer, and structured magnitude pruning."""
from copy import deepcopy
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.nn.utils import prune
from common import (
    BaselineCNN, RESULTS_DIR, evaluate, make_loaders, parameter_count,
    save_csv, set_seed, train_epochs,
)

MAGNITUDES = [0.0, 0.25, 0.5, 0.75, 0.9, 0.95]
STRUCTURED = [0.0, 0.10, 0.25, 0.50]
FINE_TUNE_EPOCHS = 3


def load_baseline():
    model = BaselineCNN()
    model.load_state_dict(torch.load(RESULTS_DIR / "baseline.pt", map_location="cpu"))
    return model


def prunable_layers(model):
    return [
        module for module in model.modules()
        if isinstance(module, (nn.Conv2d, nn.Linear))
    ]


def apply_magnitude(model, method, amount):
    layers = prunable_layers(model)
    if amount == 0.0:
        return
    if method == "global":
        prune.global_unstructured(
            [(layer, "weight") for layer in layers],
            pruning_method=prune.L1Unstructured,
            amount=amount,
        )
    else:
        for layer in layers:
            prune.l1_unstructured(layer, name="weight", amount=amount)


def apply_structured(model, fraction):
    if fraction == 0.0:
        return
    for layer in model.modules():
        if isinstance(layer, nn.Conv2d):
            prune.ln_structured(layer, name="weight", amount=fraction, n=2, dim=0)


def remove_masks(model):
    for layer in prunable_layers(model):
        if hasattr(layer, "weight_orig"):
            prune.remove(layer, "weight")


def nonzero_weights(model):
    return sum(
        int(layer.weight.detach().count_nonzero())
        for layer in prunable_layers(model)
    )


def main():
    torch.set_num_threads(1)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    _, _, test_loader = make_loaders()
    baseline = load_baseline()
    baseline_accuracy = evaluate(baseline, test_loader)
    rows = []
    global_points = []
    layer_points = []
    for method in ("global", "per_layer"):
        for amount in MAGNITUDES:
            set_seed(1000 + int(amount * 100) + (0 if method == "global" else 100))
            model = load_baseline()
            apply_magnitude(model, method, amount)
            if amount > 0.0:
                train_loader, _, _ = make_loaders()
                train_epochs(model, train_loader, FINE_TUNE_EPOCHS)
            accuracy = baseline_accuracy if amount == 0.0 else evaluate(model, test_loader)
            count = nonzero_weights(model)
            rows.append({
                "method": method,
                "sparsity_or_fraction": f"{amount:.2f}",
                "accuracy": f"{accuracy:.8f}",
                "param_count_nonzero": str(count),
                "notes": "unpruned baseline" if amount == 0.0 else "3 epochs with fixed mask",
            })
            (global_points if method == "global" else layer_points).append((amount, accuracy))
            if method == "global" and amount == 0.90:
                remove_masks(model)
                torch.save(model.state_dict(), RESULTS_DIR / "global_0.90.pt")
            print(f"{method} sparsity={amount:.2f} accuracy={accuracy:.8f} nonzero={count}")

    structured_points = []
    for fraction in STRUCTURED:
        set_seed(2000 + int(fraction * 100))
        model = load_baseline()
        apply_structured(model, fraction)
        if fraction > 0.0:
            train_loader, _, _ = make_loaders()
            train_epochs(model, train_loader, FINE_TUNE_EPOCHS)
        accuracy = baseline_accuracy if fraction == 0.0 else evaluate(model, test_loader)
        count = nonzero_weights(model)
        rows.append({
            "method": "structured",
            "sparsity_or_fraction": f"{fraction:.2f}",
            "accuracy": f"{accuracy:.8f}",
            "param_count_nonzero": str(count),
            "notes": "unpruned baseline" if fraction == 0.0 else "3 epochs with fixed Conv2d filter mask",
        })
        structured_points.append((fraction, accuracy))
        if fraction == 0.50:
            remove_masks(model)
            torch.save(model.state_dict(), RESULTS_DIR / "structured_0.50.pt")
        print(f"structured fraction={fraction:.2f} accuracy={accuracy:.8f} nonzero={count}")

    save_csv(
        RESULTS_DIR / "pruning_metrics.csv",
        ["method", "sparsity_or_fraction", "accuracy", "param_count_nonzero", "notes"],
        rows,
    )
    plt.figure(figsize=(7, 5))
    plt.plot([x for x, _ in global_points], [y for _, y in global_points], marker="o", label="global")
    plt.plot([x for x, _ in layer_points], [y for _, y in layer_points], marker="o", label="per-layer")
    plt.xlabel("Weight sparsity")
    plt.ylabel("Test accuracy")
    plt.title("Magnitude pruning accuracy vs sparsity")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "magnitude_accuracy_vs_sparsity.png", dpi=150)
    plt.close()

    plt.figure(figsize=(7, 5))
    plt.plot([x for x, _ in structured_points], [y for _, y in structured_points], marker="o")
    plt.xlabel("Filter-removal fraction")
    plt.ylabel("Test accuracy")
    plt.title("Structured pruning accuracy vs filter removal")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "structured_accuracy_vs_sparsity.png", dpi=150)
    plt.close()


if __name__ == "__main__":
    main()
