"""Pure NumPy implementation for Lab 2: a two-hidden-layer classifier."""
from pathlib import Path
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)


def load_data():
    digits = load_digits()
    X = digits.data.astype(np.float64) / 16.0
    Y = np.eye(10, dtype=np.float64)[digits.target]
    return train_test_split(
        X, Y, test_size=0.30, random_state=42, stratify=digits.target
    )


def initialize_parameters(width, rng):
    return {
        "W1": rng.normal(0.0, 0.01, size=(64, width)),
        "b1": np.zeros(width, dtype=np.float64),
        "W2": rng.normal(0.0, 0.01, size=(width, width)),
        "b2": np.zeros(width, dtype=np.float64),
        "W3": rng.normal(0.0, 0.01, size=(width, 10)),
        "b3": np.zeros(10, dtype=np.float64),
    }


def relu(z):
    return np.maximum(0.0, z)


def forward(X, params):
    Z1 = X @ params["W1"] + params["b1"]
    A1 = relu(Z1)
    Z2 = A1 @ params["W2"] + params["b2"]
    A2 = relu(Z2)
    Z3 = A2 @ params["W3"] + params["b3"]
    shifted = Z3 - np.max(Z3, axis=1, keepdims=True)
    exp_scores = np.exp(shifted)
    P = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    cache = {"X": X, "Z1": Z1, "A1": A1, "Z2": Z2, "A2": A2, "Z3": Z3, "P": P}
    return P, cache


def cross_entropy(P, Y):
    return -np.sum(Y * np.log(np.clip(P, 1e-15, 1.0))) / Y.shape[0]


def backward(cache, Y, params):
    X = cache["X"]
    Z1, A1 = cache["Z1"], cache["A1"]
    Z2, A2 = cache["Z2"], cache["A2"]
    P = cache["P"]
    batch_size = X.shape[0]

    dZ3 = (P - Y) / batch_size
    dW3 = A2.T @ dZ3
    db3 = np.sum(dZ3, axis=0)
    dA2 = dZ3 @ params["W3"].T
    dZ2 = dA2 * (Z2 > 0.0)
    dW2 = A1.T @ dZ2
    db2 = np.sum(dZ2, axis=0)
    dA1 = dZ2 @ params["W2"].T
    dZ1 = dA1 * (Z1 > 0.0)
    dW1 = X.T @ dZ1
    db1 = np.sum(dZ1, axis=0)
    return {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2, "W3": dW3, "b3": db3}


def copy_parameters(params):
    return {name: value.copy() for name, value in params.items()}


def gradient_check(X, Y):
    params = initialize_parameters(16, np.random.default_rng(2024))
    P, cache = forward(X, params)
    analytic = backward(cache, Y, params)
    rng = np.random.default_rng(81)
    selections = []
    for name, count in (("W1", 4), ("W2", 3), ("W3", 3)):
        indices = list(np.ndindex(params[name].shape))
        chosen = rng.choice(len(indices), size=count, replace=False)
        selections.extend((name, indices[int(i)]) for i in chosen)

    epsilon = 1e-5
    errors = []
    lines = ["Numerical gradient check (centered difference, epsilon=1e-5)", ""]
    for name, index in selections:
        plus = copy_parameters(params)
        minus = copy_parameters(params)
        plus[name][index] += epsilon
        minus[name][index] -= epsilon
        loss_plus = cross_entropy(forward(X, plus)[0], Y)
        loss_minus = cross_entropy(forward(X, minus)[0], Y)
        numerical = (loss_plus - loss_minus) / (2.0 * epsilon)
        analytical = analytic[name][index]
        relative_error = abs(analytical - numerical) / max(abs(analytical) + abs(numerical), 1e-12)
        errors.append(relative_error)
        lines.append(f"{name}{index}: analytical={analytical:.12e}, numerical={numerical:.12e}, relative_error={relative_error:.12e}")
    passed = sum(error < 1e-6 for error in errors)
    lines.extend(["", f"Passed: {passed}/10", f"Pass rate: {passed / 10:.0%}"])
    (RESULTS / "gradient_check.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return passed, errors


def train(X_train, Y_train, X_test, Y_test, learning_rate, width, seed):
    params = initialize_parameters(width, np.random.default_rng(seed))
    rng = np.random.default_rng(seed + 10000)
    losses = []
    for _ in range(50):
        order = rng.permutation(X_train.shape[0])
        for start in range(0, X_train.shape[0], 32):
            batch_indices = order[start:start + 32]
            P, cache = forward(X_train[batch_indices], params)
            gradients = backward(cache, Y_train[batch_indices], params)
            for name in params:
                params[name] -= learning_rate * gradients[name]
        losses.append(cross_entropy(forward(X_train, params)[0], Y_train))
    test_predictions = np.argmax(forward(X_test, params)[0], axis=1)
    test_labels = np.argmax(Y_test, axis=1)
    accuracy = float(np.mean(test_predictions == test_labels))
    return losses, accuracy


def timing(X, Y, params):
    batch = X[:32]
    labels = Y[:32]
    for _ in range(5):
        _, cache = forward(batch, params)
        backward(cache, labels, params)
    forward_times = []
    for _ in range(100):
        start = time.perf_counter()
        forward(batch, params)
        forward_times.append(time.perf_counter() - start)
    _, cache = forward(batch, params)
    backward_times = []
    for _ in range(100):
        start = time.perf_counter()
        backward(cache, labels, params)
        backward_times.append(time.perf_counter() - start)
    return float(np.median(forward_times)), float(np.median(backward_times))


def main():
    X_train, X_test, Y_train, Y_test = load_data()
    sanity_params = initialize_parameters(16, np.random.default_rng(2024))
    sanity_probabilities, sanity_cache = forward(X_train[:32], sanity_params)
    row_sum_error = float(np.max(np.abs(np.sum(sanity_probabilities, axis=1) - 1.0)))
    untrained_loss = float(cross_entropy(sanity_probabilities, Y_train[:32]))
    assert sanity_probabilities.shape == (32, 10)
    assert row_sum_error <= 1e-6

    passed, errors = gradient_check(X_train[:3], Y_train[:3])
    assert passed >= 9

    learning_rates = [0.001, 0.01, 0.1]
    widths = [16, 64]
    curves = {}
    accuracy_rows = []
    for width in widths:
        for learning_rate in learning_rates:
            losses, accuracy = train(
                X_train, Y_train, X_test, Y_test, learning_rate, width, seed=1000 + width
            )
            label = f"lr={learning_rate}, width={width}"
            curves[label] = losses
            accuracy_rows.append((learning_rate, width, accuracy))
            print(f"{label}: final test accuracy={accuracy:.6f}")

    plt.figure(figsize=(9, 6))
    epochs = np.arange(1, 51)
    for label, losses in curves.items():
        plt.plot(epochs, losses, label=label)
    plt.xlabel("Epoch")
    plt.ylabel("Training cross-entropy loss")
    plt.title("Lab 2 loss curves")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(RESULTS / "loss_curves.png", dpi=150)
    plt.close()

    with (RESULTS / "accuracy_results.csv").open("w", encoding="utf-8") as output:
        output.write("learning_rate,width,final_test_accuracy\n")
        for learning_rate, width, accuracy in accuracy_rows:
            output.write(f"{learning_rate},{width},{accuracy:.8f}\n")

    timing_forward, timing_backward = timing(X_train, Y_train, sanity_params)
    (RESULTS / "timing_results.txt").write_text(
        "Timing uses a batch of 32 and the median of 100 runs.\n"
        f"forward_median_seconds={timing_forward:.12f}\n"
        f"backward_median_seconds={timing_backward:.12f}\n"
        f"forward_median_milliseconds={timing_forward * 1000:.6f}\n"
        f"backward_median_milliseconds={timing_backward * 1000:.6f}\n",
        encoding="utf-8",
    )

    print(f"output_shape={sanity_probabilities.shape}")
    print(f"max_row_sum_error={row_sum_error:.12e}")
    print(f"untrained_loss={untrained_loss:.12f}")
    print(f"gradient_check={passed}/10, max_relative_error={max(errors):.12e}")
    print(f"forward_median_seconds={timing_forward:.12f}")
    print(f"backward_median_seconds={timing_backward:.12f}")


if __name__ == "__main__":
    main()


