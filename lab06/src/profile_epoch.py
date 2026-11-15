"""Profile one complete training epoch with labeled CPU regions."""
import torch
from torch import nn
from common import LAB_DIR, CNN, load_config, make_loaders, set_seed

def main():
    cfg = load_config(); set_seed(cfg["seed"])
    train_loader, _, _ = make_loaders(cfg)
    model = CNN(); optimizer = torch.optim.Adam(model.parameters(), lr=cfg["training"]["learning_rate"]); criterion = nn.CrossEntropyLoss()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU], record_shapes=True) as prof:
        model.train(); iterator = iter(train_loader)
        while True:
            try:
                with torch.profiler.record_function("data_loading"):
                    images, labels = next(iterator)
            except StopIteration:
                break
            with torch.profiler.record_function("forward"):
                outputs = model(images); loss = criterion(outputs, labels)
            with torch.profiler.record_function("backward"):
                optimizer.zero_grad(); loss.backward()
            with torch.profiler.record_function("optimizer_step"):
                optimizer.step()
            prof.step()
    averages = prof.key_averages(); table = averages.table(sort_by="cpu_time_total", row_limit=30)
    names = ("data_loading", "forward", "backward", "optimizer_step")
    labeled = {name: 0.0 for name in names}
    for event in averages:
        if event.key in labeled: labeled[event.key] = float(event.cpu_time_total)
    total = sum(labeled.values())
    lines = ["Lab 06 one-epoch CPU profiler", "", table, "", "Labeled region breakdown (sum of record_function CPU time):"]
    for name in names: lines.append(f"{name}: {labeled[name] / 1000.0:.3f} ms ({100.0 * labeled[name] / total:.2f}%)")
    lines.append(f"labeled total: {total / 1000.0:.3f} ms")
    output_path = LAB_DIR / "results" / "profiler_breakdown.txt"; output_path.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines[-6:]))

if __name__ == "__main__": main()
