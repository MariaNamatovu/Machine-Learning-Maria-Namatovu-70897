# Lab 06 — Training systems and optimization studies

## Scope and reproducibility

All experiments use the Lab 03 CNN exactly: Conv(1,16,3,padding=1), ReLU, MaxPool(2),
Conv(16,32,3,padding=1), ReLU, MaxPool(2), Flatten, and Linear(32×7×7,10).
Fashion-MNIST uses `ToTensor()` and `Normalize((0.2860,), (0.3530,))`, with the
seed-42 55,000/5,000 train/validation split and the official test set held out.
To keep the CPU study suite within a practical budget, every Lab 06 run uses a
real, deterministic subset: 8,000 training images, 2,000 validation images, and
2,000 test images. No values were fabricated; all reported numbers below are
from the scripts and subset. The complete settings are in [config.yaml](config.yaml).

## Task 1 — YAML-driven training and MLflow

[src/train.py](src/train.py) trains the reference CNN for five epochs with Adam
(lr 0.001), saves model and optimizer state plus epoch in
[reference_cnn.pt](results/checkpoints/reference_cnn.pt), and logs parameters,
per-epoch loss/accuracy metrics, history, checkpoint, and a plot to the local
MLflow file store. Training accuracy finished at 0.8555 and validation accuracy
finished at **0.8460** (the best validation value was 0.8575 at epoch 4).
The curve is [training_curve.png](results/training_curve.png), with tabular history
in [training_history.csv](results/training_history.csv). Tracking data is in
`results/mlruns`; view it from the repository root with
`mlflow ui --backend-store-uri lab06/results/mlruns`.

## Task 2 — CPU profiler breakdown

[src/profile_epoch.py](src/profile_epoch.py) profiles one complete training epoch
with `ProfilerActivity.CPU` and explicit `data_loading`, `forward`, `backward`,
and `optimizer_step` regions. The complete operator table and aggregated labeled
breakdown are in [profiler_breakdown.txt](results/profiler_breakdown.txt): the latest rerun measured data loading at 64.30%, forward at 13.08%, backward at 20.20%, and optimizer step at 2.41% of the labeled total. Thus this measured pass was input-loading dominated.

## Task 3 — DataLoader worker study

[src/study_num_workers.py](src/study_num_workers.py) measures the same 63-batch
training pass for each worker count, guarded by a Windows-safe `__main__` block.
Results are [num_workers.csv](results/num_workers.csv) and
[num_workers.png](results/num_workers.png). `num_workers=0` was fastest at
9,190.49 samples/s (0.870 s); 1, 2, 4, and 8 workers were progressively slower
(1,228.18, 713.04, 337.78, and 189.68 samples/s). Worker process startup and
IPC overhead dominate this small in-memory subset on this CPU.

## Task 4 — Batch-size study

[src/study_batch_sizes.py](src/study_batch_sizes.py) trains each batch size for
five real epochs. Results are in [batch_size.csv](results/batch_size.csv).
Throughput increased from 1,324.51 samples/s at batch 32 to 1,935.96 at batch
256, then slightly fell to 1,875.95 at batch 512. Validation accuracy was
0.869, 0.861, 0.856, 0.845, and 0.834 respectively. The CPU peak-memory proxy
is the maximum sampled `psutil.Process().memory_info().rss` minus the pre-run RSS;
it ranged from 0.00 MB (batch 64) to 32.71 MB (batch 512). Batch 256 was the
throughput winner, while batch 32 gave the highest accuracy.

## Task 5 — Schedules and optimizers

[src/study_schedules_optimizers.py](src/study_schedules_optimizers.py) runs five
independent ten-epoch experiments on the same subset. The three schedule runs
hold Adam at lr 0.001 constant initially: StepLR finished at loss 0.34082 /
accuracy **0.8710**, CosineAnnealingLR at 0.33299 / **0.8765**, and
ReduceLROnPlateau at 0.31432 / 0.8685. The fixed-schedule optimizer comparison
finished Adam at 0.31432 / 0.8685 and SGD+momentum (lr 0.01) at 0.33318 /
0.8640. Full loss/accuracy curves and learning-rate histories are serialized in
[lr_optimizer_comparison.csv](results/lr_optimizer_comparison.csv). Cosine had
the best final accuracy among schedules; Adam slightly exceeded SGD here.

## Task 6 — Gradient accumulation

[src/gradient_accumulation.py](src/gradient_accumulation.py) compares a true
batch of 256 with real mini-batches of 64 accumulated over four steps, for five
epochs and 32 optimizer steps per epoch. Both start at validation accuracy
0.7405 and finish at 0.8460 (accumulated) versus 0.8445 (true batch). Their
loss curves are nearly identical (final losses 0.45570 and 0.45726), supporting
similar behavior for this CNN and seed. See [gradient_accumulation.csv](results/gradient_accumulation.csv).

## Task 7 — Checkpoint and resume

[src/resume_training.py](src/resume_training.py) deliberately stops after epoch
3, saves model/optimizer state in [resume_epoch3.pt](results/checkpoints/resume_epoch3.pt),
then reloads it and trains through epoch 6. The continuous series is in
[resume_history.csv](results/resume_history.csv), and the combined loss/accuracy
plot is [resume_curves.png](results/resume_curves.png). Loss decreases from
0.47777 at epoch 3 to 0.42655 at epoch 4 while accuracy rises from 0.8370 to
0.8575, with no reset or restart artifact at the phase boundary. Epoch 6 ends
at 0.8565 accuracy after normal validation variation.

## Task 8 — Individual proposal

The concise project proposal is [proposal.md](proposal.md). It selects
post-training quantization and states the hard deployment limits of latency
≤50 ms per image and model size ≤5 MB, alongside the student name and ID.

## Source files

- [train.py](src/train.py), [profile_epoch.py](src/profile_epoch.py)
- [study_num_workers.py](src/study_num_workers.py), [study_batch_sizes.py](src/study_batch_sizes.py)
- [study_schedules_optimizers.py](src/study_schedules_optimizers.py)
- [gradient_accumulation.py](src/gradient_accumulation.py), [resume_training.py](src/resume_training.py)
