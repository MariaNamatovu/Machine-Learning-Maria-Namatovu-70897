# Lab 2: NumPy neural network

## Setup

The scikit-learn `load_digits` data has 64 input features, scaled by dividing by 16. Labels are one-hot encoded into 10 classes. A stratified 70/30 train/test split uses `random_state=42`. The model is 64 -> W -> W -> 10 with ReLU hidden layers and a numerically stable softmax output; both W=16 and W=64 are evaluated.

## Checks

The forward pass produced shape `(32, 10)` with maximum probability row-sum error `2.22e-16`. The untrained cross-entropy sanity loss was `2.302585656599`, close to ln(10). The centered-difference check used epsilon `1e-5` on 10 weights across all three weight tensors: **10/10 passed** (100%), with maximum relative error `2.93e-7`. Details are in `results/gradient_check.txt`.

## Training results

Each run shuffles mini-batches of 32 each epoch and trains for 50 epochs. Final test accuracies are:

| Learning rate | Width | Final test accuracy |
|---:|---:|---:|
| 0.001 | 16 | 0.10185185 |
| 0.010 | 16 | 0.10185185 |
| 0.100 | 16 | 0.71296296 |
| 0.001 | 64 | 0.10185185 |
| 0.010 | 64 | 0.10000000 |
| 0.100 | 64 | 0.89629630 |

The complete CSV is `results/accuracy_results.csv`; all six loss curves are in `results/loss_curves.png`.

## Timing

On a batch of 32, the median of 100 runs was **0.033500 ms forward** and **0.033350 ms backward**. Forward was slightly slower in this run, so neither pass materially dominated at these small matrix sizes. Each forward layer performs one activation matrix multiplication; backward performs additional matrix products for parameter gradients and error propagation, so it has higher matrix-multiplication FLOP cost in principle. With these small matrices, fixed NumPy call overhead and timing noise can outweigh that difference. Full values are in `results/timing_results.txt`.

