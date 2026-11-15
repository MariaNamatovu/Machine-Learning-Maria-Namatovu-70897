# Lab 3 — Architectures and their costs

## Setup

Fashion-MNIST was loaded with `torchvision.datasets.FashionMNIST`. Each image uses
`ToTensor()` followed by `Normalize((0.2860,), (0.3530,))`. The official 60,000-image
training set was split with seed 42 into 55,000 training and 5,000 validation
examples; the validation split was kept separate from training and test evaluation. The official 10,000-image test set was held out. Every loader uses batch
size 128. Validation was reserved for monitoring; the accuracies below are test-set
accuracies.

The three models were implemented exactly as specified:

- **MLP:** Flatten(784), Linear(784,256), ReLU, Linear(256,128), ReLU,
  Linear(128,10).
- **CNN:** Conv2d(1,16,3,padding=1), ReLU, MaxPool2d(2), Conv2d(16,32,3,
  padding=1), ReLU, MaxPool2d(2), Flatten, Linear(32×7×7,10).
- **Attention:** 16 non-overlapping 7×7 patches, Linear(49,64), two
  `TransformerEncoderLayer(d_model=64, nhead=4, dim_feedforward=128,
  batch_first=True)` layers, mean pooling, and Linear(64,10).

Each model was trained for 5 epochs with Adam (learning rate 1e-3). CPU latency
uses `torch.inference_mode()`, five warm-up calls, and the specified median of 100
(batch 1) or 20 (batch 128) timed calls.

## Parameters and FLOPs

| Model | Parameters | Target-band status | Deviation from [100,000,150,000] | Hand FLOPs/sample | thop FLOPs | thop parameters |
|---|---:|---|---:|---:|---:|---:|
| MLP | 235,146 | Above band | +85,146 | 469,504 | 234,752 | 235,146 |
| CNN | 20,490 | Below band | -79,510 | 2,063,488 | 1,031,744 | 20,490 |
| Attention | 70,794 | Below band | -29,206 | 2,329,856 | 50,816 | 3,850 |

Hand calculations use the required two-FLOPs-per multiply/add formulas and are
shown in [on_paper_calculations.txt](on_paper_calculations.txt). The MLP and CNN
thop values are exactly half of the hand values because thop's default MAC
accounting treats one multiply-accumulate as one operation. The attention thop
value is not a reliable full-model FLOP count: the composite
`TransformerEncoderLayer` is only partially recognized (and its reported
parameter count is incomplete). Biases, activations, pooling, softmax, layer
normalization, residual/mean-pool operations, and other bookkeeping are also
outside the hand formula set. Thus the hand count is the required comparable
arithmetic estimate, while thop is an automated tool estimate with this
limitation.

## Measured results

### Final test accuracy

| Model | Final test accuracy |
|---|---:|
| MLP | 0.880700 |
| CNN | 0.894500 |
| Attention | 0.837600 |

### Median CPU inference latency

| Model | Batch 1 (100 runs, ms) | Batch 128 (20 runs, ms) |
|---|---:|---:|
| MLP | 0.078250 | 1.141200 |
| CNN | 0.271100 | 29.480800 |
| Attention | 0.451750 | 17.532500 |

The batch-128 values are supplementary throughput-oriented measurements; the
required latency plot uses batch-1 latency.

## Cost effectiveness

CNN is the strongest parameter-efficient choice: it has the highest accuracy
(0.8945) while using only 20,490 parameters. Accuracy divided by parameter count
is approximately 4.37×10^-5 for CNN, compared with 3.75×10^-6 for MLP and
1.18×10^-5 for Attention. CNN kernels also have strong spatial locality and
reuse the same small weights over many image locations, making cache behavior
and compute per byte favorable.

MLP is the measured batch-1 latency winner (0.078250 ms), so its accuracy divided
by batch-1 latency is approximately 11.25 accuracy points per millisecond, versus
3.30 for CNN and 1.85 for Attention. Its dense layers nevertheless stream large
contiguous weight matrices and demand more memory bandwidth. Attention has
additional dependent QKV, score, softmax, and weighted-value stages, plus
intermediate activation traffic and less weight reuse per FLOP; this helps explain
its slower batch-1 latency and lower measured accuracy here despite its smaller
parameter count than the MLP. Therefore CNN is the best overall accuracy/parameter
choice, while MLP is the best latency-constrained choice in these measurements.

## Result files

- [model_costs.csv](results/model_costs.csv)
- [training_results.csv](results/training_results.csv)
- [latency_results.csv](results/latency_results.csv)
- [accuracy_vs_parameters.png](results/accuracy_vs_parameters.png)
- [accuracy_vs_latency.png](results/accuracy_vs_latency.png)
- [on_paper_calculations.txt](on_paper_calculations.txt)

The complete reproducible pipeline is [src/run_lab03.py](src/run_lab03.py).


