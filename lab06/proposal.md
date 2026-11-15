# Lab 06 Individual Project Proposal

**Student Name:** Maria Namatovu  
**Student ID:** 70897

## Problem Statement

Fashion-MNIST image classification is a compact proxy for an edge-vision workload. The project will deploy the Lab 06 convolutional neural network (CNN) to classify a 28×28 grayscale clothing image into one of ten garment categories. The goal is to preserve useful recognition quality while making inference predictable and inexpensive on CPU-only or resource-constrained devices. The work will measure accuracy, latency, memory behavior, and model size rather than optimizing accuracy in isolation.

## Dataset

The project uses the standard Fashion-MNIST dataset: 60,000 labeled training images and 10,000 held-out test images, each 28×28 pixels in grayscale across ten classes. Images are converted to tensors and normalized with the dataset statistics used throughout this repository. A fixed seed and a documented train/validation split will make experiments reproducible. The dataset is suitable for the proposal because it is public, small enough for repeated experiments, and representative of low-resolution classification at the edge.

## Success Metric

Primary success is validation/test accuracy retained after optimization, with a target of at least 88% on the held-out evaluation subset used by the experiment. The model must also satisfy both deployment constraints below: single-image CPU latency **≤ 50 ms** and serialized model size **≤ 5 MB**. Accuracy, latency, and size will be reported together so an apparent accuracy gain cannot hide a deployment regression.

## Deployment Constraints

Latency must be **≤ 50 ms per image** and model size must be **≤ 5 MB**; these are hard constraints. The target environment is CPU-only, with limited memory and no assumption of a GPU runtime. The final artifact should therefore use a fixed input shape, avoid dynamic services, and keep preprocessing deterministic. Measurements will include warm-up and repeated timing on the intended CPU class where possible.

## Planned Optimization

The selected optimization is **post-training quantization**. Dynamic or static INT8 quantization can reduce stored weights and improve CPU inference efficiency without changing the model architecture or requiring a larger training pipeline. The baseline FP32 CNN will be evaluated first; an INT8 version will then be calibrated/evaluated on representative Fashion-MNIST data. The quantized model will be accepted only if it remains within the accuracy target and meets both hard constraints. This choice is preferable to adding a new architecture because it directly addresses size and CPU latency while keeping the experiment auditable and the deployment interface unchanged.
