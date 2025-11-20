# MPRDO-PTQ: Mixed-Precision Rate–Distortion Optimized Post-Training Quantization

This repository contains the full implementation of **MPRDO-PTQ**, a mixed-precision, rate–distortion optimized post-training quantization pipeline. The method improves over standard uniform 8-bit quantization by incorporating **layer-wise sensitivity analysis**, **block-wise optimization**, **Hessian-guided refinement**, and **adaptive rounding**.

The goal is to compress a pretrained FP32 model while maintaining accuracy, enabling deployment on resource-constrained edge devices.

---

## 1. Overview

Modern deep neural networks are computationally expensive and memory-heavy. Post-training quantization reduces model size and inference cost without requiring full retraining. However, conventional uniform-bit post-training quantization can cause significant accuracy degradation.

This project introduces an enhanced PTQ approach combining:

* Mixed-precision bit allocation
* Rate–distortion optimization
* Block-wise scale optimization
* Hessian-guided sensitivity refinement
* Adaptive weight rounding (AdaRound)

The method is implemented and evaluated on **ResNet-18 with CIFAR-10**.

---

## 2. Key Features

* Mixed-precision quantization using {4, 6, 8}-bit weights
* Sensitivity-driven bit allocation
* Block-wise RD optimization with early stopping
* Hessian-based refinement to preserve second-order structure
* Adaptive rounding for weight discretization
* Per-channel weight quantization
* Full reproducibility with seed control and detailed reporting

---

## 3. Pipeline Structure

The full quantization pipeline contains the following stages:

1. Light fine-tuning of a pretrained FP32 ResNet-18
2. Sensitivity analysis for each quantizable layer
3. Automatic mixed-precision bit assignment
4. Block-wise rate–distortion optimization
5. Hessian-guided calibration
6. Adaptive rounding of sensitive layers
7. Quantized model evaluation
8. Baseline comparison with uniform 8-bit PTQ

All results, bit distributions, and configuration files are saved in `final_rdo_ptq_output/`.

---

## 4. Results Summary

**Dataset:** CIFAR-10
**Model:** ResNet-18
**Calibration samples:** 512
**Test samples:** 2000

| Method                         | Accuracy (%) | Drop (%) | Avg Bits | Compression |
| ------------------------------ | ------------ | -------- | -------- | ----------- |
| FP32                           | 74.20        | 0.00     | 32.0     | 1.00×       |
| Uniform 8-bit PTQ (Base Paper) | 37.65        | 36.55    | 8.0      | 4.00×       |
| MPRDO-PTQ (Ours)               | 54.90        | 19.30    | 4.19     | 7.64×       |

**Gains over baseline:**

* +17.25% accuracy improvement
* +3.64× higher compression
* 47% reduction in accuracy drop

---

## 5. Directory Structure

```
.
├── final_rdo_ptq_output/
│   ├── final_report.json
│   ├── comparison.png
│   ├── quant_export.pt
│   └── quant_export.json
├── info_final.ipynb
└── README.md
```

---

## 6. How to Run

### Install dependencies

```
pip install torch torchvision scipy pandas matplotlib tqdm
```

### Train, calibrate, and evaluate

Run the entire notebook:

```
info_final.ipynb
```

The script automatically:

* Loads CIFAR-10
* Fine-tunes ResNet-18
* Runs mixed-precision RDO-PTQ
* Saves metrics, figures, and quantized parameters

---

## 7. Reproducibility

* All randomness is controlled through `seed = 42`
* PyTorch deterministic behaviors are enabled where possible
* Every run generates a timestamped JSON report for comparison

---

