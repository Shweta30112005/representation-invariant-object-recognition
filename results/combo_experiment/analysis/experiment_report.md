# 511-Combination Training Experiment — Detailed Analysis Report
## Representation-Invariant Object Recognition
**Generated:** Results from exhaustive training experiment
**Representations (9):** Original, Outline, Dotted, Dashed, Sketch, Silhouette, ColorTint_Red, ColorTint_Green, ColorTint_Blue
**Total Training Combinations:** 2^9 - 1 = 511
**Result Matrix per Model:** 511 × 9

---
## 1. Experiment Overview

**Models evaluated:** ResNet-50, DINOv3, CLIP ViT-B/32, EVA-CLIP (EVA02-B/16), SigLIP-2 Base

- **ResNet-50**: 511 training combinations, 4599 total evaluations
- **DINOv3**: 511 training combinations, 4599 total evaluations
- **CLIP ViT-B/32**: 1 training combinations, 9 total evaluations
- **EVA-CLIP (EVA02-B/16)**: 1 training combinations, 9 total evaluations
- **SigLIP-2 Base**: 1 training combinations, 9 total evaluations

---
## 2. Model-Wise Inference Benchmarks

| Model | Architecture | Device | Avg Latency (ms) | Throughput (FPS) | Total Images |
|-------|-------------|--------|:-----------------:|:----------------:|:------------:|
| DINOv3 | Linear Probe | cuda (NVIDIA H200) | 5.74 | 174.2 | 600 |
| CLIP ViT-B/32 | Zero-Shot VLM | cuda (NVIDIA H200) | 6.13 | 163.2 | 600 |
| EVA-CLIP (EVA02-B/16) | Zero-Shot VLM | cuda (NVIDIA H200) | 7.14 | 140.1 | 600 |
| SigLIP-2 Base | Zero-Shot VLM | cuda (NVIDIA H200) | 11.99 | 83.4 | 600 |

---
## 3. Performance by Training Combination Size

This shows how accuracy changes as more variants are included in training.

### ResNet-50

| Combo Size | # Combos | Mean Acc (%) | Std Acc (%) | Mean F1 | Mean Precision | Mean Recall |
|:----------:|:--------:|:------------:|:-----------:|:-------:|:--------------:|:-----------:|
| 1 | 9 | 58.11 | 30.32 | 0.5548 | 0.5811 | 0.6360 |
| 2 | 36 | 76.67 | 23.52 | 0.7591 | 0.7667 | 0.8172 |
| 3 | 84 | 85.79 | 16.38 | 0.8542 | 0.8579 | 0.8875 |
| 4 | 126 | 90.31 | 9.27 | 0.9015 | 0.9031 | 0.9231 |
| 5 | 126 | 92.12 | 6.09 | 0.9202 | 0.9212 | 0.9350 |
| 6 | 84 | 93.17 | 3.33 | 0.9312 | 0.9317 | 0.9419 |
| 7 | 36 | 93.64 | 3.15 | 0.9358 | 0.9364 | 0.9459 |
| 8 | 9 | 94.17 | 2.09 | 0.9420 | 0.9417 | 0.9493 |
| 9 | 1 | 95.63 | nan | 0.9571 | 0.9563 | 0.9625 |

**Detailed breakdown per test representation:**

| Combo Size | Test Rep | Mean Acc (%) | Std (%) | Min (%) | Max (%) |
|:----------:|----------|:------------:|:-------:|:-------:|:-------:|
| 1 | Original | 56.02 | 38.80 | 13.17 | 97.83 |
| 1 | Outline | 74.15 | 16.30 | 35.50 | 92.17 |
| 1 | Dotted | 62.83 | 19.53 | 28.83 | 90.83 |
| 1 | Dashed | 72.56 | 15.25 | 39.50 | 92.00 |
| 1 | Sketch | 60.80 | 18.89 | 30.83 | 94.00 |
| 1 | Silhouette | 48.67 | 35.51 | 9.33 | 89.50 |
| 1 | ColorTint_Red | 50.20 | 41.22 | 14.00 | 98.33 |
| 1 | ColorTint_Green | 50.48 | 43.99 | 9.83 | 98.83 |
| 1 | ColorTint_Blue | 47.31 | 43.34 | 7.33 | 98.67 |
| 2 | Original | 79.75 | 29.34 | 15.83 | 99.17 |
| 2 | Outline | 84.65 | 9.35 | 55.83 | 93.83 |
| 2 | Dotted | 78.53 | 12.62 | 44.50 | 92.50 |
| 2 | Dashed | 83.07 | 10.80 | 55.33 | 93.33 |
| 2 | Sketch | 73.01 | 14.13 | 40.00 | 95.83 |
| 2 | Silhouette | 70.68 | 24.31 | 9.83 | 93.33 |
| 2 | ColorTint_Red | 74.16 | 36.86 | 9.33 | 99.17 |
| 2 | ColorTint_Green | 74.56 | 37.75 | 10.00 | 99.33 |
| 2 | ColorTint_Blue | 71.61 | 36.55 | 10.00 | 99.17 |
| 3 | Original | 91.33 | 18.21 | 10.17 | 99.33 |
| 3 | Outline | 88.15 | 7.63 | 44.00 | 93.67 |
| 3 | Dotted | 84.62 | 8.51 | 43.17 | 92.67 |
| 3 | Dashed | 87.10 | 8.41 | 43.33 | 93.83 |
| 3 | Sketch | 79.04 | 12.54 | 40.83 | 96.33 |
| 3 | Silhouette | 79.55 | 17.56 | 9.83 | 93.83 |
| 3 | ColorTint_Red | 88.79 | 22.61 | 9.50 | 99.17 |
| 3 | ColorTint_Green | 87.46 | 26.52 | 9.83 | 99.50 |
| 3 | ColorTint_Blue | 86.06 | 25.43 | 10.00 | 99.17 |
| 4 | Original | 96.15 | 8.03 | 45.67 | 99.67 |
| 4 | Outline | 90.41 | 4.15 | 72.33 | 94.00 |
| 4 | Dotted | 86.79 | 6.20 | 62.33 | 92.67 |
| 4 | Dashed | 89.76 | 4.88 | 70.50 | 93.67 |
| 4 | Sketch | 82.51 | 11.58 | 58.17 | 96.83 |
| 4 | Silhouette | 84.29 | 9.32 | 51.83 | 94.00 |
| 4 | ColorTint_Red | 95.06 | 10.98 | 27.50 | 99.00 |
| 4 | ColorTint_Green | 94.68 | 13.97 | 16.50 | 99.17 |
| 4 | ColorTint_Blue | 93.15 | 14.35 | 15.00 | 99.00 |
| 5 | Original | 97.69 | 3.96 | 57.00 | 99.50 |
| 5 | Outline | 91.35 | 2.50 | 77.33 | 93.67 |
| 5 | Dotted | 87.87 | 4.51 | 69.00 | 92.50 |
| 5 | Dashed | 90.91 | 3.27 | 70.50 | 93.50 |
| 5 | Sketch | 85.04 | 11.28 | 66.00 | 96.67 |
| 5 | Silhouette | 86.42 | 8.41 | 35.17 | 93.50 |
| 5 | ColorTint_Red | 96.98 | 6.34 | 32.50 | 99.33 |
| 5 | ColorTint_Green | 96.81 | 6.86 | 23.00 | 99.50 |
| 5 | ColorTint_Blue | 96.01 | 7.69 | 31.50 | 99.00 |
| 6 | Original | 98.28 | 0.68 | 95.67 | 99.50 |
| 6 | Outline | 91.70 | 2.12 | 78.33 | 94.17 |
| 6 | Dotted | 88.57 | 4.61 | 68.67 | 92.50 |
| 6 | Dashed | 91.47 | 2.41 | 77.67 | 93.67 |
| 6 | Sketch | 87.47 | 10.38 | 68.33 | 96.33 |
| 6 | Silhouette | 88.05 | 5.86 | 60.17 | 93.33 |
| 6 | ColorTint_Red | 97.77 | 1.12 | 90.17 | 99.17 |
| 6 | ColorTint_Green | 97.77 | 1.01 | 91.83 | 99.33 |
| 6 | ColorTint_Blue | 97.44 | 1.77 | 85.17 | 98.83 |
| 7 | Original | 98.00 | 1.82 | 88.17 | 99.17 |
| 7 | Outline | 91.98 | 1.01 | 87.50 | 93.17 |
| 7 | Dotted | 89.10 | 3.19 | 78.83 | 92.50 |
| 7 | Dashed | 91.79 | 1.09 | 87.50 | 93.33 |
| 7 | Sketch | 90.29 | 8.94 | 71.00 | 96.33 |
| 7 | Silhouette | 89.47 | 4.31 | 74.17 | 93.67 |
| 7 | ColorTint_Red | 97.73 | 1.65 | 88.83 | 99.17 |
| 7 | ColorTint_Green | 97.60 | 1.93 | 87.50 | 98.83 |
| 7 | ColorTint_Blue | 96.83 | 4.40 | 73.17 | 98.83 |
| 8 | Original | 97.91 | 1.12 | 95.17 | 98.83 |
| 8 | Outline | 92.06 | 1.34 | 90.00 | 93.83 |
| 8 | Dotted | 90.20 | 2.75 | 83.33 | 92.33 |
| 8 | Dashed | 91.87 | 1.26 | 89.50 | 93.33 |
| 8 | Sketch | 91.98 | 6.39 | 75.33 | 95.83 |
| 8 | Silhouette | 90.76 | 2.72 | 84.17 | 93.50 |
| 8 | ColorTint_Red | 97.63 | 1.04 | 95.17 | 98.67 |
| 8 | ColorTint_Green | 97.57 | 1.14 | 95.00 | 98.67 |
| 8 | ColorTint_Blue | 97.54 | 1.09 | 95.00 | 98.67 |
| 9 | Original | 98.83 | nan | 98.83 | 98.83 |
| 9 | Outline | 93.17 | nan | 93.17 | 93.17 |
| 9 | Dotted | 92.50 | nan | 92.50 | 92.50 |
| 9 | Dashed | 93.67 | nan | 93.67 | 93.67 |
| 9 | Sketch | 95.17 | nan | 95.17 | 95.17 |
| 9 | Silhouette | 92.67 | nan | 92.67 | 92.67 |
| 9 | ColorTint_Red | 98.17 | nan | 98.17 | 98.17 |
| 9 | ColorTint_Green | 98.67 | nan | 98.67 | 98.67 |
| 9 | ColorTint_Blue | 97.83 | nan | 97.83 | 97.83 |

### DINOv3

| Combo Size | # Combos | Mean Acc (%) | Std Acc (%) | Mean F1 | Mean Precision | Mean Recall |
|:----------:|:--------:|:------------:|:-----------:|:-------:|:--------------:|:-----------:|
| 1 | 9 | 81.78 | 6.28 | 0.8080 | 0.8178 | 0.8442 |
| 2 | 36 | 86.17 | 4.94 | 0.8547 | 0.8617 | 0.8790 |
| 3 | 84 | 88.34 | 3.41 | 0.8800 | 0.8835 | 0.8972 |
| 4 | 126 | 89.57 | 2.45 | 0.8938 | 0.8957 | 0.9078 |
| 5 | 126 | 90.41 | 1.84 | 0.9031 | 0.9040 | 0.9153 |
| 6 | 84 | 90.96 | 1.63 | 0.9092 | 0.9096 | 0.9199 |
| 7 | 36 | 91.34 | 1.40 | 0.9136 | 0.9134 | 0.9238 |
| 8 | 9 | 91.70 | 1.09 | 0.9174 | 0.9170 | 0.9274 |
| 9 | 1 | 91.54 | nan | 0.9155 | 0.9154 | 0.9293 |

**Detailed breakdown per test representation:**

| Combo Size | Test Rep | Mean Acc (%) | Std (%) | Min (%) | Max (%) |
|:----------:|----------|:------------:|:-------:|:-------:|:-------:|
| 1 | Original | 86.30 | 7.30 | 78.17 | 94.50 |
| 1 | Outline | 80.11 | 5.66 | 70.17 | 87.33 |
| 1 | Dotted | 73.07 | 8.40 | 60.50 | 85.67 |
| 1 | Dashed | 78.74 | 7.05 | 65.17 | 87.83 |
| 1 | Sketch | 79.57 | 4.92 | 73.83 | 91.00 |
| 1 | Silhouette | 76.31 | 5.10 | 69.33 | 85.00 |
| 1 | ColorTint_Red | 86.93 | 5.98 | 79.17 | 93.33 |
| 1 | ColorTint_Green | 86.93 | 6.46 | 81.17 | 94.17 |
| 1 | ColorTint_Blue | 88.07 | 5.67 | 82.33 | 94.83 |
| 2 | Original | 91.01 | 5.60 | 78.83 | 95.83 |
| 2 | Outline | 83.75 | 4.00 | 74.17 | 88.00 |
| 2 | Dotted | 79.07 | 6.39 | 65.50 | 86.33 |
| 2 | Dashed | 83.31 | 4.52 | 73.33 | 88.33 |
| 2 | Sketch | 83.66 | 4.36 | 76.33 | 91.67 |
| 2 | Silhouette | 81.18 | 4.40 | 71.17 | 87.17 |
| 2 | ColorTint_Red | 91.73 | 4.36 | 80.67 | 95.67 |
| 2 | ColorTint_Green | 90.29 | 5.83 | 76.67 | 95.67 |
| 2 | ColorTint_Blue | 91.51 | 5.05 | 79.17 | 95.83 |
| 3 | Original | 93.38 | 3.31 | 82.83 | 96.83 |
| 3 | Outline | 85.79 | 3.00 | 75.33 | 89.00 |
| 3 | Dotted | 81.87 | 5.03 | 65.33 | 87.33 |
| 3 | Dashed | 85.28 | 3.31 | 73.17 | 89.17 |
| 3 | Sketch | 85.11 | 4.24 | 78.33 | 93.17 |
| 3 | Silhouette | 83.20 | 3.53 | 72.17 | 89.17 |
| 3 | ColorTint_Red | 93.49 | 2.55 | 84.83 | 96.50 |
| 3 | ColorTint_Green | 93.19 | 3.07 | 82.17 | 96.00 |
| 3 | ColorTint_Blue | 93.80 | 2.62 | 84.33 | 96.00 |
| 4 | Original | 94.78 | 1.85 | 84.33 | 96.83 |
| 4 | Outline | 86.91 | 2.10 | 74.83 | 89.33 |
| 4 | Dotted | 83.41 | 3.59 | 68.00 | 87.50 |
| 4 | Dashed | 86.39 | 2.20 | 74.00 | 89.33 |
| 4 | Sketch | 86.11 | 4.30 | 79.17 | 92.50 |
| 4 | Silhouette | 84.62 | 2.80 | 75.33 | 89.83 |
| 4 | ColorTint_Red | 94.70 | 1.55 | 87.17 | 96.67 |
| 4 | ColorTint_Green | 94.53 | 2.19 | 80.67 | 97.00 |
| 4 | ColorTint_Blue | 94.64 | 1.47 | 84.33 | 96.33 |
| 5 | Original | 95.38 | 0.98 | 89.17 | 97.50 |
| 5 | Outline | 87.79 | 1.40 | 81.17 | 89.83 |
| 5 | Dotted | 84.82 | 2.70 | 75.83 | 88.17 |
| 5 | Dashed | 87.26 | 1.50 | 81.50 | 89.50 |
| 5 | Sketch | 87.08 | 4.24 | 79.83 | 92.33 |
| 5 | Silhouette | 85.66 | 2.56 | 79.33 | 89.33 |
| 5 | ColorTint_Red | 95.23 | 0.93 | 91.33 | 96.83 |
| 5 | ColorTint_Green | 95.21 | 1.45 | 90.50 | 97.00 |
| 5 | ColorTint_Blue | 95.21 | 0.78 | 89.33 | 96.50 |
| 6 | Original | 95.77 | 0.71 | 94.00 | 97.17 |
| 6 | Outline | 88.34 | 1.13 | 83.17 | 89.83 |
| 6 | Dotted | 85.59 | 2.04 | 78.33 | 88.00 |
| 6 | Dashed | 87.77 | 1.26 | 82.83 | 89.50 |
| 6 | Sketch | 88.02 | 4.37 | 80.50 | 92.67 |
| 6 | Silhouette | 86.29 | 2.45 | 81.50 | 89.50 |
| 6 | ColorTint_Red | 95.57 | 0.73 | 92.83 | 97.50 |
| 6 | ColorTint_Green | 95.80 | 1.34 | 91.33 | 97.17 |
| 6 | ColorTint_Blue | 95.47 | 0.63 | 93.50 | 96.67 |
| 7 | Original | 95.82 | 0.85 | 93.17 | 97.17 |
| 7 | Outline | 88.68 | 0.73 | 85.83 | 89.50 |
| 7 | Dotted | 86.19 | 1.90 | 81.00 | 88.50 |
| 7 | Dashed | 88.17 | 1.00 | 84.67 | 89.50 |
| 7 | Sketch | 88.97 | 3.99 | 80.17 | 91.83 |
| 7 | Silhouette | 86.85 | 2.20 | 82.33 | 89.83 |
| 7 | ColorTint_Red | 95.69 | 0.66 | 93.50 | 96.83 |
| 7 | ColorTint_Green | 96.00 | 0.84 | 93.17 | 97.67 |
| 7 | ColorTint_Blue | 95.71 | 0.42 | 94.83 | 96.33 |
| 8 | Original | 96.06 | 0.85 | 94.33 | 97.00 |
| 8 | Outline | 88.94 | 0.53 | 88.00 | 89.67 |
| 8 | Dotted | 86.89 | 0.86 | 85.33 | 87.83 |
| 8 | Dashed | 88.11 | 0.89 | 86.67 | 89.33 |
| 8 | Sketch | 89.72 | 3.10 | 81.67 | 91.67 |
| 8 | Silhouette | 87.63 | 1.58 | 83.83 | 88.83 |
| 8 | ColorTint_Red | 95.67 | 0.79 | 94.17 | 96.33 |
| 8 | ColorTint_Green | 96.37 | 0.61 | 95.00 | 96.83 |
| 8 | ColorTint_Blue | 95.91 | 0.63 | 94.83 | 96.83 |
| 9 | Original | 95.67 | nan | 95.67 | 95.67 |
| 9 | Outline | 88.83 | nan | 88.83 | 88.83 |
| 9 | Dotted | 86.17 | nan | 86.17 | 86.17 |
| 9 | Dashed | 87.17 | nan | 87.17 | 87.17 |
| 9 | Sketch | 90.83 | nan | 90.83 | 90.83 |
| 9 | Silhouette | 87.50 | nan | 87.50 | 87.50 |
| 9 | ColorTint_Red | 95.83 | nan | 95.83 | 95.83 |
| 9 | ColorTint_Green | 95.83 | nan | 95.83 | 95.83 |
| 9 | ColorTint_Blue | 96.00 | nan | 96.00 | 96.00 |

### CLIP ViT-B/32

| Combo Size | # Combos | Mean Acc (%) | Std Acc (%) | Mean F1 | Mean Precision | Mean Recall |
|:----------:|:--------:|:------------:|:-----------:|:-------:|:--------------:|:-----------:|
| 0 | 1 | 77.26 | nan | 0.7447 | 0.7726 | 0.7570 |

**Detailed breakdown per test representation:**

| Combo Size | Test Rep | Mean Acc (%) | Std (%) | Min (%) | Max (%) |
|:----------:|----------|:------------:|:-------:|:-------:|:-------:|
| 0 | Original | 88.00 | nan | 88.00 | 88.00 |
| 0 | Outline | 72.00 | nan | 72.00 | 72.00 |
| 0 | Dotted | 66.67 | nan | 66.67 | 66.67 |
| 0 | Dashed | 69.17 | nan | 69.17 | 69.17 |
| 0 | Sketch | 77.67 | nan | 77.67 | 77.67 |
| 0 | Silhouette | 62.83 | nan | 62.83 | 62.83 |
| 0 | ColorTint_Red | 86.50 | nan | 86.50 | 86.50 |
| 0 | ColorTint_Green | 86.00 | nan | 86.00 | 86.00 |
| 0 | ColorTint_Blue | 86.50 | nan | 86.50 | 86.50 |

### EVA-CLIP (EVA02-B/16)

| Combo Size | # Combos | Mean Acc (%) | Std Acc (%) | Mean F1 | Mean Precision | Mean Recall |
|:----------:|:--------:|:------------:|:-----------:|:-------:|:--------------:|:-----------:|
| 0 | 1 | 83.35 | nan | 0.8273 | 0.8335 | 0.8521 |

**Detailed breakdown per test representation:**

| Combo Size | Test Rep | Mean Acc (%) | Std (%) | Min (%) | Max (%) |
|:----------:|----------|:------------:|:-------:|:-------:|:-------:|
| 0 | Original | 91.33 | nan | 91.33 | 91.33 |
| 0 | Outline | 80.00 | nan | 80.00 | 80.00 |
| 0 | Dotted | 77.17 | nan | 77.17 | 77.17 |
| 0 | Dashed | 82.33 | nan | 82.33 | 82.33 |
| 0 | Sketch | 75.83 | nan | 75.83 | 75.83 |
| 0 | Silhouette | 71.67 | nan | 71.67 | 71.67 |
| 0 | ColorTint_Red | 92.83 | nan | 92.83 | 92.83 |
| 0 | ColorTint_Green | 87.50 | nan | 87.50 | 87.50 |
| 0 | ColorTint_Blue | 91.50 | nan | 91.50 | 91.50 |

### SigLIP-2 Base

| Combo Size | # Combos | Mean Acc (%) | Std Acc (%) | Mean F1 | Mean Precision | Mean Recall |
|:----------:|:--------:|:------------:|:-----------:|:-------:|:--------------:|:-----------:|
| 0 | 1 | 86.63 | nan | 0.8583 | 0.8663 | 0.8867 |

**Detailed breakdown per test representation:**

| Combo Size | Test Rep | Mean Acc (%) | Std (%) | Min (%) | Max (%) |
|:----------:|----------|:------------:|:-------:|:-------:|:-------:|
| 0 | Original | 89.33 | nan | 89.33 | 89.33 |
| 0 | Outline | 82.17 | nan | 82.17 | 82.17 |
| 0 | Dotted | 81.50 | nan | 81.50 | 81.50 |
| 0 | Dashed | 84.17 | nan | 84.17 | 84.17 |
| 0 | Sketch | 82.00 | nan | 82.00 | 82.00 |
| 0 | Silhouette | 76.17 | nan | 76.17 | 76.17 |
| 0 | ColorTint_Red | 94.83 | nan | 94.83 | 94.83 |
| 0 | ColorTint_Green | 94.83 | nan | 94.83 | 94.83 |
| 0 | ColorTint_Blue | 94.67 | nan | 94.67 | 94.67 |


---
## 4. Best & Worst Training Combinations

### ResNet-50

#### Top 20 Combinations (by average accuracy across all test variants)

| Rank | Combination | Size | Avg Acc (%) | Avg F1 | Avg Prec | Avg Recall | Train Time |
|:----:|-------------|:----:|:-----------:|:------:|:--------:|:----------:|:----------:|
| 1 | Org+Out+Dot+Dsh+Skt+Sil+Red+Grn+Blu | 9 | 95.63 | 0.9571 | 0.9563 | 0.9625 | 524.5s |
| 2 | Org+Dot+Dsh+Skt+Sil+Blu | 6 | 95.56 | 0.9563 | 0.9556 | 0.9619 | 477.8s |
| 3 | Dsh+Skt+Sil+Grn+Blu | 5 | 95.48 | 0.9554 | 0.9548 | 0.9599 | 332.7s |
| 4 | Org+Dsh+Skt+Sil+Grn | 5 | 95.48 | 0.9555 | 0.9548 | 0.9604 | 317.9s |
| 5 | Out+Dot+Dsh+Skt+Sil+Red+Grn+Blu | 8 | 95.46 | 0.9553 | 0.9546 | 0.9605 | 479.9s |
| 6 | Dot+Dsh+Skt+Sil+Red+Grn+Blu | 7 | 95.41 | 0.9547 | 0.9541 | 0.9599 | 421.9s |
| 7 | Org+Dsh+Skt+Sil | 4 | 95.19 | 0.9525 | 0.9519 | 0.9577 | 246.1s |
| 8 | Dot+Dsh+Skt+Sil+Grn+Blu | 6 | 95.15 | 0.9517 | 0.9515 | 0.9560 | 401.4s |
| 9 | Out+Dot+Skt+Sil+Grn+Blu | 6 | 95.09 | 0.9515 | 0.9509 | 0.9563 | 347.3s |
| 10 | Org+Dot+Dsh+Skt+Sil+Red+Blu | 7 | 95.09 | 0.9515 | 0.9509 | 0.9576 | 415.8s |
| 11 | Dot+Dsh+Skt+Sil+Red+Blu | 6 | 95.02 | 0.9507 | 0.9502 | 0.9552 | 376.3s |
| 12 | Org+Out+Dot+Dsh+Skt+Sil | 6 | 94.94 | 0.9502 | 0.9494 | 0.9560 | 365.7s |
| 13 | Dsh+Skt+Sil+Red+Grn+Blu | 6 | 94.93 | 0.9494 | 0.9493 | 0.9554 | 345.0s |
| 14 | Org+Dot+Dsh+Skt+Sil | 5 | 94.91 | 0.9498 | 0.9491 | 0.9550 | 321.1s |
| 15 | Org+Out+Dot+Dsh+Skt+Sil+Grn+Blu | 8 | 94.89 | 0.9493 | 0.9489 | 0.9545 | 454.3s |
| 16 | Dot+Skt+Sil+Grn | 4 | 94.89 | 0.9491 | 0.9489 | 0.9567 | 273.3s |
| 17 | Org+Out+Dsh+Skt+Sil+Blu | 6 | 94.87 | 0.9491 | 0.9487 | 0.9554 | 385.7s |
| 18 | Org+Dot+Dsh+Skt+Sil+Grn+Blu | 7 | 94.87 | 0.9494 | 0.9487 | 0.9553 | 436.1s |
| 19 | Org+Dsh+Skt+Sil+Red | 5 | 94.85 | 0.9488 | 0.9485 | 0.9554 | 324.0s |
| 20 | Org+Out+Dot+Skt+Sil+Grn | 6 | 94.85 | 0.9492 | 0.9485 | 0.9569 | 384.4s |

#### Bottom 20 Combinations

| Rank | Combination | Size | Avg Acc (%) | Avg F1 | Train Time |
|:----:|-------------|:----:|:-----------:|:------:|:----------:|
| 1 | Dsh+Sil | 2 | 60.76 | 0.6017 | 120.3s |
| 2 | Out+Dsh+Sil | 3 | 59.56 | 0.5953 | 178.3s |
| 3 | Dot+Sil | 2 | 58.11 | 0.5764 | 128.4s |
| 4 | Out+Sil | 2 | 58.06 | 0.5778 | 120.3s |
| 5 | Dot+Skt | 2 | 56.63 | 0.5617 | 145.4s |
| 6 | Skt+Sil | 2 | 53.98 | 0.5432 | 138.3s |
| 7 | Dsh+Skt | 2 | 52.74 | 0.5092 | 141.8s |
| 8 | Dot+Dsh+Skt | 3 | 51.67 | 0.4894 | 203.9s |
| 9 | Out+Dot+Skt | 3 | 49.87 | 0.4710 | 204.9s |
| 10 | Out+Skt | 2 | 47.43 | 0.4465 | 152.4s |
| 11 | Out+Dsh+Skt | 3 | 47.30 | 0.4404 | 195.7s |
| 12 | Skt | 1 | 44.04 | 0.4027 | 105.8s |
| 13 | Dot+Dsh | 2 | 43.52 | 0.4080 | 118.9s |
| 14 | Out+Dot | 2 | 42.76 | 0.3871 | 134.9s |
| 15 | Out+Dot+Dsh | 3 | 41.11 | 0.3811 | 189.4s |
| 16 | Dot | 1 | 40.52 | 0.3670 | 90.2s |
| 17 | Out+Dsh | 2 | 38.69 | 0.3477 | 129.6s |
| 18 | Dsh | 1 | 37.30 | 0.3382 | 82.6s |
| 19 | Sil | 1 | 36.72 | 0.3365 | 74.1s |
| 20 | Out | 1 | 34.54 | 0.2992 | 74.5s |

#### Best Combination per Size

| Size | Best Combination | Avg Acc (%) | Avg F1 | Train Images |
|:----:|-----------------|:-----------:|:------:|:------------:|
| 1 | Grn | 85.65 | 0.8486 | 2400 |
| 2 | Skt+Grn | 91.00 | 0.9105 | 4800 |
| 3 | Dsh+Skt+Blu | 94.13 | 0.9414 | 7200 |
| 4 | Org+Dsh+Skt+Sil | 95.19 | 0.9525 | 9600 |
| 5 | Dsh+Skt+Sil+Grn+Blu | 95.48 | 0.9554 | 12000 |
| 6 | Org+Dot+Dsh+Skt+Sil+Blu | 95.56 | 0.9563 | 14400 |
| 7 | Dot+Dsh+Skt+Sil+Red+Grn+Blu | 95.41 | 0.9547 | 16800 |
| 8 | Out+Dot+Dsh+Skt+Sil+Red+Grn+Blu | 95.46 | 0.9553 | 19200 |
| 9 | Org+Out+Dot+Dsh+Skt+Sil+Red+Grn+Blu | 95.63 | 0.9571 | 21600 |

### DINOv3

#### Top 20 Combinations (by average accuracy across all test variants)

| Rank | Combination | Size | Avg Acc (%) | Avg F1 | Avg Prec | Avg Recall | Train Time |
|:----:|-------------|:----:|:-----------:|:------:|:--------:|:----------:|:----------:|
| 1 | Org+Out+Dot+Skt+Sil+Red+Grn+Blu | 8 | 92.35 | 0.9247 | 0.9235 | 0.9308 | 561.7s |
| 2 | Org+Out+Dot+Dsh+Skt+Sil+Grn | 7 | 92.30 | 0.9240 | 0.9230 | 0.9323 | 492.1s |
| 3 | Org+Out+Dot+Dsh+Skt+Sil+Red+Blu | 8 | 92.19 | 0.9227 | 0.9219 | 0.9287 | 562.3s |
| 4 | Org+Dot+Dsh+Skt+Sil+Grn+Blu | 7 | 92.13 | 0.9224 | 0.9213 | 0.9286 | 497.9s |
| 5 | Org+Out+Dsh+Skt+Sil+Red+Grn | 7 | 92.09 | 0.9224 | 0.9209 | 0.9277 | 497.1s |
| 6 | Org+Out+Dot+Dsh+Skt+Sil+Grn+Blu | 8 | 92.09 | 0.9217 | 0.9209 | 0.9289 | 552.8s |
| 7 | Org+Dot+Dsh+Skt+Sil+Blu | 6 | 92.07 | 0.9216 | 0.9207 | 0.9296 | 418.8s |
| 8 | Org+Out+Dot+Dsh+Skt+Sil+Red+Grn | 8 | 92.06 | 0.9216 | 0.9206 | 0.9302 | 564.5s |
| 9 | Out+Dot+Dsh+Skt+Sil+Red+Grn+Blu | 8 | 92.04 | 0.9208 | 0.9204 | 0.9311 | 552.1s |
| 10 | Org+Out+Dot+Skt+Sil+Red+Blu | 7 | 92.00 | 0.9212 | 0.9200 | 0.9255 | 497.6s |
| 11 | Org+Out+Dot+Dsh+Skt+Sil+Blu | 7 | 91.91 | 0.9203 | 0.9191 | 0.9255 | 494.4s |
| 12 | Dot+Dsh+Skt+Sil+Red+Grn+Blu | 7 | 91.89 | 0.9203 | 0.9189 | 0.9261 | 493.8s |
| 13 | Dot+Dsh+Skt+Sil+Red+Grn | 6 | 91.85 | 0.9195 | 0.9185 | 0.9263 | 427.4s |
| 14 | Org+Out+Dot+Skt+Sil+Grn+Blu | 7 | 91.83 | 0.9191 | 0.9183 | 0.9262 | 497.9s |
| 15 | Org+Out+Dot+Dsh+Skt+Sil+Red | 7 | 91.83 | 0.9193 | 0.9183 | 0.9258 | 492.2s |
| 16 | Org+Dot+Dsh+Skt+Sil+Grn | 6 | 91.78 | 0.9186 | 0.9178 | 0.9247 | 426.5s |
| 17 | Out+Dot+Dsh+Skt+Sil+Red+Blu | 7 | 91.76 | 0.9183 | 0.9176 | 0.9283 | 503.8s |
| 18 | Org+Out+Dsh+Skt+Sil+Blu | 6 | 91.76 | 0.9183 | 0.9176 | 0.9268 | 423.1s |
| 19 | Dsh+Skt+Sil+Red+Grn+Blu | 6 | 91.76 | 0.9190 | 0.9176 | 0.9253 | 422.6s |
| 20 | Org+Dsh+Skt+Sil+Red+Grn | 6 | 91.69 | 0.9179 | 0.9169 | 0.9238 | 433.9s |

#### Bottom 20 Combinations

| Rank | Combination | Size | Avg Acc (%) | Avg F1 | Train Time |
|:----:|-------------|:----:|:-----------:|:------:|:----------:|
| 1 | Org+Grn | 2 | 83.59 | 0.8196 | 154.0s |
| 2 | Org+Red+Grn | 3 | 83.54 | 0.8302 | 226.2s |
| 3 | Red+Grn | 2 | 83.33 | 0.8184 | 145.0s |
| 4 | Out+Dsh | 2 | 83.24 | 0.8248 | 145.2s |
| 5 | Org+Grn+Blu | 3 | 83.19 | 0.8259 | 226.3s |
| 6 | Dot+Dsh | 2 | 83.17 | 0.8182 | 143.7s |
| 7 | Org+Red | 2 | 83.13 | 0.8247 | 149.9s |
| 8 | Grn+Blu | 2 | 83.06 | 0.8198 | 139.5s |
| 9 | Dot+Sil | 2 | 83.00 | 0.8202 | 140.1s |
| 10 | Red+Blu | 2 | 82.87 | 0.8126 | 148.9s |
| 11 | Dsh | 1 | 82.65 | 0.8190 | 67.0s |
| 12 | Org+Blu | 2 | 82.61 | 0.8137 | 144.0s |
| 13 | Out | 1 | 82.59 | 0.8182 | 75.3s |
| 14 | Org+Red+Blu | 3 | 82.50 | 0.8116 | 219.8s |
| 15 | Skt | 1 | 82.30 | 0.8185 | 83.3s |
| 16 | Out+Dot | 2 | 82.06 | 0.8110 | 146.4s |
| 17 | Dot | 1 | 81.50 | 0.7968 | 74.7s |
| 18 | Grn | 1 | 81.17 | 0.8021 | 81.1s |
| 19 | Sil | 1 | 79.11 | 0.7742 | 74.5s |
| 20 | Org | 1 | 79.06 | 0.7854 | 81.6s |

#### Best Combination per Size

| Size | Best Combination | Avg Acc (%) | Avg F1 | Train Images |
|:----:|-----------------|:-----------:|:------:|:------------:|
| 1 | Blu | 84.04 | 0.8282 | 2400 |
| 2 | Org+Dsh | 89.28 | 0.8918 | 4800 |
| 3 | Out+Skt+Grn | 90.26 | 0.9010 | 7200 |
| 4 | Dsh+Skt+Sil+Blu | 90.87 | 0.9097 | 9600 |
| 5 | Dot+Dsh+Skt+Sil+Blu | 91.59 | 0.9160 | 12000 |
| 6 | Org+Dot+Dsh+Skt+Sil+Blu | 92.07 | 0.9216 | 14400 |
| 7 | Org+Out+Dot+Dsh+Skt+Sil+Grn | 92.30 | 0.9240 | 16800 |
| 8 | Org+Out+Dot+Skt+Sil+Red+Grn+Blu | 92.35 | 0.9247 | 19200 |
| 9 | Org+Out+Dot+Dsh+Skt+Sil+Red+Grn+Blu | 91.54 | 0.9155 | 21600 |

### CLIP ViT-B/32

#### Top 20 Combinations (by average accuracy across all test variants)

| Rank | Combination | Size | Avg Acc (%) | Avg F1 | Avg Prec | Avg Recall | Train Time |
|:----:|-------------|:----:|:-----------:|:------:|:--------:|:----------:|:----------:|
| 1 | Zero-Shot | 0 | 77.26 | 0.7447 | 0.7726 | 0.7570 | 0.0s |

#### Bottom 20 Combinations

| Rank | Combination | Size | Avg Acc (%) | Avg F1 | Train Time |
|:----:|-------------|:----:|:-----------:|:------:|:----------:|
| 1 | Zero-Shot | 0 | 77.26 | 0.7447 | 0.0s |

#### Best Combination per Size

| Size | Best Combination | Avg Acc (%) | Avg F1 | Train Images |
|:----:|-----------------|:-----------:|:------:|:------------:|
| 0 | Zero-Shot | 77.26 | 0.7447 | 0 |

### EVA-CLIP (EVA02-B/16)

#### Top 20 Combinations (by average accuracy across all test variants)

| Rank | Combination | Size | Avg Acc (%) | Avg F1 | Avg Prec | Avg Recall | Train Time |
|:----:|-------------|:----:|:-----------:|:------:|:--------:|:----------:|:----------:|
| 1 | Zero-Shot | 0 | 83.35 | 0.8273 | 0.8335 | 0.8521 | 0.0s |

#### Bottom 20 Combinations

| Rank | Combination | Size | Avg Acc (%) | Avg F1 | Train Time |
|:----:|-------------|:----:|:-----------:|:------:|:----------:|
| 1 | Zero-Shot | 0 | 83.35 | 0.8273 | 0.0s |

#### Best Combination per Size

| Size | Best Combination | Avg Acc (%) | Avg F1 | Train Images |
|:----:|-----------------|:-----------:|:------:|:------------:|
| 0 | Zero-Shot | 83.35 | 0.8273 | 0 |

### SigLIP-2 Base

#### Top 20 Combinations (by average accuracy across all test variants)

| Rank | Combination | Size | Avg Acc (%) | Avg F1 | Avg Prec | Avg Recall | Train Time |
|:----:|-------------|:----:|:-----------:|:------:|:--------:|:----------:|:----------:|
| 1 | Zero-Shot | 0 | 86.63 | 0.8582 | 0.8663 | 0.8866 | 0.0s |

#### Bottom 20 Combinations

| Rank | Combination | Size | Avg Acc (%) | Avg F1 | Train Time |
|:----:|-------------|:----:|:-----------:|:------:|:----------:|
| 1 | Zero-Shot | 0 | 86.63 | 0.8582 | 0.0s |

#### Best Combination per Size

| Size | Best Combination | Avg Acc (%) | Avg F1 | Train Images |
|:----:|-----------------|:-----------:|:------:|:------------:|
| 0 | Zero-Shot | 86.63 | 0.8582 | 0 |


---
## 5. Variant Contribution Analysis

This shows how much each training variant contributes to performance when included vs excluded.

### ResNet-50

#### Average Accuracy Delta When Variant is Included in Training

| Training Variant | Avg Δ Accuracy (%) | Impact |
|-----------------|:------------------:|--------|
| Original | +5.16 | 🟢 Positive |
| ColorTint_Green | +5.08 | 🟢 Positive |
| ColorTint_Blue | +5.08 | 🟢 Positive |
| ColorTint_Red | +4.55 | 🟢 Positive |
| Sketch | +3.30 | 🟢 Positive |
| Silhouette | +2.02 | 🟢 Positive |
| Dotted | +1.67 | 🟢 Positive |
| Dashed | +1.34 | 🟢 Positive |
| Outline | +1.05 | 🟢 Positive |

#### Contribution Matrix (Training Variant → Test Variant)

| Training \ Test | Org | Out | Dot | Dsh | Skt | Sil | Red | Grn | Blu |
|---------------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Org | +7.89 | +0.53 | -0.10 | +0.47 | +1.07 | +4.31 | +9.95 | +10.26 | +12.08 |
| Out | -0.65 | +4.65 | +2.63 | +5.43 | +0.27 | -1.24 | -0.39 | -0.70 | -0.51 |
| Dot | -0.06 | +2.19 | +9.38 | +2.39 | +0.76 | +1.33 | -0.15 | -0.42 | -0.41 |
| Dsh | -0.52 | +4.61 | +4.23 | +5.80 | +0.56 | -0.64 | -0.61 | -0.80 | -0.57 |
| Skt | +0.74 | +1.24 | +1.44 | +1.29 | +23.73 | +0.60 | +0.34 | -0.20 | +0.52 |
| Sil | +1.68 | -0.17 | -1.08 | -0.29 | -0.07 | +16.30 | +1.28 | +0.30 | +0.22 |
| Red | +6.93 | +0.06 | +0.27 | +0.06 | +0.84 | +4.16 | +10.37 | +9.72 | +8.53 |
| Grn | +7.42 | +0.35 | +0.96 | +0.48 | +1.30 | +4.04 | +9.80 | +11.19 | +10.13 |
| Blu | +7.52 | +0.20 | +1.12 | +0.29 | +1.38 | +4.52 | +7.69 | +9.94 | +13.02 |

### DINOv3

#### Average Accuracy Delta When Variant is Included in Training

| Training Variant | Avg Δ Accuracy (%) | Impact |
|-----------------|:------------------:|--------|
| Sketch | +1.40 | 🟢 Positive |
| Dashed | +1.26 | 🟢 Positive |
| Dotted | +1.11 | 🟢 Positive |
| Silhouette | +1.11 | 🟢 Positive |
| Outline | +0.96 | 🟢 Positive |
| ColorTint_Green | +0.88 | 🟢 Positive |
| Original | +0.87 | 🟢 Positive |
| ColorTint_Red | +0.84 | 🟢 Positive |
| ColorTint_Blue | +0.72 | 🟢 Positive |

#### Contribution Matrix (Training Variant → Test Variant)

| Training \ Test | Org | Out | Dot | Dsh | Skt | Sil | Red | Grn | Blu |
|---------------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Org | +2.15 | +0.15 | +0.04 | -0.09 | -0.14 | +0.53 | +1.47 | +2.24 | +1.48 |
| Out | +0.22 | +2.59 | +1.97 | +2.24 | +0.25 | +0.81 | +0.32 | +0.18 | +0.07 |
| Dot | +0.09 | +1.35 | +5.73 | +1.50 | +0.29 | +1.01 | +0.10 | -0.17 | +0.08 |
| Dsh | +0.26 | +2.38 | +3.10 | +3.20 | +0.56 | +0.97 | +0.23 | +0.52 | +0.11 |
| Skt | +0.36 | +0.52 | +0.49 | +0.85 | +8.78 | +0.57 | +0.34 | +0.33 | +0.32 |
| Sil | +0.11 | +0.84 | +1.23 | +0.79 | +0.18 | +5.90 | +0.05 | +0.57 | +0.29 |
| Red | +1.93 | +0.30 | +0.23 | -0.02 | -0.08 | +0.27 | +2.07 | +1.56 | +1.31 |
| Grn | +1.65 | +0.31 | +0.13 | +0.20 | +0.05 | +0.21 | +1.31 | +2.81 | +1.30 |
| Blu | +1.70 | +0.23 | -0.06 | -0.02 | -0.13 | +0.18 | +1.43 | +1.28 | +1.91 |


---
## 6. Single vs Multi-Variant Training Comparison

Compares: (a) training on same variant as test, (b) training on all 9 variants, (c) best combo found.

### ResNet-50

| Test Variant | Same-as-Test (%) | All Combined (%) | Best Combo (%) | Best Combo | Best Size |
|-------------|:----------------:|:----------------:|:--------------:|------------|:---------:|
| Original | 97.67 | 98.83 | 99.67 | Org+Out+Skt+Red | 4 |
| Outline | 92.17 | 93.17 | 94.17 | Org+Out+Dsh+Skt+Grn+Blu | 6 |
| Dotted | 90.83 | 92.50 | 92.67 | Org+Dot+Grn | 3 |
| Dashed | 91.17 | 93.67 | 93.83 | Out+Sil+Red | 3 |
| Sketch | 94.00 | 95.17 | 96.83 | Org+Dsh+Skt+Sil | 4 |
| Silhouette | 89.50 | 92.67 | 94.00 | Dot+Skt+Sil+Red | 4 |
| ColorTint_Red | 97.33 | 98.17 | 99.33 | Org+Skt+Sil+Red+Blu | 5 |
| ColorTint_Green | 98.00 | 98.67 | 99.50 | Dsh+Red+Grn | 3 |
| ColorTint_Blue | 98.67 | 97.83 | 99.17 | Grn+Blu | 2 |

### DINOv3

| Test Variant | Same-as-Test (%) | All Combined (%) | Best Combo (%) | Best Combo | Best Size |
|-------------|:----------------:|:----------------:|:--------------:|------------|:---------:|
| Original | 93.67 | 95.67 | 97.50 | Org+Out+Skt+Red+Blu | 5 |
| Outline | 87.33 | 88.83 | 89.83 | Org+Out+Dsh+Sil+Grn | 5 |
| Dotted | 85.67 | 86.17 | 88.50 | Org+Out+Dot+Dsh+Skt+Sil+Grn | 7 |
| Dashed | 87.83 | 87.17 | 89.50 | Out+Dot+Dsh+Skt+Sil | 5 |
| Sketch | 91.00 | 90.83 | 93.17 | Out+Dsh+Skt | 3 |
| Silhouette | 85.00 | 87.50 | 89.83 | Org+Dot+Dsh+Sil | 4 |
| ColorTint_Red | 93.00 | 95.83 | 97.50 | Dot+Dsh+Skt+Sil+Red+Blu | 6 |
| ColorTint_Green | 94.17 | 95.83 | 97.67 | Org+Out+Dsh+Skt+Sil+Red+Blu | 7 |
| ColorTint_Blue | 94.83 | 96.00 | 96.83 | Org+Out+Dot+Skt+Sil+Red+Grn+Blu | 8 |

### CLIP ViT-B/32

| Test Variant | Same-as-Test (%) | All Combined (%) | Best Combo (%) | Best Combo | Best Size |
|-------------|:----------------:|:----------------:|:--------------:|------------|:---------:|
| Original | — | — | 88.00 | Zero-Shot | 0 |
| Outline | — | — | 72.00 | Zero-Shot | 0 |
| Dotted | — | — | 66.67 | Zero-Shot | 0 |
| Dashed | — | — | 69.17 | Zero-Shot | 0 |
| Sketch | — | — | 77.67 | Zero-Shot | 0 |
| Silhouette | — | — | 62.83 | Zero-Shot | 0 |
| ColorTint_Red | — | — | 86.50 | Zero-Shot | 0 |
| ColorTint_Green | — | — | 86.00 | Zero-Shot | 0 |
| ColorTint_Blue | — | — | 86.50 | Zero-Shot | 0 |

### EVA-CLIP (EVA02-B/16)

| Test Variant | Same-as-Test (%) | All Combined (%) | Best Combo (%) | Best Combo | Best Size |
|-------------|:----------------:|:----------------:|:--------------:|------------|:---------:|
| Original | — | — | 91.33 | Zero-Shot | 0 |
| Outline | — | — | 80.00 | Zero-Shot | 0 |
| Dotted | — | — | 77.17 | Zero-Shot | 0 |
| Dashed | — | — | 82.33 | Zero-Shot | 0 |
| Sketch | — | — | 75.83 | Zero-Shot | 0 |
| Silhouette | — | — | 71.67 | Zero-Shot | 0 |
| ColorTint_Red | — | — | 92.83 | Zero-Shot | 0 |
| ColorTint_Green | — | — | 87.50 | Zero-Shot | 0 |
| ColorTint_Blue | — | — | 91.50 | Zero-Shot | 0 |

### SigLIP-2 Base

| Test Variant | Same-as-Test (%) | All Combined (%) | Best Combo (%) | Best Combo | Best Size |
|-------------|:----------------:|:----------------:|:--------------:|------------|:---------:|
| Original | — | — | 89.33 | Zero-Shot | 0 |
| Outline | — | — | 82.17 | Zero-Shot | 0 |
| Dotted | — | — | 81.50 | Zero-Shot | 0 |
| Dashed | — | — | 84.17 | Zero-Shot | 0 |
| Sketch | — | — | 82.00 | Zero-Shot | 0 |
| Silhouette | — | — | 76.17 | Zero-Shot | 0 |
| ColorTint_Red | — | — | 94.83 | Zero-Shot | 0 |
| ColorTint_Green | — | — | 94.83 | Zero-Shot | 0 |
| ColorTint_Blue | — | — | 94.67 | Zero-Shot | 0 |


---
## 7. Key Findings & Insights

### ResNet-50

1. **Best overall combination**: `Org+Out+Dot+Dsh+Skt+Sil+Red+Grn+Blu` (avg accuracy: 95.63%)
2. **Optimal combo size**: 9 variants (avg accuracy: 95.63%)
4. **Best self-transfer**: ColorTint_Blue (98.7%)
5. **Worst self-transfer**: Silhouette (89.5%)

### DINOv3

1. **Best overall combination**: `Org+Out+Dot+Skt+Sil+Red+Grn+Blu` (avg accuracy: 92.35%)
2. **Optimal combo size**: 8 variants (avg accuracy: 91.70%)
3. **Diminishing returns**: Performance plateaus around 7 variants
4. **Best self-transfer**: ColorTint_Blue (94.8%)
5. **Worst self-transfer**: Silhouette (85.0%)

### CLIP ViT-B/32

1. **Best overall combination**: `Zero-Shot` (avg accuracy: 77.26%)
2. **Optimal combo size**: 0 variants (avg accuracy: 77.26%)

### EVA-CLIP (EVA02-B/16)

1. **Best overall combination**: `Zero-Shot` (avg accuracy: 83.35%)
2. **Optimal combo size**: 0 variants (avg accuracy: 83.35%)

### SigLIP-2 Base

1. **Best overall combination**: `Zero-Shot` (avg accuracy: 86.63%)
2. **Optimal combo size**: 0 variants (avg accuracy: 86.63%)


---
## 8. Output Files

| File | Description |
|------|------------|
| `combo_results.csv` | Full 511×9 results for all models |
| `combo_classwise.csv` | Per-class metrics for all combinations |
| `combo_inference.csv` | Model-wise inference benchmarks |
| `analysis/performance_by_combo_size.csv` | Aggregated stats by combination size |
| `analysis/best_worst_combinations.csv` | Top/Bottom 20 combos per model |
| `analysis/variant_contribution.csv` | Impact of each variant on accuracy |
| `analysis/single_vs_multi_comparison.csv` | Single vs multi-variant comparison |
| `analysis/*_accuracy_pct_matrix.csv` | 511×9 accuracy matrices per model |
| `analysis/*_f1_matrix.csv` | 511×9 F1 matrices per model |
| `analysis/*_precision_matrix.csv` | 511×9 precision matrices per model |
| `analysis/*_recall_matrix.csv` | 511×9 recall matrices per model |
