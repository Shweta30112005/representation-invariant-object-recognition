# Representation-Invariant Object Recognition
*A Comprehensive Empirical Benchmark, Combinatorial Invariance Analysis, Reliability Stress-Testing, and Closed-Loop Continual Optimization across Vision Architectures.*

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.5+](https://img.shields.io/badge/PyTorch-2.5+-ee4c2c.svg)](https://pytorch.org/)
[![Hardware: NVIDIA H200](https://img.shields.io/badge/Hardware-NVIDIA%20H200-76B900.svg)](https://www.nvidia.com)
[![Release: v1.0](https://img.shields.io/badge/Release-v1.0-brightgreen.svg)](https://github.com/Shweta30112005/representation-invariant-object-recognition/releases/tag/v1.0)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 1. Project Overview & Problem Statement

Human vision effortlessly identifies objects regardless of visual modality—whether rendered as a glossy photograph, minimalist outline, broken dashed contour, hand-drawn sketch, or solid silhouette. In contrast, deep neural networks frequently rely on **shortcut learning**, exploiting superficial surface statistics (textures, localized color distributions, dense edge continuity) and degrading catastrophically when encountering visual domain shifts.

This benchmark rigorously addresses this core failure mode through three sequential research questions:
1. **The Combinatorial Invariance Question:** *How do distinct visual representations interact during training, and what is the minimal spanning subset required to guarantee cross-modal representation invariance?*
2. **The Decision-Boundary Reliability Question:** *How fragile are model predictions under subtle spatial and stroke perturbations at test time, and does high confidence conceal out-of-distribution calibration failure?*
3. **The Closed-Loop Optimization & Continual Learning Question:** *Can failure diagnostics dynamically guide loss regularization and targeted augmentations to rescue representation collapse, and can new visual data be assimilated continually without catastrophic forgetting?*

---

## 2. Sequential Research Roadmap

```
┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                                           SEQUENTIAL RESEARCH ROADMAP                                                           │
├───────────────────┬───────────────────┬───────────────────┬───────────────────┬─────────────────────────────────┬───────────────────────────────┤
│      Phase 1      │      Phase 2      │      Phase 3      │      Phase 4      │             Phase 5             │            Phase 6            │
│ Baseline Transfer │  Interpretability │ Resolution Scaling│ Foundation Models │ 511-Combination Power Set       │ Multi-Criteria Top 10,        │
│  & Texture Bias   │  & Grad-CAM Maps  │  & Patch Collapse │  & Zero-Shot VLMs │ & Minimal Invariance Set        │ TTA & Continual Feedback Loop │
├───────────────────┼───────────────────┼───────────────────┼───────────────────┼─────────────────────────────────┼───────────────────────────────┤
│ • 9 Modalities    │ • 1,800 CAM maps  │ • 32px to 448px   │ • CLIP / EVA-CLIP │ • 2^9 - 1 = 511 combinations    │ • 7-metric Composite (CRS)    │
│ • Texture Gap     │ • Texture vs Shape│ • 64px transition │ • SigLIP-2 Base   │ • 511x9 Transfer Matrices       │ • 10-view TTA Stress Test     │
│ • ResNet: 10.3%   │   attention       │ • Patch Collapse  │ • Zero-shot 86.6% │ • Minimal Set: Org+Dsh+Skt+Sil  │ • Failure Feedback Retraining │
│   worst-case      │                   │   boundary        │   reference       │ • ResNet-50 / DINOv3 / ViT-B/16 │ • EWC Continual Learning      │
└───────────────────┴───────────────────┴───────────────────┴───────────────────┴─────────────────────────────────┴───────────────────────────────┘
```

---

## 3. Benchmark Dimensions & Modality Synthesis

Ten semantic object categories (`car`, `cat`, `chair`, `circle`, `cup`, `dog`, `rectangle`, `square`, `table`, `triangle`) spanning rigid vehicles, natural quadrupeds, household furniture, and pure geometric primitives were algorithmically transformed into **9 systematic visual modalities**:

| Representation | Key | Algorithmic Synthesis Pipeline | Invariance Mechanism Tested |
| :--- | :---: | :--- | :--- |
| **`Original`** | `Org` | Natural RGB photographic imagery | Standard photographic baseline grounding |
| **`Outline`** | `Out` | Canny edge detection with hysteresis thresholding | Continuous structural boundary & shape invariance |
| **`Dotted`** | `Dot` | Equidistant circular point sampling along perimeters | Resistance to discrete spatial sampling gaps |
| **`Dashed`** | `Dsh` | Segmented alternating stroke contours | Gestalt contour grouping & broken-line integration |
| **`Sketch`** | `Skt` | Difference-of-Gaussians (DoG) filter with pencil shading | Artistic stroke variations & hand-drawn stylization |
| **`Silhouette`** | `Sil` | Binary thresholded solid filled mask | Global shape profile vs. interior texture reliance |
| **`ColorTint_Red`** | `Red` | Amplified red chrominance channel ($\Delta \text{Ch} > 1.8\times$) | Chromatic shift robustness |
| **`ColorTint_Green`** | `Grn` | Amplified green chrominance channel ($\Delta \text{Ch} > 1.8\times$) | Chromatic shift robustness |
| **`ColorTint_Blue`** | `Blu` | Amplified blue chrominance channel ($\Delta \text{Ch} > 1.8\times$) | Chromatic shift robustness |

---

## 4. Sequential Research Progress & Empirical Findings

### Phase 1: Baseline Representation Transfer & The "Texture Gap"
*In our initial experiments, we trained standard models on natural photographs (`Train_Original`) and evaluated their out-of-distribution transfer to non-photographic representations, and vice versa.*

- **The ConvNet Texture Gap**: When trained exclusively on natural photos, **ResNet-50 collapses to 10.3% on `Outline` and 38.0% on `Silhouette`**. The network overfits to high-frequency textures and local chromatic patterns, rendering it incapable of recognizing pure geometric boundaries.
- **Asymmetric Representation Transfer**: Training on abstract representations transfers better to natural photos than vice versa: **DINOv3** trained solely on `Outline` retains **75.2%** accuracy on `Original` photos, and **77.2%** when trained solely on `Silhouette`.
- **Joint Multi-Representation Training**: Training simultaneously on all representations closes the gap for supervised models, driving ResNet-50 to **96.9%** and ViT-B/16 to **94.1%**.

---

### Phase 2: Mechanistic Explainability via Grad-CAM Activation Mapping
*To explain why convolutional networks fail on non-photographic modalities while transformers demonstrate higher resilience, we generated 1,800 Grad-CAM activation heatmaps across all models, classes, and representations (20 images per class).*

- **Convolutional Texture Fixation**: ResNet-50 activation maps concentrate heavily on interior texture regions (e.g., fur on animals, surface reflections on cars). When the object is rendered as an outline or silhouette, activations disperse erratically, triggering misclassifications.
- **Transformer Boundary Attention**: ViT-B/16 and DINOv3 feature maps distribute attention uniformly along outer structural boundaries, maintaining spatial correspondence even when surface textures are eliminated.
- **The Discontinuity Penalty**: When continuous outlines are replaced with dotted or dashed lines, convolutional feature activations experience spatial disruption, dropping by **$3.2\%$ to $8.7\%$** at standard resolution.

---

### Phase 3: Input Resolution Scaling & The Discovery of "Patch Collapse"
We evaluated all models across 6 resolutions: $32\times32$, $64\times64$, $112\times112$, $224\times224$ (baseline), $336\times336$, and $448\times448$.

- **The "Patch Collapse" Phenomenon Under Downsampling ($32\times32$)**:
  - In **ViT-B/16**, inputting a $32\times32$ image produces only $(32/16)\times(32/16) = 2\times2 = \mathbf{4}$ **patch tokens**. On broken contours (`Dashed`), thin 1-pixel line segments average out within the $16\times16$ window, causing accuracy to collapse to **12.00%** (random chance).
  - Conversely, **ResNet-50** maintains **56.33%** on `Dashed` at $32\times32$ due to dense overlapping $3\times3$ convolutional kernels that preserve spatial transitions.
  - **DINOv3** maintains **57.33%** on `Dashed` and **59.00%** on `Outline` at $32\times32$ ($5\times$ higher than supervised ViT), confirming that self-supervised pretraining provides shape resilience under severe downsampling.
- **The $64\times64$ Phase Transition**:
  - Between $32\text{px}$ and $64\text{px}$, ViT-B/16 undergoes a phase transition, leaping from **12.0% $\rightarrow$ 65.0% (+53.0%)** on `Dashed` and **22.5% $\rightarrow$ 78.7% (+56.2%)** on `Outline`.
  - Scaling beyond $224\text{px}$ to $448\text{px}$ yields marginal accuracy gains ($<0.4\%$) while quadrupling inference latency and memory footprints.

---

### Phase 4: Zero-Shot Foundation Models & Multimodal LLM Evaluation
*We next evaluated whether web-scale multimodal pretraining eliminates the need for representation-specific training by testing zero-shot Vision-Language Models (VLMs) and an autoregressive MLLM with zero fine-tuning.*

- **SigLIP-2 Base Leads Zero-Shot Robustness (86.63%)**: SigLIP-2 Base demonstrates outstanding out-of-the-box representation invariance, maintaining $>81\%$ accuracy on fragmented `Dotted` (81.50%) and `Dashed` (84.17%) lines, and $>94\%$ across color tints.
- **EVA-CLIP (83.35%) & CLIP ViT-B/32 (77.26%)**: EVA-CLIP shows strong chromatic stability (92.83% on red tint) but degrades on solid `Silhouette` (71.67%). CLIP ViT-B/32 suffers from patch tokenization limits on fine contours (`Outline` 72.00%, `Silhouette` 62.83%).
- **Qwen3-VL-8B (69.8%)**: While capable of zero-shot recognition, the large autoregressive MLLM degrades on fragmented contours (`Dashed` 62.9%, `Silhouette` 59.2%) and incurs significant computational overhead (68.4 ms latency).

---

### Phase 5: The Exhaustive 511-Combination Experiment & The Minimal Invariance Spanning Set
*Phases 1–4 demonstrated that single-variant training fails while joint 9-variant training succeeds. We conducted an exhaustive combinatorial evaluation across the entire power set of non-empty representation mixtures:*
$$\mathcal{S} = \mathcal{P}(\mathcal{R}) \setminus \{\emptyset\}, \quad |\mathcal{S}| = \sum_{k=1}^{9} \binom{9}{k} = 2^9 - 1 = \mathbf{511} \text{ combinations}$$

On an **NVIDIA H200 GPU**, we trained and evaluated:
- **ResNet-50**: 511 combinations trained $\times$ 9 test modalities = **4,599 evaluations** (46.4 hours).
- **DINOv3 (Linear Probe)**: 511 combinations trained $\times$ 9 test modalities = **4,599 evaluations** (49.9 hours).
- **ViT-B/16**: 511 combinations trained $\times$ 9 test modalities = **4,599 evaluations** (47.2 hours).
- Generated complete **$511 \times 9$ transfer matrices** per model architecture.

```
                 [ Representation Generalization by Training Size ]
   100% ┤                                                  ╭─────── ResNet-50 (95.63%)
        │                                       ╭──────────╯
    90% ┤                             ╭─────────┴────────────────── DINOv3 (91.54%)
        │                  ╭──────────╯
    80% ┤        ╭─────────╯  ◄─── DINOv3 Zero-Shot/Single-Variant Advantage (81.78%)
        │  ╭─────╯
    70% ┤  │
        │  │
    60% ┤  * ◄─── ResNet-50 Single-Variant Vulnerability (58.11%)
        └──┴─────────┴─────────┴─────────┴─────────┴─────────┴─────────┴─────────┴────
         Size 1    Size 2    Size 3    Size 4    Size 5    Size 6    Size 7    Size 9
```

1. **The Combinatorial Scaling Law & Diminishing Returns**:
   - ResNet-50 cross-modal accuracy scales non-linearly with representation diversity:
     $$\text{Size 1 } (58.11\%) \longrightarrow \text{Size 2 } (76.67\%) \longrightarrow \text{Size 3 } (85.79\%) \longrightarrow \text{Size 4 } (90.31\%) \longrightarrow \text{Size 9 } (95.63\%)$$
   - Moving from Size 1 to Size 4 yields a **$+32.20\%$ surge in generalization**. Beyond Size 4, returns diminish sharply: adding the remaining 5 variants yields only an incremental **$+5.32\%$**.

2. **The Minimal Invariance Spanning Set**:
   We discovered that an optimal 4-variant combination matches full 9-variant performance:
   $$\mathcal{M}^* = \{\text{Original}, \text{Dashed}, \text{Sketch}, \text{Silhouette}\}$$
   - ResNet-50 trained on $\mathcal{M}^*$ achieves **95.19% cross-modal accuracy** (within $0.44\%$ of full 9-variant training at 95.63%) while saving **55.6% training data and computational budget**.
   - **Mechanism**: Combining natural texture (`Original`) with broken contours (`Dashed`), hand-drawn strokes (`Sketch`), and pure silhouette geometry (`Silhouette`) covers the orthogonal basis of visual invariance.

3. **Self-Supervised Foundation Inductive Bias (DINOv3 vs. ResNet-50)**:
   - When trained on only a **single representation (Size 1)**, **DINOv3 achieves 81.78%** mean accuracy across all 9 test modalities—outperforming ResNet-50 (58.11%) by **$+23.67\%$**.
   - DINOv3's self-supervised pretraining imparts an intrinsic topological shape-bias, preventing representation collapse even under zero representational diversity.

---

## 5. Master Benchmark Tables

### Table 1: Master Model Scorecard & Computational Efficiency Benchmark
*Benchmarked under unified hardware on an NVIDIA H200 GPU (141GB VRAM, batch size 32).*

| Architecture | Paradigm | Training Combinations | Total Test Passes | Peak Cross-Modal Acc (%) | Single-Variant Floor (%) | Avg Latency (ms/img) | Throughput (FPS) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **ResNet-50** | Supervised CNN | 511 | 4,599 | **95.63%** | 58.11% | **4.20 ms** | **238.1** |
| **ViT-B/16** | Supervised ViT | 511 | 4,599 | **94.11%** | 52.40% | 4.80 ms | 208.3 |
| **DINOv3** | SSL Linear Probe | 511 | 4,599 | **92.35%** | **81.78%** | 5.74 ms | 174.2 |
| **SigLIP-2 Base** | Zero-Shot VLM | 1 (Zero-Shot) | 9 | **86.63%** | 86.63% | 11.99 ms | 83.4 |
| **EVA-CLIP (EVA02-B/16)** | Zero-Shot VLM | 1 (Zero-Shot) | 9 | **83.35%** | 83.35% | 7.14 ms | 140.1 |
| **CLIP ViT-B/32** | Zero-Shot VLM | 1 (Zero-Shot) | 9 | **77.26%** | 77.26% | 6.13 ms | 163.2 |
| **Qwen3-VL-8B** | Multimodal LLM | 1 (Zero-Shot) | 9 | **69.80%** | 69.80% | 68.40 ms | 14.6 |

---

### Table 2: Combinatorial Scaling Matrix by Training Mixture Size ($k = 1 \dots 9$)

Mean test accuracy across all 9 target representations as a function of the number of representation variants included in training:

| Combo Size ($k$) | Number of Combos $\binom{9}{k}$ | ResNet-50 Mean Acc (%) | ResNet-50 Std (%) | DINOv3 Mean Acc (%) | DINOv3 Std (%) | ViT-B/16 Mean Acc (%) | ViT-B/16 Std (%) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | 9 | 58.11% | 30.32% | **81.78%** | 6.28% | 52.40% | 28.14% |
| **2** | 36 | 76.67% | 23.52% | **86.17%** | 4.94% | 74.15% | 21.08% |
| **3** | 84 | 85.79% | 16.38% | **88.34%** | 3.41% | 83.90% | 14.82% |
| **4** | 126 | **90.31%** | 9.27% | **89.57%** | 2.45% | 88.74% | 8.16% |
| **5** | 126 | **92.12%** | 6.09% | **90.41%** | 1.84% | 90.82% | 5.22% |
| **6** | 84 | **93.17%** | 3.33% | **90.96%** | 1.63% | 91.95% | 3.10% |
| **7** | 36 | **93.64%** | 3.15% | **91.34%** | 1.40% | 92.51% | 2.80% |
| **8** | 9 | **94.17%** | 2.09% | **91.70%** | 1.09% | 93.10% | 1.95% |
| **9** | 1 | **95.63%** | — | **91.54%** | — | **94.11%** | — |

---

### Table 3: Top-Performing Training Combinations per Size (ResNet-50)

| Size | Best Combination | Shorthand | Mean Accuracy (%) | F1 Score | Training Images | Compute Savings |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: |
| **1** | Green Color Tint | `Grn` | 85.65% | 0.8486 | 2,400 | 88.9% |
| **2** | Sketch + Green Tint | `Skt+Grn` | 91.00% | 0.9105 | 4,800 | 77.8% |
| **3** | Dashed + Sketch + Blue Tint | `Dsh+Skt+Blu` | 94.13% | 0.9414 | 7,200 | 66.7% |
| **4** | **Original + Dashed + Sketch + Silhouette** | `Org+Dsh+Skt+Sil` | **95.19%** | **0.9525** | **9,600** | **55.6%** |
| **5** | Dashed + Sketch + Silhouette + Green + Blue | `Dsh+Skt+Sil+Grn+Blu` | 95.48% | 0.9554 | 12,000 | 44.4% |
| **6** | Original + Dotted + Dashed + Sketch + Silhouette + Blue | `Org+Dot+Dsh+Skt+Sil+Blu` | 95.56% | 0.9563 | 14,400 | 33.3% |
| **7** | Dotted + Dashed + Sketch + Silhouette + Red + Green + Blue | `Dot+Dsh+Skt+Sil+Red+Grn+Blu` | 95.41% | 0.9547 | 16,800 | 22.2% |
| **8** | Outline + Dotted + Dashed + Sketch + Silhouette + Red + Green + Blue | `Out+Dot+Dsh+Skt+Sil+Red+Grn+Blu` | 95.46% | 0.9553 | 19,200 | 11.1% |
| **9** | All 9 Representations Combined | `Org+Out+Dot+Dsh+Skt+Sil+Red+Grn+Blu` | **95.63%** | **0.9571** | 21,600 | 0.0% |

---

### Table 4: Representation-Wise Accuracy Breakdown (Test Modality %)

Test accuracy across each individual representation variant under full multi-representation training and zero-shot baseline evaluation:

| Model | Setup | Org | Out | Dot | Dsh | Skt | Sil | Red | Grn | Blu | Overall Mean |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **ResNet-50** | Size 9 Combined | **98.8%** | **93.2%** | **92.5%** | **93.7%** | **95.2%** | **92.7%** | **98.2%** | **98.7%** | **97.8%** | **95.63%** |
| **ViT-B/16** | Size 9 Combined | 97.5% | 91.0% | 89.8% | 91.2% | 93.4% | 90.1% | 96.8% | 97.2% | 96.9% | **94.11%** |
| **DINOv3** | Size 8 Best | 96.0% | 88.8% | 86.2% | 87.2% | 90.8% | 87.5% | 95.8% | 95.8% | 96.0% | **92.35%** |
| **SigLIP-2 Base** | Zero-Shot | 89.3% | 82.2% | 81.5% | 84.2% | 82.0% | 76.2% | 94.8% | 94.8% | 94.7% | **86.63%** |
| **EVA-CLIP** | Zero-Shot | 91.3% | 80.0% | 77.2% | 82.3% | 75.8% | 71.7% | 92.8% | 87.5% | 91.5% | **83.35%** |
| **CLIP ViT-B/32** | Zero-Shot | 88.0% | 72.0% | 66.7% | 69.2% | 77.7% | 62.8% | 86.5% | 86.0% | 86.5% | **77.26%** |
| **Qwen3-VL-8B** | Zero-Shot | 84.0% | 68.0% | 64.0% | 62.9% | 72.0% | 59.2% | 73.0% | 72.5% | 72.8% | **69.80%** |

---

### Table 5: Input Resolution Degradation & Broken Contour Failure Analysis

Accuracy on continuous (`Outline`) versus broken (`Dashed`) contours across input resolutions ($32\times32 \rightarrow 448\times448$):

| Architecture | 32×32 (Outline) | 32×32 (Dashed) | 64×64 (Dashed) | 112×112 (Dashed) | 224×224 (Baseline) | 448×448 (Dashed) | 32px Dash Collapse |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **ResNet-50** | 76.17% | **56.33%** | 83.67% | 92.00% | 94.50% | 94.17% | $-38.17\%$ |
| **ViT-B/16** | 22.50% | **12.00%** | 65.00% | 87.50% | 92.00% | 92.00% | $\mathbf{-80.00\%}$ |
| **DINOv3** | 59.00% | **57.33%** | 81.33% | 84.67% | 87.00% | 85.50% | $\mathbf{-29.67\%}$ |
| **CLIP ViT-B/32** | 68.33% | 55.50% | 66.17% | 70.17% | 70.33% | 70.00% | $-14.83\%$ |
| **EVA-CLIP** | 59.67% | 40.33% | 45.50% | 46.50% | 57.33% | 56.83% | $-17.00\%$ |
| **SigLIP-2 Base** | 71.33% | 68.50% | 76.00% | 82.50% | 84.17% | 84.17% | $-15.67\%$ |

---

## 6. Key Visualizations & Research Dashboards

<p align="center">
  <img src="results/visualizations/fig1_architecture_invariance_profiles.png" width="48%" alt="Architecture Invariance Profiles" />
  <img src="results/visualizations/fig2_texture_gap_and_discontinuity_penalty.png" width="48%" alt="Texture Gap and Discontinuity Penalty" />
</p>
<p align="center">
  <img src="results/visualizations/fig3_pareto_robustness_vs_throughput.png" width="48%" alt="Pareto Robustness vs Throughput" />
  <img src="results/visualizations/resolution_dashboard.png" width="48%" alt="Resolution Scaling Dashboard" />
</p>

---

## 7. Phase 6: Multi-Criteria Top-10 Selection, TTA & Continual Feedback Loop

Phase 6 moves beyond raw arithmetic mean accuracy by introducing a multi-criteria evaluation, reliability calibration, and closed-loop optimization framework across the 511 combinations for **ResNet-50**, **DINOv3**, and **ViT-B/16**:

### 1. Multi-Criteria Composite Robustness Score (CRS)
Evaluated combinations across 7 normalized empirical dimensions:

$$
\text{CRS} = 0.20 \hat{\text{μ}}_{\text{acc}} + 0.25 \hat{\text{WC}} + 0.10 (1 - \hat{\text{σ}}_{\text{acc}}) + 0.15 \hat{\text{μ}}_{\text{F1}} + 0.10 \hat{\text{WC}}_{\text{class}} + 0.10 \hat{\text{η}}_{\text{Pareto}} + 0.10 \hat{\text{Δ}}_{\text{OOD}}
$$

<p align="center">
  <img src="results/top10_selection/visualizations/pareto_frontier_ResNet-50.png" width="48%" alt="Pareto Frontier ResNet-50" />
  <img src="results/top10_selection/visualizations/radar_top10_ResNet-50.png" width="48%" alt="Radar Top-10 ResNet-50" />
</p>

### 2. 10-View Test-Time Augmentation (TTA) Reliability & Calibration Stress Testing
Evaluated 10 stochastic views per test image (tilts $\pm 12^\circ$, flips, zooms, translations, contour erasing for line drawings, and blur/color jitter for photos):

<p align="center">
  <img src="results/tta_calibration/visualizations/flip_rate_barplot.png" width="48%" alt="Flip Rate Barplot" />
  <img src="results/tta_calibration/visualizations/ece_comparison_barplot.png" width="48%" alt="ECE Comparison Barplot" />
</p>

- **Decision-Boundary Stability (Flip Rate):**
  - **ResNet-50:** **0.78%** mean flip rate across Top 10 combos (< 1 in 100 images flip), exhibiting exceptional convolutional boundary rigidity.
  - **ViT-B/16:** **0.74%** flip rate on Rank 2 (`Out+Dot+Dsh+Skt+Sil`) and **1.52%** overall.
  - **DINOv3:** **2.05%** mean flip rate (sensitive to stroke thickness shifts in `Sketch`).
- **Expected Calibration Error (ECE Reduction):**
  - Uncalibrated models exhibited severe overconfidence ($\text{ECE} \approx 0.142 - 0.221$).
  - Platt temperature scaling ($T \approx 1.20 - 1.22$) reduced ECE to **$0.040 - 0.047$** (a **67% to 78% reduction in calibration error**).

---

### 3. Closed-Loop Feedback Optimization & Parameter Sensitivity Study
We evaluated 4 systematic parameter configurations to identify optimal training dynamics:

<p align="center">
  <img src="results/feedback_loop/parameter_variations/visualizations/config_comparison_accuracy.png" width="48%" alt="Config Comparison Accuracy" />
  <img src="results/feedback_loop/parameter_variations/visualizations/config_radar_tradeoffs.png" width="48%" alt="Config Radar Tradeoffs" />
</p>

* **Config 1 (Standard Baseline):** $\alpha=0.05, T=1.22$, moderate augmentations, uniform weighting.
* **Config 2 (High Regularization):** $\alpha=0.12, T=1.35$. Slashes ECE by **82.2%** (down to $0.025$).
* **Config 3 (Aggressive Structural Stress):** Heavy erasing ($p=0.45$) and $25^\circ$ shear. Lifts worst-case floors by up to **+2.71%**.
* **Config 4 (Adaptive Fragility-Weighted Universal Winner):** Dynamic $2.0\times$ loss weight on fragile modalities (`Dotted`, `Dashed`, `Sketch`, `Silhouette`), achieving the highest accuracy across all three models (**ResNet-50: 96.06%**, **DINOv3: 94.49%**, **ViT-B/16: 93.94%**) and lowest flip rate (**0.22%**).

---

### Table 6: Calibrated Top Combinations Benchmark Summary

| Model | Top Rank Combo | Size | Baseline Acc | Improved Acc (Config 4) | 95% Confidence Interval | Worst-Case Floor | Final ECE | Flip Rate | Pareto Takeaway |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **ResNet-50** | `Dot+Dsh+Skt+Sil+Grn` | **5** | 94.61% | **96.06%** | [95.48%, 96.64%] | **93.35%** | **0.0333** | **0.22%** | **Primary Pareto Winner**: 44.4% compute/data savings. |
| **DINOv3** | `Org+Out+Dot+Dsh+Skt+Sil+Grn` | 7 | 92.30% | **94.49%** | [93.81%, 95.17%] | **91.31%** | 0.0333 | 0.59% | **SSL Representation Ceiling**: Optimal shape tokens. |
| **ViT-B/16** | `Dot+Dsh+Skt+Sil+Red+Grn` | 6 | 92.26% | **93.94%** | [93.24%, 94.64%] | **89.98%** | 0.0333 | 0.42% | **Transformer Baseline**: Strong global self-attention. |
| **ViT-B/16** | `Out+Dot+Dsh+Skt+Sil` | **5** | 92.24% | **93.07%** | [92.36%, 93.72%] | 89.14% | 0.0447 | **0.74%** | **Compact Transformer Set**: Low variance ($\sigma = 2.10\%$). |
| **SigLIP-2** | Zero-Shot Baseline | N/A | 86.63% | 86.63% | N/A | 76.50% | 0.1250 | N/A | Top combinations outperform SigLIP-2 by **+6.4% to +9.4%**. |

---

### 4. Continual Learning (Learning Without Forgetting via EWC)
To add new images or representations incrementally without retraining from scratch:
**Elastic Weight Consolidation (EWC):** Computes the diagonal Fisher Information Matrix (FIM) over invariant anchors.

$$
\mathcal{L}_{\mathrm{total}} = \mathcal{L}_{\mathrm{new}} + \frac{\lambda}{2} \sum_{i} F_i (\theta_i - \theta_i^*)^2
$$

Retains anchor prototypes to guarantee **0% catastrophic forgetting** on fragile line-drawing modalities.

---

### 5. Formal Statistical Significance Testing

<p align="center">
  <img src="results/statistical_testing/visualizations/fig_mcnemar_contingency.png" width="48%" alt="McNemar Contingency" />
  <img src="results/statistical_testing/visualizations/fig_wilcoxon_significance.png" width="48%" alt="Wilcoxon Significance" />
</p>

* **McNemar's Paired Chi-Square Test (Pre vs. Post Feedback):**
  * ResNet-50 ($\chi^2 = 15.02, p = 1.06 \times 10^{-4}$), DINOv3 ($\chi^2 = 17.65, p = 2.66 \times 10^{-5}$), ViT-B/16 ($\chi^2 = 13.22, p = 2.76 \times 10^{-4}$).
  * **Verdict:** $p < 0.001$ firmly rejects the Null Hypothesis across all three architectures, proving that feedback loop gains are statistically significant.
* **Wilcoxon Signed-Rank Test & Cohen's $d$:**
  * ResNet-50 is statistically superior to DINOv3 ($p = 0.0039$, Cohen's $d = -9.07$, Large Effect Size) and ViT-B/16 ($p = 0.0039$, Cohen's $d = -8.80$).

---

## 8. Quickstart: Reproducing All Results (Step-by-Step)

### Option A: Instant Verification with Pre-Trained Weights (~1 minute)
All **150 model checkpoints** across Top-10 combinations and parameter variations are saved as lightweight delta/linear heads (**7.33 MB total**):

```bash
# 1. Clone repository & install dependencies
git clone https://github.com/Shweta30112005/representation-invariant-object-recognition.git
cd representation-invariant-object-recognition
pip install -r requirements.txt

# 2. Download pre-trained 150-model release bundle (6.57 MB)
curl -L -O https://github.com/Shweta30112005/representation-invariant-object-recognition/releases/download/v1.0/checkpoints_bundle.tar.gz
tar -xzf checkpoints_bundle.tar.gz -C results/

# 3. Verify any model checkpoint in Python
python -c "
import torch
chk = torch.load('results/model_checkpoints/ResNet-50/rank01_Dot_Dsh_Skt_Sil_Grn_after_feedback_Config4_AdaptiveWinner.pt')
print('Model:', chk['model_name'], '| Calibrated Acc:', chk['metrics']['mean_acc'], '%')
"

# 4. Reproduce Statistical Testing Tables & Plots
python scripts/21_statistical_testing.py
```

### Option B: Full Pipeline Reproduction from Scratch

```bash
# 1. Multi-Criteria Top-10 Selection & Pareto CRS Extraction
python scripts/12_top10_selection.py

# 2. 10-View Test-Time Augmentation (TTA) Reliability & Calibration
python scripts/13_tta_calibration.py --views 10 --samples 60

# 3. Iterative Feedback Loop Failure Diagnosis & Retraining
python scripts/14_feedback_loop.py

# 4. Parameter Sensitivity Study & Research Experiment Suite
python scripts/17_parameter_variation_study.py
python scripts/18_research_experiment_suite.py

# 5. Generate & Verify All 150 Model Checkpoints
python scripts/19_train_and_save_top10.py

# 6. Continual Learning with EWC (Zero Catastrophic Forgetting)
python scripts/20_continual_learning_ewc.py --epochs 2

# 7. Statistical Hypothesis Testing
python scripts/21_statistical_testing.py
```

---

## 9. Structured Repository Hierarchy

```
results/
├── model_checkpoints/                          # Complete 150-model suite (< 7.4 MB)
│   ├── ResNet-50/                              # 50 checkpoints (Top 10 x 5 states)
│   ├── DINOv3/                                 # 50 checkpoints (Top 10 x 5 states)
│   ├── ViT-B_16/                               # 50 checkpoints (Top 10 x 5 states)
│   └── checkpoint_manifest.csv                 # Master manifest with metrics & SHA256 hashes
│
├── checkpoints_bundle.tar.gz                   # Release archive of all 150 checkpoints (6.57 MB)
│
├── statistical_testing/                        # Rigorous statistical significance testing
│   ├── statistical_hypothesis_results.csv      # McNemar & Wilcoxon test outputs
│   ├── statistical_testing_report.md           # Formal paper report
│   └── visualizations/                         # Contingency and Cohen's d plots
│
├── feedback_loop/                              # Phase 6 feedback loop deliverables
│   ├── parameter_variations/                   # 4-Configuration parameter sensitivity study
│   │   ├── master_parameter_sensitivity.csv    # 120-record sensitivity matrix
│   │   └── visualizations/                     # Radar & bar chart comparisons
│   └── research_experiments/                   # Scientific research experiment suite
│       ├── empirical_retrain_results.csv       # Live backpropagation evaluation logs
│       └── research_findings_report.md         # Comprehensive research chapter
│
├── top10_selection/                            # Multi-criteria rankings & Pareto frontiers
│   ├── top10_{model}.csv                       # Top 10 with explicit why_chosen rationales
│   └── visualizations/                         # Pareto charts, radar plots, CRS breakdowns
│
├── tta_calibration/                            # 10-view TTA reliability & calibration logs
│   ├── tta_results_{model}.csv                 # Per-combination reliability outputs
│   └── visualizations/                         # ECE calibration diagrams & flip-rate bars
│
├── combo_experiment/                           # Exhaustive 511-combination power-set matrices
│   ├── matrix_511x9_{model}.csv                # Full 511x9 transfer matrices
│   └── combo_results.csv                       # Consolidated master raw results (13,824 rows)
│
└── phase6_report/                              # Master deliverables & thesis chapter
    ├── phase6_tables.csv                       # Consolidated master table for all 30 top combos
    ├── phase6_report.md                        # Master thesis markdown chapter
    └── phase6_dashboard.html                   # Interactive self-contained HTML dashboard
```

---

