# Top 10 Combination Selection Justification — DINOv3

## Composite Robustness Score (CRS) Framework
Each combination is scored using 7 weighted metrics:
- Mean Accuracy (20%) — overall performance
- Worst-Case Accuracy (25%) — robustness under worst variant
- Stability/1-σ (10%) — consistency across variants
- Mean F1 (15%) — class-balanced performance
- Worst-Case Class Accuracy (10%) — no class left behind
- Pareto Efficiency (10%) — accuracy per training image
- OOD Transfer Gap (10%) — generalization to unseen variants

## Selected Combinations

### Rank 1: `Org+Out+Dot+Dsh+Skt+Sil+Grn` (Size 7)
- **CRS Score**: 0.8735
- **Mean Accuracy**: 92.30%
- **Worst-Case Accuracy**: 88.50% (variant with lowest performance)
- **Std Deviation**: 3.49% (lower = more consistent)
- **Mean F1**: 0.9240
- **Worst Class Accuracy**: 58.24%
- **Pareto Efficiency**: 0.005494 (acc/training_image)
- **OOD Transfer**: 95.67%
- **Per-Variant Accuracy**: Org: 96.5%, Out: 89.2%, Dot: 88.5%, Dsh: 88.7%, Skt: 91.3%, Sil: 88.7%, Red: 95.7%, Grn: 96.5%, Blu: 95.7%
- **Selection Rationale**: strong worst-case (≥85%); highly consistent (σ<4%); strong OOD transfer (>90%)

### Rank 2: `Org+Out+Dot+Dsh+Skt+Sil+Red+Blu` (Size 8)
- **CRS Score**: 0.8639
- **Mean Accuracy**: 92.19%
- **Worst-Case Accuracy**: 87.83% (variant with lowest performance)
- **Std Deviation**: 3.68% (lower = more consistent)
- **Mean F1**: 0.9227
- **Worst Class Accuracy**: 56.98%
- **Pareto Efficiency**: 0.004801 (acc/training_image)
- **OOD Transfer**: 96.83%
- **Per-Variant Accuracy**: Org: 96.3%, Out: 88.0%, Dot: 87.8%, Dsh: 89.3%, Skt: 91.5%, Sil: 88.5%, Red: 95.8%, Grn: 96.8%, Blu: 95.5%
- **Selection Rationale**: strong worst-case (≥85%); highly consistent (σ<4%); strong OOD transfer (>90%)

### Rank 3: `Org+Dot+Dsh+Skt+Sil+Blu` (Size 6)
- **CRS Score**: 0.8540
- **Mean Accuracy**: 92.07%
- **Worst-Case Accuracy**: 87.83% (variant with lowest performance)
- **Std Deviation**: 3.49% (lower = more consistent)
- **Mean F1**: 0.9216
- **Worst Class Accuracy**: 58.24%
- **Pareto Efficiency**: 0.006394 (acc/training_image)
- **OOD Transfer**: 93.28%
- **Per-Variant Accuracy**: Org: 96.7%, Out: 89.2%, Dot: 88.0%, Dsh: 89.0%, Skt: 91.5%, Sil: 87.8%, Red: 95.0%, Grn: 95.7%, Blu: 95.8%
- **Selection Rationale**: strong worst-case (≥85%); highly consistent (σ<4%); strong OOD transfer (>90%)

### Rank 4: `Org+Out+Dot+Dsh+Skt+Sil+Grn+Blu` (Size 8)
- **CRS Score**: 0.8499
- **Mean Accuracy**: 92.09%
- **Worst-Case Accuracy**: 87.50% (variant with lowest performance)
- **Std Deviation**: 3.67% (lower = more consistent)
- **Mean F1**: 0.9217
- **Worst Class Accuracy**: 58.06%
- **Pareto Efficiency**: 0.004796 (acc/training_image)
- **OOD Transfer**: 94.50%
- **Per-Variant Accuracy**: Org: 96.3%, Out: 89.2%, Dot: 87.5%, Dsh: 88.7%, Skt: 91.7%, Sil: 88.0%, Red: 94.5%, Grn: 96.7%, Blu: 96.3%
- **Selection Rationale**: strong worst-case (≥85%); highly consistent (σ<4%); strong OOD transfer (>90%)

### Rank 5: `Org+Out+Dot+Dsh+Skt+Sil+Red` (Size 7)
- **CRS Score**: 0.8498
- **Mean Accuracy**: 91.83%
- **Worst-Case Accuracy**: 87.17% (variant with lowest performance)
- **Std Deviation**: 3.49% (lower = more consistent)
- **Mean F1**: 0.9193
- **Worst Class Accuracy**: 59.77%
- **Pareto Efficiency**: 0.005466 (acc/training_image)
- **OOD Transfer**: 95.08%
- **Per-Variant Accuracy**: Org: 96.3%, Out: 89.2%, Dot: 87.2%, Dsh: 89.5%, Skt: 91.3%, Sil: 87.3%, Red: 95.5%, Grn: 94.8%, Blu: 95.3%
- **Selection Rationale**: strong worst-case (≥85%); highly consistent (σ<4%); strong OOD transfer (>90%)

### Rank 6: `Out+Dot+Dsh+Skt+Sil+Red` (Size 6)
- **CRS Score**: 0.8485
- **Mean Accuracy**: 91.59%
- **Worst-Case Accuracy**: 87.17% (variant with lowest performance)
- **Std Deviation**: 3.35% (lower = more consistent)
- **Mean F1**: 0.9165
- **Worst Class Accuracy**: 60.47%
- **Pareto Efficiency**: 0.006361 (acc/training_image)
- **OOD Transfer**: 95.06%
- **Per-Variant Accuracy**: Org: 96.2%, Out: 89.3%, Dot: 87.2%, Dsh: 88.8%, Skt: 91.7%, Sil: 87.3%, Red: 94.8%, Grn: 95.5%, Blu: 93.5%
- **Selection Rationale**: strong worst-case (≥85%); highly consistent (σ<4%); strong OOD transfer (>90%)

### Rank 7: `Org+Out+Dot+Dsh+Skt+Sil+Blu` (Size 7)
- **CRS Score**: 0.8481
- **Mean Accuracy**: 91.91%
- **Worst-Case Accuracy**: 86.67% (variant with lowest performance)
- **Std Deviation**: 3.67% (lower = more consistent)
- **Mean F1**: 0.9203
- **Worst Class Accuracy**: 59.55%
- **Pareto Efficiency**: 0.005471 (acc/training_image)
- **OOD Transfer**: 95.67%
- **Per-Variant Accuracy**: Org: 95.8%, Out: 89.5%, Dot: 86.7%, Dsh: 88.8%, Skt: 91.7%, Sil: 87.5%, Red: 95.3%, Grn: 96.0%, Blu: 95.8%
- **Selection Rationale**: strong worst-case (≥85%); highly consistent (σ<4%); strong OOD transfer (>90%)

### Rank 8: `Out+Dot+Dsh+Skt+Sil+Red+Blu` (Size 7)
- **CRS Score**: 0.8451
- **Mean Accuracy**: 91.76%
- **Worst-Case Accuracy**: 87.50% (variant with lowest performance)
- **Std Deviation**: 3.36% (lower = more consistent)
- **Mean F1**: 0.9183
- **Worst Class Accuracy**: 57.58%
- **Pareto Efficiency**: 0.005462 (acc/training_image)
- **OOD Transfer**: 94.42%
- **Per-Variant Accuracy**: Org: 94.8%, Out: 89.0%, Dot: 87.5%, Dsh: 88.7%, Skt: 91.7%, Sil: 88.0%, Red: 96.3%, Grn: 94.0%, Blu: 95.8%
- **Selection Rationale**: strong worst-case (≥85%); highly consistent (σ<4%); strong OOD transfer (>90%)

### Rank 9: `Org+Out+Dot+Dsh+Skt+Sil+Red+Grn` (Size 8)
- **CRS Score**: 0.8409
- **Mean Accuracy**: 92.06%
- **Worst-Case Accuracy**: 86.50% (variant with lowest performance)
- **Std Deviation**: 3.95% (lower = more consistent)
- **Mean F1**: 0.9216
- **Worst Class Accuracy**: 57.14%
- **Pareto Efficiency**: 0.004795 (acc/training_image)
- **OOD Transfer**: 95.50%
- **Per-Variant Accuracy**: Org: 96.8%, Out: 89.0%, Dot: 86.5%, Dsh: 87.8%, Skt: 91.3%, Sil: 88.8%, Red: 95.8%, Grn: 96.8%, Blu: 95.5%
- **Selection Rationale**: strong worst-case (≥85%); highly consistent (σ<4%); strong OOD transfer (>90%)

### Rank 10: `Out+Dot+Skt+Sil+Grn` (Size 5)
- **CRS Score**: 0.8409
- **Mean Accuracy**: 91.54%
- **Worst-Case Accuracy**: 87.17% (variant with lowest performance)
- **Std Deviation**: 3.22% (lower = more consistent)
- **Mean F1**: 0.9163
- **Worst Class Accuracy**: 58.70%
- **Pareto Efficiency**: 0.007628 (acc/training_image)
- **OOD Transfer**: 93.08%
- **Per-Variant Accuracy**: Org: 95.0%, Out: 89.2%, Dot: 87.2%, Dsh: 88.3%, Skt: 92.0%, Sil: 87.8%, Red: 94.2%, Grn: 95.3%, Blu: 94.8%
- **Selection Rationale**: strong worst-case (≥85%); highly consistent (σ<4%); efficient (only 5 variants needed); strong OOD transfer (>90%)
