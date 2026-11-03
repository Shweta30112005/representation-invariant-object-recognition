# Top 10 Combination Selection Justification — ViT-B/16

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

### Rank 1: `Dot+Dsh+Skt+Sil+Red+Grn` (Size 6)
- **CRS Score**: 0.8911
- **Mean Accuracy**: 92.26%
- **Worst-Case Accuracy**: 87.67% (variant with lowest performance)
- **Std Deviation**: 1.80% (lower = more consistent)
- **Mean F1**: 0.9205
- **Worst Class Accuracy**: 72.46%
- **Pareto Efficiency**: 0.006407 (acc/training_image)
- **OOD Transfer**: 92.22%
- **Per-Variant Accuracy**: Org: 91.2%, Out: 93.0%, Dot: 93.2%, Dsh: 92.7%, Skt: 94.3%, Sil: 87.7%, Red: 93.2%, Grn: 92.7%, Blu: 92.5%
- **Selection Rationale**: strong worst-case (≥85%); highly consistent (σ<4%); strong OOD transfer (>90%)

### Rank 2: `Out+Dot+Dsh+Skt+Sil` (Size 5)
- **CRS Score**: 0.8910
- **Mean Accuracy**: 92.24%
- **Worst-Case Accuracy**: 88.33% (variant with lowest performance)
- **Std Deviation**: 2.10% (lower = more consistent)
- **Mean F1**: 0.9213
- **Worst Class Accuracy**: 69.64%
- **Pareto Efficiency**: 0.007687 (acc/training_image)
- **OOD Transfer**: 91.12%
- **Per-Variant Accuracy**: Org: 90.5%, Out: 94.7%, Dot: 93.5%, Dsh: 94.5%, Skt: 94.7%, Sil: 88.3%, Red: 91.7%, Grn: 91.5%, Blu: 90.8%
- **Selection Rationale**: strong worst-case (≥85%); highly consistent (σ<4%); efficient (only 5 variants needed); strong OOD transfer (>90%)

### Rank 3: `Org+Out+Dot+Skt+Sil+Grn+Blu` (Size 7)
- **CRS Score**: 0.8871
- **Mean Accuracy**: 92.00%
- **Worst-Case Accuracy**: 88.33% (variant with lowest performance)
- **Std Deviation**: 1.91% (lower = more consistent)
- **Mean F1**: 0.9193
- **Worst Class Accuracy**: 73.77%
- **Pareto Efficiency**: 0.005476 (acc/training_image)
- **OOD Transfer**: 91.08%
- **Per-Variant Accuracy**: Org: 95.0%, Out: 91.2%, Dot: 91.0%, Dsh: 90.5%, Skt: 93.8%, Sil: 88.3%, Red: 91.7%, Grn: 93.3%, Blu: 93.2%
- **Selection Rationale**: strong worst-case (≥85%); highly consistent (σ<4%); strong OOD transfer (>90%)

### Rank 4: `Out+Dot+Dsh+Skt+Sil+Red` (Size 6)
- **CRS Score**: 0.8862
- **Mean Accuracy**: 91.93%
- **Worst-Case Accuracy**: 88.50% (variant with lowest performance)
- **Std Deviation**: 1.91% (lower = more consistent)
- **Mean F1**: 0.9189
- **Worst Class Accuracy**: 69.01%
- **Pareto Efficiency**: 0.006384 (acc/training_image)
- **OOD Transfer**: 92.28%
- **Per-Variant Accuracy**: Org: 92.7%, Out: 91.8%, Dot: 90.0%, Dsh: 91.8%, Skt: 95.3%, Sil: 88.5%, Red: 93.0%, Grn: 90.7%, Blu: 93.5%
- **Selection Rationale**: strong worst-case (≥85%); highly consistent (σ<4%); strong OOD transfer (>90%)

### Rank 5: `Dot+Dsh+Skt+Sil` (Size 4)
- **CRS Score**: 0.8862
- **Mean Accuracy**: 90.81%
- **Worst-Case Accuracy**: 87.83% (variant with lowest performance)
- **Std Deviation**: 1.78% (lower = more consistent)
- **Mean F1**: 0.9067
- **Worst Class Accuracy**: 72.29%
- **Pareto Efficiency**: 0.009460 (acc/training_image)
- **OOD Transfer**: 90.57%
- **Per-Variant Accuracy**: Org: 89.0%, Out: 92.2%, Dot: 90.2%, Dsh: 92.3%, Skt: 94.2%, Sil: 87.8%, Red: 90.7%, Grn: 90.2%, Blu: 90.8%
- **Selection Rationale**: strong worst-case (≥85%); highly consistent (σ<4%); efficient (only 4 variants needed); strong OOD transfer (>90%)

### Rank 6: `Org+Out+Dot+Dsh+Skt+Sil+Red+Grn+Blu` (Size 9)
- **CRS Score**: 0.8835
- **Mean Accuracy**: 92.50%
- **Worst-Case Accuracy**: 89.00% (variant with lowest performance)
- **Std Deviation**: 1.67% (lower = more consistent)
- **Mean F1**: 0.9241
- **Worst Class Accuracy**: 66.67%
- **Pareto Efficiency**: 0.004282 (acc/training_image)
- **OOD Transfer**: 92.50%
- **Per-Variant Accuracy**: Org: 95.2%, Out: 92.0%, Dot: 92.7%, Dsh: 91.7%, Skt: 94.2%, Sil: 89.0%, Red: 93.5%, Grn: 91.5%, Blu: 92.8%
- **Selection Rationale**: strong worst-case (≥85%); highly consistent (σ<4%); strong OOD transfer (>90%)

### Rank 7: `Out+Dot+Skt+Sil+Red` (Size 5)
- **CRS Score**: 0.8812
- **Mean Accuracy**: 91.76%
- **Worst-Case Accuracy**: 88.00% (variant with lowest performance)
- **Std Deviation**: 1.95% (lower = more consistent)
- **Mean F1**: 0.9179
- **Worst Class Accuracy**: 64.29%
- **Pareto Efficiency**: 0.007647 (acc/training_image)
- **OOD Transfer**: 92.21%
- **Per-Variant Accuracy**: Org: 92.7%, Out: 91.8%, Dot: 89.2%, Dsh: 91.5%, Skt: 94.5%, Sil: 88.0%, Red: 93.5%, Grn: 91.5%, Blu: 93.2%
- **Selection Rationale**: strong worst-case (≥85%); highly consistent (σ<4%); efficient (only 5 variants needed); strong OOD transfer (>90%)

### Rank 8: `Out+Dot+Skt+Sil` (Size 4)
- **CRS Score**: 0.8808
- **Mean Accuracy**: 90.98%
- **Worst-Case Accuracy**: 88.00% (variant with lowest performance)
- **Std Deviation**: 1.66% (lower = more consistent)
- **Mean F1**: 0.9097
- **Worst Class Accuracy**: 64.94%
- **Pareto Efficiency**: 0.009477 (acc/training_image)
- **OOD Transfer**: 91.33%
- **Per-Variant Accuracy**: Org: 91.8%, Out: 90.3%, Dot: 90.7%, Dsh: 89.5%, Skt: 93.2%, Sil: 88.0%, Red: 92.7%, Grn: 89.8%, Blu: 92.8%
- **Selection Rationale**: strong worst-case (≥85%); highly consistent (σ<4%); efficient (only 4 variants needed); strong OOD transfer (>90%)

### Rank 9: `Org+Dot+Dsh+Skt+Sil+Red+Grn+Blu` (Size 8)
- **CRS Score**: 0.8804
- **Mean Accuracy**: 93.15%
- **Worst-Case Accuracy**: 87.67% (variant with lowest performance)
- **Std Deviation**: 2.48% (lower = more consistent)
- **Mean F1**: 0.9313
- **Worst Class Accuracy**: 64.62%
- **Pareto Efficiency**: 0.004851 (acc/training_image)
- **OOD Transfer**: 91.83%
- **Per-Variant Accuracy**: Org: 95.8%, Out: 91.8%, Dot: 92.0%, Dsh: 91.5%, Skt: 94.5%, Sil: 87.7%, Red: 94.7%, Grn: 94.8%, Blu: 95.5%
- **Selection Rationale**: strong worst-case (≥85%); highly consistent (σ<4%); excellent class balance (F1>0.93); strong OOD transfer (>90%)

### Rank 10: `Org+Out+Dot+Skt+Sil+Red+Grn` (Size 7)
- **CRS Score**: 0.8802
- **Mean Accuracy**: 92.11%
- **Worst-Case Accuracy**: 88.83% (variant with lowest performance)
- **Std Deviation**: 1.80% (lower = more consistent)
- **Mean F1**: 0.9198
- **Worst Class Accuracy**: 65.79%
- **Pareto Efficiency**: 0.005483 (acc/training_image)
- **OOD Transfer**: 91.67%
- **Per-Variant Accuracy**: Org: 95.2%, Out: 90.8%, Dot: 91.5%, Dsh: 90.8%, Skt: 93.2%, Sil: 88.8%, Red: 94.2%, Grn: 92.0%, Blu: 92.5%
- **Selection Rationale**: strong worst-case (≥85%); highly consistent (σ<4%); strong OOD transfer (>90%)
