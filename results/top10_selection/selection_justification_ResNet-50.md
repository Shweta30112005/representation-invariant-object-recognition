# Top 10 Combination Selection Justification — ResNet-50

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

### Rank 1: `Dot+Dsh+Skt+Sil+Grn` (Size 5)
- **CRS Score**: 0.8947
- **Mean Accuracy**: 94.61%
- **Worst-Case Accuracy**: 91.50% (variant with lowest performance)
- **Std Deviation**: 2.42% (lower = more consistent)
- **Mean F1**: 0.9465
- **Worst Class Accuracy**: 68.75%
- **Pareto Efficiency**: 0.007884 (acc/training_image)
- **OOD Transfer**: 95.46%
- **Per-Variant Accuracy**: Org: 97.5%, Out: 92.2%, Dot: 91.5%, Dsh: 91.8%, Skt: 96.3%, Sil: 92.5%, Red: 96.7%, Grn: 97.5%, Blu: 95.5%
- **Selection Rationale**: strong worst-case (≥85%); highly consistent (σ<4%); efficient (only 5 variants needed); excellent class balance (F1>0.93); strong OOD transfer (>90%)

### Rank 2: `Org+Dot+Dsh+Skt+Sil` (Size 5)
- **CRS Score**: 0.8946
- **Mean Accuracy**: 94.91%
- **Worst-Case Accuracy**: 90.83% (variant with lowest performance)
- **Std Deviation**: 2.85% (lower = more consistent)
- **Mean F1**: 0.9498
- **Worst Class Accuracy**: 68.97%
- **Pareto Efficiency**: 0.007909 (acc/training_image)
- **OOD Transfer**: 96.21%
- **Per-Variant Accuracy**: Org: 98.7%, Out: 92.5%, Dot: 90.8%, Dsh: 92.3%, Skt: 95.7%, Sil: 91.8%, Red: 97.3%, Grn: 98.0%, Blu: 97.0%
- **Selection Rationale**: strong worst-case (≥85%); highly consistent (σ<4%); efficient (only 5 variants needed); excellent class balance (F1>0.93); strong OOD transfer (>90%)

### Rank 3: `Dot+Dsh+Skt+Sil+Red+Blu` (Size 6)
- **CRS Score**: 0.8943
- **Mean Accuracy**: 95.02%
- **Worst-Case Accuracy**: 91.17% (variant with lowest performance)
- **Std Deviation**: 2.82% (lower = more consistent)
- **Mean F1**: 0.9507
- **Worst Class Accuracy**: 70.49%
- **Pareto Efficiency**: 0.006599 (acc/training_image)
- **OOD Transfer**: 96.33%
- **Per-Variant Accuracy**: Org: 98.0%, Out: 92.8%, Dot: 91.2%, Dsh: 92.8%, Skt: 95.5%, Sil: 91.3%, Red: 98.2%, Grn: 98.2%, Blu: 97.2%
- **Selection Rationale**: strong worst-case (≥85%); highly consistent (σ<4%); excellent class balance (F1>0.93); strong OOD transfer (>90%)

### Rank 4: `Out+Dot+Dsh+Skt+Sil+Red+Grn+Blu` (Size 8)
- **CRS Score**: 0.8940
- **Mean Accuracy**: 95.46%
- **Worst-Case Accuracy**: 91.83% (variant with lowest performance)
- **Std Deviation**: 2.75% (lower = more consistent)
- **Mean F1**: 0.9553
- **Worst Class Accuracy**: 68.49%
- **Pareto Efficiency**: 0.004972 (acc/training_image)
- **OOD Transfer**: 98.83%
- **Per-Variant Accuracy**: Org: 98.8%, Out: 92.8%, Dot: 91.8%, Dsh: 92.8%, Skt: 94.3%, Sil: 93.5%, Red: 98.5%, Grn: 98.2%, Blu: 98.3%
- **Selection Rationale**: strong worst-case (≥85%); highly consistent (σ<4%); excellent class balance (F1>0.93); strong OOD transfer (>90%)

### Rank 5: `Org+Dot+Dsh+Skt+Sil+Blu` (Size 6)
- **CRS Score**: 0.8938
- **Mean Accuracy**: 95.56%
- **Worst-Case Accuracy**: 92.33% (variant with lowest performance)
- **Std Deviation**: 2.62% (lower = more consistent)
- **Mean F1**: 0.9563
- **Worst Class Accuracy**: 65.48%
- **Pareto Efficiency**: 0.006636 (acc/training_image)
- **OOD Transfer**: 95.94%
- **Per-Variant Accuracy**: Org: 99.0%, Out: 92.5%, Dot: 92.3%, Dsh: 92.8%, Skt: 96.2%, Sil: 93.3%, Red: 97.5%, Grn: 97.8%, Blu: 98.5%
- **Selection Rationale**: strong worst-case (≥85%); highly consistent (σ<4%); excellent class balance (F1>0.93); strong OOD transfer (>90%)

### Rank 6: `Out+Dot+Skt+Sil+Grn+Blu` (Size 6)
- **CRS Score**: 0.8931
- **Mean Accuracy**: 95.09%
- **Worst-Case Accuracy**: 91.50% (variant with lowest performance)
- **Std Deviation**: 2.71% (lower = more consistent)
- **Mean F1**: 0.9515
- **Worst Class Accuracy**: 68.57%
- **Pareto Efficiency**: 0.006604 (acc/training_image)
- **OOD Transfer**: 96.11%
- **Per-Variant Accuracy**: Org: 98.3%, Out: 92.2%, Dot: 91.5%, Dsh: 92.0%, Skt: 95.3%, Sil: 93.3%, Red: 98.0%, Grn: 98.2%, Blu: 97.0%
- **Selection Rationale**: strong worst-case (≥85%); highly consistent (σ<4%); excellent class balance (F1>0.93); strong OOD transfer (>90%)

### Rank 7: `Org+Out+Dot+Dsh+Skt+Sil` (Size 6)
- **CRS Score**: 0.8919
- **Mean Accuracy**: 94.94%
- **Worst-Case Accuracy**: 91.67% (variant with lowest performance)
- **Std Deviation**: 2.57% (lower = more consistent)
- **Mean F1**: 0.9502
- **Worst Class Accuracy**: 66.67%
- **Pareto Efficiency**: 0.006593 (acc/training_image)
- **OOD Transfer**: 97.33%
- **Per-Variant Accuracy**: Org: 98.2%, Out: 92.5%, Dot: 91.7%, Dsh: 92.2%, Skt: 95.5%, Sil: 92.5%, Red: 97.5%, Grn: 96.5%, Blu: 98.0%
- **Selection Rationale**: strong worst-case (≥85%); highly consistent (σ<4%); excellent class balance (F1>0.93); strong OOD transfer (>90%)

### Rank 8: `Org+Out+Dot+Skt+Sil` (Size 5)
- **CRS Score**: 0.8900
- **Mean Accuracy**: 94.78%
- **Worst-Case Accuracy**: 91.50% (variant with lowest performance)
- **Std Deviation**: 2.89% (lower = more consistent)
- **Mean F1**: 0.9483
- **Worst Class Accuracy**: 65.33%
- **Pareto Efficiency**: 0.007898 (acc/training_image)
- **OOD Transfer**: 95.83%
- **Per-Variant Accuracy**: Org: 98.7%, Out: 92.0%, Dot: 91.5%, Dsh: 91.5%, Skt: 95.8%, Sil: 91.7%, Red: 97.7%, Grn: 97.8%, Blu: 96.3%
- **Selection Rationale**: strong worst-case (≥85%); highly consistent (σ<4%); efficient (only 5 variants needed); excellent class balance (F1>0.93); strong OOD transfer (>90%)

### Rank 9: `Org+Out+Dot+Dsh+Skt+Sil+Red+Grn+Blu` (Size 9)
- **CRS Score**: 0.8892
- **Mean Accuracy**: 95.63%
- **Worst-Case Accuracy**: 92.50% (variant with lowest performance)
- **Std Deviation**: 2.57% (lower = more consistent)
- **Mean F1**: 0.9571
- **Worst Class Accuracy**: 66.67%
- **Pareto Efficiency**: 0.004427 (acc/training_image)
- **OOD Transfer**: 95.63%
- **Per-Variant Accuracy**: Org: 98.8%, Out: 93.2%, Dot: 92.5%, Dsh: 93.7%, Skt: 95.2%, Sil: 92.7%, Red: 98.2%, Grn: 98.7%, Blu: 97.8%
- **Selection Rationale**: strong worst-case (≥85%); highly consistent (σ<4%); excellent class balance (F1>0.93); strong OOD transfer (>90%)

### Rank 10: `Org+Dot+Dsh+Skt+Sil+Grn` (Size 6)
- **CRS Score**: 0.8878
- **Mean Accuracy**: 94.78%
- **Worst-Case Accuracy**: 91.50% (variant with lowest performance)
- **Std Deviation**: 2.69% (lower = more consistent)
- **Mean F1**: 0.9484
- **Worst Class Accuracy**: 66.20%
- **Pareto Efficiency**: 0.006582 (acc/training_image)
- **OOD Transfer**: 96.00%
- **Per-Variant Accuracy**: Org: 97.8%, Out: 92.7%, Dot: 91.7%, Dsh: 92.3%, Skt: 94.2%, Sil: 91.5%, Red: 98.0%, Grn: 97.5%, Blu: 97.3%
- **Selection Rationale**: strong worst-case (≥85%); highly consistent (σ<4%); excellent class balance (F1>0.93); strong OOD transfer (>90%)
