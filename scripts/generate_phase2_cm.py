import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

out_dir = "publication_vault/phase_2_physical_6_feature/graphs"
os.makedirs(out_dir, exist_ok=True)

# Mathematically derived directly from your CSV:
# FP = 1851 | Recall = 3.06% | Precision = 19.70% | Accuracy = 71.26%
# Resulting Matrix: TN=39794, FP=1851, FN=14382, TP=454
cm = np.array([[39794, 1851],
               [14382, 454]])

plt.figure(figsize=(8, 6))
# Using a 'Reds' color map to visually signify the danger/errors in Phase 2
sns.heatmap(cm, annot=True, fmt='d', cmap='Reds', cbar=True,
            xticklabels=['Normal Riding', 'Crash Event'],
            yticklabels=['Normal Riding', 'Crash Event'],
            annot_kws={"size": 14, "weight": "bold"})

plt.title('Domain Gap Evidence: LSTM Zero-Shot on Physical Edge Data', fontsize=14, pad=15)
plt.ylabel('Actual Real-World Class', fontsize=12, fontweight='bold')
plt.xlabel('Model Prediction (Trained Only on Simulation)', fontsize=12, fontweight='bold')

plt.tight_layout()
save_path = os.path.join(out_dir, 'figure_10_phase2_confusion_matrix.png')
plt.savefig(save_path, dpi=300)
print(f"Saved: {save_path}")