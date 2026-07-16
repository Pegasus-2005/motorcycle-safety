import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

out_dir = "publication_vault/phase_1_simulation_14_feature/graphs"
os.makedirs(out_dir, exist_ok=True)

# Mathematically derived from your Phase 1 sequence_baseline_darus_matrix.csv
# Total Negatives (Normal Riding) = 55,438 | Total Positives (Crash) = 2,937

# 1. Hybrid CNN-GRU
cm_hybrid = np.array([[53955, 1483], 
                      [270, 2668]])

# 2. GRU
cm_gru = np.array([[54887, 551], 
                   [230, 2707]])

# 3. LSTM
cm_lstm = np.array([[55178, 260], 
                    [207, 2730]])

fig, axes = plt.subplots(1, 3, figsize=(20, 6))
cmap = 'Blues'

# Plot Hybrid
sns.heatmap(cm_hybrid, annot=True, fmt='d', cmap=cmap, cbar=False, ax=axes[0],
            xticklabels=['Normal', 'Crash'], yticklabels=['Normal', 'Crash'],
            annot_kws={"size": 14, "weight": "bold"})
axes[0].set_title('Hybrid (CNN-GRU)\nFP: 1483 | FN: 270', fontsize=14, pad=10)
axes[0].set_ylabel('Actual Class', fontsize=12)

# Plot GRU
sns.heatmap(cm_gru, annot=True, fmt='d', cmap=cmap, cbar=False, ax=axes[1],
            xticklabels=['Normal', 'Crash'], yticklabels=['Normal', 'Crash'],
            annot_kws={"size": 14, "weight": "bold"})
axes[1].set_title('GRU\nFP: 551 | FN: 230', fontsize=14, pad=10)
axes[1].set_xlabel('Predicted Class', fontsize=12)

# Plot LSTM
sns.heatmap(cm_lstm, annot=True, fmt='d', cmap=cmap, cbar=False, ax=axes[2],
            xticklabels=['Normal', 'Crash'], yticklabels=['Normal', 'Crash'],
            annot_kws={"size": 14, "weight": "bold"})
axes[2].set_title('LSTM (Proposed)\nFP: 260 | FN: 207', fontsize=14, pad=10)

plt.suptitle('Appendix: Comparative Confusion Matrices of Sequence Architectures (Phase 1 Simulation)', 
             fontsize=16, fontweight='bold', y=1.05)
plt.tight_layout()

save_path = os.path.join(out_dir, 'figure_11_appendix_all_matrices.png')
plt.savefig(save_path, dpi=300, bbox_inches='tight')
print(f"Saved: {save_path}")