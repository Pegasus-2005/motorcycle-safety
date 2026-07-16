import matplotlib.pyplot as plt
import numpy as np
import os

# Output directory (We'll put this in the root of the vault as an executive summary graphic)
out_dir = "publication_vault"
os.makedirs(out_dir, exist_ok=True)

# The Data
architectures = ['Hybrid', 'GRU', 'LSTM']

# Phase 1: 14-Feature Simulation Data (From sequence_baseline_darus_matrix.csv)
p1_recall = [90.81, 92.17, 92.95]
p1_fp = [1483, 551, 260]

# Phase 2: 6-Feature Physical Data (From sequence_baseline_boubezoul_matrix.csv)
p2_recall = [18.09, 6.64, 3.06]
p2_fp = [13014, 2968, 1851]

x = np.arange(len(architectures))
width = 0.35

plt.style.use('seaborn-v0_8-whitegrid')
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Subplot 1: The Crash Sensitivity Collapse (Recall)
rects1 = ax1.bar(x - width/2, p1_recall, width, label='Phase 1 (Simulated 14-Feat)', color='#2ecc71')
rects2 = ax1.bar(x + width/2, p2_recall, width, label='Phase 2 (Physical 6-Feat)', color='#e74c3c')

ax1.set_ylabel('Crash Detection Rate (Recall %)', fontsize=12)
ax1.set_title('The Sensitivity Collapse (Domain Gap)', fontsize=14, pad=15)
ax1.set_xticks(x)
ax1.set_xticklabels(architectures, fontsize=12)
ax1.legend()

# Add value labels
for rect in rects1 + rects2:
    height = rect.get_height()
    ax1.annotate(f'{height:.1f}%',
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 3), textcoords="offset points",
                ha='center', va='bottom', fontweight='bold')

# Subplot 2: The False Alarm Explosion
rects3 = ax2.bar(x - width/2, p1_fp, width, label='Phase 1 (Simulated 14-Feat)', color='#2ecc71')
rects4 = ax2.bar(x + width/2, p2_fp, width, label='Phase 2 (Physical 6-Feat)', color='#e74c3c')

ax2.set_ylabel('Total False Positives', fontsize=12)
ax2.set_title('The False Alarm Explosion (Real-World Noise)', fontsize=14, pad=15)
ax2.set_xticks(x)
ax2.set_xticklabels(architectures, fontsize=12)
ax2.legend()

# Add value labels
for rect in rects3 + rects4:
    height = rect.get_height()
    ax2.annotate(f'{int(height)}',
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 3), textcoords="offset points",
                ha='center', va='bottom', fontweight='bold')

plt.suptitle('Sim-to-Real Domain Gap: Phase 1 vs. Phase 2 Performance Degradation', fontsize=16, fontweight='bold', y=1.05)
plt.tight_layout()

# Save the figure
save_path = os.path.join(out_dir, 'figure_9_master_domain_gap_comparison.png')
plt.savefig(save_path, dpi=300, bbox_inches='tight')
print(f"Saved: {save_path}")