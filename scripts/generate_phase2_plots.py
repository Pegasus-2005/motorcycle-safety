import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from math import pi

# Create graphs directory if it doesn't exist
out_dir = "publication_vault/phase_2_physical_6_feature/graphs"
os.makedirs(out_dir, exist_ok=True)

# Data from our physical testing (Objective Data)
models = ['Random Forest', 'Gradient Boosting', 'AdaBoost', 'MLP', 'Hybrid (CNN-GRU)', 'GRU', 'LSTM']
categories = ['Traditional (Static)', 'Traditional (Static)', 'Traditional (Static)', 'Traditional (Static)', 'Sequence (Temporal)', 'Sequence (Temporal)', 'Sequence (Temporal)']

accuracy = [35.71, 37.18, 39.96, 40.08, 55.44, 70.22, 71.26]
precision = [18.47, 17.39, 17.96, 16.13, 17.09, 24.92, 19.70]
recall = [42.14, 36.87, 35.82, 30.31, 18.09, 6.64, 3.06]
f1_scores = [25.68, 23.63, 23.93, 21.05, 17.58, 10.49, 5.30]
false_positives = [27961, 26328, 24586, 23691, 13014, 2968, 1851]
roc_auc = [None, None, None, None, 34.12, 33.05, 48.22] # Only calculated for sequence in standard script

plt.style.use('seaborn-v0_8-whitegrid')
trad_color = '#e74c3c' # Red for traditional
seq_color = '#3498db'  # Blue for sequence

# ---------------------------------------------------------
# 1. False Positive vs Recall Trade-off (Scatter Plot)
# ---------------------------------------------------------
plt.figure(figsize=(10, 6))
for i in range(len(models)):
    color = trad_color if 'Traditional' in categories[i] else seq_color
    plt.scatter(false_positives[i], recall[i], s=200, c=color, label=categories[i] if i in [0, 4] else "")
    plt.annotate(models[i], (false_positives[i], recall[i]), xytext=(5, 5), textcoords='offset points', fontsize=10)

plt.title('Performance Frontier: Crash Sensitivity vs. False Alarms (Physical Data)', fontsize=14, pad=15)
plt.xlabel('Total False Positives (Lower is Better)', fontsize=12)
plt.ylabel('Recall / Crash Sensitivity % (Higher is Better)', fontsize=12)
plt.legend(loc='upper right')
plt.tight_layout()
plt.savefig(os.path.join(out_dir, 'figure_5_pareto_frontier.png'), dpi=300)
plt.close()

# ---------------------------------------------------------
# 2. Grouped False Positive Bar Chart
# ---------------------------------------------------------
plt.figure(figsize=(12, 6))
bars = plt.bar(models, false_positives, color=[trad_color]*4 + [seq_color]*3)
plt.title('System Reliability: False Alarm Generation Across Architectures', fontsize=14, pad=15)
plt.ylabel('Total False Positives', fontsize=12)
plt.xticks(rotation=30, ha='right', fontsize=11)

# Add value labels on top of bars
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 500, f'{int(yval)}', ha='center', va='bottom', fontweight='bold')

# Custom legend
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor=trad_color, label='Static Inference'),
                   Patch(facecolor=seq_color, label='Temporal Inference')]
plt.legend(handles=legend_elements, loc='upper right')

plt.tight_layout()
plt.savefig(os.path.join(out_dir, 'figure_6_false_positives_bar.png'), dpi=300)
plt.close()

# ---------------------------------------------------------
# 3. Overall Accuracy Comparison
# ---------------------------------------------------------
plt.figure(figsize=(10, 6))
bars = plt.bar(models, accuracy, color=[trad_color]*4 + [seq_color]*3)
plt.title('Overall Classification Accuracy (Physical Edge Data)', fontsize=14, pad=15)
plt.ylabel('Accuracy (%)', fontsize=12)
plt.xticks(rotation=30, ha='right', fontsize=11)
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 1, f'{yval:.2f}%', ha='center', va='bottom')
plt.tight_layout()
plt.savefig(os.path.join(out_dir, 'figure_7_accuracy_bar.png'), dpi=300)
plt.close()

# ---------------------------------------------------------
# 4. F1-Score Radar Chart (Sequence Models Only)
# ---------------------------------------------------------
# We isolate the sequence models to show their specific balance
seq_models = ['Hybrid', 'GRU', 'LSTM']
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
N = len(metrics)

# Data specifically for the radar
values = [
    [55.44, 17.09, 18.09, 17.58], # Hybrid
    [70.22, 24.92, 6.64, 10.49],  # GRU
    [71.26, 19.70, 3.06, 5.30]    # LSTM
]

angles = [n / float(N) * 2 * pi for n in range(N)]
angles += angles[:1]

fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

for i in range(len(seq_models)):
    vals = values[i]
    vals += vals[:1] # Close the loop
    ax.plot(angles, vals, linewidth=2, linestyle='solid', label=seq_models[i])
    ax.fill(angles, vals, alpha=0.1)

plt.xticks(angles[:-1], metrics, size=12)
ax.set_rlabel_position(0)
plt.yticks([20, 40, 60, 80], ["20%", "40%", "60%", "80%"], color="grey", size=10)
plt.ylim(0, 80)
plt.title('Metric Balance of Sequence Architectures (Phase 2)', size=15, pad=20)
plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
plt.tight_layout()
plt.savefig(os.path.join(out_dir, 'figure_8_sequence_radar.png'), dpi=300)
plt.close()

print("[SUCCESS] Phase 2 graphing suite generated. 4 figures saved to publication vault.")