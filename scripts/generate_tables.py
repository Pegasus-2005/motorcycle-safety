import pandas as pd
import matplotlib.pyplot as plt
import os

def render_table_to_png(csv_path, output_png_path, title):
    if not os.path.exists(csv_path):
        print(f"[ERROR] Could not find {csv_path}")
        return
        
    df = pd.read_csv(csv_path)
    
    # Format the metrics cleanly
    for col in df.columns:
        if any(metric in col for metric in ['Accuracy', 'Recall', 'Precision', 'F1', 'ROC', 'Optimized']):
            df[col] = df[col].apply(lambda x: f"{x:.4f}" if isinstance(x, (int, float)) else x)

    # Create the visual plot
    fig, ax = plt.subplots(figsize=(12, 0.8 * len(df) + 1.5)) 
    ax.axis('off')
    
    table = ax.table(cellText=df.values, colLabels=df.columns, cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.8)
    
    # IEEE Styling for the header
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight='bold', color='white')
            cell.set_facecolor('#2c3e50') # Dark blue header
        else:
            cell.set_facecolor('#ecf0f1' if row % 2 == 0 else '#ffffff') # Alternating rows

    plt.title(title, fontweight="bold", fontsize=14, pad=20)
    plt.tight_layout()
    plt.savefig(output_png_path, dpi=300, bbox_inches='tight')
    print(f"Saved Table Image: {output_png_path}")

if __name__ == "__main__":
    # 1. Phase 1 Simulation Results
    render_table_to_png(
        "publication_vault/phase_1_simulation_14_feature/matrices/sequence_baseline_darus_matrix.csv",
        "publication_vault/phase_1_simulation_14_feature/graphs/table_1_simulation_results.png",
        "Table 1: Phase 1 (Simulation) - 14-Feature Sequence Model Benchmarks"
    )
    
    # 2. Phase 2 Sequence Results
    render_table_to_png(
        "publication_vault/phase_2_physical_6_feature/matrices/sequence_baseline_boubezoul_matrix.csv",
        "publication_vault/phase_2_physical_6_feature/graphs/table_2_physical_sequence_results.png",
        "Table 2: Phase 2 (Physical) - Sequence Architecture Edge Benchmarks"
    )
    
    # 3. Phase 2 Traditional Results
    render_table_to_png(
        "publication_vault/phase_2_physical_6_feature/matrices/traditional_baseline_boubezoul_matrix.csv",
        "publication_vault/phase_2_physical_6_feature/graphs/table_3_physical_traditional_results.png",
        "Table 3: Phase 2 (Physical) - Traditional Static Model Benchmarks"
    )