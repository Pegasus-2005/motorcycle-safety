import os
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, recall_score
from sklearn.preprocessing import StandardScaler

# --- CONFIGURATION ---
BASE_DIR = "boubezoul_physical_test"
CLASS_0_DIR = os.path.join(BASE_DIR, "class_0_maneuvers")
CLASS_1_DIR = os.path.join(BASE_DIR, "class_1_falls")
SEQ_LENGTH = 50
FEATURES = ['MotoBody_linaccX', 'MotoBody_linaccY', 'MotoBody_linaccZ', 'MotoBody_angvelX', 'MotoBody_angvelY', 'MotoBody_angvelZ']
OUTPUT_DIR = "publication_vault"

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

def fit_darus_scaler():
    filepath = "data/raw/TrainingData.csv"
    if not os.path.exists(filepath): filepath = "data/raw/trainingData.csv"
    df = pd.read_csv(filepath, low_memory=False)
    X_train_raw = df[FEATURES].apply(pd.to_numeric, errors='coerce').fillna(0).values
    scaler = StandardScaler()
    scaler.fit(X_train_raw)
    return scaler

def parse_file(filepath, label, scaler):
    try: df = pd.read_csv(filepath, sep='\t', encoding='latin-1', low_memory=False)
    except: return None, None
    if len(df.columns) < 7: return None, None
    for col in df.columns[1:7]: df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna(subset=df.columns[1:7])
    if len(df) == 0: return None, None
    
    data = df.iloc[:, 1:7].values
    data_50hz = data[::20, :] 
    if data_50hz.shape[0] == 0 or data_50hz.shape[1] != 6: return None, None
    data_50hz[:, 3:] = data_50hz[:, 3:] * (np.pi / 180.0) 
    data_50hz = scaler.transform(data_50hz) 
    
    X_seq, y_seq = [], []
    for i in range(len(data_50hz) - SEQ_LENGTH + 1):
        window = data_50hz[i : i + SEQ_LENGTH, :]
        if not np.isnan(window).any():
            X_seq.append(window)
            y_seq.append(label)
    return np.array(X_seq), np.array(y_seq)

def load_data(scaler):
    X_all, y_all = [], []
    print("[SYSTEM] Loading Physical Data for Visualization...")
    for directory, label in [(CLASS_0_DIR, 0), (CLASS_1_DIR, 1)]:
        if os.path.exists(directory):
            for f in os.listdir(directory):
                if f.endswith(".csv"):
                    X, y = parse_file(os.path.join(directory, f), label, scaler)
                    if X is not None and len(X) > 0:
                        X_all.append(X)
                        y_all.append(y)
    return np.vstack(X_all), np.concatenate(y_all)

def plot_confusion_matrix(y_true, y_pred, threshold):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.set_theme(style='whitegrid', font_scale=1.2)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Normal Driving', 'Crash Event'],
                yticklabels=['Normal Driving', 'Crash Event'])
    plt.title(f'Phase 3: Domain Adapted Confusion Matrix (Threshold: {threshold})', fontsize=14, pad=15)
    plt.ylabel('Actual Event', fontweight='bold')
    plt.xlabel('Predicted Event', fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'figure_13_phase3_confusion_matrix.png'), dpi=300)
    print("[SUCCESS] Saved figure_13_phase3_confusion_matrix.png")
    plt.close()

def plot_pareto_curve(thresholds, recalls, false_alarms):
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # Plot Recall on primary Y axis
    color = 'tab:blue'
    ax1.set_xlabel('Decision Boundary Threshold', fontweight='bold')
    ax1.set_ylabel('Recall (Crash Detection Rate)', color=color, fontweight='bold')
    ax1.plot(thresholds, recalls, marker='o', linewidth=2, color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # Plot False Alarms on secondary Y axis
    ax2 = ax1.twinx()  
    color = 'tab:red'
    ax2.set_ylabel('Total False Alarms', color=color, fontweight='bold')
    ax2.plot(thresholds, false_alarms, marker='s', linewidth=2, color=color, linestyle='--')
    ax2.tick_params(axis='y', labelcolor=color)
    
    plt.title('Sim-to-Real Domain Gap: Pareto Trade-off Curve', fontsize=14, pad=15)
    fig.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'figure_14_pareto_tradeoff.png'), dpi=300)
    print("[SUCCESS] Saved figure_14_pareto_tradeoff.png")
    plt.close()

def main():
    scaler = fit_darus_scaler()
    X_final, y_final = load_data(scaler)
    
    # Isolate the exact same 90% Unseen Set
    np.random.seed(42)
    idx_0 = np.where(y_final == 0)[0]
    idx_1 = np.where(y_final == 1)[0]
    np.random.shuffle(idx_0)
    np.random.shuffle(idx_1)
    split_0, split_1 = max(1, int(len(idx_0) * 0.10)), max(1, int(len(idx_1) * 0.10))
    test_idx = np.concatenate((idx_0[split_0:], idx_1[split_1:]))
    X_test, y_test = X_final[test_idx], y_final[test_idx]
    
    print("[MODEL] Loading Transfer-Learned Model...")
    model = tf.keras.models.load_model("src/paper_model_lstm_transfer.keras", compile=False)
    y_pred_prob = model.predict(X_test, batch_size=512, verbose=0).flatten()
    
    # 1. Generate Confusion Matrix at exactly 0.90 Threshold
    target_threshold = 0.90
    y_pred_class_90 = (y_pred_prob >= target_threshold).astype(int)
    plot_confusion_matrix(y_test, y_pred_class_90, target_threshold)
    
    # 2. Generate the Pareto Trade-off Curve
    print("[SYSTEM] Calculating threshold sweep for Pareto curve...")
    thresholds = [0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    recalls = []
    false_alarms = []
    
    for t in thresholds:
        y_pred_class = (y_pred_prob >= t).astype(int)
        rec = recall_score(y_test, y_pred_class, zero_division=0)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred_class).ravel()
        recalls.append(rec)
        false_alarms.append(fp)
        
    plot_pareto_curve(thresholds, recalls, false_alarms)
    print("="*60)
    print("ALL VISUAL PROOFS GENERATED SUCCESSFULLY!")
    print("="*60)

if __name__ == "__main__":
    import matplotlib
    matplotlib.use('Agg') # Ensures it runs cleanly in terminal without opening GUI windows
    main()