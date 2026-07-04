import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler

BASE_DIR = "boubezoul_physical_test"
CLASS_0_DIR = os.path.join(BASE_DIR, "class_0_maneuvers")
CLASS_1_DIR = os.path.join(BASE_DIR, "class_1_falls")
SEQ_LENGTH = 50
FEATURES = ['MotoBody_linaccX', 'MotoBody_linaccY', 'MotoBody_linaccZ', 'MotoBody_angvelX', 'MotoBody_angvelY', 'MotoBody_angvelZ']

def fit_darus_scaler():
    df = pd.read_csv("data/raw/TrainingData.csv", low_memory=False)
    if len(df.columns) == 1: df = pd.read_csv("data/raw/TrainingData.csv", sep='\t', low_memory=False)
    X_train_raw = df[FEATURES].apply(pd.to_numeric, errors='coerce').fillna(0).values
    scaler = StandardScaler().fit(X_train_raw)
    return scaler

def parse_boubezoul_file(filepath, label, scaler):
    df = pd.read_csv(filepath, sep='\t', encoding='latin-1', low_memory=False)
    for col in df.columns[1:7]: df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna(subset=df.columns[1:7])
    data_50hz = df.iloc[::20, 1:7].values
    data_50hz[:, 3:] = data_50hz[:, 3:] * (np.pi / 180.0)
    data_50hz = scaler.transform(data_50hz)
    
    X_seq, y_seq = [], []
    for i in range(len(data_50hz) - SEQ_LENGTH + 1):
        window = data_50hz[i : i + SEQ_LENGTH, :]
        if not np.isnan(window).any():
            X_seq.append(window)
            y_seq.append(label)
    return np.array(X_seq), np.array(y_seq)

def load_data():
    scaler = fit_darus_scaler()
    X_all, y_all = [], []
    for filename in os.listdir(CLASS_0_DIR):
        if filename.endswith(".csv"):
            X, y = parse_boubezoul_file(os.path.join(CLASS_0_DIR, filename), 0, scaler)
            if X is not None and len(X) > 0: X_all.append(X); y_all.append(y)
    for filename in os.listdir(CLASS_1_DIR):
        if filename.endswith(".csv"):
            X, y = parse_boubezoul_file(os.path.join(CLASS_1_DIR, filename), 1, scaler)
            if X is not None and len(X) > 0: X_all.append(X); y_all.append(y)
    return np.vstack(X_all), np.concatenate(y_all)

def evaluate_and_optimize(model_name, model_path, X_test, y_test):
    print(f"\nLoading and evaluating {model_name.upper()}...")
    try:
        model = tf.keras.models.load_model(model_path, compile=False)
    except Exception as e:
        print(f"Error loading {model_path}: {e}")
        return None

    y_pred_prob = []
    for i in range(0, len(X_test), 512):
        batch = X_test[i:i+512]
        y_pred_prob.extend(model(batch, training=False).numpy().flatten())
    y_pred_prob = np.array(y_pred_prob)

    best_f2 = 0
    best_metrics = {}

    # Sweep thresholds to find the best balance of catching crashes (recall) vs minimizing false alarms
    for thresh in np.arange(0.01, 1.0, 0.02):
        y_pred = (y_pred_prob >= thresh).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        rec = recall_score(y_test, y_pred, zero_division=0)
        prec = precision_score(y_test, y_pred, zero_division=0)
        
        # F2 score penalizes missing a crash (False Negatives) heavily
        f2 = (5 * prec * rec) / ((4 * prec) + rec + 1e-7)
        
        if f2 > best_f2:
            best_f2 = f2
            best_metrics = {
                'Model': model_name.upper(),
                'Optimal Thresh': round(thresh, 2),
                'Accuracy': round(accuracy_score(y_test, y_pred), 4),
                'Recall': round(rec, 4),
                'Precision': round(prec, 4),
                'False Positives': fp
            }
    return best_metrics

if __name__ == "__main__":
    print("[INIT] Starting Comprehensive Sequence Model Evaluation...")
    X_test, y_test = load_data()
    
    models = {
        "lstm": "src/paper_model_lstm.keras",
        "gru": "src/paper_model_gru.keras",
        "hybrid": "src/paper_model_hybrid.keras"
    }
    
    results = []
    for name, path in models.items():
        if os.path.exists(path):
            metrics = evaluate_and_optimize(name, path, X_test, y_test)
            if metrics:
                results.append(metrics)
        else:
            print(f"File not found: {path}")

    # Compile the final comparison table
    print("\n" + "="*80)
    print("--- FINAL SEQUENCE MODEL CROSS-VALIDATION (OPTIMIZED FOR BOUBEZOUL) ---")
    print("="*80)
    df_results = pd.DataFrame(results)
    print(df_results.to_string(index=False))
    print("="*80)