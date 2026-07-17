import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import confusion_matrix, recall_score
from sklearn.preprocessing import StandardScaler

# --- CONFIGURATION ---
BASE_DIR = "boubezoul_physical_test"
CLASS_0_DIR = os.path.join(BASE_DIR, "class_0_maneuvers")
CLASS_1_DIR = os.path.join(BASE_DIR, "class_1_falls")
SEQ_LENGTH = 50
FEATURES = ['MotoBody_linaccX', 'MotoBody_linaccY', 'MotoBody_linaccZ', 'MotoBody_angvelX', 'MotoBody_angvelY', 'MotoBody_angvelZ']

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
    print("[SYSTEM] Loading Physical Data...")
    for directory, label in [(CLASS_0_DIR, 0), (CLASS_1_DIR, 1)]:
        if os.path.exists(directory):
            for f in os.listdir(directory):
                if f.endswith(".csv"):
                    X, y = parse_file(os.path.join(directory, f), label, scaler)
                    if X is not None and len(X) > 0:
                        X_all.append(X)
                        y_all.append(y)
    return np.vstack(X_all), np.concatenate(y_all)

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
    
    print("\n--- SWEEPING DECISION BOUNDARIES ---")
    thresholds_to_test = [0.50, 0.90, 0.95, 0.99, 0.999, 0.9999]
    
    for t in thresholds_to_test:
        y_pred_class = (y_pred_prob >= t).astype(int)
        rec = recall_score(y_test, y_pred_class, zero_division=0)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred_class).ravel()
        print(f"Threshold: {t:.4f} | Recall: {rec:.4f} | False Alarms: {fp}")

if __name__ == "__main__":
    main()