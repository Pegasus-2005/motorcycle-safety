import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler

# --- CONFIGURATION ---
BASE_DIR = "boubezoul_physical_test"
CLASS_0_DIR = os.path.join(BASE_DIR, "class_0_maneuvers")
CLASS_1_DIR = os.path.join(BASE_DIR, "class_1_falls")
SEQ_LENGTH = 50

FEATURES = [
    'MotoBody_linaccX', 'MotoBody_linaccY', 'MotoBody_linaccZ',
    'MotoBody_angvelX', 'MotoBody_angvelY', 'MotoBody_angvelZ'
]

def fit_darus_scaler():
    """Calibrates the scaler to the original simulation data."""
    print("[SYSTEM] Calibrating DaRUS Base Scaler...")
    filepath = "data/raw/trainingData.csv"
    if not os.path.exists(filepath):
        filepath = "data/raw/TrainingData.csv" 
    df = pd.read_csv(filepath, low_memory=False)
    X_train_raw = df[FEATURES].apply(pd.to_numeric, errors='coerce').fillna(0).values
    scaler = StandardScaler()
    scaler.fit(X_train_raw)
    return scaler

def parse_boubezoul_file(filepath, label, scaler):
    """Your exact logic to ingest physical edge data, with safety guards."""
    try:
        df = pd.read_csv(filepath, sep='\t', encoding='latin-1', low_memory=False)
    except:
        return None, None
        
    if len(df.columns) < 7:
        return None, None
        
    for col in df.columns[1:7]:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna(subset=df.columns[1:7])
    
    if len(df) == 0:
        return None, None
        
    data = df.iloc[:, 1:7].values
    data_50hz = data[::20, :] 
    
    if data_50hz.shape[0] == 0 or data_50hz.shape[1] != 6:
        return None, None
        
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
    print("[SYSTEM] Loading Physical Data into RAM (Skipping Corrupt Files)...")
    if os.path.exists(CLASS_0_DIR):
        for f in os.listdir(CLASS_0_DIR):
            if f.endswith(".csv"):
                X, y = parse_boubezoul_file(os.path.join(CLASS_0_DIR, f), 0, scaler)
                if X is not None and len(X) > 0:
                    X_all.append(X)
                    y_all.append(y)
    if os.path.exists(CLASS_1_DIR):
        for f in os.listdir(CLASS_1_DIR):
            if f.endswith(".csv"):
                X, y = parse_boubezoul_file(os.path.join(CLASS_1_DIR, f), 1, scaler)
                if X is not None and len(X) > 0:
                    X_all.append(X)
                    y_all.append(y)
    return np.vstack(X_all), np.concatenate(y_all)

def print_metrics(y_true, y_pred, title):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    print(f"\n--- {title} ---")
    print(f"Accuracy    : {acc:.4f}")
    print(f"Precision   : {prec:.4f}")
    print(f"Recall      : {rec:.4f}  <-- THE TARGET METRIC")
    print(f"F1-Score    : {f1:.4f}")
    print(f"False Alarms: {fp}")
    print(f"Missed Crashes: {fn}")
    print("---------------------------------------")

def main():
    print("="*60)
    print(" PHASE 3: TRANSFER LEARNING EXPERIMENT")
    print("="*60)
    
    scaler = fit_darus_scaler()
    X_final, y_final = load_data(scaler)
    
    print("[SYSTEM] Executing Stratified 10/90 Split...")
    np.random.seed(42)
    idx_0 = np.where(y_final == 0)[0]
    idx_1 = np.where(y_final == 1)[0]
    np.random.shuffle(idx_0)
    np.random.shuffle(idx_1)
    
    split_0 = max(1, int(len(idx_0) * 0.10))
    split_1 = max(1, int(len(idx_1) * 0.10))
    
    train_idx = np.concatenate((idx_0[:split_0], idx_1[:split_1]))
    test_idx = np.concatenate((idx_0[split_0:], idx_1[split_1:]))
    np.random.shuffle(train_idx)
    np.random.shuffle(test_idx)
    
    X_train, y_train = X_final[train_idx], y_final[train_idx]
    X_test, y_test = X_final[test_idx], y_final[test_idx]
    print(f"Adaptation Set (10%): {len(X_train)} samples")
    print(f"Unseen Eval Set (90%): {len(X_test)} samples\n")
    
    # --- NEW: BUILD SKELETON AND LOAD WEIGHTS ("BRAIN TRANSPLANT") ---
    print("[MODEL] Constructing Native Architecture and Loading Mac Weights...")
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.InputLayer(shape=(SEQ_LENGTH, len(FEATURES))))
    model.add(tf.keras.layers.LSTM(32, return_sequences=False))
    model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.Dense(16, activation='relu'))
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
    
    # Load just the raw math from the .keras file to bypass the version crash
    model_path = "src/paper_model_lstm.keras"
    model.load_weights(model_path)
    
    # EVALUATE BEFORE TRANSFER LEARNING
    threshold = 0.9897
    y_pred_prob = model.predict(X_test, batch_size=512, verbose=0).flatten()
    y_pred_class = (y_pred_prob >= threshold).astype(int)
    print_metrics(y_test, y_pred_class, "BEFORE TRANSFER LEARNING (Zero-Shot)")
    
    # --- EXECUTE TRANSFER LEARNING ---
    print("\n[TRANSFER] Freezing early layers and fine-tuning decision boundary...")
    model.layers[0].trainable = False 
    
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005), 
                  loss='binary_crossentropy', 
                  metrics=['accuracy'])
    
    model.fit(X_train, y_train, epochs=5, batch_size=256, verbose=1, 
              class_weight={0: 1.0, 1: 5.0}) 
    
    # EVALUATE AFTER TRANSFER LEARNING
    print("\n[EVALUATION] Testing on the 90% Unseen Physical Data...")
    y_pred_prob_after = model.predict(X_test, batch_size=512, verbose=0).flatten()
    y_pred_class_after = (y_pred_prob_after >= 0.5).astype(int) 
    print_metrics(y_test, y_pred_class_after, "AFTER TRANSFER LEARNING (Domain Adapted)")
    
    final_path = "src/paper_model_lstm_transfer.keras"
    model.save(final_path)
    print(f"\n[SUCCESS] Final domain-adapted model saved to {final_path}")
    print("="*60)

if __name__ == "__main__":
    main()