import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Bypass Mac Metal allocation bug
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
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

# Optimal Safety-Critical Boundaries calculated from your Phase 1 training logs
THRESHOLDS = {
    "lstm": 0.9897,
    "gru": 0.9194,     
    "hybrid": 0.9995   
}

def fit_darus_scaler():
    """Reads the simulation data to learn the exact scaling factors the LSTM expects."""
    print("[SCALER] Calibrating Sim-to-Real Domain Shift Scaler...")
    filepath = "data/raw/TrainingData.csv"
    if not os.path.exists(filepath):
        print("CRITICAL ERROR: Cannot find TrainingData.csv to fit scaler.")
        exit(1)
        
    df = pd.read_csv(filepath, low_memory=False)
    if len(df.columns) == 1:
        df = pd.read_csv(filepath, sep='\t', low_memory=False)
        
    X_train_raw = df[FEATURES].apply(pd.to_numeric, errors='coerce').fillna(0).values
    
    scaler = StandardScaler()
    scaler.fit(X_train_raw)
    print("[SCALER] Mathematical scale successfully calibrated to DaRUS Physics.")
    return scaler

def parse_boubezoul_file(filepath, label, scaler):
    try:
        df = pd.read_csv(filepath, sep='\t', encoding='latin-1', low_memory=False)
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return None, None

    for col in df.columns[1:7]:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        
    df = df.dropna(subset=df.columns[1:7])
    data = df.iloc[:, 1:7].values
    
    # 1. DOWNSAMPLING: 1000Hz to 50Hz (Take every 20th sample)
    data_50hz = data[::20, :]
    
    # 2. UNIT ALIGNMENT: Convert Gyro (columns 3, 4, 5) from degrees/s to radians/s
    data_50hz[:, 3:] = data_50hz[:, 3:] * (np.pi / 180.0)
    
    # 3. DOMAIN SHIFT FIX: Scale the physical data to match the simulation's expectations
    data_50hz = scaler.transform(data_50hz)
    
    # 4. SEQUENCE GENERATION: Rolling 50-timestep windows
    X_seq, y_seq = [], []
    for i in range(len(data_50hz) - SEQ_LENGTH + 1):
        window = data_50hz[i : i + SEQ_LENGTH, :]
        if not np.isnan(window).any():
            X_seq.append(window)
            y_seq.append(label)
            
    return np.array(X_seq), np.array(y_seq)

def load_and_compile_physical_data(scaler):
    print("[DATA PIPELINE] Ingesting Boubezoul Physical Crash Data...")
    X_all, y_all = [], []
    
    if os.path.exists(CLASS_0_DIR):
        for filename in os.listdir(CLASS_0_DIR):
            if filename.endswith(".csv"):
                X, y = parse_boubezoul_file(os.path.join(CLASS_0_DIR, filename), 0, scaler)
                if X is not None and len(X) > 0:
                    X_all.append(X)
                    y_all.append(y)
                
    if os.path.exists(CLASS_1_DIR):
        for filename in os.listdir(CLASS_1_DIR):
            if filename.endswith(".csv"):
                X, y = parse_boubezoul_file(os.path.join(CLASS_1_DIR, filename), 1, scaler)
                if X is not None and len(X) > 0:
                    X_all.append(X)
                    y_all.append(y)

    X_final = np.vstack(X_all)
    y_final = np.concatenate(y_all)
    
    print(f"[DATA PIPELINE] Successfully compiled {len(X_final)} scaled sequence windows.")
    return X_final, y_final

def evaluate_model(model_name, filepath, X_test, y_test, threshold):
    print(f"--- EVALUATING {model_name.upper()} ---")
    try:
        model = tf.keras.models.load_model(filepath, compile=False)
    except Exception as e:
        print(f"Error loading {filepath}: {e}\n")
        return

    print("Running sequence inference...")
    y_pred_prob = []
    
    # Manual Batching Loop to avoid Mac Threading deadlocks
    batch_size = 512
    num_samples = len(X_test)
    
    for i in range(0, num_samples, batch_size):
        end = min(i + batch_size, num_samples)
        batch = X_test[i:end]
        batch_pred = model(batch, training=False)
        y_pred_prob.extend(batch_pred.numpy().flatten())
            
    y_pred_prob = np.array(y_pred_prob)
    y_pred_class = (y_pred_prob >= threshold).astype(int)
    
    acc = accuracy_score(y_test, y_pred_class)
    prec = precision_score(y_test, y_pred_class, zero_division=0)
    rec = recall_score(y_test, y_pred_class, zero_division=0)
    f1 = f1_score(y_test, y_pred_class, zero_division=0)
    
    try:
        roc_auc = roc_auc_score(y_test, y_pred_prob)
    except ValueError:
        roc_auc = 0.0
        
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred_class).ravel()
    
    print(f"Accuracy           : {acc:.4f}")
    print(f"Precision          : {prec:.4f}")
    print(f"Recall (Sensitivity): {rec:.4f}")
    print(f"F1-Score           : {f1:.4f}")
    print(f"ROC-AUC            : {roc_auc:.4f}")
    print(f"True Positives     : {tp}")
    print(f"False Positives    : {fp}  <-- FALSE ALARMS")
    print(f"True Negatives     : {tn}")
    print(f"False Negatives    : {fn}\n")
    print("="*60)

if __name__ == "__main__":
    # 1. Fit the scaler on the simulation data
    master_scaler = fit_darus_scaler()
    
    # 2. Scale and compile the real-world physical data
    X_test, y_test = load_and_compile_physical_data(master_scaler)
    
    if X_test is not None:
        models_to_test = {
            "lstm": "src/paper_model_lstm.keras",
            "gru": "src/paper_model_gru.keras",
            "hybrid": "src/paper_model_hybrid.keras"
        }
        
        for name, path in models_to_test.items():
            if os.path.exists(path):
                evaluate_model(name, path, X_test, y_test, THRESHOLDS[name])