import os
import sys
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

# --- CONFIGURATION ---
BASE_DIR = "boubezoul_physical_test"
CLASS_0_DIR = os.path.join(BASE_DIR, "class_0_maneuvers")
CLASS_1_DIR = os.path.join(BASE_DIR, "class_1_falls")

FEATURES = [
    'MotoBody_linaccX', 'MotoBody_linaccY', 'MotoBody_linaccZ',
    'MotoBody_angvelX', 'MotoBody_angvelY', 'MotoBody_angvelZ'
]

def load_physical_flat_data():
    print("[TRADITIONAL PIPELINE] Ingesting and flattening real-world streams...")
    X_list, y_list = [], []
    
    def process_file(filepath, label):
        df = pd.read_csv(filepath, sep='\t', encoding='latin-1', low_memory=False)
        # Force numeric conversion to strip KML tags
        for col in df.columns[1:7]:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df = df.dropna(subset=df.columns[1:7])
        
        data = df.iloc[::20, 1:7].values  # Downsample 1000Hz to 50Hz
        data[:, 3:] = data[:, 3:] * (np.pi / 180.0)  # Gyro degrees to rad
        X_list.append(data)
        y_list.append(np.full(len(data), label))

    if os.path.exists(CLASS_0_DIR):
        for filename in os.listdir(CLASS_0_DIR):
            if filename.endswith(".csv"):
                process_file(os.path.join(CLASS_0_DIR, filename), 0)

    if os.path.exists(CLASS_1_DIR):
        for filename in os.listdir(CLASS_1_DIR):
            if filename.endswith(".csv"):
                process_file(os.path.join(CLASS_1_DIR, filename), 1)

    return np.vstack(X_list), np.concatenate(y_list)

def load_simulated_training_data():
    print("[TRADITIONAL PIPELINE] Bypassing legacy loader to directly map 6-Feature DaRUS Simulation Data...")
    filepath = "data/raw/TrainingData.csv"
    
    if not os.path.exists(filepath):
        print(f"CRITICAL ERROR: File not found at {filepath}")
        exit(1)

    # 1. Robust Loading: Detect if file is comma or tab separated
    df = pd.read_csv(filepath, low_memory=False)
    if len(df.columns) == 1:  # If pandas reads the whole row as 1 column, it's tab-separated
        df = pd.read_csv(filepath, sep='\t', low_memory=False)

    # 2. Dynamic Target Detection (Searches for 'crash', 'Crash', 'label', etc.)
    target_col = None
    for col in df.columns:
        if 'crash' in col.lower() or 'label' in col.lower() or 'class' in col.lower():
            target_col = col
            break

    if target_col is None:
        print(f"CRITICAL ERROR: Target column not found. Available columns: {list(df.columns)[:10]}")
        exit(1)

    print(f"[TRADITIONAL PIPELINE] Mapped Target: '{target_col}'")

    # 3. Safe Extraction
    try:
        # Extract, force numeric, and format for Scikit-Learn
        X_train = df[FEATURES].apply(pd.to_numeric, errors='coerce').fillna(0).values
        y_train = df[target_col].apply(pd.to_numeric, errors='coerce').fillna(0).values.astype(int)
    except KeyError as e:
        print(f"CRITICAL ERROR: Missing expected column: {e}")
        exit(1)

    print(f"[TRADITIONAL PIPELINE] Successfully loaded {X_train.shape[0]} flat training samples.")
    return X_train, y_train

if __name__ == "__main__":
    X_train, y_train = load_simulated_training_data()
    X_test, y_test = load_physical_flat_data()
    
    # Define the 5 traditional classifiers matching Rodegast et al.
    models = {
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        # "SVM": SVC(probability=True, random_state=42), # Excluded due to O(n^3) complexity on 450k+ telemetry samples
        "Neural Net (MLP)": MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=200, random_state=42),
        "AdaBoost": AdaBoostClassifier(random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42)
    }
    
    results = []
    
    # 1. Evaluate Naïve Kinematic Threshold Baseline (Rule-based control)
    # Triggers if the total acceleration vector length exceeds 2.5G (~24.5 m/s²)
    accel_magnitudes = np.linalg.norm(X_test[:, :3], axis=1)
    y_pred_thresh = (accel_magnitudes > 24.5).astype(int)
    
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred_thresh).ravel()
    results.append({
        "Model": "Naïve Threshold",
        "Accuracy": accuracy_score(y_test, y_pred_thresh),
        "Precision": precision_score(y_test, y_pred_thresh, zero_division=0),
        "Recall": recall_score(y_test, y_pred_thresh, zero_division=0),
        "False Positives": fp
    })

    # 2. Train and evaluate Scikit-Learn baseline classifiers
    for name, model in models.items():
        print(f"[TRAINING baselines] Fitting {name} model...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        
        results.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred, zero_division=0),
            "Recall": recall_score(y_test, y_pred, zero_division=0),
            "False Positives": fp
        })
        
    # Print the final execution comparative matrix
    df_results = pd.DataFrame(results)
    print("\n" + "="*70)
    print("--- TRADITIONAL BASELINES PHYSICAL EVALUATION (RODEGAST COMPARISON) ---")
    print("="*70)
    print(df_results.to_string(index=False, formatters={
        'Accuracy': '{:,.4f}'.format, 'Precision': '{:,.4f}'.format, 'Recall': '{:,.4f}'.format
    }))
    print("="*70)
    
    # Save isolated results matrix to prevent file overwrites
    os.makedirs("reports", exist_ok=True)
    df_results.to_csv("reports/traditional_baseline_boubezoul_matrix.csv", index=False)
    print("[SUCCESS] Matrix saved to reports/traditional_baseline_boubezoul_matrix.csv")