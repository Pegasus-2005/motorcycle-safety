import os
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import GroupShuffleSplit

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data", "raw")
TRAIN_PATH = os.path.join(DATA_DIR, "trainingData.csv")
TEST_PATH = os.path.join(DATA_DIR, "testData.csv")
MODEL_DIR = os.path.join(BASE_DIR, "src")
MODEL_PATH = os.path.join(MODEL_DIR, "crash_model.pkl")
REPORT_DIR = os.path.join(BASE_DIR, "reports")
os.makedirs(REPORT_DIR, exist_ok=True)

FEATURES = [
    'Time', 'MotoBody_angaccRes', 'MotoBody_angaccX', 'MotoBody_angaccY', 'MotoBody_angaccZ',
    'FrontDef_angdispX', 'FrontDef_angdispY', 'FrontDef_angdispZ', 'MotoBody_angvelRes',
    'MotoBody_angvelX', 'MotoBody_angvelY', 'MotoBody_angvelZ', 'MotoRW_linaccRes',
    'MotoRW_linaccX', 'MotoRW_linaccY', 'MotoRW_linaccZ', 'MotoFW_linaccRes',
    'MotoFW_linaccX', 'MotoFW_linaccY', 'MotoFW_linaccZ', 'MotoBody_linaccRes',
    'MotoBody_linaccX', 'MotoBody_linaccY', 'MotoBody_linaccZ', 'MotoFW_linvelRes',
    'MotoFW_linvelX', 'MotoFW_linvelY', 'MotoFW_linvelZ', 'MotoRW_linvelRes',
    'MotoRW_linvelX', 'MotoRW_linvelY', 'MotoRW_linvelZ', 'MotoBody_linvelRes',
    'MotoBody_linvelX', 'MotoBody_linvelY', 'MotoBody_linvelZ', 'Frontdef_angacc',
    'Frontdef_angpos', 'JointRW_angvel', 'Frontdef_angvel', 'JointFW_angvel',
    'SwitchSidesensor_left_car', 'SwitchSidesensor_right_road', 'SwitchSidesensor_right_car',
    'Sensor_left_road', 'Sensor_left_car', 'Sensor_right_road', 'Sensor_right_car',
    'SwitchSidesensor_left_road', 'FW_cnt_Force', 'RW_cnt_Force', 'angveldiff',
    'angaccdiff', 'sensorLeft', 'sensorRight'
]

TARGET = 'CrashLabelML'

def load_data(path, is_training_file=False):
    df = pd.read_csv(path)
    df['angveldiff'] = df.FW_angvel_Y - df.RW_angvel_Y
    df['angaccdiff'] = df.FW_angacc_Y - df.RW_angacc_Y
    df['sensorLeft'] = np.logical_or(df.SwitchSidesensor_left_road, df.SwitchSidesensor_left_car).astype(int)
    df['sensorRight'] = np.logical_or(df.SwitchSidesensor_right_road, df.SwitchSidesensor_right_car).astype(int)
    df['FW_cnt_Force'] = df.FW_Car_cnt_force + df.FW_Road_cnt_force
    df['RW_cnt_Force'] = df.RW_Car_cnt_force + df.RW_Road_cnt_force

    # Dynamically generate a RunID based on the Time column resetting
    # This is crucial for preventing data leakage during validation splits
    if 'Time' in df.columns:
        df['RunID'] = (df['Time'] < df['Time'].shift(1, fill_value=0)).cumsum()
    else:
        df['RunID'] = 0 
        
    return df

def metrics_dict(y_true, y_pred, y_prob):
    labels = [0, 1]
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=labels).ravel()

    return {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'precision': float(precision_score(y_true, y_pred, zero_division=0)),
        'recall': float(recall_score(y_true, y_pred, zero_division=0)),
        'f1': float(f1_score(y_true, y_pred, zero_division=0)),
        'roc_auc': float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else None,
        'false_positive_rate': float(fp / (fp + tn)) if (fp + tn) > 0 else None,
        'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)
    }

def evaluate_control_scenarios(model):
    scenario_dir = os.path.join(DATA_DIR, "ControlScenarios")
    if not os.path.exists(scenario_dir):
        print(f"\nScenario folder not found: {scenario_dir}")
        return None

    all_files = sorted([f for f in os.listdir(scenario_dir) if f.lower().endswith(".csv")])
    if not all_files:
        print(f"\nNo CSV files found in: {scenario_dir}")
        return None

    rows = []
    for fname in all_files:
        path = os.path.join(scenario_dir, fname)
        df = load_data(path)

        X = df[FEATURES].copy()
        y = df[TARGET].astype(int).copy()

        prob = model.predict_proba(X)[:, 1]
        pred = (prob >= 0.5).astype(int)

        metrics = metrics_dict(y, pred, prob)
        metrics["scenario"] = fname
        metrics["is_crash_scenario"] = int("iso" in fname.lower())
        rows.append(metrics)

    out_df = pd.DataFrame(rows)
    out_path = os.path.join(REPORT_DIR, "control_scenario_metrics.csv")
    out_df.to_csv(out_path, index=False)

    print("\nCONTROL SCENARIO RESULTS")
    print(out_df[["scenario", "is_crash_scenario", "tn", "fp", "fn", "tp", "accuracy", "precision", "recall"]].to_string(index=False))
    print(f"\nSaved scenario report to: {out_path}")
    return out_df

def main():
    print("Loading datasets and extracting runs...")
    train_df = load_data(TRAIN_PATH, is_training_file=True)
    test_df = load_data(TEST_PATH)

    X = train_df[FEATURES].copy()
    y = train_df[TARGET].astype(int).copy()
    groups = train_df['RunID']

    print(f"Detected {groups.nunique()} unique simulation runs in training data.")

    # SUPERVISOR FIX: Group-based split to prevent data leakage
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, val_idx = next(gss.split(X, y, groups=groups))

    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    X_test = test_df[FEATURES].copy()
    y_test = test_df[TARGET].astype(int).copy()

    print(f"Train split: {len(X_train)} samples | Validation split: {len(X_val)} samples")

    print("\nTraining Random Forest model...")
    model = RandomForestClassifier(
        n_estimators=150,
        max_depth=18,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced"
    )
    model.fit(X_train, y_train)

    # Predictions
    train_prob = model.predict_proba(X_train)[:, 1]
    val_prob = model.predict_proba(X_val)[:, 1]
    test_prob = model.predict_proba(X_test)[:, 1]

    train_pred = (train_prob >= 0.5).astype(int)
    val_pred = (val_prob >= 0.5).astype(int)
    test_pred = (test_prob >= 0.5).astype(int)

    # Metrics computation
    train_metrics = metrics_dict(y_train, train_pred, train_prob)
    val_metrics = metrics_dict(y_val, val_pred, val_prob)
    test_metrics = metrics_dict(y_test, test_pred, test_prob)

    print("\nTRAIN METRICS (Run-Isolated)")
    print(train_metrics)
    print("\nVALIDATION METRICS (Run-Isolated)")
    print(val_metrics)
    print("\nTEST METRICS")
    print(test_metrics)

    joblib.dump(model, MODEL_PATH)
    print(f"\nBaseline model saved securely to {MODEL_PATH}")

    # Save strict performance reports
    results = {
        "train": train_metrics,
        "validation": val_metrics,
        "test": test_metrics
    }
    with open(os.path.join(REPORT_DIR, "rf_metrics.json"), "w") as f:
        json.dump(results, f, indent=2)

    # Run Supervisor's required control evaluations
    evaluate_control_scenarios(model)

if __name__ == "__main__":
    main()