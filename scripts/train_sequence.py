import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
from sklearn.utils.class_weight import compute_class_weight

# --- PATH CONFIGURATION ---
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data", "raw")
TRAIN_PATH = os.path.join(DATA_DIR, "trainingData.csv")
TEST_PATH = os.path.join(DATA_DIR, "testData.csv")
MODEL_DIR = os.path.join(BASE_DIR, "src")
REPORT_DIR = os.path.join(BASE_DIR, "reports")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

# Selected physical features ensuring no overfitting on irrelevant variables
FEATURES = [
    'MotoBody_angaccY', 'MotoBody_angvelY', 'MotoBody_linaccX', 'MotoBody_linaccZ',
    'MotoFW_linaccX', 'MotoFW_linaccZ', 'MotoRW_linaccX', 'MotoRW_linaccZ',
    'FW_cnt_Force', 'RW_cnt_Force', 'angveldiff', 'angaccdiff', 'sensorLeft', 'sensorRight'
]
TARGET = 'CrashLabelML'
WINDOW_SIZE = 50
BATCH_SIZE = 512  # Upsized for more stable gradient steps on Mac architecture

def load_and_engineer_data(path):
    """Loads CSV and derives essential physical features."""
    df = pd.read_csv(path)
    df['angveldiff'] = df.FW_angvel_Y - df.RW_angvel_Y
    df['angaccdiff'] = df.FW_angacc_Y - df.RW_angacc_Y
    df['sensorLeft'] = np.logical_or(df.SwitchSidesensor_left_road, df.SwitchSidesensor_left_car).astype(int)
    df['sensorRight'] = np.logical_or(df.SwitchSidesensor_right_road, df.SwitchSidesensor_right_car).astype(int)
    df['FW_cnt_Force'] = df.FW_Car_cnt_force + df.FW_Road_cnt_force
    df['RW_cnt_Force'] = df.RW_Car_cnt_force + df.RW_Road_cnt_force
    
    # Crucial Data Leakage Prevention: Group by continuous simulation runs
    if 'Time' in df.columns:
        df['RunID'] = (df['Time'] < df['Time'].shift(1, fill_value=0)).cumsum()
    else:
        df['RunID'] = 0 
    return df

def create_tf_dataset(df, scaler, is_training=False):
    """Creates a memory-efficient sequence generator to prevent RAM crashes."""
    df = df.reset_index(drop=True) 
    X_scaled = scaler.transform(df[FEATURES])
    y = df[TARGET].values

    dataset_list = []
    for run_id, group in df.groupby('RunID'):
        idx = group.index
        X_group = X_scaled[idx]
        y_group = y[idx]
        
        if len(X_group) <= WINDOW_SIZE:
            continue
            
        ds = tf.keras.utils.timeseries_dataset_from_array(
            data=X_group,
            targets=y_group[WINDOW_SIZE-1:], # Map window sequence to final timestep outcome
            sequence_length=WINDOW_SIZE,
            sequence_stride=1,
            batch_size=BATCH_SIZE,
            shuffle=is_training
        )
        dataset_list.append(ds)

    if not dataset_list:
        return None
        
    full_dataset = dataset_list[0]
    for ds in dataset_list[1:]:
        full_dataset = full_dataset.concatenate(ds)
        
    return full_dataset

def build_advanced_model(model_type, input_shape):
    """Procedural generator to isolate model mechanics precisely for ablation study."""
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.InputLayer(shape=input_shape))
    
    if model_type == 'LSTM':
        model.add(tf.keras.layers.LSTM(32, return_sequences=False))
    elif model_type == 'GRU':
        model.add(tf.keras.layers.GRU(32, return_sequences=False))
    elif model_type == 'Hybrid':
        # Conv1D extracts multi-axis structural sensor signatures before temporal encoding
        model.add(tf.keras.layers.Conv1D(filters=16, kernel_size=3, activation='relu'))
        model.add(tf.keras.layers.GRU(32, return_sequences=False))
        
    model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.Dense(16, activation='relu'))
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
    )
    return model

def evaluate_model_pipeline(model, test_ds, threshold, model_name):
    """Strict evaluation utilizing the mathematically tuned threshold."""
    y_true, y_probs = [], []
    for X_batch, y_batch in test_ds:
        preds = model.predict(X_batch, verbose=0)
        y_true.extend(y_batch.numpy())
        y_probs.extend(preds.flatten())
        
    y_true = np.array(y_true)
    y_probs = np.array(y_probs)
    y_pred = (y_probs >= threshold).astype(int)
    
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    fpr_w, tpr_w, _ = roc_curve(y_true, y_probs)
    roc_auc = auc(fpr_w, tpr_w)
    
    return {
        "Architecture": model_name,
        "Accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "Precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "Recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "F1_Score": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "ROC_AUC": round(float(roc_auc), 4),
        "False_Positives": int(fp),
        "False_Negatives (Missed)": int(fn)
    }

def main():
    print("\n[PUBLICATION ENGINE] Loading Datasets and Enforcing Run-Isolation...")
    train_df = load_and_engineer_data(TRAIN_PATH)
    test_df = load_and_engineer_data(TEST_PATH)

    # 80/20 Run-Isolated Split to prevent temporal data leakage
    runs = train_df['RunID'].unique()
    np.random.seed(42)
    np.random.shuffle(runs)
    split_idx = int(len(runs) * 0.8)
    
    train_split_df = train_df[train_df['RunID'].isin(runs[:split_idx])]
    val_split_df = train_df[train_df['RunID'].isin(runs[split_idx:])]

    # Calculate exact class weights to penalize False Alarms during gradient descent
    y_train_flat = train_split_df[TARGET].values
    classes = np.unique(y_train_flat)
    weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train_flat)
    class_weight_dict = {int(classes[i]): float(weights[i]) for i in range(len(classes))}
    print(f"[PUBLICATION ENGINE] Imbalance weights calculated: {class_weight_dict}")

    scaler = StandardScaler()
    scaler.fit(train_split_df[FEATURES])

    print("[PUBLICATION ENGINE] Compiling Sequence Windows...")
    train_ds = create_tf_dataset(train_split_df, scaler, is_training=True)
    val_ds = create_tf_dataset(val_split_df, scaler, is_training=False)
    test_ds = create_tf_dataset(test_df, scaler, is_training=False)

    input_shape = (WINDOW_SIZE, len(FEATURES))
    architectures = ['GRU', 'LSTM', 'Hybrid']
    master_results = []

    for arch in architectures:
        print(f"\n==================================================")
        print(f"--- Training Candidate Architecture: {arch} ---")
        print(f"==================================================")
        model = build_advanced_model(arch, input_shape)
        
        # UPGRADE: Patience increased to 5 for complete convergence
        early_stop = tf.keras.callbacks.EarlyStopping(
            monitor='val_auc', mode='max', patience=5, restore_best_weights=True
        )
        
        # UPGRADE: Epochs increased to 40
        model.fit(
            train_ds, validation_data=val_ds, epochs=40,
            callbacks=[early_stop], class_weight=class_weight_dict, verbose=1
        )
        
        # --- ACADEMIC THRESHOLD TUNING (F-Beta Optimization) ---
        print(f"\nOptimizing Safety-Critical Boundary for {arch}...")
        y_val_true, y_val_probs = [], []
        for X_batch, y_batch in val_ds:
            preds = model.predict(X_batch, verbose=0)
            y_val_true.extend(y_batch.numpy())
            y_val_probs.extend(preds.flatten())
            
        # Ensure proper array types to prevent thresholding errors
        y_val_true = np.array(y_val_true)
        y_val_probs = np.array(y_val_probs)
            
        fpr, tpr, thresholds = roc_curve(y_val_true, y_val_probs)
        
        # Beta=2 heavily prioritizes catching crashes (Recall) while minimizing False Alarms
        beta = 2
        eps = 1e-9
        
        # Calculate raw F-beta dynamically across the curve
        f_beta_scores = []
        for thresh in thresholds:
            p = (y_val_probs >= thresh).astype(int)
            precision = precision_score(y_val_true, p, zero_division=0)
            recall = recall_score(y_val_true, p, zero_division=0)
            fb = ((1 + beta**2) * precision * recall) / (((beta**2) * precision) + recall + eps)
            f_beta_scores.append(fb)
            
        optimal_threshold = float(thresholds[np.argmax(f_beta_scores)])
        print(f"Optimal Boundary (Beta=2): {optimal_threshold:.4f} (Replaces naive 0.5)")
        
        # Execute rigorous test evaluation
        metrics = evaluate_model_pipeline(model, test_ds, optimal_threshold, arch)
        metrics["Optimized_Threshold"] = optimal_threshold
        master_results.append(metrics)
        
        # Save production weights
        model.save(os.path.join(MODEL_DIR, f"paper_model_{arch.lower()}.keras"))

    # Compile Benchmark Matrix Table
    summary_df = pd.DataFrame(master_results)
    print("\n================ FINAL PUBLICATION BENCHMARK MATRIX ================")
    print(summary_df.to_string(index=False))
    
    csv_path = os.path.join(REPORT_DIR, "academic_comparison_matrix.csv")
    summary_df.to_csv(csv_path, index=False)
    print(f"\nMatrix successfully generated and saved to: {csv_path}")

if __name__ == "__main__":
    main()