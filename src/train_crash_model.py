import os
import numpy as np
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

# ─────────────────────────────────────────────
# STEP 1: SET DATA PATH (matches your layout)
# motorcycle-safety/
#   ├─ src/
#   └─ data/raw/
#       ├─ trainingData.csv
#       ├─ testData.csv
#       └─ ControlScenarios/...
# ─────────────────────────────────────────────

HERE = os.path.dirname(__file__)
DATAPATH = os.path.join(HERE, "..", "data", "raw")


# ─────────────────────────────────────────────
# STEP 2: LOAD TRAINING AND TEST DATA
# ─────────────────────────────────────────────

print("Loading trainingData.csv ...")
train_path = os.path.join(DATAPATH, "trainingData.csv")
data_train = pd.read_csv(train_path)

print("Loading testData.csv ...")
test_path = os.path.join(DATAPATH, "testData.csv")
data_test = pd.read_csv(test_path)

print(f"Training samples: {len(data_train)}")
print(f"Test samples    : {len(data_test)}")


# ─────────────────────────────────────────────
# STEP 3: CREATE DERIVED FEATURES (from LoadData.py)
#  - angular velocity difference front vs rear
#  - angular acceleration difference front vs rear
#  - left/right sensor OR-combination
#  - combined contact forces
# ─────────────────────────────────────────────

def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    df["angveldiff"] = df["FW_angvel_Y"] - df["RW_angvel_Y"]
    df["angaccdiff"] = df["FW_angacc_Y"] - df["RW_angacc_Y"]

    df["sensorLeft"] = (
        df["SwitchSidesensor_left_road"] | df["SwitchSidesensor_left_car"]
    ).astype(int)

    df["sensorRight"] = (
        df["SwitchSidesensor_right_road"] | df["SwitchSidesensor_right_car"]
    ).astype(int)

    df["FW_cnt_Force"] = df["FW_Car_cnt_force"] + df["FW_Road_cnt_force"]
    df["RW_cnt_Force"] = df["RW_Car_cnt_force"] + df["RW_Road_cnt_force"]

    return df


data_train = add_derived_features(data_train)
data_test = add_derived_features(data_test)


# ─────────────────────────────────────────────
# STEP 4: SELECT FEATURES (X) AND LABEL (y)
# Based on LoadData.py description and derived features
# ─────────────────────────────────────────────

feature_cols = [
    "Time",
    # Motorcycle body angular acceleration
    "MotoBody_angaccRes", "MotoBody_angaccX", "MotoBody_angaccY", "MotoBody_angaccZ",
    # Front deflection
    "FrontDef_angdispX", "FrontDef_angdispY", "FrontDef_angdispZ",
    # Motorcycle body angular velocity
    "MotoBody_angvelRes", "MotoBody_angvelX", "MotoBody_angvelY", "MotoBody_angvelZ",
    # Rear wheel linear acceleration
    "MotoRW_linaccRes", "MotoRW_linaccX", "MotoRW_linaccY", "MotoRW_linaccZ",
    # Front wheel linear acceleration
    "MotoFW_linaccRes", "MotoFW_linaccX", "MotoFW_linaccY", "MotoFW_linaccZ",
    # Motorcycle body linear acceleration
    "MotoBody_linaccRes", "MotoBody_linaccX", "MotoBody_linaccY", "MotoBody_linaccZ",
    # Front wheel linear velocity
    "MotoFW_linvelRes", "MotoFW_linvelX", "MotoFW_linvelY", "MotoFW_linvelZ",
    # Rear wheel linear velocity
    "MotoRW_linvelRes", "MotoRW_linvelX", "MotoRW_linvelY", "MotoRW_linvelZ",
    # Motorcycle body linear velocity
    "MotoBody_linvelRes", "MotoBody_linvelX", "MotoBody_linvelY", "MotoBody_linvelZ",
    # Front deflection dynamics
    "Frontdef_angacc", "Frontdef_angpos", "Frontdef_angvel",
    # Joint velocities
    "JointRW_angvel", "JointFW_angvel",
    # Derived features
    "angveldiff", "angaccdiff",
    "sensorLeft", "sensorRight",
    "FW_cnt_Force", "RW_cnt_Force",
]

label_col = "CrashLabelML"

# Ensure all columns exist
missing_cols = [c for c in feature_cols + [label_col] if c not in data_train.columns]
if missing_cols:
    raise ValueError(f"Missing columns in trainingData.csv: {missing_cols}")

X_train = data_train[feature_cols].to_numpy()
y_train = data_train[label_col].to_numpy()

X_test = data_test[feature_cols].to_numpy()
y_test = data_test[label_col].to_numpy()

print(f"\nNumber of features: {len(feature_cols)}")
print(f"Crash samples in training: {int(y_train.sum())} / {len(y_train)}")
print(f"Crash samples in test    : {int(y_test.sum())} / {len(y_test)}")


# ─────────────────────────────────────────────
# STEP 5: TRAIN RANDOM FOREST CLASSIFIER
# ─────────────────────────────────────────────

print("\nTraining RandomForestClassifier ...")
model = RandomForestClassifier(
    n_estimators=150,  # slightly more trees for stability
    max_depth=18,      # limit depth to avoid overfitting
    random_state=42,
    n_jobs=-1,         # use all cores
)

model.fit(X_train, y_train)
print("Training finished.")


# ─────────────────────────────────────────────
# STEP 6: EVALUATE ON TEST SET
# ─────────────────────────────────────────────

y_pred = model.predict(X_test)

print("\n============================================================")
print("                    TEST SET PERFORMANCE                    ")
print("============================================================")
print(f"Accuracy  : {accuracy_score(y_test, y_pred) * 100:.2f}%")
print(f"Precision : {precision_score(y_test, y_pred) * 100:.2f}%")
print(f"Recall    : {recall_score(y_test, y_pred) * 100:.2f}%")
print(f"F1 Score  : {f1_score(y_test, y_pred) * 100:.2f}%")

cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(cm)
print(f"  True Non‑Crash (correct) : {cm[0][0]}")
print(f"  False Alarm (wrong alert): {cm[0][1]}")
print(f"  Missed Crash (dangerous) : {cm[1][0]}")
print(f"  True Crash (correct)     : {cm[1][1]}")

print("\nFull classification report:")
print(classification_report(y_test, y_pred, target_names=["No Crash", "Crash"]))


# ─────────────────────────────────────────────
# STEP 7: SAVE MODEL FOR LATER USE
# ─────────────────────────────────────────────

model_path = os.path.join(HERE, "crash_model.pkl")
joblib.dump(model, model_path)
print(f"\nModel saved to: {model_path}")
print("Load later with: joblib.load('src/crash_model.pkl')")


# ─────────────────────────────────────────────
# STEP 8: OPTIONAL — VALIDATE ON ONE SCENARIO
# Example: Curbstone_5_Out.csv (NON‑CRASH)
# ─────────────────────────────────────────────

scenario_rel_path = os.path.join("ControlScenarios", "Curbstone_5_Out.csv")
scenario_file = os.path.join(DATAPATH, scenario_rel_path)

if os.path.exists(scenario_file):
    print("\n--- Scenario check: Curbstone_5_Out.csv (expected NON‑CRASH) ---")
    scenario = pd.read_csv(scenario_file)
    scenario = add_derived_features(scenario)

    # Ensure same feature set
    missing_scen_cols = [c for c in feature_cols + [label_col] if c not in scenario.columns]
    if missing_scen_cols:
        print(f"Missing columns in scenario file: {missing_scen_cols}")
    else:
        X_scen = scenario[feature_cols].to_numpy()
        y_scen = scenario[label_col].to_numpy()
        y_scen_pred = model.predict(X_scen)

        print(f"Scenario accuracy: {accuracy_score(y_scen, y_scen_pred) * 100:.2f}%")
        crash_pct = y_scen_pred.mean() * 100
        print(f"Crash predictions in this scenario: {crash_pct:.2f}% (should be ~0% for curbstone)")
else:
    print("\nScenario file Curbstone_5_Out.csv not found, skipping scenario validation.")
