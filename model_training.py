import os
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Configuration
DATA_PATH = "student_datasheet.csv"
MODEL_PATH = "model.pkl"

# These are the columns the model expects to find in the CSV
FEATURE_COLS = [
    "AttendancePercent_Norm",
    "AssignmentScore_Norm",
    "Test1_Norm",
    "Test2_Norm",
    "Test3_Norm",
    "FinalExam_Norm",
]

def create_dummy_data_if_missing(path: str):
    """Creates a dummy CSV file for testing if one does not exist."""
    if os.path.exists(path):
        return

    print(f"⚠️ '{path}' not found. Creating dummy data for training...")
    
    # Generate 200 random students
    np.random.seed(42)
    n = 200
    data = {
        "StudentID": range(1, n + 1),
        "Name": [f"Student_{i}" for i in range(1, n + 1)],
        "AttendancePercent": np.random.randint(40, 100, n),
        "AssignmentScore": np.random.randint(50, 100, n),
        "Test1": np.random.randint(30, 100, n),
        "Test2": np.random.randint(30, 100, n),
        "Test3": np.random.randint(30, 100, n),
        "FinalExam": np.random.randint(30, 100, n),
    }
    
    df = pd.DataFrame(data)
    
    # Normalize (Simple min-max scaling simulation 0.0 to 1.0)
    for col in ["AttendancePercent", "AssignmentScore", "Test1", "Test2", "Test3", "FinalExam"]:
        df[f"{col}_Norm"] = df[col] / 100.0

    df["Total"] = df[["Test1", "Test2", "Test3", "FinalExam"]].sum(axis=1) + df["AssignmentScore"]
    
    # Logic to assign Pass/Fail (Make it somewhat realistic based on scores)
    # If Average norm score < 0.5, mark as Fail
    avg_score = df[[c for c in df.columns if "_Norm" in c]].mean(axis=1)
    df["PassFail"] = avg_score.apply(lambda x: "Fail" if x < 0.6 else "Pass")

    df.to_csv(path, index=False)
    print(f"✅ Created dummy dataset at: {path}")


def load_data(path: str):
    """Loads CSV, validates columns, and prepares X/y."""
    # 1. Ensure file exists
    create_dummy_data_if_missing(path)
    
    df = pd.read_csv(path)

    # 2. Check for missing columns
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"❌ Input CSV is missing required feature columns: {missing}")

    if "PassFail" not in df.columns:
        raise ValueError("❌ Input CSV is missing the target column 'PassFail'")

    # 3. Filter and Map Target
    # We map 'Fail' to 1 because we want to predict the probability of failure.
    df = df[df["PassFail"].isin(["Pass", "Fail"])].copy()
    df["target"] = df["PassFail"].map({"Pass": 0, "Fail": 1})

    print(f"📊 Data Loaded: {len(df)} rows.")
    print(f"   - Pass Count: {len(df[df['target']==0])}")
    print(f"   - Fail Count: {len(df[df['target']==1])}")

    X = df[FEATURE_COLS]
    y = df["target"]
    
    return X, y


def train_and_save_model():
    print("🚀 Starting Model Training...")
    
    try:
        X, y = load_data(DATA_PATH)
    except Exception as e:
        print(e)
        return

    # Split Data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Train Model
    # class_weight='balanced' helps if there are very few failures compared to passes
    model = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    print("\n------------------------------------------------")
    print(f"✅ Model Trained. Test Accuracy: {acc:.2%}")
    print("------------------------------------------------")
    print("Detailed Classification Report:\n")
    print(classification_report(y_test, y_pred, target_names=["Pass", "Fail"]))
    print("------------------------------------------------")

    # Save Model
    payload = {
        "model": model,
        "feature_cols": FEATURE_COLS,
    }
    
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(payload, f)

    print(f"💾 Model and feature columns saved to '{MODEL_PATH}'")


if __name__ == "__main__":
    train_and_save_model()