# model_training.py
import pandas as pd
from sklearn.model_selection import train_test_split # pyright: ignore[reportMissingModuleSource]
from sklearn.linear_model import LogisticRegression # type: ignore
import pickle

DATA_PATH = "student_datasheet.csv"   # yahi naam se file rakhna
MODEL_PATH = "model.pkl"


def load_data(path: str):
    df = pd.read_csv(path)

    # Sirf Pass/Fail rows
    df = df[df["PassFail"].isin(["Pass", "Fail"])].copy()
    df["target"] = df["PassFail"].map({"Pass": 0, "Fail": 1})

    # Normalized feature columns (tumhari sheet me present hain)
    feature_cols = [
        "AttendancePercent_Norm",
        "AssignmentScore_Norm",
        "Test1_Norm",
        "Test2_Norm",
        "Test3_Norm",
        "FinalExam_Norm",
    ]

    X = df[feature_cols]
    y = df["target"]
    return X, y, feature_cols


def train_and_save_model():
    X, y, feature_cols = load_data(DATA_PATH)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    print("Train accuracy:", model.score(X_train, y_train))
    print("Test accuracy:", model.score(X_test, y_test))

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(
            {
                "model": model,
                "feature_cols": feature_cols,
            },
            f,
        )

    print(f"Model saved to {MODEL_PATH}")


if __name__ == "__main__":
    train_and_save_model()
