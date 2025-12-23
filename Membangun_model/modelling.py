import mlflow
import mlflow.sklearn
import pandas as pd
import joblib
import os
import shutil
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def train_model(data_path):
    # LOAD DATA
    df = pd.read_csv(data_path)

    # LOAD ENCODERS (JIKA DIPAKAI)
    encoders = joblib.load("label_encoders.pkl")

    # HAPUS KOLOM LEAKAGE
    leakage_cols = ["math score_bin", "reading score_bin", "writing score_bin", "average_score"]
    df = df.drop(columns=leakage_cols, errors="ignore")

    # FEATURE & TARGET
    y = df["performance_level"]
    X = df.drop(["performance_level"], axis=1)

    # TRAIN TEST SPLIT
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    input_example = X_train.head(3)

    # SET MLflow LOCAL
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("Kriteria2_Basic")

    with mlflow.start_run(run_name="Kriteria2-Basic"):

        # ENABLE AUTOLOG
        mlflow.autolog()

        # DEFINE MODEL
        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            random_state=42
        )

        # TRAIN
        model.fit(X_train, y_train)

        # PREDIKSI & EVALUASI
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print("Accuracy:", acc)

        # SIMPAN MODEL
        model_dir = "model"

        # HAPUS FOLDER MODEL LAMA JIKA ADA
        if os.path.exists(model_dir):
            shutil.rmtree(model_dir)

        try:
            # Simpan model ke folder lokal
            mlflow.sklearn.save_model(
                sk_model=model,
                path=model_dir,
                input_example=input_example
            )
            # Upload folder model ke MLflow artifacts
            mlflow.log_artifacts(model_dir, artifact_path="model")
            print("Model successfully saved and logged in MLflow!")
        except Exception as e:
            print("Failed to save/log model:", e)
            raise e

    print("Training completed — logged locally with autolog.")

if __name__ == "__main__":
    train_model("StudentsPerformance_preprocessed.csv")