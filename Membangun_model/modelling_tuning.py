import mlflow
import mlflow.sklearn
import dagshub
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_curve, auc
)
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.inspection import permutation_importance
from sklearn.utils import estimator_html_repr
import numpy as np
import os

# DAGSHUB CONFIG
dagshub.init(
    repo_owner='lyaaaaa04',
    repo_name='SMSML_alyanoviyanti',
    mlflow=True
)
mlflow.set_tracking_uri("https://dagshub.com/lyaaaaa04/SMSML_alyanoviyanti.mlflow")
mlflow.set_experiment("Kriteria 2")

# MACRO ROC CURVE
def plot_macro_roc_curve(y_test, y_score, classes, filename):
    y_bin = label_binarize(y_test, classes=classes)
    fpr = {}
    tpr = {}
    roc_auc = {}

    for i in range(len(classes)):
        fpr[i], tpr[i], _ = roc_curve(y_bin[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(len(classes))]))
    mean_tpr = np.zeros_like(all_fpr)

    for i in range(len(classes)):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

    mean_tpr /= len(classes)
    macro_auc = auc(all_fpr, mean_tpr)

    plt.figure(figsize=(6, 5))
    plt.plot(all_fpr, mean_tpr, label=f"Macro-average ROC (AUC = {macro_auc:.3f})")
    plt.title("ROC Curve - Macro Average (Multi-Class)")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.savefig(filename)
    plt.close()


# TRAIN MODEL
def train_model(data_path):

    # Load data
    df = pd.read_csv(data_path)

    # LOAD ENCODERS PREPROCESSING
    encoders = joblib.load("label_encoders.pkl")

    # Encode kolom binned: math score_bin, reading score_bin, writing score_bin
    # bin_cols = ["math score_bin", "reading score_bin", "writing score_bin"]
    # for col in bin_cols:
        # if col in df.columns:
            # df[col] = encoders[col].transform(df[col])

    # Encode target performance_level
    # df["performance_level"] = encoders["performance_level"].transform(df["performance_level"])

    leakage_cols = ["math score_bin", "reading score_bin", "writing score_bin", "average_score"]
    df = df.drop(columns=leakage_cols, errors="ignore")

    # FEATURE & TARGET
    y = df["performance_level"]
    X = df.drop(["performance_level"], axis=1)

    classes = sorted(y.unique())

    # TRAIN TEST SPLIT
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    input_example = X_train.iloc[:2]

    # MLflow RUN
    with mlflow.start_run(run_name="RandomForest-Tuned"):

        # Hyperparameter Tuning
        param_dist = {
            "n_estimators": [100, 150, 200, 300, 500],
            "max_depth": [5, 10, 15, 20, None],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "max_features": ["sqrt", "log2", None]
        }

        tuner = RandomizedSearchCV(
            estimator=RandomForestClassifier(random_state=42),
            param_distributions=param_dist,
            n_iter=15,
            cv=3,
            scoring="accuracy",
            random_state=42,
            n_jobs=-1
        )

        tuner.fit(X_train, y_train)

        model = tuner.best_estimator_

        mlflow.log_params(tuner.best_params_)

        y_pred = model.predict(X_test)
        y_score = model.predict_proba(X_test)

        # METRICS
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='weighted')
        rec = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        report = classification_report(y_test, y_pred, output_dict=True)

        # MANUAL LOGGING
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)
        mlflow.log_metric("f1_score", f1)

        # CONFUSION MATRIX
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.title("Confusion Matrix")
        plt.savefig("confusion_matrix_tuned.png")
        plt.close()
        mlflow.log_artifact("confusion_matrix_tuned.png")

        #  FEATURE IMPORTANCE
        importances = model.feature_importances_
        plt.figure(figsize=(8, 5))
        sns.barplot(x=importances, y=X.columns)
        plt.title("Feature Importance")
        plt.tight_layout()
        plt.savefig("feature_importance_tuned.png")
        plt.close()
        mlflow.log_artifact("feature_importance_tuned.png")

        #  ROC MACRO AVERAGE
        plot_macro_roc_curve(
            y_test, y_score, classes, filename="roc_macro_avg_tuned.png"
        )
        mlflow.log_artifact("roc_macro_avg_tuned.png")

        #  estimator_html
        html_repr = estimator_html_repr(model)
        with open("estimator_html_tuned.html", "w", encoding="utf-8") as f:
            f.write(html_repr)
        mlflow.log_artifact("estimator_html_tuned.html")

        #  metric_info JSON
        with open("metric_info_tuned.json", "w") as f:
            json.dump({
                "accuracy": acc,
                "precision_weighted": prec,
                "recall_weighted": rec,
                "f1_weighted": f1,
                "classification_report": report
            }, f, indent=4)
        mlflow.log_artifact("metric_info_tuned.json")

        #  SAVE MODEL
        model_dir = "model_tuned"
        mlflow.sklearn.save_model(
            sk_model=model,
            path=model_dir,
            input_example=input_example
        )
        mlflow.log_artifacts(model_dir, artifact_path="model_tuned")

    print("Training Completed — Logged to MLflow on DagsHub")

if __name__ == "__main__":
    train_model("StudentsPerformance_preprocessed.csv")
