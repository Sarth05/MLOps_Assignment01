"""Train, compare, evaluate, persist, and optionally log classification models."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    RocCurveDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline

from heart_ml.features import FEATURES, build_preprocessor, split_xy


def evaluate_model(model, x_train, y_train, x_test, y_test, cv):
    cv_results = cross_validate(
        model, x_train, y_train, cv=cv,
        scoring={"accuracy": "accuracy", "precision": "precision",
                 "recall": "recall", "roc_auc": "roc_auc"},
        n_jobs=-1, error_score="raise",
    )
    model.fit(x_train, y_train)
    pred = model.predict(x_test)
    proba = model.predict_proba(x_test)[:, 1]
    metrics = {
        "accuracy": float(accuracy_score(y_test, pred)),
        "precision": float(precision_score(y_test, pred, zero_division=0)),
        "recall": float(recall_score(y_test, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, proba)),
        "cv_accuracy_mean": float(cv_results["test_accuracy"].mean()),
        "cv_accuracy_std": float(cv_results["test_accuracy"].std()),
        "cv_precision_mean": float(cv_results["test_precision"].mean()),
        "cv_recall_mean": float(cv_results["test_recall"].mean()),
        "cv_roc_auc_mean": float(cv_results["test_roc_auc"].mean()),
    }
    return model, metrics, pred, proba


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/raw/heart.csv")
    parser.add_argument("--output", default="models")
    parser.add_argument("--reports", default="reports/model")
    parser.add_argument("--experiment", default="heart-disease-assignment")
    args = parser.parse_args()

    data_path, model_dir, report_dir = Path(args.data), Path(args.output), Path(args.reports)
    model_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(data_path)
    x, y = split_xy(frame)
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42, stratify=y
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    candidates = {
        "logistic_regression": LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42),
        "random_forest": RandomForestClassifier(
            n_estimators=300, min_samples_leaf=2, class_weight="balanced",
            random_state=42, n_jobs=-1,
        ),
    }
    mlflow.set_experiment(args.experiment)
    results = {}
    fitted = {}
    for name, estimator in candidates.items():
        pipeline = Pipeline([("preprocessor", build_preprocessor()), ("classifier", estimator)])
        with mlflow.start_run(run_name=name):
            model, metrics, pred, proba = evaluate_model(
                pipeline, x_train, y_train, x_test, y_test, cv
            )
            params = estimator.get_params()
            mlflow.log_params({k: str(v) for k, v in params.items()})
            mlflow.log_param("model_name", name)
            mlflow.log_param("random_state", 42)
            mlflow.log_param("test_size", 0.2)
            mlflow.log_param("cv_folds", 5)
            mlflow.log_metrics(metrics)
            results[name] = metrics
            fitted[name] = model

            cm = confusion_matrix(y_test, pred)
            plt.figure(figsize=(5, 4))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
            plt.title(f"Confusion Matrix — {name}")
            plt.xlabel("Predicted")
            plt.ylabel("Actual")
            cm_path = report_dir / f"{name}_confusion_matrix.png"
            plt.tight_layout()
            plt.savefig(cm_path, dpi=160)
            plt.close()
            mlflow.log_artifact(str(cm_path))

            RocCurveDisplay.from_predictions(y_test, proba)
            plt.title(f"ROC Curve — {name}")
            roc_path = report_dir / f"{name}_roc_curve.png"
            plt.tight_layout()
            plt.savefig(roc_path, dpi=160)
            plt.close()
            mlflow.log_artifact(str(roc_path))

            report_path = report_dir / f"{name}_classification_report.json"
            report_path.write_text(json.dumps(
                classification_report(y_test, pred, output_dict=True, zero_division=0), indent=2
            ))
            mlflow.log_artifact(str(report_path))
            mlflow.sklearn.log_model(model, artifact_path="model", serialization_format="cloudpickle")

    # Selection rule is fixed and transparent: highest mean CV ROC-AUC.
    selected = max(results, key=lambda name: results[name]["cv_roc_auc_mean"])
    final_model = fitted[selected]
    model_path = model_dir / "heart_disease_pipeline.joblib"
    joblib.dump(final_model, model_path)
    metadata = {
        "selected_model": selected,
        "features": FEATURES,
        "target_mapping": "original 0 -> 0; original 1-4 -> 1",
        "selection_rule": "highest mean 5-fold CV ROC-AUC on training split",
        "random_state": 42,
        "test_size": 0.2,
        "results": results,
        "note": "Educational model only; not clinically validated.",
    }
    metadata_path = model_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2))
    (report_dir / "metrics.json").write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata, indent=2))
    print(f"Saved model to {model_path}")


if __name__ == "__main__":
    main()
