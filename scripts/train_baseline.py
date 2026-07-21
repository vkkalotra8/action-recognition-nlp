import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder, StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parent.parent
FEATURES_ROOT = PROJECT_ROOT / "data" / "features"
MODELS_ROOT = PROJECT_ROOT / "models"
RESULTS_ROOT = PROJECT_ROOT / "results" / "reports"


def load_dataset(split_name: str) -> tuple[np.ndarray, np.ndarray]:
    """Load combined features and labels for one split."""
    features_path = FEATURES_ROOT / f"{split_name}_features.npy"
    labels_path = FEATURES_ROOT / f"{split_name}_labels.npy"

    if not features_path.exists():
        raise FileNotFoundError(
            f"Feature file not found: {features_path}"
        )

    if not labels_path.exists():
        raise FileNotFoundError(
            f"Label file not found: {labels_path}"
        )

    X = np.load(features_path)
    y = np.load(labels_path)

    return X, y


def main() -> None:
    print("=" * 70)
    print("TRAINING VIDEO-ONLY LOGISTIC REGRESSION BASELINE")
    print("=" * 70)

    MODELS_ROOT.mkdir(parents=True, exist_ok=True)
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)

    X_train, y_train_text = load_dataset("train")
    X_val, y_val_text = load_dataset("val")

    print(f"Training features   : {X_train.shape}")
    print(f"Training labels     : {y_train_text.shape}")
    print(f"Validation features : {X_val.shape}")
    print(f"Validation labels   : {y_val_text.shape}")

    print("\nEncoding class labels...")

    label_encoder = LabelEncoder()

    y_train = label_encoder.fit_transform(y_train_text)
    y_val = label_encoder.transform(y_val_text)

    print(f"Classes detected: {len(label_encoder.classes_)}")

    for index, class_name in enumerate(label_encoder.classes_):
        print(f"  {index}: {class_name}")

    print("\nScaling feature vectors...")

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    print("Feature scaling completed.")

    print("\nTraining Logistic Regression...")

    model = LogisticRegression(
        max_iter=3000,
        solver="lbfgs",
        class_weight="balanced",
        random_state=42,
    )

    model.fit(X_train_scaled, y_train)

    print("Model training completed.")

    print("\nEvaluating on validation split...")

    y_val_pred = model.predict(X_val_scaled)

    validation_accuracy = accuracy_score(
        y_val,
        y_val_pred,
    )

    report = classification_report(
        y_val,
        y_val_pred,
        target_names=label_encoder.classes_,
        digits=4,
        zero_division=0,
    )

    print(f"\nValidation accuracy: {validation_accuracy:.4f}")
    print("\nValidation classification report:")
    print(report)

    model_path = MODELS_ROOT / "logistic_regression_baseline.joblib"
    scaler_path = MODELS_ROOT / "feature_scaler.joblib"
    encoder_path = MODELS_ROOT / "label_encoder.joblib"

    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    joblib.dump(label_encoder, encoder_path)

    report_path = RESULTS_ROOT / "validation_classification_report.txt"

    with report_path.open("w", encoding="utf-8") as file:
        file.write(
            "VIDEO-ONLY LOGISTIC REGRESSION BASELINE\n"
        )
        file.write("=" * 70 + "\n")
        file.write(
            f"Validation accuracy: "
            f"{validation_accuracy:.4f}\n\n"
        )
        file.write(report)

    metrics = {
        "model": "LogisticRegression",
        "validation_accuracy": float(validation_accuracy),
        "training_samples": int(X_train.shape[0]),
        "validation_samples": int(X_val.shape[0]),
        "feature_dimension": int(X_train.shape[1]),
        "number_of_classes": int(
            len(label_encoder.classes_)
        ),
        "classes": label_encoder.classes_.tolist(),
    }

    metrics_path = RESULTS_ROOT / "validation_metrics.json"

    with metrics_path.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    print("\nSaved files:")
    print(f"  Model         : {model_path}")
    print(f"  Scaler        : {scaler_path}")
    print(f"  Label encoder : {encoder_path}")
    print(f"  Report        : {report_path}")
    print(f"  Metrics       : {metrics_path}")

    print("\nPASS: Baseline model trained and saved successfully.")


if __name__ == "__main__":
    main()