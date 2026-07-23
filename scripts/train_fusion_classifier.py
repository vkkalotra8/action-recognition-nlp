from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder, StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]

FUSION_ROOT = PROJECT_ROOT / "data" / "fusion_features"
MODELS_ROOT = PROJECT_ROOT / "models"
REPORTS_ROOT = PROJECT_ROOT / "results" / "reports"

MODEL_OUTPUT_PATH = (
    MODELS_ROOT / "oracle_fusion_logistic_regression.joblib"
)

SCALER_OUTPUT_PATH = (
    MODELS_ROOT / "oracle_fusion_feature_scaler.joblib"
)

LABEL_ENCODER_OUTPUT_PATH = (
    MODELS_ROOT / "oracle_fusion_label_encoder.joblib"
)

REPORT_OUTPUT_PATH = (
    REPORTS_ROOT / "oracle_fusion_validation_report.txt"
)


def load_split(
    split_name: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Load fused features and labels for one split."""

    features_path = (
        FUSION_ROOT / f"{split_name}_fused_features.npy"
    )

    labels_path = (
        FUSION_ROOT / f"{split_name}_labels.npy"
    )

    if not features_path.exists():
        raise FileNotFoundError(
            f"Fused feature file not found: {features_path}"
        )

    if not labels_path.exists():
        raise FileNotFoundError(
            f"Label file not found: {labels_path}"
        )

    features = np.load(
        features_path
    )

    labels = np.load(
        labels_path,
        allow_pickle=True,
    )

    return features, labels


def main() -> None:
    print("=" * 70)
    print("TRAINING ORACLE VISION-LANGUAGE FUSION CLASSIFIER")
    print("=" * 70)

    train_features, train_labels_text = load_split(
        "train"
    )

    val_features, val_labels_text = load_split(
        "val"
    )

    if train_features.shape[1] != 896:
        raise ValueError(
            f"Expected 896 training features, "
            f"received {train_features.shape[1]}."
        )

    if val_features.shape[1] != 896:
        raise ValueError(
            f"Expected 896 validation features, "
            f"received {val_features.shape[1]}."
        )

    if not np.isfinite(train_features).all():
        raise ValueError(
            "Training features contain NaN or infinite values."
        )

    if not np.isfinite(val_features).all():
        raise ValueError(
            "Validation features contain NaN or infinite values."
        )

    label_encoder = LabelEncoder()

    train_labels = label_encoder.fit_transform(
        train_labels_text
    )

    val_labels = label_encoder.transform(
        val_labels_text
    )

    scaler = StandardScaler()

    train_features_scaled = scaler.fit_transform(
        train_features
    )

    val_features_scaled = scaler.transform(
        val_features
    )

    model = LogisticRegression(
        max_iter=3000,
        random_state=42,
    )

    model.fit(
        train_features_scaled,
        train_labels,
    )

    val_predictions = model.predict(
        val_features_scaled
    )

    validation_accuracy = accuracy_score(
        val_labels,
        val_predictions,
    )

    report = classification_report(
        val_labels,
        val_predictions,
        target_names=label_encoder.classes_,
        digits=4,
        zero_division=0,
    )

    MODELS_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    REPORTS_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        model,
        MODEL_OUTPUT_PATH,
    )

    joblib.dump(
        scaler,
        SCALER_OUTPUT_PATH,
    )

    joblib.dump(
        label_encoder,
        LABEL_ENCODER_OUTPUT_PATH,
    )

    report_text = "\n".join(
        [
            "ORACLE VISION-LANGUAGE FUSION VALIDATION REPORT",
            "=" * 70,
            "",
            "WARNING:",
            (
                "This model uses the ground-truth class text embedding "
                "inside each fused feature. It contains target leakage "
                "and must only be interpreted as an oracle experiment."
            ),
            "",
            f"Training feature shape: {train_features.shape}",
            f"Validation feature shape: {val_features.shape}",
            f"Validation accuracy: {validation_accuracy:.4f}",
            "",
            report,
        ]
    )

    REPORT_OUTPUT_PATH.write_text(
        report_text,
        encoding="utf-8",
    )

    print(
        f"Training features   : {train_features.shape}"
    )

    print(
        f"Validation features : {val_features.shape}"
    )

    print(
        f"Validation accuracy : {validation_accuracy:.4f}"
    )

    print("\nClassification report:")
    print(report)

    print("Files saved:")

    print(
        f"  Model         : {MODEL_OUTPUT_PATH}"
    )

    print(
        f"  Scaler        : {SCALER_OUTPUT_PATH}"
    )

    print(
        f"  Label encoder : {LABEL_ENCODER_OUTPUT_PATH}"
    )

    print(
        f"  Report        : {REPORT_OUTPUT_PATH}"
    )

    print("\n" + "=" * 70)
    print("PASS: ORACLE FUSION CLASSIFIER TRAINED")
    print("=" * 70)


if __name__ == "__main__":
    main()