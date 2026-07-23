from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

FUSION_ROOT = PROJECT_ROOT / "data" / "fusion_features"
MODELS_ROOT = PROJECT_ROOT / "models"
REPORTS_ROOT = PROJECT_ROOT / "results" / "reports"
PLOTS_ROOT = PROJECT_ROOT / "results" / "plots"

MODEL_PATH = (
    MODELS_ROOT / "oracle_fusion_logistic_regression.joblib"
)

SCALER_PATH = (
    MODELS_ROOT / "oracle_fusion_feature_scaler.joblib"
)

LABEL_ENCODER_PATH = (
    MODELS_ROOT / "oracle_fusion_label_encoder.joblib"
)

TEST_REPORT_PATH = (
    REPORTS_ROOT / "oracle_fusion_test_report.txt"
)

TEST_METRICS_PATH = (
    REPORTS_ROOT / "oracle_fusion_test_metrics.json"
)

MISCLASSIFICATIONS_PATH = (
    REPORTS_ROOT / "oracle_fusion_test_misclassifications.json"
)

CONFUSION_MATRIX_PATH = (
    PLOTS_ROOT / "oracle_fusion_test_confusion_matrix.png"
)


def load_test_data() -> tuple[np.ndarray, np.ndarray]:
    """Load the oracle fused test features and labels."""

    features_path = (
        FUSION_ROOT / "test_fused_features.npy"
    )

    labels_path = (
        FUSION_ROOT / "test_labels.npy"
    )

    if not features_path.exists():
        raise FileNotFoundError(
            f"Test features not found: {features_path}"
        )

    if not labels_path.exists():
        raise FileNotFoundError(
            f"Test labels not found: {labels_path}"
        )

    features = np.load(
        features_path
    )

    labels = np.load(
        labels_path,
        allow_pickle=True,
    )

    return features, labels


def save_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: np.ndarray,
) -> None:
    """Create and save the test confusion matrix."""

    matrix = confusion_matrix(
        y_true,
        y_pred,
    )

    PLOTS_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure, axis = plt.subplots(
        figsize=(11, 9),
    )

    image = axis.imshow(
        matrix,
        aspect="auto",
    )

    figure.colorbar(
        image,
        ax=axis,
    )

    axis.set_xticks(
        np.arange(len(class_names))
    )

    axis.set_yticks(
        np.arange(len(class_names))
    )

    axis.set_xticklabels(
        class_names,
        rotation=45,
        ha="right",
    )

    axis.set_yticklabels(
        class_names
    )

    axis.set_xlabel(
        "Predicted class"
    )

    axis.set_ylabel(
        "True class"
    )

    axis.set_title(
        "Oracle Fusion Test Confusion Matrix"
    )

    for row_index in range(matrix.shape[0]):
        for column_index in range(matrix.shape[1]):
            axis.text(
                column_index,
                row_index,
                str(matrix[row_index, column_index]),
                ha="center",
                va="center",
            )

    figure.tight_layout()

    figure.savefig(
        CONFUSION_MATRIX_PATH,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(figure)


def main() -> None:
    print("=" * 70)
    print("EVALUATING ORACLE VISION-LANGUAGE FUSION CLASSIFIER")
    print("=" * 70)

    for required_path in (
        MODEL_PATH,
        SCALER_PATH,
        LABEL_ENCODER_PATH,
    ):
        if not required_path.exists():
            raise FileNotFoundError(
                f"Required model file not found: {required_path}"
            )

    test_features, test_labels_text = load_test_data()

    if test_features.shape[1] != 896:
        raise ValueError(
            f"Expected 896 test features, "
            f"received {test_features.shape[1]}."
        )

    if test_features.shape[0] != test_labels_text.shape[0]:
        raise ValueError(
            "Test feature count does not match label count."
        )

    if not np.isfinite(test_features).all():
        raise ValueError(
            "Test features contain NaN or infinite values."
        )

    model = joblib.load(
        MODEL_PATH
    )

    scaler = joblib.load(
        SCALER_PATH
    )

    label_encoder = joblib.load(
        LABEL_ENCODER_PATH
    )

    test_labels = label_encoder.transform(
        test_labels_text
    )

    test_features_scaled = scaler.transform(
        test_features
    )

    predictions = model.predict(
        test_features_scaled
    )

    accuracy = accuracy_score(
        test_labels,
        predictions,
    )

    macro_precision = precision_score(
        test_labels,
        predictions,
        average="macro",
        zero_division=0,
    )

    macro_recall = recall_score(
        test_labels,
        predictions,
        average="macro",
        zero_division=0,
    )

    macro_f1 = f1_score(
        test_labels,
        predictions,
        average="macro",
        zero_division=0,
    )

    weighted_f1 = f1_score(
        test_labels,
        predictions,
        average="weighted",
        zero_division=0,
    )

    report = classification_report(
        test_labels,
        predictions,
        target_names=label_encoder.classes_,
        digits=4,
        zero_division=0,
    )

    REPORTS_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    report_text = "\n".join(
        [
            "ORACLE VISION-LANGUAGE FUSION TEST REPORT",
            "=" * 70,
            "",
            "WARNING:",
            (
                "This model uses the ground-truth class text embedding "
                "inside every fused test feature. It contains target "
                "leakage and is not a valid unseen-video evaluation."
            ),
            "",
            f"Test feature shape: {test_features.shape}",
            f"Accuracy: {accuracy:.4f}",
            f"Macro precision: {macro_precision:.4f}",
            f"Macro recall: {macro_recall:.4f}",
            f"Macro F1-score: {macro_f1:.4f}",
            f"Weighted F1-score: {weighted_f1:.4f}",
            "",
            report,
        ]
    )

    TEST_REPORT_PATH.write_text(
        report_text,
        encoding="utf-8",
    )

    metrics = {
        "experiment": "oracle_true_label_concatenation",
        "contains_label_leakage": True,
        "test_samples": int(
            test_features.shape[0]
        ),
        "feature_dimension": int(
            test_features.shape[1]
        ),
        "accuracy": float(
            accuracy
        ),
        "macro_precision": float(
            macro_precision
        ),
        "macro_recall": float(
            macro_recall
        ),
        "macro_f1": float(
            macro_f1
        ),
        "weighted_f1": float(
            weighted_f1
        ),
    }

    with TEST_METRICS_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metrics,
            file,
            indent=4,
        )

    incorrect_indices = np.where(
        predictions != test_labels
    )[0]

    misclassifications = [
        {
            "sample_index": int(index),
            "true_class": str(
                test_labels_text[index]
            ),
            "predicted_class": str(
                label_encoder.inverse_transform(
                    [predictions[index]]
                )[0]
            ),
        }
        for index in incorrect_indices
    ]

    with MISCLASSIFICATIONS_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            misclassifications,
            file,
            indent=4,
        )

    save_confusion_matrix(
        test_labels,
        predictions,
        label_encoder.classes_,
    )

    print(
        f"Test features      : {test_features.shape}"
    )

    print(
        f"Accuracy           : {accuracy:.4f}"
    )

    print(
        f"Macro precision    : {macro_precision:.4f}"
    )

    print(
        f"Macro recall       : {macro_recall:.4f}"
    )

    print(
        f"Macro F1-score     : {macro_f1:.4f}"
    )

    print(
        f"Weighted F1-score  : {weighted_f1:.4f}"
    )

    print(
        f"Misclassifications : {len(misclassifications)}"
    )

    print("\nClassification report:")
    print(report)

    print("Files saved:")

    print(
        f"  Report             : {TEST_REPORT_PATH}"
    )

    print(
        f"  Metrics            : {TEST_METRICS_PATH}"
    )

    print(
        f"  Misclassifications : {MISCLASSIFICATIONS_PATH}"
    )

    print(
        f"  Confusion matrix   : {CONFUSION_MATRIX_PATH}"
    )

    print("\n" + "=" * 70)
    print("PASS: ORACLE FUSION CLASSIFIER EVALUATED")
    print("=" * 70)


if __name__ == "__main__":
    main()