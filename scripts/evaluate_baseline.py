import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
    precision_score,
    recall_score,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
FEATURES_ROOT = PROJECT_ROOT / "data" / "features"
MODELS_ROOT = PROJECT_ROOT / "models"
REPORTS_ROOT = PROJECT_ROOT / "results" / "reports"
PLOTS_ROOT = PROJECT_ROOT / "results" / "plots"


def load_test_dataset() -> tuple[np.ndarray, np.ndarray]:
    """Load test features and text labels."""
    features_path = FEATURES_ROOT / "test_features.npy"
    labels_path = FEATURES_ROOT / "test_labels.npy"

    if not features_path.exists():
        raise FileNotFoundError(
            f"Test features not found: {features_path}"
        )

    if not labels_path.exists():
        raise FileNotFoundError(
            f"Test labels not found: {labels_path}"
        )

    X_test = np.load(features_path)
    y_test_text = np.load(labels_path)

    return X_test, y_test_text


def load_saved_components():
    """Load the trained model, scaler, and label encoder."""
    model_path = MODELS_ROOT / "logistic_regression_baseline.joblib"
    scaler_path = MODELS_ROOT / "feature_scaler.joblib"
    encoder_path = MODELS_ROOT / "label_encoder.joblib"

    for path in (model_path, scaler_path, encoder_path):
        if not path.exists():
            raise FileNotFoundError(
                f"Required saved file not found: {path}"
            )

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    label_encoder = joblib.load(encoder_path)

    return model, scaler, label_encoder


def save_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
) -> Path:
    """Create and save the confusion matrix plot."""
    matrix = confusion_matrix(y_true, y_pred)

    figure, axis = plt.subplots(figsize=(12, 10))

    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=class_names,
    )

    display.plot(
        ax=axis,
        cmap="Blues",
        xticks_rotation=45,
        values_format="d",
        colorbar=False,
    )

    axis.set_title(
        "Video-Only Baseline Confusion Matrix"
    )

    figure.tight_layout()

    output_path = PLOTS_ROOT / "test_confusion_matrix.png"

    figure.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(figure)

    return output_path


def print_misclassification_summary(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
) -> list[dict]:
    """Find and print the most common incorrect class pairs."""
    matrix = confusion_matrix(y_true, y_pred)

    errors = []

    for true_index in range(len(class_names)):
        for predicted_index in range(len(class_names)):
            if true_index == predicted_index:
                continue

            count = int(matrix[true_index, predicted_index])

            if count > 0:
                errors.append(
                    {
                        "true_class": class_names[true_index],
                        "predicted_class": class_names[predicted_index],
                        "count": count,
                    }
                )

    errors.sort(
        key=lambda item: item["count"],
        reverse=True,
    )

    print("\nMost common misclassifications:")

    if not errors:
        print("  No misclassifications found.")
    else:
        for error in errors[:10]:
            print(
                f"  {error['true_class']} -> "
                f"{error['predicted_class']}: "
                f"{error['count']}"
            )

    return errors


def main() -> None:
    print("=" * 70)
    print("TESTING VIDEO-ONLY LOGISTIC REGRESSION BASELINE")
    print("=" * 70)

    REPORTS_ROOT.mkdir(parents=True, exist_ok=True)
    PLOTS_ROOT.mkdir(parents=True, exist_ok=True)

    X_test, y_test_text = load_test_dataset()

    print(f"Test features: {X_test.shape}")
    print(f"Test labels  : {y_test_text.shape}")

    model, scaler, label_encoder = load_saved_components()

    print("\nSaved model components loaded successfully.")

    y_test = label_encoder.transform(y_test_text)

    X_test_scaled = scaler.transform(X_test)

    print("Test feature scaling completed.")

    y_pred = model.predict(X_test_scaled)

    accuracy = accuracy_score(y_test, y_pred)

    macro_precision = precision_score(
        y_test,
        y_pred,
        average="macro",
        zero_division=0,
    )

    macro_recall = recall_score(
        y_test,
        y_pred,
        average="macro",
        zero_division=0,
    )

    macro_f1 = f1_score(
        y_test,
        y_pred,
        average="macro",
        zero_division=0,
    )

    weighted_f1 = f1_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0,
    )

    class_names = label_encoder.classes_.tolist()

    report = classification_report(
        y_test,
        y_pred,
        target_names=class_names,
        digits=4,
        zero_division=0,
    )

    print("\nOfficial test results:")
    print(f"Accuracy        : {accuracy:.4f}")
    print(f"Macro precision : {macro_precision:.4f}")
    print(f"Macro recall    : {macro_recall:.4f}")
    print(f"Macro F1-score  : {macro_f1:.4f}")
    print(f"Weighted F1     : {weighted_f1:.4f}")

    print("\nTest classification report:")
    print(report)

    confusion_matrix_path = save_confusion_matrix(
        y_true=y_test,
        y_pred=y_pred,
        class_names=class_names,
    )

    errors = print_misclassification_summary(
        y_true=y_test,
        y_pred=y_pred,
        class_names=class_names,
    )

    report_path = (
        REPORTS_ROOT
        / "test_classification_report.txt"
    )

    with report_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        file.write(
            "VIDEO-ONLY LOGISTIC REGRESSION TEST RESULTS\n"
        )
        file.write("=" * 70 + "\n\n")
        file.write(f"Accuracy: {accuracy:.4f}\n")
        file.write(
            f"Macro precision: {macro_precision:.4f}\n"
        )
        file.write(
            f"Macro recall: {macro_recall:.4f}\n"
        )
        file.write(
            f"Macro F1-score: {macro_f1:.4f}\n"
        )
        file.write(
            f"Weighted F1-score: {weighted_f1:.4f}\n\n"
        )
        file.write(report)

    metrics = {
        "model": "LogisticRegression",
        "split": "test",
        "accuracy": float(accuracy),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "test_samples": int(X_test.shape[0]),
        "feature_dimension": int(X_test.shape[1]),
        "number_of_classes": len(class_names),
        "classes": class_names,
    }

    metrics_path = REPORTS_ROOT / "test_metrics.json"

    with metrics_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metrics,
            file,
            indent=2,
        )

    errors_path = (
        REPORTS_ROOT
        / "test_misclassifications.json"
    )

    with errors_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            errors,
            file,
            indent=2,
        )

    print("\nSaved files:")
    print(f"  Report           : {report_path}")
    print(f"  Metrics          : {metrics_path}")
    print(f"  Misclassifications: {errors_path}")
    print(f"  Confusion matrix : {confusion_matrix_path}")

    print("\nPASS: Test evaluation completed successfully.")


if __name__ == "__main__":
    main()