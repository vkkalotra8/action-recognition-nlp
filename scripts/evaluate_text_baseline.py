from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

FEATURES_ROOT = PROJECT_ROOT / "data" / "features"
TEXT_ROOT = PROJECT_ROOT / "data" / "text_embeddings"
MODELS_ROOT = PROJECT_ROOT / "models"
REPORTS_ROOT = PROJECT_ROOT / "results" / "reports"

CLASS_EMBEDDINGS_PATH = (
    TEXT_ROOT / "class_embeddings.npy"
)

CLASS_NAMES_PATH = (
    TEXT_ROOT / "class_names.json"
)

MODEL_PATH = (
    MODELS_ROOT / "text_only_logistic_regression.joblib"
)

SCALER_PATH = (
    MODELS_ROOT / "text_only_feature_scaler.joblib"
)

LABEL_ENCODER_PATH = (
    MODELS_ROOT / "text_only_label_encoder.joblib"
)

TEST_REPORT_PATH = (
    REPORTS_ROOT / "text_only_test_report.txt"
)

TEST_METRICS_PATH = (
    REPORTS_ROOT / "text_only_test_metrics.json"
)

MISCLASSIFICATIONS_PATH = (
    REPORTS_ROOT / "text_only_test_misclassifications.json"
)


def load_json(file_path: Path):
    """Load a JSON file."""

    if not file_path.exists():
        raise FileNotFoundError(
            f"Required JSON file not found: {file_path}"
        )

    with file_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def load_test_labels() -> np.ndarray:
    """Load the original test labels."""

    labels_path = (
        FEATURES_ROOT / "test_labels.npy"
    )

    if not labels_path.exists():
        raise FileNotFoundError(
            f"Test labels not found: {labels_path}"
        )

    return np.load(
        labels_path,
        allow_pickle=True,
    )


def build_text_features(
    labels: np.ndarray,
    class_names: list[str],
    class_embeddings: np.ndarray,
) -> np.ndarray:
    """
    Attach each sample's ground-truth class embedding.

    This is an oracle experiment and contains target leakage.
    """

    label_to_embedding = {
        class_name: class_embeddings[index]
        for index, class_name in enumerate(class_names)
    }

    features = np.stack(
        [
            label_to_embedding[str(label)]
            for label in labels
        ],
        axis=0,
    )

    return features.astype(np.float32)


def main() -> None:
    print("=" * 70)
    print("EVALUATING TEXT-ONLY ORACLE BASELINE")
    print("=" * 70)

    required_paths = (
        CLASS_EMBEDDINGS_PATH,
        CLASS_NAMES_PATH,
        MODEL_PATH,
        SCALER_PATH,
        LABEL_ENCODER_PATH,
    )

    for required_path in required_paths:
        if not required_path.exists():
            raise FileNotFoundError(
                f"Required file not found: {required_path}"
            )

    class_names = load_json(
        CLASS_NAMES_PATH
    )

    class_embeddings = np.load(
        CLASS_EMBEDDINGS_PATH
    ).astype(np.float32)

    test_labels_text = load_test_labels()

    if class_embeddings.shape != (
        len(class_names),
        384,
    ):
        raise ValueError(
            f"Unexpected class-embedding shape: "
            f"{class_embeddings.shape}"
        )

    test_features = build_text_features(
        test_labels_text,
        class_names,
        class_embeddings,
    )

    if test_features.shape != (
        test_labels_text.shape[0],
        384,
    ):
        raise ValueError(
            f"Unexpected test feature shape: "
            f"{test_features.shape}"
        )

    if not np.isfinite(test_features).all():
        raise ValueError(
            "Test text features contain NaN or infinite values."
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

    REPORTS_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    report_text = "\n".join(
        [
            "TEXT-ONLY ORACLE TEST REPORT",
            "=" * 70,
            "",
            "WARNING:",
            (
                "This experiment constructs each test feature using "
                "the sample's ground-truth class text embedding. It "
                "therefore contains target leakage and is not a valid "
                "unseen-video evaluation."
            ),
            "",
            f"Test feature shape: {test_features.shape}",
            f"Accuracy: {accuracy:.4f}",
            f"Macro precision: {macro_precision:.4f}",
            f"Macro recall: {macro_recall:.4f}",
            f"Macro F1-score: {macro_f1:.4f}",
            f"Weighted F1-score: {weighted_f1:.4f}",
            f"Misclassifications: {len(misclassifications)}",
            "",
            report,
        ]
    )

    TEST_REPORT_PATH.write_text(
        report_text,
        encoding="utf-8",
    )

    metrics = {
        "experiment": "text_only_true_label_oracle",
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
        "misclassification_count": int(
            len(misclassifications)
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

    with MISCLASSIFICATIONS_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            misclassifications,
            file,
            indent=4,
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

    print("\n" + "=" * 70)
    print("PASS: TEXT-ONLY ORACLE BASELINE EVALUATED")
    print("=" * 70)


if __name__ == "__main__":
    main()