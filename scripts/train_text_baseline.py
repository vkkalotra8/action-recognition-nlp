from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder, StandardScaler


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

MODEL_OUTPUT_PATH = (
    MODELS_ROOT / "text_only_logistic_regression.joblib"
)

SCALER_OUTPUT_PATH = (
    MODELS_ROOT / "text_only_feature_scaler.joblib"
)

LABEL_ENCODER_OUTPUT_PATH = (
    MODELS_ROOT / "text_only_label_encoder.joblib"
)

REPORT_OUTPUT_PATH = (
    REPORTS_ROOT / "text_only_validation_report.txt"
)


def load_json(file_path: Path):
    if not file_path.exists():
        raise FileNotFoundError(
            f"Required JSON file not found: {file_path}"
        )

    with file_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def load_labels(split_name: str) -> np.ndarray:
    labels_path = (
        FEATURES_ROOT / f"{split_name}_labels.npy"
    )

    if not labels_path.exists():
        raise FileNotFoundError(
            f"Label file not found: {labels_path}"
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
    print("TRAINING TEXT-ONLY BASELINE")
    print("=" * 70)

    class_embeddings = np.load(
        CLASS_EMBEDDINGS_PATH
    )

    class_names = load_json(
        CLASS_NAMES_PATH
    )

    train_labels_text = load_labels(
        "train"
    )

    val_labels_text = load_labels(
        "val"
    )

    train_features = build_text_features(
        train_labels_text,
        class_names,
        class_embeddings,
    )

    val_features = build_text_features(
        val_labels_text,
        class_names,
        class_embeddings,
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
        max_iter=2000,
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
            "TEXT-ONLY LOGISTIC REGRESSION BASELINE",
            "=" * 70,
            "",
            "WARNING:",
            (
                "This experiment uses the ground-truth class text "
                "embedding for every sample. It contains target leakage "
                "and is included only for ablation analysis."
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
    print("PASS: TEXT-ONLY BASELINE TRAINED")
    print("=" * 70)


if __name__ == "__main__":
    main()