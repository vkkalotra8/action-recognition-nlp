from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch import nn


PROJECT_ROOT = Path(__file__).resolve().parents[1]

VISUAL_ROOT = PROJECT_ROOT / "data" / "features"
TEXT_ROOT = PROJECT_ROOT / "data" / "text_embeddings"
MODELS_ROOT = PROJECT_ROOT / "models"
REPORTS_ROOT = PROJECT_ROOT / "results" / "reports"
PLOTS_ROOT = PROJECT_ROOT / "results" / "plots"

MODEL_PATH = (
    MODELS_ROOT / "semantic_projection_model.pth"
)

SCALER_PATH = (
    MODELS_ROOT / "semantic_projection_scaler.joblib"
)

CLASS_EMBEDDINGS_PATH = (
    TEXT_ROOT / "class_embeddings.npy"
)

CLASS_NAMES_PATH = (
    TEXT_ROOT / "class_names.json"
)

TEST_REPORT_PATH = (
    REPORTS_ROOT / "semantic_projection_test_report.txt"
)

TEST_METRICS_PATH = (
    REPORTS_ROOT / "semantic_projection_test_metrics.json"
)

MISCLASSIFICATIONS_PATH = (
    REPORTS_ROOT / "semantic_projection_test_misclassifications.json"
)

CONFUSION_MATRIX_PATH = (
    PLOTS_ROOT / "semantic_projection_test_confusion_matrix.png"
)

INPUT_DIMENSION = 512
OUTPUT_DIMENSION = 384


class SemanticProjectionModel(nn.Module):
    """Project visual features into the semantic text space."""

    def __init__(
        self,
        input_dimension: int = INPUT_DIMENSION,
        output_dimension: int = OUTPUT_DIMENSION,
    ) -> None:
        super().__init__()

        self.projection = nn.Linear(
            input_dimension,
            output_dimension,
        )

    def forward(
        self,
        features: torch.Tensor,
    ) -> torch.Tensor:
        projected = self.projection(
            features
        )

        return nn.functional.normalize(
            projected,
            p=2,
            dim=1,
        )


def load_json(file_path: Path):
    """Load a JSON file."""

    if not file_path.exists():
        raise FileNotFoundError(
            f"Required JSON file not found: {file_path}"
        )

    with file_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def load_test_data() -> tuple[np.ndarray, np.ndarray]:
    """Load visual test features and labels."""

    features_path = (
        VISUAL_ROOT / "test_features.npy"
    )

    labels_path = (
        VISUAL_ROOT / "test_labels.npy"
    )

    if not features_path.exists():
        raise FileNotFoundError(
            f"Test feature file not found: {features_path}"
        )

    if not labels_path.exists():
        raise FileNotFoundError(
            f"Test label file not found: {labels_path}"
        )

    features = np.load(
        features_path
    ).astype(np.float32)

    labels = np.load(
        labels_path,
        allow_pickle=True,
    )

    return features, labels


def save_confusion_matrix(
    true_indices: np.ndarray,
    predicted_indices: np.ndarray,
    class_names: list[str],
) -> None:
    """Create and save the semantic projection confusion matrix."""

    matrix = confusion_matrix(
        true_indices,
        predicted_indices,
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
        class_names,
    )

    axis.set_xlabel(
        "Predicted class"
    )

    axis.set_ylabel(
        "True class"
    )

    axis.set_title(
        "Leakage-Safe Semantic Projection Test Confusion Matrix"
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
    print("EVALUATING LEAKAGE-SAFE SEMANTIC PROJECTION")
    print("=" * 70)

    for required_path in (
        MODEL_PATH,
        SCALER_PATH,
        CLASS_EMBEDDINGS_PATH,
        CLASS_NAMES_PATH,
    ):
        if not required_path.exists():
            raise FileNotFoundError(
                f"Required file not found: {required_path}"
            )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Device: {device}")

    test_features, test_labels_text = load_test_data()

    if test_features.shape[1] != INPUT_DIMENSION:
        raise ValueError(
            f"Expected test feature dimension "
            f"{INPUT_DIMENSION}, received "
            f"{test_features.shape[1]}."
        )

    if test_features.shape[0] != test_labels_text.shape[0]:
        raise ValueError(
            "Test feature count does not match label count."
        )

    if not np.isfinite(test_features).all():
        raise ValueError(
            "Test features contain NaN or infinite values."
        )

    class_names = load_json(
        CLASS_NAMES_PATH
    )

    class_embeddings = np.load(
        CLASS_EMBEDDINGS_PATH
    ).astype(np.float32)

    if class_embeddings.shape != (
        len(class_names),
        OUTPUT_DIMENSION,
    ):
        raise ValueError(
            f"Unexpected class embedding shape: "
            f"{class_embeddings.shape}"
        )

    scaler = joblib.load(
        SCALER_PATH
    )

    test_features_scaled = scaler.transform(
        test_features
    ).astype(np.float32)

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=device,
        weights_only=False,
    )

    model = SemanticProjectionModel(
        input_dimension=checkpoint["input_dimension"],
        output_dimension=checkpoint["output_dimension"],
    ).to(device)

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    feature_tensor = torch.from_numpy(
        test_features_scaled
    ).to(device)

    class_embedding_tensor = torch.from_numpy(
        class_embeddings
    ).to(device)

    class_embedding_tensor = nn.functional.normalize(
        class_embedding_tensor,
        p=2,
        dim=1,
    )

    with torch.no_grad():
        projected_features = model(
            feature_tensor
        )

        similarity_scores = (
            projected_features
            @ class_embedding_tensor.T
        )

        predicted_indices = similarity_scores.argmax(
            dim=1
        ).cpu().numpy()

        top_scores = similarity_scores.max(
            dim=1
        ).values.cpu().numpy()

    class_to_index = {
        class_name: index
        for index, class_name in enumerate(class_names)
    }

    true_indices = np.array(
        [
            class_to_index[str(label)]
            for label in test_labels_text
        ],
        dtype=np.int64,
    )

    accuracy = accuracy_score(
        true_indices,
        predicted_indices,
    )

    macro_precision = precision_score(
        true_indices,
        predicted_indices,
        average="macro",
        zero_division=0,
    )

    macro_recall = recall_score(
        true_indices,
        predicted_indices,
        average="macro",
        zero_division=0,
    )

    macro_f1 = f1_score(
        true_indices,
        predicted_indices,
        average="macro",
        zero_division=0,
    )

    weighted_f1 = f1_score(
        true_indices,
        predicted_indices,
        average="weighted",
        zero_division=0,
    )

    report = classification_report(
        true_indices,
        predicted_indices,
        target_names=class_names,
        digits=4,
        zero_division=0,
    )

    incorrect_indices = np.where(
        predicted_indices != true_indices
    )[0]

    misclassifications = [
        {
            "sample_index": int(index),
            "true_class": str(
                test_labels_text[index]
            ),
            "predicted_class": class_names[
                int(predicted_indices[index])
            ],
            "predicted_similarity": float(
                top_scores[index]
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
            "LEAKAGE-SAFE SEMANTIC PROJECTION TEST REPORT",
            "=" * 70,
            "",
            "This model receives only visual features during inference.",
            (
                "Predictions are produced by comparing the projected "
                "visual vector with all class text embeddings."
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
        "experiment": "leakage_safe_semantic_projection",
        "contains_label_leakage": False,
        "test_samples": int(
            test_features.shape[0]
        ),
        "visual_feature_dimension": INPUT_DIMENSION,
        "semantic_dimension": OUTPUT_DIMENSION,
        "number_of_classes": len(class_names),
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
        "misclassification_count": len(
            misclassifications
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

    save_confusion_matrix(
        true_indices,
        predicted_indices,
        class_names,
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
    print("PASS: LEAKAGE-SAFE SEMANTIC PROJECTION EVALUATED")
    print("=" * 70)


if __name__ == "__main__":
    main()