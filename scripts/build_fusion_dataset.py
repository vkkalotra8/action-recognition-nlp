from __future__ import annotations

import json
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]

FEATURES_ROOT = PROJECT_ROOT / "data" / "features"
TEXT_EMBEDDINGS_ROOT = PROJECT_ROOT / "data" / "text_embeddings"
OUTPUT_ROOT = PROJECT_ROOT / "data" / "fusion_features"

CLASS_EMBEDDINGS_PATH = (
    TEXT_EMBEDDINGS_ROOT / "class_embeddings.npy"
)

TEXT_CLASS_NAMES_PATH = (
    TEXT_EMBEDDINGS_ROOT / "class_names.json"
)

VISUAL_CLASS_NAMES_PATH = (
    FEATURES_ROOT / "class_names.json"
)

SPLITS = (
    "train",
    "val",
    "test",
)


def load_json(file_path: Path):
    """Load a JSON file."""

    if not file_path.exists():
        raise FileNotFoundError(
            f"Required JSON file not found: {file_path}"
        )

    with file_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def load_split(
    split_name: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Load visual features and labels for one dataset split."""

    features_path = (
        FEATURES_ROOT / f"{split_name}_features.npy"
    )

    labels_path = (
        FEATURES_ROOT / f"{split_name}_labels.npy"
    )

    if not features_path.exists():
        raise FileNotFoundError(
            f"Feature file not found: {features_path}"
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


def build_label_to_embedding(
    class_names: list[str],
    class_embeddings: np.ndarray,
) -> dict[str, np.ndarray]:
    """Map each action-class name to its text embedding."""

    if class_embeddings.shape[0] != len(class_names):
        raise ValueError(
            "The number of class embeddings does not match "
            "the number of class names."
        )

    return {
        class_name: class_embeddings[index]
        for index, class_name in enumerate(class_names)
    }


def build_fused_features(
    visual_features: np.ndarray,
    labels: np.ndarray,
    label_to_embedding: dict[str, np.ndarray],
) -> np.ndarray:
    """
    Concatenate each visual feature with the text embedding
    associated with its true class label.

    This is an oracle or label-informed fusion experiment.
    """

    if visual_features.ndim != 2:
        raise ValueError(
            "Visual features must be a 2D array."
        )

    if labels.ndim != 1:
        raise ValueError(
            "Labels must be a 1D array."
        )

    if visual_features.shape[0] != labels.shape[0]:
        raise ValueError(
            "Feature count does not match label count."
        )

    text_features: list[np.ndarray] = []

    for label in labels:
        label_name = str(label)

        if label_name not in label_to_embedding:
            raise KeyError(
                f"No text embedding found for label: "
                f"{label_name}"
            )

        text_features.append(
            label_to_embedding[label_name]
        )

    text_feature_matrix = np.stack(
        text_features,
        axis=0,
    ).astype(np.float32)

    fused_features = np.concatenate(
        [
            visual_features.astype(np.float32),
            text_feature_matrix,
        ],
        axis=1,
    ).astype(np.float32)

    return fused_features


def save_split(
    split_name: str,
    fused_features: np.ndarray,
    labels: np.ndarray,
) -> None:
    """Save one fused dataset split."""

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.save(
        OUTPUT_ROOT / f"{split_name}_fused_features.npy",
        fused_features,
    )

    np.save(
        OUTPUT_ROOT / f"{split_name}_labels.npy",
        labels,
    )


def main() -> None:
    print("=" * 70)
    print("BUILDING ORACLE VISION-LANGUAGE FUSION DATASET")
    print("=" * 70)

    if not CLASS_EMBEDDINGS_PATH.exists():
        raise FileNotFoundError(
            f"Class embeddings not found: "
            f"{CLASS_EMBEDDINGS_PATH}"
        )

    class_embeddings = np.load(
        CLASS_EMBEDDINGS_PATH
    )

    text_class_names = load_json(
        TEXT_CLASS_NAMES_PATH
    )

    visual_class_names = load_json(
        VISUAL_CLASS_NAMES_PATH
    )

    if visual_class_names != text_class_names:
        raise ValueError(
            "Visual and text class orders do not match."
        )

    if class_embeddings.ndim != 2:
        raise ValueError(
            "Class embeddings must be a 2D array."
        )

    if class_embeddings.shape != (
        len(text_class_names),
        384,
    ):
        raise ValueError(
            "Expected class embeddings shape "
            f"({len(text_class_names)}, 384), but received "
            f"{class_embeddings.shape}."
        )

    if not np.isfinite(class_embeddings).all():
        raise ValueError(
            "Class embeddings contain invalid values."
        )

    label_to_embedding = build_label_to_embedding(
        text_class_names,
        class_embeddings,
    )

    split_metadata: dict[str, dict] = {}

    for split_name in SPLITS:
        visual_features, labels = load_split(
            split_name
        )

        fused_features = build_fused_features(
            visual_features,
            labels,
            label_to_embedding,
        )

        expected_dimension = (
            visual_features.shape[1]
            + class_embeddings.shape[1]
        )

        if fused_features.shape != (
            visual_features.shape[0],
            expected_dimension,
        ):
            raise ValueError(
                f"Unexpected fused shape for {split_name}: "
                f"{fused_features.shape}"
            )

        if not np.isfinite(fused_features).all():
            raise ValueError(
                f"{split_name} fused features contain "
                "NaN or infinite values."
            )

        save_split(
            split_name,
            fused_features,
            labels,
        )

        split_metadata[split_name] = {
            "number_of_samples": int(
                fused_features.shape[0]
            ),
            "visual_dimension": int(
                visual_features.shape[1]
            ),
            "text_dimension": int(
                class_embeddings.shape[1]
            ),
            "fused_dimension": int(
                fused_features.shape[1]
            ),
            "feature_dtype": str(
                fused_features.dtype
            ),
        }

        print(
            f"{split_name:<5} "
            f"visual={visual_features.shape} "
            f"text=({visual_features.shape[0]}, "
            f"{class_embeddings.shape[1]}) "
            f"fused={fused_features.shape}"
        )

    metadata = {
        "fusion_type": "oracle_true_label_concatenation",
        "warning": (
            "This dataset uses the ground-truth class embedding "
            "for each sample and therefore contains label leakage. "
            "It must only be used as an oracle experiment."
        ),
        "class_names": text_class_names,
        "number_of_classes": len(text_class_names),
        "text_embedding_model": "all-MiniLM-L6-v2",
        "splits": split_metadata,
    }

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    with (
        OUTPUT_ROOT / "fusion_metadata.json"
    ).open("w", encoding="utf-8") as file:
        json.dump(
            metadata,
            file,
            indent=4,
        )

    with (
        OUTPUT_ROOT / "class_names.json"
    ).open("w", encoding="utf-8") as file:
        json.dump(
            text_class_names,
            file,
            indent=4,
        )

    print("\n" + "=" * 70)
    print("PASS: ORACLE FUSION DATASET CREATED")
    print("=" * 70)

    print(
        f"Files saved in: {OUTPUT_ROOT}"
    )

    print(
        "\nWARNING: These fused features contain "
        "ground-truth label information."
    )


if __name__ == "__main__":
    main()