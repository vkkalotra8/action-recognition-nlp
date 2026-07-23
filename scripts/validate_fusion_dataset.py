from __future__ import annotations

import json
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]

VISUAL_ROOT = PROJECT_ROOT / "data" / "features"
TEXT_ROOT = PROJECT_ROOT / "data" / "text_embeddings"
FUSION_ROOT = PROJECT_ROOT / "data" / "fusion_features"

SPLITS = ("train", "val", "test")

VISUAL_DIMENSION = 512
TEXT_DIMENSION = 384
FUSED_DIMENSION = VISUAL_DIMENSION + TEXT_DIMENSION


def load_json(file_path: Path):
    if not file_path.exists():
        raise FileNotFoundError(
            f"Required JSON file not found: {file_path}"
        )

    with file_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def main() -> None:
    print("=" * 70)
    print("ORACLE FUSION DATASET VALIDATION")
    print("=" * 70)

    class_embeddings = np.load(
        TEXT_ROOT / "class_embeddings.npy"
    )

    text_class_names = load_json(
        TEXT_ROOT / "class_names.json"
    )

    fusion_class_names = load_json(
        FUSION_ROOT / "class_names.json"
    )

    fusion_metadata = load_json(
        FUSION_ROOT / "fusion_metadata.json"
    )

    if text_class_names != fusion_class_names:
        raise ValueError(
            "Text and fusion class orders do not match."
        )

    if class_embeddings.shape != (
        len(text_class_names),
        TEXT_DIMENSION,
    ):
        raise ValueError(
            f"Unexpected text embedding shape: "
            f"{class_embeddings.shape}"
        )

    label_to_embedding = {
        class_name: class_embeddings[index]
        for index, class_name in enumerate(text_class_names)
    }

    for split_name in SPLITS:
        visual_features = np.load(
            VISUAL_ROOT / f"{split_name}_features.npy"
        )

        original_labels = np.load(
            VISUAL_ROOT / f"{split_name}_labels.npy",
            allow_pickle=True,
        )

        fused_features = np.load(
            FUSION_ROOT
            / f"{split_name}_fused_features.npy"
        )

        fused_labels = np.load(
            FUSION_ROOT / f"{split_name}_labels.npy",
            allow_pickle=True,
        )

        expected_shape = (
            visual_features.shape[0],
            FUSED_DIMENSION,
        )

        if fused_features.shape != expected_shape:
            raise ValueError(
                f"{split_name}: expected fused shape "
                f"{expected_shape}, got {fused_features.shape}"
            )

        if fused_labels.shape != original_labels.shape:
            raise ValueError(
                f"{split_name}: label shapes do not match."
            )

        if not np.array_equal(
            fused_labels,
            original_labels,
        ):
            raise ValueError(
                f"{split_name}: fused labels differ "
                "from original labels."
            )

        if not np.isfinite(fused_features).all():
            raise ValueError(
                f"{split_name}: fused features contain "
                "NaN or infinite values."
            )

        visual_part = fused_features[
            :,
            :VISUAL_DIMENSION,
        ]

        text_part = fused_features[
            :,
            VISUAL_DIMENSION:,
        ]

        if not np.allclose(
            visual_part,
            visual_features,
            atol=1e-6,
        ):
            raise ValueError(
                f"{split_name}: visual section does not "
                "match the original visual features."
            )

        expected_text_part = np.stack(
            [
                label_to_embedding[str(label)]
                for label in original_labels
            ],
            axis=0,
        ).astype(np.float32)

        if not np.allclose(
            text_part,
            expected_text_part,
            atol=1e-6,
        ):
            raise ValueError(
                f"{split_name}: text section does not match "
                "the expected label embeddings."
            )

        split_metadata = fusion_metadata[
            "splits"
        ][split_name]

        if split_metadata["number_of_samples"] != (
            fused_features.shape[0]
        ):
            raise ValueError(
                f"{split_name}: metadata sample count mismatch."
            )

        if split_metadata["fused_dimension"] != (
            FUSED_DIMENSION
        ):
            raise ValueError(
                f"{split_name}: metadata dimension mismatch."
            )

        print(
            f"{split_name:<5} "
            f"samples={fused_features.shape[0]} "
            f"visual={visual_part.shape[1]} "
            f"text={text_part.shape[1]} "
            f"fused={fused_features.shape[1]}"
        )

    print(
        f"\nFusion type: "
        f"{fusion_metadata['fusion_type']}"
    )

    print(
        "Label leakage warning recorded: "
        f"{bool(fusion_metadata.get('warning'))}"
    )

    print("\n" + "=" * 70)
    print("PASS: ORACLE FUSION DATASET IS VALID")
    print("=" * 70)


if __name__ == "__main__":
    main()