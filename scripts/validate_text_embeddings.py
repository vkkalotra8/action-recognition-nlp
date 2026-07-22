from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]

EMBEDDINGS_ROOT = PROJECT_ROOT / "data" / "text_embeddings"

DESCRIPTION_EMBEDDINGS_PATH = (
    EMBEDDINGS_ROOT / "description_embeddings.npy"
)

CLASS_EMBEDDINGS_PATH = (
    EMBEDDINGS_ROOT / "class_embeddings.npy"
)

CLASS_NAMES_PATH = (
    EMBEDDINGS_ROOT / "class_names.json"
)

DESCRIPTION_METADATA_PATH = (
    EMBEDDINGS_ROOT / "description_metadata.json"
)

EMBEDDING_METADATA_PATH = (
    EMBEDDINGS_ROOT / "embedding_metadata.json"
)


def load_json(file_path: Path):
    if not file_path.exists():
        raise FileNotFoundError(
            f"Required file not found: {file_path}"
        )

    with file_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def main() -> None:
    print("=" * 70)
    print("TEXT EMBEDDING VALIDATION")
    print("=" * 70)

    description_embeddings = np.load(
        DESCRIPTION_EMBEDDINGS_PATH
    )

    class_embeddings = np.load(
        CLASS_EMBEDDINGS_PATH
    )

    class_names = load_json(
        CLASS_NAMES_PATH
    )

    description_metadata = load_json(
        DESCRIPTION_METADATA_PATH
    )

    embedding_metadata = load_json(
        EMBEDDING_METADATA_PATH
    )

    print(
        f"Description embeddings shape : "
        f"{description_embeddings.shape}"
    )

    print(
        f"Class embeddings shape       : "
        f"{class_embeddings.shape}"
    )

    print(
        f"Number of class names        : "
        f"{len(class_names)}"
    )

    print(
        f"Metadata records             : "
        f"{len(description_metadata)}"
    )

    # -----------------------------------------------------
    # Basic shape checks
    # -----------------------------------------------------

    if description_embeddings.shape != (45, 384):
        raise ValueError(
            "Expected description embeddings shape (45, 384)."
        )

    if class_embeddings.shape != (9, 384):
        raise ValueError(
            "Expected class embeddings shape (9, 384)."
        )

    if len(class_names) != 9:
        raise ValueError(
            "Expected exactly 9 class names."
        )

    if len(description_metadata) != 45:
        raise ValueError(
            "Expected exactly 45 metadata records."
        )

    # -----------------------------------------------------
    # Numerical checks
    # -----------------------------------------------------

    if not np.isfinite(description_embeddings).all():
        raise ValueError(
            "Description embeddings contain invalid values."
        )

    if not np.isfinite(class_embeddings).all():
        raise ValueError(
            "Class embeddings contain invalid values."
        )

    description_norms = np.linalg.norm(
        description_embeddings,
        axis=1,
    )

    class_norms = np.linalg.norm(
        class_embeddings,
        axis=1,
    )

    print(
        f"\nDescription norm range       : "
        f"{description_norms.min():.6f} "
        f"to {description_norms.max():.6f}"
    )

    print(
        f"Class norm range             : "
        f"{class_norms.min():.6f} "
        f"to {class_norms.max():.6f}"
    )

    if not np.allclose(
        description_norms,
        1.0,
        atol=1e-5,
    ):
        raise ValueError(
            "Some description embeddings are not normalized."
        )

    if not np.allclose(
        class_norms,
        1.0,
        atol=1e-5,
    ):
        raise ValueError(
            "Some class embeddings are not normalized."
        )

    # -----------------------------------------------------
    # Metadata consistency checks
    # -----------------------------------------------------

    metadata_class_names = [
        record["class_name"]
        for record in description_metadata
    ]

    class_counts = Counter(
        metadata_class_names
    )

    print("\nDescriptions per class:")

    for class_name in class_names:
        count = class_counts[class_name]

        print(
            f"  {class_name:<15} : {count}"
        )

        if count != 5:
            raise ValueError(
                f"{class_name} does not have exactly 5 descriptions."
            )

    descriptions = [
        record["description"].strip()
        for record in description_metadata
    ]

    normalized_descriptions = [
        description.lower()
        for description in descriptions
    ]

    duplicate_count = (
        len(normalized_descriptions)
        - len(set(normalized_descriptions))
    )

    print(
        f"\nDuplicate descriptions       : "
        f"{duplicate_count}"
    )

    if duplicate_count != 0:
        raise ValueError(
            "Duplicate descriptions were found."
        )

    expected_model = "all-MiniLM-L6-v2"

    if embedding_metadata["model_name"] != expected_model:
        raise ValueError(
            "Unexpected embedding model name."
        )

    if embedding_metadata["embedding_dimension"] != 384:
        raise ValueError(
            "Unexpected embedding dimension."
        )

    print("\n" + "=" * 70)
    print("PASS: ALL TEXT EMBEDDING CHECKS COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()