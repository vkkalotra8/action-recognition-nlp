from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer


# ---------------------------------------------------------
# Project paths
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DESCRIPTIONS_PATH = (
    PROJECT_ROOT / "metadata" / "action_descriptions.json"
)

OUTPUT_ROOT = PROJECT_ROOT / "data" / "text_embeddings"

MODEL_NAME = "all-MiniLM-L6-v2"


# ---------------------------------------------------------
# Load and validate action descriptions
# ---------------------------------------------------------

def load_action_descriptions(
    file_path: Path,
) -> dict[str, list[str]]:
    """
    Load action descriptions from a JSON file.

    Expected format:
    {
        "ClassName": [
            "Description one.",
            "Description two."
        ]
    }
    """

    if not file_path.exists():
        raise FileNotFoundError(
            f"Action description file not found: {file_path}"
        )

    with file_path.open("r", encoding="utf-8") as file:
        action_descriptions = json.load(file)

    if not isinstance(action_descriptions, dict):
        raise TypeError(
            "The action description JSON must contain an object."
        )

    if not action_descriptions:
        raise ValueError(
            "The action description file is empty."
        )

    for class_name, descriptions in action_descriptions.items():
        if not isinstance(class_name, str) or not class_name.strip():
            raise ValueError(
                "Every class name must be a non-empty string."
            )

        if not isinstance(descriptions, list):
            raise TypeError(
                f"Descriptions for {class_name} must be a list."
            )

        if not descriptions:
            raise ValueError(
                f"No descriptions found for class: {class_name}"
            )

        for description in descriptions:
            if not isinstance(description, str):
                raise TypeError(
                    f"Every description for {class_name} must be text."
                )

            if not description.strip():
                raise ValueError(
                    f"An empty description was found for {class_name}."
                )

    return action_descriptions


# ---------------------------------------------------------
# Generate description and class embeddings
# ---------------------------------------------------------

def create_embeddings(
    action_descriptions: dict[str, list[str]],
    model: SentenceTransformer,
) -> tuple[
    list[str],
    list[str],
    list[str],
    np.ndarray,
    np.ndarray,
]:
    """
    Encode every action description.

    Returns:
        class_names:
            Ordered list of class names.

        description_class_names:
            Class name associated with every description.

        descriptions:
            Flat list containing every description.

        description_embeddings:
            Embedding for every individual description.

        class_embeddings:
            Average embedding for every action class.
    """

    class_names = list(action_descriptions.keys())

    description_class_names: list[str] = []
    descriptions: list[str] = []

    for class_name in class_names:
        for description in action_descriptions[class_name]:
            description_class_names.append(class_name)
            descriptions.append(description)

    print(f"Encoding {len(descriptions)} descriptions...")

    description_embeddings = model.encode(
        descriptions,
        convert_to_numpy=True,
        show_progress_bar=True,
        normalize_embeddings=True,
    ).astype(np.float32)

    class_embedding_list: list[np.ndarray] = []

    start_index = 0

    for class_name in class_names:
        number_of_descriptions = len(
            action_descriptions[class_name]
        )

        end_index = start_index + number_of_descriptions

        current_embeddings = description_embeddings[
            start_index:end_index
        ]

        mean_embedding = current_embeddings.mean(axis=0)

        # Normalize the averaged class embedding.
        norm = np.linalg.norm(mean_embedding)

        if norm == 0:
            raise ValueError(
                f"Zero-length class embedding produced for {class_name}."
            )

        mean_embedding = mean_embedding / norm

        class_embedding_list.append(
            mean_embedding.astype(np.float32)
        )

        start_index = end_index

    class_embeddings = np.stack(
        class_embedding_list,
        axis=0,
    )

    return (
        class_names,
        description_class_names,
        descriptions,
        description_embeddings,
        class_embeddings,
    )


# ---------------------------------------------------------
# Validate generated embeddings
# ---------------------------------------------------------

def validate_embeddings(
    class_names: list[str],
    descriptions: list[str],
    description_embeddings: np.ndarray,
    class_embeddings: np.ndarray,
) -> None:
    """
    Verify that the generated embedding arrays are valid.
    """

    expected_description_count = len(descriptions)
    expected_class_count = len(class_names)

    if description_embeddings.ndim != 2:
        raise ValueError(
            "Description embeddings must be a 2D array."
        )

    if class_embeddings.ndim != 2:
        raise ValueError(
            "Class embeddings must be a 2D array."
        )

    if description_embeddings.shape[0] != expected_description_count:
        raise ValueError(
            "Description embedding count does not match "
            "the number of descriptions."
        )

    if class_embeddings.shape[0] != expected_class_count:
        raise ValueError(
            "Class embedding count does not match "
            "the number of classes."
        )

    if description_embeddings.shape[1] != class_embeddings.shape[1]:
        raise ValueError(
            "Description and class embedding dimensions do not match."
        )

    if not np.isfinite(description_embeddings).all():
        raise ValueError(
            "Description embeddings contain NaN or infinite values."
        )

    if not np.isfinite(class_embeddings).all():
        raise ValueError(
            "Class embeddings contain NaN or infinite values."
        )


# ---------------------------------------------------------
# Save generated files
# ---------------------------------------------------------

def save_embeddings(
    class_names: list[str],
    description_class_names: list[str],
    descriptions: list[str],
    description_embeddings: np.ndarray,
    class_embeddings: np.ndarray,
) -> None:
    """
    Save embeddings and their supporting metadata.
    """

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.save(
        OUTPUT_ROOT / "description_embeddings.npy",
        description_embeddings,
    )

    np.save(
        OUTPUT_ROOT / "class_embeddings.npy",
        class_embeddings,
    )

    with (
        OUTPUT_ROOT / "class_names.json"
    ).open("w", encoding="utf-8") as file:
        json.dump(
            class_names,
            file,
            indent=4,
        )

    description_metadata = [
        {
            "description_index": index,
            "class_name": class_name,
            "description": description,
        }
        for index, (class_name, description) in enumerate(
            zip(
                description_class_names,
                descriptions,
                strict=True,
            )
        )
    ]

    with (
        OUTPUT_ROOT / "description_metadata.json"
    ).open("w", encoding="utf-8") as file:
        json.dump(
            description_metadata,
            file,
            indent=4,
        )

    embedding_metadata = {
        "model_name": MODEL_NAME,
        "number_of_classes": len(class_names),
        "number_of_descriptions": len(descriptions),
        "descriptions_per_class": {
            class_name: description_class_names.count(class_name)
            for class_name in class_names
        },
        "embedding_dimension": int(
            class_embeddings.shape[1]
        ),
        "description_embeddings_shape": list(
            description_embeddings.shape
        ),
        "class_embeddings_shape": list(
            class_embeddings.shape
        ),
        "embedding_dtype": str(
            class_embeddings.dtype
        ),
        "normalized_embeddings": True,
    }

    with (
        OUTPUT_ROOT / "embedding_metadata.json"
    ).open("w", encoding="utf-8") as file:
        json.dump(
            embedding_metadata,
            file,
            indent=4,
        )


# ---------------------------------------------------------
# Main program
# ---------------------------------------------------------

def main() -> None:
    print("=" * 70)
    print("NLP ACTION DESCRIPTION ENCODING")
    print("=" * 70)

    print(f"Description file : {DESCRIPTIONS_PATH}")
    print(f"Output directory : {OUTPUT_ROOT}")
    print(f"Sentence model   : {MODEL_NAME}")

    action_descriptions = load_action_descriptions(
        DESCRIPTIONS_PATH
    )

    total_descriptions = sum(
        len(descriptions)
        for descriptions in action_descriptions.values()
    )

    print(f"\nClasses loaded      : {len(action_descriptions)}")
    print(f"Descriptions loaded : {total_descriptions}")

    for class_name, descriptions in action_descriptions.items():
        print(
            f"  {class_name:<15} : "
            f"{len(descriptions)} descriptions"
        )

    print("\nLoading Sentence-BERT model...")

    model = SentenceTransformer(MODEL_NAME)

    (
        class_names,
        description_class_names,
        descriptions,
        description_embeddings,
        class_embeddings,
    ) = create_embeddings(
        action_descriptions,
        model,
    )

    validate_embeddings(
        class_names,
        descriptions,
        description_embeddings,
        class_embeddings,
    )

    save_embeddings(
        class_names,
        description_class_names,
        descriptions,
        description_embeddings,
        class_embeddings,
    )

    print("\n" + "=" * 70)
    print("ENCODING COMPLETED SUCCESSFULLY")
    print("=" * 70)

    print(
        "Description embeddings : "
        f"{description_embeddings.shape}"
    )

    print(
        "Class embeddings       : "
        f"{class_embeddings.shape}"
    )

    print(
        "Embedding data type     : "
        f"{class_embeddings.dtype}"
    )

    print(f"Files saved in         : {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()