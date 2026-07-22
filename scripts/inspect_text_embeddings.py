from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


PROJECT_ROOT = Path(__file__).resolve().parents[1]

EMBEDDINGS_ROOT = PROJECT_ROOT / "data" / "text_embeddings"

CLASS_EMBEDDINGS_PATH = (
    EMBEDDINGS_ROOT / "class_embeddings.npy"
)

CLASS_NAMES_PATH = (
    EMBEDDINGS_ROOT / "class_names.json"
)

DESCRIPTION_METADATA_PATH = (
    EMBEDDINGS_ROOT / "description_metadata.json"
)


def load_json(file_path: Path):
    """Load a JSON file."""

    if not file_path.exists():
        raise FileNotFoundError(
            f"Required file not found: {file_path}"
        )

    with file_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def normalize_class_name(
    requested_name: str,
    class_names: list[str],
) -> str:
    """
    Match a user-provided class name without requiring exact casing.

    For example, 'playingguitar' matches 'PlayingGuitar'.
    """

    normalized_lookup = {
        class_name.lower(): class_name
        for class_name in class_names
    }

    requested_key = requested_name.strip().lower()

    if requested_key not in normalized_lookup:
        available_classes = ", ".join(class_names)

        raise ValueError(
            f"Unknown class: {requested_name}\n"
            f"Available classes: {available_classes}"
        )

    return normalized_lookup[requested_key]


def print_descriptions(
    class_name: str,
    description_metadata: list[dict],
) -> None:
    """Print all descriptions associated with one class."""

    descriptions = [
        record["description"]
        for record in description_metadata
        if record["class_name"] == class_name
    ]

    print("\nDescriptions:")

    for index, description in enumerate(
        descriptions,
        start=1,
    ):
        print(
            f"  {index}. {description}"
        )


def print_nearest_classes(
    selected_index: int,
    class_names: list[str],
    similarity_matrix: np.ndarray,
    top_k: int,
) -> None:
    """Print the most similar classes excluding the selected class."""

    similarities = similarity_matrix[
        selected_index
    ]

    ranked_indices = np.argsort(
        similarities
    )[::-1]

    ranked_indices = [
        index
        for index in ranked_indices
        if index != selected_index
    ]

    number_to_show = min(
        top_k,
        len(class_names) - 1,
    )

    print(
        f"\nTop {number_to_show} nearest classes:"
    )

    for rank, class_index in enumerate(
        ranked_indices[:number_to_show],
        start=1,
    ):
        print(
            f"  {rank}. "
            f"{class_names[class_index]:<15} "
            f"similarity={similarities[class_index]:.4f}"
        )


def inspect_class(
    requested_name: str,
    top_k: int,
) -> None:
    """Inspect one class embedding and its nearest neighbors."""

    class_embeddings = np.load(
        CLASS_EMBEDDINGS_PATH
    )

    class_names = load_json(
        CLASS_NAMES_PATH
    )

    description_metadata = load_json(
        DESCRIPTION_METADATA_PATH
    )

    if class_embeddings.ndim != 2:
        raise ValueError(
            "Class embeddings must be a 2D array."
        )

    if class_embeddings.shape[0] != len(class_names):
        raise ValueError(
            "Embedding count does not match class-name count."
        )

    class_name = normalize_class_name(
        requested_name,
        class_names,
    )

    selected_index = class_names.index(
        class_name
    )

    selected_embedding = class_embeddings[
        selected_index
    ]

    similarity_matrix = cosine_similarity(
        class_embeddings
    )

    print("=" * 70)
    print("TEXT EMBEDDING INSPECTION")
    print("=" * 70)

    print(
        f"Selected class       : {class_name}"
    )

    print(
        f"Class index          : {selected_index}"
    )

    print(
        f"Embedding shape      : {selected_embedding.shape}"
    )

    print(
        f"Embedding dtype      : {selected_embedding.dtype}"
    )

    print(
        f"Embedding norm       : "
        f"{np.linalg.norm(selected_embedding):.6f}"
    )

    print(
        f"Minimum value        : "
        f"{selected_embedding.min():.6f}"
    )

    print(
        f"Maximum value        : "
        f"{selected_embedding.max():.6f}"
    )

    print(
        f"Mean value           : "
        f"{selected_embedding.mean():.6f}"
    )

    print(
        f"Standard deviation   : "
        f"{selected_embedding.std():.6f}"
    )

    print(
        "First 10 values      : "
        f"{selected_embedding[:10]}"
    )

    print_descriptions(
        class_name,
        description_metadata,
    )

    print_nearest_classes(
        selected_index,
        class_names,
        similarity_matrix,
        top_k,
    )

    print("\n" + "=" * 70)
    print("PASS: EMBEDDING INSPECTION COMPLETED")
    print("=" * 70)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Inspect one action-class text embedding and "
            "its nearest semantic classes."
        )
    )

    parser.add_argument(
        "--class-name",
        default="Archery",
        help=(
            "Action class to inspect. "
            "Example: PlayingGuitar"
        ),
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help=(
            "Number of nearest classes to display."
        ),
    )

    arguments = parser.parse_args()

    if arguments.top_k < 1:
        parser.error(
            "--top-k must be at least 1."
        )

    return arguments


def main() -> None:
    arguments = parse_arguments()

    inspect_class(
        requested_name=arguments.class_name,
        top_k=arguments.top_k,
    )


if __name__ == "__main__":
    main()