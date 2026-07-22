from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


PROJECT_ROOT = Path(__file__).resolve().parents[1]

EMBEDDINGS_ROOT = PROJECT_ROOT / "data" / "text_embeddings"
REPORTS_ROOT = PROJECT_ROOT / "results" / "reports"

CLASS_EMBEDDINGS_PATH = (
    EMBEDDINGS_ROOT / "class_embeddings.npy"
)

CLASS_NAMES_PATH = (
    EMBEDDINGS_ROOT / "class_names.json"
)

CSV_MATRIX_OUTPUT_PATH = (
    REPORTS_ROOT / "text_similarity_matrix.csv"
)

NUMPY_MATRIX_OUTPUT_PATH = (
    REPORTS_ROOT / "text_similarity_matrix.npy"
)

REPORT_OUTPUT_PATH = (
    REPORTS_ROOT / "text_similarity_report.txt"
)


def load_class_names(file_path: Path) -> list[str]:
    """Load and validate the ordered action class names."""

    if not file_path.exists():
        raise FileNotFoundError(
            f"Class-name file not found: {file_path}"
        )

    with file_path.open("r", encoding="utf-8") as file:
        class_names = json.load(file)

    if not isinstance(class_names, list):
        raise TypeError(
            "Class names must be stored as a JSON list."
        )

    if not class_names:
        raise ValueError(
            "The class-name list is empty."
        )

    for class_name in class_names:
        if not isinstance(class_name, str) or not class_name.strip():
            raise ValueError(
                "Every class name must be a non-empty string."
            )

    return class_names


def create_class_pairs(
    class_names: list[str],
    similarity_matrix: np.ndarray,
) -> list[tuple[str, str, float]]:
    """
    Create unique action-class pairs.

    Only the upper half of the matrix is used, so a pair such as
    Archery-Basketball is not repeated as Basketball-Archery.
    """

    pairs: list[tuple[str, str, float]] = []

    number_of_classes = len(class_names)

    for first_index in range(number_of_classes):
        for second_index in range(
            first_index + 1,
            number_of_classes,
        ):
            similarity = float(
                similarity_matrix[
                    first_index,
                    second_index,
                ]
            )

            pairs.append(
                (
                    class_names[first_index],
                    class_names[second_index],
                    similarity,
                )
            )

    return pairs


def save_csv_similarity_matrix(
    class_names: list[str],
    similarity_matrix: np.ndarray,
) -> None:
    """Save the similarity matrix as a human-readable CSV file."""

    REPORTS_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    with CSV_MATRIX_OUTPUT_PATH.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.writer(file)

        writer.writerow(
            ["Class"] + class_names
        )

        for class_name, row in zip(
            class_names,
            similarity_matrix,
            strict=True,
        ):
            writer.writerow(
                [class_name]
                + [
                    f"{value:.6f}"
                    for value in row
                ]
            )


def save_numpy_similarity_matrix(
    similarity_matrix: np.ndarray,
) -> None:
    """
    Save the similarity matrix as a NumPy file.

    This preserves the original floating-point precision and allows
    later scripts to load the matrix directly.
    """

    REPORTS_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.save(
        NUMPY_MATRIX_OUTPUT_PATH,
        similarity_matrix,
    )


def save_similarity_report(
    class_names: list[str],
    similarity_matrix: np.ndarray,
    sorted_pairs: list[tuple[str, str, float]],
) -> None:
    """Save the most and least similar class pairs in a text report."""

    most_similar_pairs = sorted_pairs[:10]

    least_similar_pairs = list(
        reversed(sorted_pairs[-10:])
    )

    lines: list[str] = []

    lines.append(
        "TEXT EMBEDDING SEMANTIC SIMILARITY REPORT"
    )
    lines.append("=" * 70)
    lines.append("")
    lines.append(
        "Model: all-MiniLM-L6-v2"
    )
    lines.append(
        f"Number of classes: {len(class_names)}"
    )
    lines.append(
        f"Similarity matrix shape: {similarity_matrix.shape}"
    )
    lines.append(
        "Similarity metric: cosine similarity"
    )

    lines.append("")
    lines.append("MOST SIMILAR CLASS PAIRS")
    lines.append("-" * 70)

    for rank, (
        first_class,
        second_class,
        similarity,
    ) in enumerate(
        most_similar_pairs,
        start=1,
    ):
        lines.append(
            f"{rank:>2}. "
            f"{first_class:<15} <-> "
            f"{second_class:<15} : "
            f"{similarity:.4f}"
        )

    lines.append("")
    lines.append("LEAST SIMILAR CLASS PAIRS")
    lines.append("-" * 70)

    for rank, (
        first_class,
        second_class,
        similarity,
    ) in enumerate(
        least_similar_pairs,
        start=1,
    ):
        lines.append(
            f"{rank:>2}. "
            f"{first_class:<15} <-> "
            f"{second_class:<15} : "
            f"{similarity:.4f}"
        )

    REPORT_OUTPUT_PATH.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def main() -> None:
    print("=" * 70)
    print("TEXT EMBEDDING SEMANTIC SIMILARITY ANALYSIS")
    print("=" * 70)

    if not CLASS_EMBEDDINGS_PATH.exists():
        raise FileNotFoundError(
            f"Class embeddings not found: "
            f"{CLASS_EMBEDDINGS_PATH}"
        )

    class_embeddings = np.load(
        CLASS_EMBEDDINGS_PATH
    )

    class_names = load_class_names(
        CLASS_NAMES_PATH
    )

    if class_embeddings.ndim != 2:
        raise ValueError(
            "Class embeddings must be a 2D NumPy array."
        )

    if class_embeddings.shape[0] != len(class_names):
        raise ValueError(
            "The number of class embeddings does not match "
            "the number of class names."
        )

    if not np.isfinite(class_embeddings).all():
        raise ValueError(
            "Class embeddings contain NaN or infinite values."
        )

    similarity_matrix = cosine_similarity(
        class_embeddings
    ).astype(np.float32)

    expected_shape = (
        len(class_names),
        len(class_names),
    )

    if similarity_matrix.shape != expected_shape:
        raise ValueError(
            f"Unexpected similarity matrix shape: "
            f"{similarity_matrix.shape}"
        )

    if not np.isfinite(similarity_matrix).all():
        raise ValueError(
            "Similarity matrix contains invalid values."
        )

    class_pairs = create_class_pairs(
        class_names,
        similarity_matrix,
    )

    sorted_pairs = sorted(
        class_pairs,
        key=lambda item: item[2],
        reverse=True,
    )

    save_csv_similarity_matrix(
        class_names,
        similarity_matrix,
    )

    save_numpy_similarity_matrix(
        similarity_matrix,
    )

    save_similarity_report(
        class_names,
        similarity_matrix,
        sorted_pairs,
    )

    print(
        f"Class embeddings shape : "
        f"{class_embeddings.shape}"
    )

    print(
        f"Similarity matrix shape: "
        f"{similarity_matrix.shape}"
    )

    print(
        f"Unique class pairs     : "
        f"{len(class_pairs)}"
    )

    print("\nTop 5 most similar class pairs:")

    for (
        first_class,
        second_class,
        similarity,
    ) in sorted_pairs[:5]:
        print(
            f"  {first_class:<15} <-> "
            f"{second_class:<15} : "
            f"{similarity:.4f}"
        )

    print("\nTop 5 least similar class pairs:")

    for (
        first_class,
        second_class,
        similarity,
    ) in reversed(sorted_pairs[-5:]):
        print(
            f"  {first_class:<15} <-> "
            f"{second_class:<15} : "
            f"{similarity:.4f}"
        )

    print("\nFiles saved:")

    print(
        f"  CSV matrix   : {CSV_MATRIX_OUTPUT_PATH}"
    )

    print(
        f"  NumPy matrix : {NUMPY_MATRIX_OUTPUT_PATH}"
    )

    print(
        f"  Text report  : {REPORT_OUTPUT_PATH}"
    )

    print("\n" + "=" * 70)
    print("PASS: SEMANTIC SIMILARITY ANALYSIS COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()