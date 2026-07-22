from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]

REPORTS_ROOT = PROJECT_ROOT / "results" / "reports"
PLOTS_ROOT = PROJECT_ROOT / "results" / "plots"
EMBEDDINGS_ROOT = PROJECT_ROOT / "data" / "text_embeddings"

SIMILARITY_MATRIX_PATH = (
    REPORTS_ROOT / "text_similarity_matrix.npy"
)

CLASS_NAMES_PATH = (
    EMBEDDINGS_ROOT / "class_names.json"
)

OUTPUT_PATH = (
    PLOTS_ROOT / "text_similarity_heatmap.png"
)


def load_class_names(file_path: Path) -> list[str]:
    """Load the ordered action class names."""

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

    return class_names


def main() -> None:
    print("=" * 70)
    print("TEXT SIMILARITY HEATMAP")
    print("=" * 70)

    if not SIMILARITY_MATRIX_PATH.exists():
        raise FileNotFoundError(
            f"Similarity matrix not found: "
            f"{SIMILARITY_MATRIX_PATH}"
        )

    similarity_matrix = np.load(
        SIMILARITY_MATRIX_PATH
    )

    class_names = load_class_names(
        CLASS_NAMES_PATH
    )

    expected_shape = (
        len(class_names),
        len(class_names),
    )

    if similarity_matrix.shape != expected_shape:
        raise ValueError(
            f"Expected matrix shape {expected_shape}, "
            f"but received {similarity_matrix.shape}."
        )

    if not np.isfinite(similarity_matrix).all():
        raise ValueError(
            "Similarity matrix contains NaN or infinite values."
        )

    PLOTS_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure, axis = plt.subplots(
        figsize=(11, 9),
    )

    image = axis.imshow(
        similarity_matrix,
        vmin=0.0,
        vmax=1.0,
        aspect="auto",
    )

    color_bar = figure.colorbar(
        image,
        ax=axis,
    )

    color_bar.set_label(
        "Cosine Similarity",
        rotation=270,
        labelpad=20,
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
        class_names
    )

    axis.set_xlabel(
        "Action Class"
    )

    axis.set_ylabel(
        "Action Class"
    )

    axis.set_title(
        "Sentence-BERT Action-Class Semantic Similarity"
    )

    for row_index in range(len(class_names)):
        for column_index in range(len(class_names)):
            value = similarity_matrix[
                row_index,
                column_index,
            ]

            axis.text(
                column_index,
                row_index,
                f"{value:.2f}",
                ha="center",
                va="center",
            )

    figure.tight_layout()

    figure.savefig(
        OUTPUT_PATH,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(figure)

    print(
        f"Similarity matrix shape : "
        f"{similarity_matrix.shape}"
    )

    print(
        f"Number of class labels  : "
        f"{len(class_names)}"
    )

    print(
        f"Heatmap saved to        : "
        f"{OUTPUT_PATH}"
    )

    print("\n" + "=" * 70)
    print("PASS: TEXT SIMILARITY HEATMAP CREATED")
    print("=" * 70)


if __name__ == "__main__":
    main()