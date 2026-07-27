from __future__ import annotations

from typing import Any

import numpy as np


def rank_predictions(
    class_names: list[str],
    similarity_scores: np.ndarray,
    top_k: int,
) -> list[dict[str, Any]]:
    """
    Rank action classes by cosine similarity.

    Returns a reusable list of dictionaries for terminal output,
    JSON reports, and future Streamlit integration.
    """

    if not class_names:
        raise ValueError(
            "Class names cannot be empty."
        )

    if similarity_scores.ndim != 1:
        raise ValueError(
            "Similarity scores must be a 1D array."
        )

    if len(class_names) != similarity_scores.shape[0]:
        raise ValueError(
            "Class-name count does not match score count."
        )

    if not np.isfinite(
        similarity_scores
    ).all():
        raise ValueError(
            "Similarity scores contain NaN or infinite values."
        )

    if top_k < 1:
        raise ValueError(
            "top_k must be at least 1."
        )

    number_to_show = min(
        top_k,
        len(class_names),
    )

    ranked_indices = np.argsort(
        similarity_scores
    )[::-1]

    ranked_predictions: list[
        dict[str, Any]
    ] = []

    for rank, class_index in enumerate(
        ranked_indices[:number_to_show],
        start=1,
    ):
        class_index = int(
            class_index
        )

        ranked_predictions.append(
            {
                "rank": rank,
                "class_name": class_names[
                    class_index
                ],
                "similarity": float(
                    similarity_scores[
                        class_index
                    ]
                ),
            }
        )

    return ranked_predictions


def get_confidence_level(
    top_similarity: float,
) -> str:
    """
    Convert the top cosine similarity into a descriptive level.

    These levels are interpretation aids, not calibrated
    probabilities.
    """

    if top_similarity >= 0.70:
        return "High"

    if top_similarity >= 0.40:
        return "Moderate"

    if top_similarity >= 0.20:
        return "Low"

    return "Very low"


def get_separation_level(
    score_gap: float | None,
) -> str:
    """
    Interpret the difference between the top two similarity scores.

    Returns a special value when no second prediction is available.
    """

    if score_gap is None:
        return "Not available"

    if score_gap >= 0.30:
        return "Clear separation"

    if score_gap >= 0.15:
        return "Moderate separation"

    if score_gap >= 0.05:
        return "Small separation"

    return "Very small separation"


def select_representative_description(
    predicted_class: str,
    action_descriptions: dict[str, list[str]],
) -> str:
    """
    Select a representative natural-language description.

    The second description is preferred because it is usually
    more specific. The first description is used as a fallback.
    """

    if predicted_class not in action_descriptions:
        raise KeyError(
            "No descriptions found for predicted class: "
            f"{predicted_class}"
        )

    descriptions = action_descriptions[
        predicted_class
    ]

    if not isinstance(
        descriptions,
        list,
    ):
        raise TypeError(
            "Action descriptions must be stored as a list."
        )

    valid_descriptions = [
        description.strip()
        for description in descriptions
        if (
            isinstance(
                description,
                str,
            )
            and description.strip()
        )
    ]

    if not valid_descriptions:
        raise ValueError(
            "No valid descriptions found for class: "
            f"{predicted_class}"
        )

    if len(valid_descriptions) >= 2:
        return valid_descriptions[1]

    return valid_descriptions[0]


def generate_selection_explanation(
    predicted_class: str,
    top_similarity: float,
    second_class: str | None,
    second_similarity: float | None,
    score_gap: float | None,
) -> str:
    """
    Generate a faithful explanation based on ranking and score gap.
    """

    if (
        second_class is None
        or second_similarity is None
        or score_gap is None
    ):
        return (
            "The projected semantic representation of the video "
            f"was most similar to the {predicted_class} class "
            f"embedding, with a cosine similarity score of "
            f"{top_similarity:.4f}. No alternative class was "
            "included in the requested ranking."
        )

    if score_gap >= 0.30:
        return (
            "The projected semantic representation of the video "
            f"was most similar to the {predicted_class} class "
            "embedding. Its similarity score of "
            f"{top_similarity:.4f} was clearly higher than the "
            f"next-best class, {second_class}, which scored "
            f"{second_similarity:.4f}."
        )

    if score_gap >= 0.15:
        return (
            f"The model preferred {predicted_class}, although "
            f"{second_class} was also a meaningful alternative. "
            "The two similarity scores were "
            f"{top_similarity:.4f} and "
            f"{second_similarity:.4f}, respectively."
        )

    return (
        f"The model selected {predicted_class}, but the prediction "
        f"is relatively ambiguous because {second_class} received "
        "a nearby similarity score. The top two scores were "
        f"{top_similarity:.4f} and "
        f"{second_similarity:.4f}."
    )


def build_reliability_note(
    confidence_level: str,
    separation_level: str,
) -> str:
    """
    Build a reliability note without presenting similarity
    as probability.
    """

    base_note = (
        "These values are cosine similarity scores, not calibrated "
        "probabilities."
    )

    if separation_level == "Not available":
        return (
            f"{base_note} A score-gap interpretation is unavailable "
            "because only one prediction was requested."
        )

    cautious_result = (
        confidence_level in {
            "Low",
            "Very low",
        }
        or separation_level in {
            "Small separation",
            "Very small separation",
        }
    )

    if cautious_result:
        return (
            f"{base_note} The prediction should be interpreted "
            "cautiously because the semantic evidence is weak or "
            "the top alternatives are close."
        )

    return (
        f"{base_note} The top result has a reasonably distinct "
        "semantic advantage over the next candidate."
    )


def build_explanation_metadata(
    class_names: list[str],
    similarity_scores: np.ndarray,
    action_descriptions: dict[str, list[str]],
    top_k: int = 3,
) -> dict[str, Any]:
    """
    Build all structured facts required for one explanation.
    """

    ranked_predictions = rank_predictions(
        class_names=class_names,
        similarity_scores=similarity_scores,
        top_k=top_k,
    )

    if not ranked_predictions:
        raise ValueError(
            "No ranked predictions were produced."
        )

    top_prediction = ranked_predictions[0]

    predicted_class = str(
        top_prediction["class_name"]
    )

    top_similarity = float(
        top_prediction["similarity"]
    )

    second_class: str | None
    second_similarity: float | None
    score_gap: float | None

    if len(ranked_predictions) >= 2:
        second_prediction = ranked_predictions[1]

        second_class = str(
            second_prediction["class_name"]
        )

        second_similarity = float(
            second_prediction["similarity"]
        )

        score_gap = (
            top_similarity
            - second_similarity
        )
    else:
        second_class = None
        second_similarity = None
        score_gap = None

    confidence_level = get_confidence_level(
        top_similarity
    )

    separation_level = get_separation_level(
        score_gap
    )

    representative_description = (
        select_representative_description(
            predicted_class=predicted_class,
            action_descriptions=action_descriptions,
        )
    )

    explanation = generate_selection_explanation(
        predicted_class=predicted_class,
        top_similarity=top_similarity,
        second_class=second_class,
        second_similarity=second_similarity,
        score_gap=score_gap,
    )

    reliability_note = build_reliability_note(
        confidence_level=confidence_level,
        separation_level=separation_level,
    )

    return {
        "predicted_class": predicted_class,
        "top_similarity": top_similarity,
        "confidence_level": confidence_level,
        "second_class": second_class,
        "second_similarity": second_similarity,
        "score_gap": score_gap,
        "separation_level": separation_level,
        "representative_description": (
            representative_description
        ),
        "explanation": explanation,
        "reliability_note": reliability_note,
        "top_predictions": ranked_predictions,
        "score_type": "cosine_similarity",
        "calibrated_probability": False,
        "uses_ground_truth_label": False,
    }


def format_optional_float(
    value: float | None,
    decimal_places: int = 4,
) -> str:
    """
    Format an optional numerical value for a readable report.
    """

    if value is None:
        return "Not available"

    return f"{value:.{decimal_places}f}"


def format_explanation_report(
    explanation_metadata: dict[str, Any],
) -> str:
    """
    Convert structured explanation metadata into readable text.
    """

    required_fields = {
        "predicted_class",
        "top_similarity",
        "confidence_level",
        "score_gap",
        "separation_level",
        "representative_description",
        "explanation",
        "reliability_note",
        "top_predictions",
    }

    missing_fields = required_fields.difference(
        explanation_metadata
    )

    if missing_fields:
        raise KeyError(
            "Explanation metadata is missing fields: "
            f"{sorted(missing_fields)}"
        )

    score_gap = explanation_metadata[
        "score_gap"
    ]

    lines = [
        "PREDICTION",
        "-" * 70,
        (
            "Predicted action        : "
            f"{explanation_metadata['predicted_class']}"
        ),
        (
            "Semantic similarity     : "
            f"{explanation_metadata['top_similarity']:.4f}"
        ),
        (
            "Confidence level        : "
            f"{explanation_metadata['confidence_level']}"
        ),
        (
            "Score separation        : "
            f"{explanation_metadata['separation_level']}"
        ),
        (
            "Top-two score gap       : "
            f"{format_optional_float(score_gap)}"
        ),
        "",
        "Representative description:",
        str(
            explanation_metadata[
                "representative_description"
            ]
        ),
        "",
        "WHY THIS CLASS WAS SELECTED",
        "-" * 70,
        str(
            explanation_metadata[
                "explanation"
            ]
        ),
        "",
        "TOP PREDICTIONS",
        "-" * 70,
    ]

    for prediction in explanation_metadata[
        "top_predictions"
    ]:
        lines.append(
            f"{prediction['rank']}. "
            f"{prediction['class_name']:<15} "
            f"similarity="
            f"{prediction['similarity']:.4f}"
        )

    lines.extend(
        [
            "",
            "RELIABILITY NOTE",
            "-" * 70,
            str(
                explanation_metadata[
                    "reliability_note"
                ]
            ),
        ]
    )

    return "\n".join(lines)