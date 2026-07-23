from __future__ import annotations

import json
import re
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

REPORTS_ROOT = PROJECT_ROOT / "results" / "reports"

VIDEO_REPORT_PATH = (
    REPORTS_ROOT / "test_classification_report.txt"
)

PROJECTION_REPORT_PATH = (
    REPORTS_ROOT / "semantic_projection_test_report.txt"
)

OUTPUT_JSON_PATH = (
    REPORTS_ROOT / "week4_classwise_comparison.json"
)

OUTPUT_TEXT_PATH = (
    REPORTS_ROOT / "week4_classwise_comparison.txt"
)


def parse_classification_report(
    file_path: Path,
) -> dict[str, dict[str, float]]:
    """
    Parse class-wise precision, recall, F1-score, and support
    from a saved sklearn classification report.
    """

    if not file_path.exists():
        raise FileNotFoundError(
            f"Report file not found: {file_path}"
        )

    lines = file_path.read_text(
        encoding="utf-8"
    ).splitlines()

    class_metrics: dict[
        str,
        dict[str, float],
    ] = {}

    pattern = re.compile(
        r"^\s*"
        r"([A-Za-z][A-Za-z0-9]*)"
        r"\s+"
        r"([0-9.]+)"
        r"\s+"
        r"([0-9.]+)"
        r"\s+"
        r"([0-9.]+)"
        r"\s+"
        r"([0-9]+)"
        r"\s*$"
    )

    excluded_names = {
        "accuracy",
        "macro",
        "weighted",
    }

    for line in lines:
        match = pattern.match(line)

        if match is None:
            continue

        class_name = match.group(1)

        if class_name.lower() in excluded_names:
            continue

        class_metrics[class_name] = {
            "precision": float(
                match.group(2)
            ),
            "recall": float(
                match.group(3)
            ),
            "f1": float(
                match.group(4)
            ),
            "support": int(
                match.group(5)
            ),
        }

    if not class_metrics:
        raise ValueError(
            f"No class-wise metrics found in {file_path}"
        )

    return class_metrics


def main() -> None:
    print("=" * 78)
    print("WEEK 4 CLASS-WISE RESULT ANALYSIS")
    print("=" * 78)

    video_metrics = parse_classification_report(
        VIDEO_REPORT_PATH
    )

    projection_metrics = parse_classification_report(
        PROJECTION_REPORT_PATH
    )

    video_classes = set(
        video_metrics
    )

    projection_classes = set(
        projection_metrics
    )

    if video_classes != projection_classes:
        raise ValueError(
            "Video and projection reports contain "
            "different class sets."
        )

    rows: list[dict] = []

    for class_name in sorted(video_classes):
        video_f1 = video_metrics[
            class_name
        ]["f1"]

        projection_f1 = projection_metrics[
            class_name
        ]["f1"]

        difference = (
            projection_f1 - video_f1
        )

        if difference > 0.0001:
            outcome = "Improved"
        elif difference < -0.0001:
            outcome = "Worse"
        else:
            outcome = "Unchanged"

        rows.append(
            {
                "class_name": class_name,
                "video_precision": (
                    video_metrics[
                        class_name
                    ]["precision"]
                ),
                "video_recall": (
                    video_metrics[
                        class_name
                    ]["recall"]
                ),
                "video_f1": video_f1,
                "projection_precision": (
                    projection_metrics[
                        class_name
                    ]["precision"]
                ),
                "projection_recall": (
                    projection_metrics[
                        class_name
                    ]["recall"]
                ),
                "projection_f1": (
                    projection_f1
                ),
                "f1_difference": (
                    difference
                ),
                "outcome": outcome,
                "support": (
                    video_metrics[
                        class_name
                    ]["support"]
                ),
            }
        )

    improved = [
        row
        for row in rows
        if row["outcome"] == "Improved"
    ]

    worse = [
        row
        for row in rows
        if row["outcome"] == "Worse"
    ]

    unchanged = [
        row
        for row in rows
        if row["outcome"] == "Unchanged"
    ]

    lines = [
        "WEEK 4 CLASS-WISE F1 COMPARISON",
        "=" * 78,
        "",
        (
            f"{'Class':<18}"
            f"{'Video F1':>12}"
            f"{'Projection F1':>16}"
            f"{'Difference':>13}"
            f"{'Outcome':>12}"
        ),
        "-" * 78,
    ]

    for row in rows:
        lines.append(
            f"{row['class_name']:<18}"
            f"{row['video_f1']:>12.4f}"
            f"{row['projection_f1']:>16.4f}"
            f"{row['f1_difference']:>+13.4f}"
            f"{row['outcome']:>12}"
        )

    lines.extend(
        [
            "",
            "SUMMARY",
            "-" * 78,
            (
                f"Improved classes : "
                f"{len(improved)}"
            ),
            (
                f"Worse classes    : "
                f"{len(worse)}"
            ),
            (
                f"Unchanged classes: "
                f"{len(unchanged)}"
            ),
            "",
            "Best class-wise changes:",
        ]
    )

    for row in sorted(
        rows,
        key=lambda item: item[
            "f1_difference"
        ],
        reverse=True,
    )[:3]:
        lines.append(
            f"  {row['class_name']:<15} "
            f"{row['f1_difference']:+.4f}"
        )

    lines.append("")
    lines.append(
        "Largest decreases:"
    )

    for row in sorted(
        rows,
        key=lambda item: item[
            "f1_difference"
        ],
    )[:3]:
        lines.append(
            f"  {row['class_name']:<15} "
            f"{row['f1_difference']:+.4f}"
        )

    if not improved:
        lines.extend(
            [
                "",
                "INTERPRETATION",
                "-" * 78,
                (
                    "No action class improved under the leakage-safe "
                    "semantic projection."
                ),
                (
                    "Bowling and PlayingGuitar were unchanged, while "
                    "the remaining classes showed lower F1-scores than "
                    "the video-only baseline."
                ),
                (
                    "The largest decreases occurred for Basketball, "
                    "Archery, and Typing."
                ),
            ]
        )

    OUTPUT_TEXT_PATH.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    with OUTPUT_JSON_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            rows,
            file,
            indent=4,
        )

    print(
        f"{'Class':<18}"
        f"{'Video F1':>12}"
        f"{'Projection F1':>16}"
        f"{'Difference':>13}"
        f"{'Outcome':>12}"
    )

    print("-" * 78)

    for row in rows:
        print(
            f"{row['class_name']:<18}"
            f"{row['video_f1']:>12.4f}"
            f"{row['projection_f1']:>16.4f}"
            f"{row['f1_difference']:>+13.4f}"
            f"{row['outcome']:>12}"
        )

    print("\nSummary:")

    print(
        f"  Improved : {len(improved)}"
    )

    print(
        f"  Worse    : {len(worse)}"
    )

    print(
        f"  Unchanged: {len(unchanged)}"
    )

    print("\nFiles saved:")

    print(
        f"  JSON   : {OUTPUT_JSON_PATH}"
    )

    print(
        f"  Report : {OUTPUT_TEXT_PATH}"
    )

    print("\n" + "=" * 78)
    print("PASS: CLASS-WISE ANALYSIS COMPLETED")
    print("=" * 78)


if __name__ == "__main__":
    main()