from __future__ import annotations

import csv
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

REPORTS_ROOT = PROJECT_ROOT / "results" / "reports"

VIDEO_METRICS_PATH = (
    REPORTS_ROOT / "test_metrics.json"
)

TEXT_ONLY_METRICS_PATH = (
    REPORTS_ROOT / "text_only_test_metrics.json"
)

ORACLE_FUSION_METRICS_PATH = (
    REPORTS_ROOT / "oracle_fusion_test_metrics.json"
)

SEMANTIC_PROJECTION_METRICS_PATH = (
    REPORTS_ROOT / "semantic_projection_test_metrics.json"
)

CSV_OUTPUT_PATH = (
    REPORTS_ROOT / "week4_ablation_results.csv"
)

JSON_OUTPUT_PATH = (
    REPORTS_ROOT / "week4_ablation_results.json"
)

MARKDOWN_OUTPUT_PATH = (
    REPORTS_ROOT / "week4_ablation_results.md"
)

TEXT_REPORT_OUTPUT_PATH = (
    REPORTS_ROOT / "week4_ablation_report.txt"
)


def load_json(file_path: Path) -> dict:
    """Load and validate one JSON metrics file."""

    if not file_path.exists():
        raise FileNotFoundError(
            f"Required metrics file not found: {file_path}"
        )

    with file_path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, dict):
        raise TypeError(
            f"Expected a JSON object in: {file_path}"
        )

    return data


def get_metric(
    metrics: dict,
    metric_name: str,
) -> float:
    """Read one required numerical metric."""

    if metric_name not in metrics:
        raise KeyError(
            f"Metric '{metric_name}' is missing."
        )

    return float(
        metrics[metric_name]
    )


def build_experiment_rows() -> list[dict]:
    """Load the four experiments and build comparison rows."""

    video_metrics = load_json(
        VIDEO_METRICS_PATH
    )

    text_only_metrics = load_json(
        TEXT_ONLY_METRICS_PATH
    )

    oracle_fusion_metrics = load_json(
        ORACLE_FUSION_METRICS_PATH
    )

    semantic_projection_metrics = load_json(
        SEMANTIC_PROJECTION_METRICS_PATH
    )

    rows = [
        {
            "experiment": "Video-only baseline",
            "feature_type": "Visual",
            "feature_dimension": int(
                video_metrics["feature_dimension"]
            ),
            "contains_label_leakage": False,
            "valid_unseen_video_inference": True,
            "test_accuracy": get_metric(
                video_metrics,
                "accuracy",
            ),
            "macro_precision": get_metric(
                video_metrics,
                "macro_precision",
            ),
            "macro_recall": get_metric(
                video_metrics,
                "macro_recall",
            ),
            "macro_f1": get_metric(
                video_metrics,
                "macro_f1",
            ),
            "weighted_f1": get_metric(
                video_metrics,
                "weighted_f1",
            ),
            "interpretation": (
                "Valid visual-only reference model."
            ),
        },
        {
            "experiment": "Text-only oracle",
            "feature_type": "Text",
            "feature_dimension": int(
                text_only_metrics["feature_dimension"]
            ),
            "contains_label_leakage": bool(
                text_only_metrics[
                    "contains_label_leakage"
                ]
            ),
            "valid_unseen_video_inference": False,
            "test_accuracy": get_metric(
                text_only_metrics,
                "accuracy",
            ),
            "macro_precision": get_metric(
                text_only_metrics,
                "macro_precision",
            ),
            "macro_recall": get_metric(
                text_only_metrics,
                "macro_recall",
            ),
            "macro_f1": get_metric(
                text_only_metrics,
                "macro_f1",
            ),
            "weighted_f1": get_metric(
                text_only_metrics,
                "weighted_f1",
            ),
            "interpretation": (
                "Oracle result because the true class selects "
                "the text embedding."
            ),
        },
        {
            "experiment": "Oracle concatenation",
            "feature_type": "Visual + true-label text",
            "feature_dimension": int(
                oracle_fusion_metrics["feature_dimension"]
            ),
            "contains_label_leakage": bool(
                oracle_fusion_metrics[
                    "contains_label_leakage"
                ]
            ),
            "valid_unseen_video_inference": False,
            "test_accuracy": get_metric(
                oracle_fusion_metrics,
                "accuracy",
            ),
            "macro_precision": get_metric(
                oracle_fusion_metrics,
                "macro_precision",
            ),
            "macro_recall": get_metric(
                oracle_fusion_metrics,
                "macro_recall",
            ),
            "macro_f1": get_metric(
                oracle_fusion_metrics,
                "macro_f1",
            ),
            "weighted_f1": get_metric(
                oracle_fusion_metrics,
                "weighted_f1",
            ),
            "interpretation": (
                "Upper-bound experiment with target leakage."
            ),
        },
        {
            "experiment": "Leakage-safe semantic projection",
            "feature_type": "Visual projected to text space",
            "feature_dimension": int(
                semantic_projection_metrics[
                    "semantic_dimension"
                ]
            ),
            "contains_label_leakage": bool(
                semantic_projection_metrics[
                    "contains_label_leakage"
                ]
            ),
            "valid_unseen_video_inference": True,
            "test_accuracy": get_metric(
                semantic_projection_metrics,
                "accuracy",
            ),
            "macro_precision": get_metric(
                semantic_projection_metrics,
                "macro_precision",
            ),
            "macro_recall": get_metric(
                semantic_projection_metrics,
                "macro_recall",
            ),
            "macro_f1": get_metric(
                semantic_projection_metrics,
                "macro_f1",
            ),
            "weighted_f1": get_metric(
                semantic_projection_metrics,
                "weighted_f1",
            ),
            "interpretation": (
                "Valid semantic prediction from visual features only."
            ),
        },
    ]

    return rows


def save_csv(rows: list[dict]) -> None:
    """Save the ablation results as CSV."""

    fieldnames = list(
        rows[0].keys()
    )

    with CSV_OUTPUT_PATH.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for row in rows:
            formatted_row = row.copy()

            for metric_name in (
                "test_accuracy",
                "macro_precision",
                "macro_recall",
                "macro_f1",
                "weighted_f1",
            ):
                formatted_row[metric_name] = (
                    f"{row[metric_name]:.4f}"
                )

            writer.writerow(
                formatted_row
            )


def save_json(rows: list[dict]) -> None:
    """Save the ablation results as JSON."""

    with JSON_OUTPUT_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            rows,
            file,
            indent=4,
        )


def save_markdown(rows: list[dict]) -> None:
    """Save a compact Markdown comparison table."""

    lines = [
        "| Experiment | Feature dimension | Leakage | Valid inference | Test accuracy | Macro F1 | Weighted F1 |",
        "|---|---:|---|---|---:|---:|---:|",
    ]

    for row in rows:
        leakage = (
            "Yes"
            if row["contains_label_leakage"]
            else "No"
        )

        valid_inference = (
            "Yes"
            if row["valid_unseen_video_inference"]
            else "No"
        )

        lines.append(
            "| "
            f"{row['experiment']} | "
            f"{row['feature_dimension']} | "
            f"{leakage} | "
            f"{valid_inference} | "
            f"{row['test_accuracy']:.4f} | "
            f"{row['macro_f1']:.4f} | "
            f"{row['weighted_f1']:.4f} |"
        )

    MARKDOWN_OUTPUT_PATH.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def save_text_report(rows: list[dict]) -> None:
    """Save a readable ablation report."""

    video_row = rows[0]
    safe_row = rows[3]

    accuracy_difference = (
        safe_row["test_accuracy"]
        - video_row["test_accuracy"]
    )

    macro_f1_difference = (
        safe_row["macro_f1"]
        - video_row["macro_f1"]
    )

    lines = [
        "WEEK 4 VISION-LANGUAGE ABLATION REPORT",
        "=" * 80,
        "",
        (
            f"{'Experiment':<38}"
            f"{'Dim':>7}"
            f"{'Leakage':>10}"
            f"{'Accuracy':>12}"
            f"{'Macro F1':>12}"
        ),
        "-" * 78,
    ]

    for row in rows:
        leakage = (
            "Yes"
            if row["contains_label_leakage"]
            else "No"
        )

        lines.append(
            f"{row['experiment']:<38}"
            f"{row['feature_dimension']:>7}"
            f"{leakage:>10}"
            f"{row['test_accuracy']:>12.4f}"
            f"{row['macro_f1']:>12.4f}"
        )

    lines.extend(
        [
            "",
            "VALID COMPARISON",
            "-" * 78,
            (
                "Video-only baseline test accuracy: "
                f"{video_row['test_accuracy']:.4f}"
            ),
            (
                "Leakage-safe projection test accuracy: "
                f"{safe_row['test_accuracy']:.4f}"
            ),
            (
                "Accuracy difference "
                "(projection minus video-only): "
                f"{accuracy_difference:+.4f}"
            ),
            (
                "Macro F1 difference "
                "(projection minus video-only): "
                f"{macro_f1_difference:+.4f}"
            ),
            "",
            "INTERPRETATION",
            "-" * 78,
            (
                "The text-only and concatenation experiments reached "
                "perfect scores because they used the ground-truth "
                "class embedding and therefore contain label leakage."
            ),
            (
                "The leakage-safe semantic projection is a valid "
                "unseen-video model because it receives only visual "
                "features during inference."
            ),
            (
                "The semantic projection performed below the video-only "
                "baseline, showing that the simple linear mapping lost "
                "some discriminative visual information while aligning "
                "the representation with the text-embedding space."
            ),
            (
                "The oracle scores must not be presented as real "
                "performance improvements."
            ),
        ]
    )

    TEXT_REPORT_OUTPUT_PATH.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def print_comparison(rows: list[dict]) -> None:
    """Display the comparison table in the terminal."""

    print("=" * 90)
    print("WEEK 4 MODEL COMPARISON")
    print("=" * 90)

    print(
        f"{'Experiment':<38}"
        f"{'Dim':>7}"
        f"{'Leakage':>10}"
        f"{'Accuracy':>11}"
        f"{'Macro F1':>11}"
    )

    print("-" * 88)

    for row in rows:
        leakage = (
            "Yes"
            if row["contains_label_leakage"]
            else "No"
        )

        print(
            f"{row['experiment']:<38}"
            f"{row['feature_dimension']:>7}"
            f"{leakage:>10}"
            f"{row['test_accuracy']:>11.4f}"
            f"{row['macro_f1']:>11.4f}"
        )


def main() -> None:
    REPORTS_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    rows = build_experiment_rows()

    save_csv(
        rows
    )

    save_json(
        rows
    )

    save_markdown(
        rows
    )

    save_text_report(
        rows
    )

    print_comparison(
        rows
    )

    print("\nFiles saved:")

    print(
        f"  CSV      : {CSV_OUTPUT_PATH}"
    )

    print(
        f"  JSON     : {JSON_OUTPUT_PATH}"
    )

    print(
        f"  Markdown : {MARKDOWN_OUTPUT_PATH}"
    )

    print(
        f"  Report   : {TEXT_REPORT_OUTPUT_PATH}"
    )

    print("\n" + "=" * 90)
    print("PASS: WEEK 4 ABLATION COMPARISON CREATED")
    print("=" * 90)


if __name__ == "__main__":
    main()