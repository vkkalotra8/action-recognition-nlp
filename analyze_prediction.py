from pathlib import Path
import json

report_path = Path(
    "results/predictions/"
    "difficult_drumming_test_2_prediction.json"
)

if not report_path.exists():
    raise FileNotFoundError(
        f"Report not found: {report_path}"
    )

with report_path.open(
    "r",
    encoding="utf-8",
) as file:
    report = json.load(file)

print("=" * 70)
print("DIFFICULT EXAMPLE ANALYSIS")
print("=" * 70)

print(
    f"Predicted class      : {report['predicted_class']}"
)

print(
    f"Top similarity       : {report['top_similarity']:.4f}"
)

print(
    f"Confidence level     : {report['confidence_level']}"
)

print(
    f"Separation level     : {report['separation_level']}"
)

print(
    f"Top-two score gap    : {report['score_gap']:.4f}"
)

print()
print("TOP PREDICTIONS")
print("-" * 70)

for item in report["top_predictions"]:
    print(
        f"{item['rank']}. "
        f"{item['class_name']:<16} "
        f"similarity={item['similarity']:.4f}"
    )

print()
print("Explanation")
print("-" * 70)
print(report["explanation"])

print()
print("Reliability note")
print("-" * 70)
print(report["reliability_note"])

print("=" * 70)
