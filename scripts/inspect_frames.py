from collections import Counter
from pathlib import Path

import cv2


PROJECT_ROOT = Path(__file__).resolve().parent.parent
FRAMES_ROOT = PROJECT_ROOT / "data" / "frames"

SPLITS = ("train", "val", "test")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def get_image_files(video_folder: Path) -> list[Path]:
    """Return all supported image files inside one video folder."""
    return sorted(
        file_path
        for file_path in video_folder.iterdir()
        if file_path.is_file()
        and file_path.suffix.lower() in IMAGE_EXTENSIONS
    )


def inspect_split(split_name: str) -> dict:
    """Inspect one split and return its summary."""
    split_path = FRAMES_ROOT / split_name

    summary = {
        "classes": set(),
        "video_count": 0,
        "frame_count": 0,
        "unreadable_frames": [],
        "empty_video_folders": [],
        "frame_counts_per_video": [],
        "videos_per_class": Counter(),
    }

    print("\n" + "=" * 70)
    print(f"INSPECTING SPLIT: {split_name.upper()}")
    print("=" * 70)

    if not split_path.exists():
        print(f"ERROR: Split folder does not exist: {split_path}")
        return summary

    class_folders = sorted(
        folder for folder in split_path.iterdir() if folder.is_dir()
    )

    if not class_folders:
        print(f"ERROR: No class folders found in: {split_path}")
        return summary

    for class_folder in class_folders:
        class_name = class_folder.name
        summary["classes"].add(class_name)

        video_folders = sorted(
            folder for folder in class_folder.iterdir() if folder.is_dir()
        )

        summary["videos_per_class"][class_name] = len(video_folders)

        for video_folder in video_folders:
            summary["video_count"] += 1

            image_files = get_image_files(video_folder)
            frame_count = len(image_files)

            summary["frame_count"] += frame_count
            summary["frame_counts_per_video"].append(frame_count)

            if frame_count == 0:
                summary["empty_video_folders"].append(video_folder)

            for image_path in image_files:
                image = cv2.imread(str(image_path))

                if image is None:
                    summary["unreadable_frames"].append(image_path)

    print(f"Classes found       : {len(summary['classes'])}")
    print(f"Videos found        : {summary['video_count']}")
    print(f"Frames found        : {summary['frame_count']}")

    print("\nVideos per class:")

    for class_name, count in sorted(summary["videos_per_class"].items()):
        print(f"  {class_name:<20} {count}")

    frame_counts = summary["frame_counts_per_video"]

    if frame_counts:
        print("\nFrames per video:")
        print(f"  Minimum: {min(frame_counts)}")
        print(f"  Maximum: {max(frame_counts)}")
        print(f"  Average: {sum(frame_counts) / len(frame_counts):.2f}")

        frame_count_distribution = Counter(frame_counts)

        print("\nFrame-count distribution:")

        for count, number_of_videos in sorted(
            frame_count_distribution.items()
        ):
            print(
                f"  {count} frames: {number_of_videos} video folders"
            )

    print(
        f"\nEmpty video folders : "
        f"{len(summary['empty_video_folders'])}"
    )
    print(
        f"Unreadable frames   : "
        f"{len(summary['unreadable_frames'])}"
    )

    return summary


def compare_split_classes(all_summaries: dict) -> None:
    """Check whether all splits contain the same classes."""
    print("\n" + "=" * 70)
    print("CLASS CONSISTENCY CHECK")
    print("=" * 70)

    split_class_sets = {
        split_name: summary["classes"]
        for split_name, summary in all_summaries.items()
    }

    reference_split = SPLITS[0]
    reference_classes = split_class_sets[reference_split]

    all_match = True

    for split_name in SPLITS[1:]:
        current_classes = split_class_sets[split_name]

        missing_classes = reference_classes - current_classes
        extra_classes = current_classes - reference_classes

        if missing_classes or extra_classes:
            all_match = False

            print(f"\nDifference found in {split_name}:")

            if missing_classes:
                print(
                    "  Missing classes: "
                    + ", ".join(sorted(missing_classes))
                )

            if extra_classes:
                print(
                    "  Extra classes: "
                    + ", ".join(sorted(extra_classes))
                )

    if all_match:
        print("PASS: Train, validation, and test contain the same classes.")

    print("\nClasses:")

    for class_name in sorted(reference_classes):
        print(f"  - {class_name}")


def print_problem_details(all_summaries: dict) -> None:
    """Print the paths of any detected problems."""
    print("\n" + "=" * 70)
    print("PROBLEM DETAILS")
    print("=" * 70)

    problems_found = False

    for split_name, summary in all_summaries.items():
        if summary["empty_video_folders"]:
            problems_found = True
            print(f"\nEmpty video folders in {split_name}:")

            for folder in summary["empty_video_folders"]:
                print(f"  {folder}")

        if summary["unreadable_frames"]:
            problems_found = True
            print(f"\nUnreadable frames in {split_name}:")

            for frame_path in summary["unreadable_frames"]:
                print(f"  {frame_path}")

    if not problems_found:
        print("No empty video folders or unreadable frames were found.")


def print_final_result(all_summaries: dict) -> None:
    """Print the final dataset readiness result."""
    print("\n" + "=" * 70)
    print("FINAL RESULT")
    print("=" * 70)

    has_missing_split = any(
        not (FRAMES_ROOT / split_name).exists()
        for split_name in SPLITS
    )

    has_empty_folders = any(
        summary["empty_video_folders"]
        for summary in all_summaries.values()
    )

    has_unreadable_frames = any(
        summary["unreadable_frames"]
        for summary in all_summaries.values()
    )

    class_sets = [
        all_summaries[split_name]["classes"]
        for split_name in SPLITS
    ]

    classes_match = all(
        class_set == class_sets[0]
        for class_set in class_sets[1:]
    )

    if (
        not has_missing_split
        and not has_empty_folders
        and not has_unreadable_frames
        and classes_match
    ):
        print("PASS: The extracted-frame dataset is ready.")
        print("You can continue to CNN feature extraction.")
    else:
        print("FAIL: The dataset has one or more problems.")
        print("Review the messages above before continuing.")


def main() -> None:
    print("=" * 70)
    print("EXTRACTED FRAME DATASET INSPECTION")
    print("=" * 70)
    print(f"Frames root: {FRAMES_ROOT}")

    if not FRAMES_ROOT.exists():
        print(f"\nERROR: Frames folder not found: {FRAMES_ROOT}")
        return

    all_summaries = {}

    for split_name in SPLITS:
        all_summaries[split_name] = inspect_split(split_name)

    compare_split_classes(all_summaries)
    print_problem_details(all_summaries)
    print_final_result(all_summaries)


if __name__ == "__main__":
    main()