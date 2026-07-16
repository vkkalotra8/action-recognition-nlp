from pathlib import Path

import cv2


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = PROJECT_ROOT / "data" / "raw_videos"
CLASSES_FILE = PROJECT_ROOT / "metadata" / "classes.txt"


def load_classes() -> list[str]:
    """Load class names from metadata/classes.txt."""
    if not CLASSES_FILE.exists():
        raise FileNotFoundError(f"Classes file not found: {CLASSES_FILE}")

    classes = [
        line.strip()
        for line in CLASSES_FILE.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    if not classes:
        raise ValueError("No classes were found in metadata/classes.txt")

    return classes


def check_video(video_path: Path) -> bool:
    """Return True when OpenCV can open and read at least one frame."""
    capture = cv2.VideoCapture(str(video_path))

    if not capture.isOpened():
        capture.release()
        return False

    success, frame = capture.read()
    capture.release()

    return success and frame is not None


def main() -> None:
    classes = load_classes()
    total_videos = 0
    unreadable_videos: list[Path] = []

    print(f"Dataset directory: {DATASET_DIR}\n")

    for class_name in classes:
        class_dir = DATASET_DIR / class_name

        if not class_dir.exists():
            print(f"[MISSING] {class_name}: folder not found")
            continue

        videos = sorted(
            path
            for path in class_dir.iterdir()
            if path.suffix.lower() in {".avi", ".mp4", ".mov", ".mkv"}
        )

        total_videos += len(videos)
        print(f"{class_name}: {len(videos)} videos")

        for video_path in videos:
            if not check_video(video_path):
                unreadable_videos.append(video_path)

    print("\nDataset check complete")
    print(f"Total videos: {total_videos}")
    print(f"Unreadable videos: {len(unreadable_videos)}")

    if unreadable_videos:
        print("\nUnreadable files:")
        for path in unreadable_videos:
            print(f"  - {path}")
    else:
        print("All videos were opened successfully.")


if __name__ == "__main__":
    main()