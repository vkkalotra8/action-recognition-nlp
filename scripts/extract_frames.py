from pathlib import Path

import cv2
from tqdm import tqdm


# ==========================
# Project Paths
# ==========================
PROJECT_ROOT = Path(__file__).resolve().parents[1]

VIDEO_SPLIT_DIR = PROJECT_ROOT / "data" / "splits"
FRAME_OUTPUT_DIR = PROJECT_ROOT / "data" / "frames"

SPLITS = ["train", "val", "test"]
VIDEO_EXTENSIONS = {".avi", ".mp4", ".mov", ".mkv"}

FRAMES_PER_VIDEO = 16
FRAME_WIDTH = 224
FRAME_HEIGHT = 224

# If a requested frame cannot be read,
# try previous frames.
MAX_FALLBACK = 10


# ==========================
# Helper Functions
# ==========================
def get_video_files(directory: Path) -> list[Path]:
    """Return all supported videos recursively."""

    if not directory.exists():
        return []

    return sorted(
        path
        for path in directory.rglob("*")
        if path.is_file()
        and path.suffix.lower() in VIDEO_EXTENSIONS
    )


def calculate_frame_indices(
    total_frames: int,
    samples: int,
) -> list[int]:
    """Calculate evenly spaced frame indices."""

    if total_frames <= 0:
        return []

    if samples <= 1:
        return [0]

    if total_frames == 1:
        return [0] * samples

    return [
        round(i * (total_frames - 1) / (samples - 1))
        for i in range(samples)
    ]


def read_frame_with_fallback(
    capture: cv2.VideoCapture,
    requested_index: int,
):
    """
    Try reading the requested frame.
    If it fails, move backwards a few frames.
    """

    for offset in range(MAX_FALLBACK + 1):

        candidate = max(0, requested_index - offset)

        capture.set(
            cv2.CAP_PROP_POS_FRAMES,
            candidate,
        )

        success, frame = capture.read()

        if success and frame is not None:
            return True, frame, candidate

    return False, None, -1


# ==========================
# Frame Extraction
# ==========================
def extract_video_frames(
    video_path: Path,
    output_directory: Path,
) -> int:

    capture = cv2.VideoCapture(str(video_path))

    if not capture.isOpened():
        print(f"\n[ERROR] Cannot open {video_path.name}")
        return 0

    total_frames = int(
        capture.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    frame_indices = calculate_frame_indices(
        total_frames,
        FRAMES_PER_VIDEO,
    )

    if not frame_indices:
        capture.release()
        return 0

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    saved_count = 0

    for output_index, requested_index in enumerate(frame_indices):

        success, frame, actual_index = read_frame_with_fallback(
            capture,
            requested_index,
        )

        if not success:
            print(
                f"\n[WARNING] Could not read frame "
                f"{requested_index} from "
                f"{video_path.name}"
            )
            continue

        frame = cv2.resize(
            frame,
            (FRAME_WIDTH, FRAME_HEIGHT),
            interpolation=cv2.INTER_AREA,
        )

        output_path = (
            output_directory
            / f"frame_{output_index:03d}.jpg"
        )

        if cv2.imwrite(str(output_path), frame):
            saved_count += 1

        if actual_index != requested_index:
            print(
                f"\n[FALLBACK] "
                f"{video_path.name}: "
                f"{requested_index} -> {actual_index}"
            )

    capture.release()

    return saved_count


# ==========================
# Process Dataset Split
# ==========================
def process_split(split_name: str):

    split_video_dir = VIDEO_SPLIT_DIR / split_name

    if not split_video_dir.exists():
        print(f"[ERROR] Missing {split_video_dir}")
        return 0, 0

    videos = get_video_files(split_video_dir)

    print("\n" + "=" * 60)
    print(f"{split_name.upper()} SET")
    print("=" * 60)
    print(f"Videos: {len(videos)}")

    total_saved = 0
    incomplete = 0

    for video_path in tqdm(
        videos,
        desc=split_name,
        unit="video",
    ):

        class_name = video_path.parent.name
        video_name = video_path.stem

        output_directory = (
            FRAME_OUTPUT_DIR
            / split_name
            / class_name
            / video_name
        )

        saved = extract_video_frames(
            video_path,
            output_directory,
        )

        total_saved += saved

        if saved != FRAMES_PER_VIDEO:
            incomplete += 1

            print(
                f"\n[WARNING] "
                f"{video_path.name}: "
                f"{saved}/{FRAMES_PER_VIDEO}"
            )

    return total_saved, incomplete


# ==========================
# Main
# ==========================
def main():

    FRAME_OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    total_frames = 0
    total_incomplete = 0

    print("=" * 60)
    print("FRAME EXTRACTION")
    print("=" * 60)

    for split in SPLITS:

        saved, incomplete = process_split(split)

        total_frames += saved
        total_incomplete += incomplete

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    print(f"Saved frames : {total_frames}")
    print(f"Incomplete videos : {total_incomplete}")

    if total_incomplete == 0:
        print("\nSUCCESS: All videos produced 16 frames.")
    else:
        print(
            "\nWARNING: Some videos produced fewer "
            "than 16 frames."
        )


if __name__ == "__main__":
    main()