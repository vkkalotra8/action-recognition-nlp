from pathlib import Path

import cv2


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FRAMES_DIR = PROJECT_ROOT / "data" / "frames"

SPLITS = ["train", "val", "test"]

EXPECTED_FRAMES_PER_VIDEO = 16
EXPECTED_WIDTH = 224
EXPECTED_HEIGHT = 224


def verify_image(image_path: Path) -> tuple[bool, bool]:
    """
    Check whether an image can be opened and whether
    its dimensions are correct.

    Returns:
        readable: True if OpenCV loads the image
        correct_size: True if image size is 224 x 224
    """

    image = cv2.imread(str(image_path))

    if image is None:
        return False, False

    height, width = image.shape[:2]

    correct_size = (
        width == EXPECTED_WIDTH
        and height == EXPECTED_HEIGHT
    )

    return True, correct_size


def main() -> None:
    print("=" * 70)
    print("FRAME VERIFICATION STARTED")
    print("=" * 70)
    print(f"Frames folder: {FRAMES_DIR}")
    print(f"Expected frames per video: {EXPECTED_FRAMES_PER_VIDEO}")
    print(
        f"Expected frame size: "
        f"{EXPECTED_WIDTH} x {EXPECTED_HEIGHT}"
    )

    if not FRAMES_DIR.exists():
        raise FileNotFoundError(
            f"Frames directory not found: {FRAMES_DIR}"
        )

    missing_split_folders: list[Path] = []
    empty_class_folders: list[Path] = []
    incomplete_video_folders: list[tuple[Path, int]] = []
    unreadable_images: list[Path] = []
    incorrect_size_images: list[tuple[Path, int, int]] = []

    total_classes = 0
    total_video_folders = 0
    total_images = 0

    for split_name in SPLITS:
        split_dir = FRAMES_DIR / split_name

        print()
        print("=" * 70)
        print(f"CHECKING SPLIT: {split_name.upper()}")
        print("=" * 70)

        if not split_dir.exists():
            print(f"[MISSING] {split_dir}")
            missing_split_folders.append(split_dir)
            continue

        class_folders = sorted(
            folder
            for folder in split_dir.iterdir()
            if folder.is_dir()
        )

        if not class_folders:
            print(f"[WARNING] No class folders found in {split_dir}")
            continue

        split_video_count = 0
        split_image_count = 0

        for class_folder in class_folders:
            total_classes += 1

            video_folders = sorted(
                folder
                for folder in class_folder.iterdir()
                if folder.is_dir()
            )

            if not video_folders:
                empty_class_folders.append(class_folder)

            class_video_count = 0
            class_image_count = 0

            for video_folder in video_folders:
                total_video_folders += 1
                split_video_count += 1
                class_video_count += 1

                image_files = sorted(
                    video_folder.glob("frame_*.jpg")
                )

                image_count = len(image_files)

                total_images += image_count
                split_image_count += image_count
                class_image_count += image_count

                if image_count != EXPECTED_FRAMES_PER_VIDEO:
                    incomplete_video_folders.append(
                        (video_folder, image_count)
                    )

                for image_path in image_files:
                    readable, correct_size = verify_image(image_path)

                    if not readable:
                        unreadable_images.append(image_path)
                        continue

                    if not correct_size:
                        image = cv2.imread(str(image_path))
                        height, width = image.shape[:2]

                        incorrect_size_images.append(
                            (image_path, width, height)
                        )

            print(
                f"{class_folder.name}: "
                f"{class_video_count} video folders, "
                f"{class_image_count} frames"
            )

        print()
        print(
            f"{split_name.upper()} TOTAL: "
            f"{split_video_count} video folders, "
            f"{split_image_count} frames"
        )

    expected_total_images = (
        total_video_folders
        * EXPECTED_FRAMES_PER_VIDEO
    )

    print()
    print("=" * 70)
    print("FINAL VERIFICATION SUMMARY")
    print("=" * 70)

    print(f"Class folders checked: {total_classes}")
    print(f"Video folders checked: {total_video_folders}")
    print(f"Expected total images: {expected_total_images}")
    print(f"Actual total images: {total_images}")
    print(f"Missing split folders: {len(missing_split_folders)}")
    print(f"Empty class folders: {len(empty_class_folders)}")
    print(
        f"Video folders without exactly "
        f"{EXPECTED_FRAMES_PER_VIDEO} frames: "
        f"{len(incomplete_video_folders)}"
    )
    print(f"Unreadable images: {len(unreadable_images)}")
    print(
        f"Images not sized "
        f"{EXPECTED_WIDTH} x {EXPECTED_HEIGHT}: "
        f"{len(incorrect_size_images)}"
    )

    if missing_split_folders:
        print()
        print("Missing split folders:")

        for folder in missing_split_folders:
            print(f"  - {folder}")

    if empty_class_folders:
        print()
        print("Empty class folders:")

        for folder in empty_class_folders:
            print(f"  - {folder}")

    if incomplete_video_folders:
        print()
        print("Incomplete video folders:")

        for folder, frame_count in incomplete_video_folders[:20]:
            print(
                f"  - {folder} "
                f"contains {frame_count} frames"
            )

        if len(incomplete_video_folders) > 20:
            remaining = len(incomplete_video_folders) - 20
            print(f"  ... and {remaining} more")

    if unreadable_images:
        print()
        print("Unreadable images:")

        for image_path in unreadable_images[:20]:
            print(f"  - {image_path}")

        if len(unreadable_images) > 20:
            remaining = len(unreadable_images) - 20
            print(f"  ... and {remaining} more")

    if incorrect_size_images:
        print()
        print("Images with incorrect dimensions:")

        for image_path, width, height in incorrect_size_images[:20]:
            print(
                f"  - {image_path}: "
                f"{width} x {height}"
            )

        if len(incorrect_size_images) > 20:
            remaining = len(incorrect_size_images) - 20
            print(f"  ... and {remaining} more")

    all_checks_passed = (
        not missing_split_folders
        and not empty_class_folders
        and not incomplete_video_folders
        and not unreadable_images
        and not incorrect_size_images
        and total_images == expected_total_images
    )

    print()
    print("=" * 70)

    if all_checks_passed:
        print("[SUCCESS] All extracted frames passed verification.")
    else:
        print(
            "[WARNING] Verification found one or more issues. "
            "Review the details above."
        )

    print("=" * 70)


if __name__ == "__main__":
    main()