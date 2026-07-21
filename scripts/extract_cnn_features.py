import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from tqdm import tqdm
from torchvision.models import ResNet18_Weights, resnet18


PROJECT_ROOT = Path(__file__).resolve().parent.parent
FRAMES_ROOT = PROJECT_ROOT / "data" / "frames"
FEATURES_ROOT = PROJECT_ROOT / "data" / "features"

VALID_SPLITS = ("train", "val", "test")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def parse_arguments() -> argparse.Namespace:
    """Read command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Extract one mean-pooled ResNet18 feature vector "
            "for every video folder."
        )
    )

    parser.add_argument(
        "--split",
        choices=VALID_SPLITS,
        required=True,
        help="Dataset split to process.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Number of frames processed together. Default: 16.",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional number of videos to process for testing.",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recreate feature files that already exist.",
    )

    return parser.parse_args()


def load_feature_extractor(
    device: torch.device,
) -> tuple[nn.Module, object]:
    """Load pretrained ResNet18 without its classifier."""
    weights = ResNet18_Weights.DEFAULT

    model = resnet18(weights=weights)

    # Replace the ImageNet classifier.
    # ResNet18 will now return 512 visual features.
    model.fc = nn.Identity()

    model = model.to(device)
    model.eval()

    preprocess = weights.transforms()

    return model, preprocess


def find_video_folders(split_path: Path) -> list[Path]:
    """Return all video folders inside one dataset split."""
    video_folders = []

    for class_folder in sorted(split_path.iterdir()):
        if not class_folder.is_dir():
            continue

        for video_folder in sorted(class_folder.iterdir()):
            if video_folder.is_dir():
                video_folders.append(video_folder)

    return video_folders


def find_frame_files(video_folder: Path) -> list[Path]:
    """Return all image frames inside one video folder."""
    return sorted(
        frame_path
        for frame_path in video_folder.iterdir()
        if (
            frame_path.is_file()
            and frame_path.suffix.lower() in IMAGE_EXTENSIONS
        )
    )


def load_frame_batch(
    frame_paths: list[Path],
    preprocess: object,
) -> torch.Tensor:
    """Load and preprocess a group of frames."""
    processed_frames = []

    for frame_path in frame_paths:
        with Image.open(frame_path) as image:
            rgb_image = image.convert("RGB")
            processed_frame = preprocess(rgb_image)

        processed_frames.append(processed_frame)

    return torch.stack(processed_frames)


def extract_video_feature(
    frame_paths: list[Path],
    model: nn.Module,
    preprocess: object,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    """
    Extract frame features and average them into one video feature.
    """
    frame_feature_batches = []

    for start_index in range(0, len(frame_paths), batch_size):
        batch_paths = frame_paths[
            start_index : start_index + batch_size
        ]

        input_batch = load_frame_batch(
            frame_paths=batch_paths,
            preprocess=preprocess,
        ).to(device)

        with torch.inference_mode():
            batch_features = model(input_batch)

        frame_feature_batches.append(batch_features.cpu())

    # Example shape:
    # 16 frames × 512 features
    all_frame_features = torch.cat(
        frame_feature_batches,
        dim=0,
    )

    # Mean pooling:
    # (16, 512) becomes (512,)
    video_feature = all_frame_features.mean(dim=0)

    return video_feature.numpy().astype(np.float32)


def build_output_path(
    video_folder: Path,
    split_name: str,
) -> Path:
    """Create the output feature path for one video."""
    class_name = video_folder.parent.name
    video_name = video_folder.name

    output_directory = (
        FEATURES_ROOT
        / split_name
        / class_name
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    return output_directory / f"{video_name}.npy"


def save_metadata(
    split_name: str,
    metadata: list[dict],
) -> None:
    """Save information about generated feature files."""
    metadata_path = (
        FEATURES_ROOT
        / split_name
        / "metadata.json"
    )

    with metadata_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metadata,
            file,
            indent=2,
        )


def main() -> None:
    args = parse_arguments()

    split_path = FRAMES_ROOT / args.split

    if not split_path.exists():
        raise FileNotFoundError(
            f"Split folder not found: {split_path}"
        )

    if args.batch_size <= 0:
        raise ValueError(
            "Batch size must be greater than zero."
        )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("=" * 70)
    print("VIDEO-LEVEL CNN FEATURE EXTRACTION")
    print("=" * 70)
    print(f"Split           : {args.split}")
    print(f"Frames folder   : {split_path}")
    print(f"Features folder : {FEATURES_ROOT / args.split}")
    print(f"Device          : {device}")
    print(f"Batch size      : {args.batch_size}")
    print(f"Overwrite       : {args.overwrite}")

    model, preprocess = load_feature_extractor(device)

    video_folders = find_video_folders(split_path)

    if args.limit is not None:
        video_folders = video_folders[: args.limit]

    print(f"Videos detected : {len(video_folders)}")

    created_count = 0
    skipped_count = 0
    failed_count = 0
    metadata = []

    progress_bar = tqdm(
        video_folders,
        desc=f"Extracting {args.split} features",
        unit="video",
    )

    for video_folder in progress_bar:
        class_name = video_folder.parent.name
        video_name = video_folder.name

        output_path = build_output_path(
            video_folder=video_folder,
            split_name=args.split,
        )

        if output_path.exists() and not args.overwrite:
            skipped_count += 1

            metadata.append(
                {
                    "split": args.split,
                    "class_name": class_name,
                    "video_name": video_name,
                    "frame_count": len(
                        find_frame_files(video_folder)
                    ),
                    "feature_path": str(
                        output_path.relative_to(PROJECT_ROOT)
                    ),
                    "status": "skipped_existing",
                }
            )

            continue

        frame_paths = find_frame_files(video_folder)

        if not frame_paths:
            failed_count += 1
            print(
                f"\nWARNING: No frames found in {video_folder}"
            )
            continue

        try:
            video_feature = extract_video_feature(
                frame_paths=frame_paths,
                model=model,
                preprocess=preprocess,
                device=device,
                batch_size=args.batch_size,
            )

            if video_feature.shape != (512,):
                raise ValueError(
                    "Unexpected feature shape: "
                    f"{video_feature.shape}"
                )

            np.save(output_path, video_feature)

            created_count += 1

            metadata.append(
                {
                    "split": args.split,
                    "class_name": class_name,
                    "video_name": video_name,
                    "frame_count": len(frame_paths),
                    "feature_shape": list(
                        video_feature.shape
                    ),
                    "feature_path": str(
                        output_path.relative_to(PROJECT_ROOT)
                    ),
                    "status": "created",
                }
            )

        except Exception as error:
            failed_count += 1

            print(
                f"\nERROR processing {video_folder}: "
                f"{error}"
            )

    save_metadata(
        split_name=args.split,
        metadata=metadata,
    )

    print("\n" + "=" * 70)
    print("EXTRACTION SUMMARY")
    print("=" * 70)
    print(f"Created feature files : {created_count}")
    print(f"Skipped existing files: {skipped_count}")
    print(f"Failed videos         : {failed_count}")
    print(f"Total examined        : {len(video_folders)}")

    if failed_count == 0:
        print("\nPASS: Feature extraction completed successfully.")
    else:
        print(
            "\nWARNING: Some videos failed. "
            "Review the errors above."
        )


if __name__ == "__main__":
    main()