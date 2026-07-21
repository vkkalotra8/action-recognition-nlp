import json
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent
FEATURES_ROOT = PROJECT_ROOT / "data" / "features"

SPLITS = ("train", "val", "test")


def load_split(split_name: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load all video feature files and labels for one split."""
    split_path = FEATURES_ROOT / split_name

    features = []
    labels = []
    video_ids = []

    class_folders = sorted(
        folder for folder in split_path.iterdir() if folder.is_dir()
    )

    for class_folder in class_folders:
        class_name = class_folder.name

        feature_files = sorted(class_folder.glob("*.npy"))

        for feature_file in feature_files:
            feature_vector = np.load(feature_file)

            if feature_vector.shape != (512,):
                raise ValueError(
                    f"Unexpected shape in {feature_file}: "
                    f"{feature_vector.shape}"
                )

            features.append(feature_vector)
            labels.append(class_name)
            video_ids.append(feature_file.stem)

    if not features:
        raise ValueError(
            f"No feature files found for split: {split_name}"
        )

    X = np.stack(features).astype(np.float32)
    y = np.array(labels)
    video_ids_array = np.array(video_ids)

    return X, y, video_ids_array.tolist()


def save_split(
    split_name: str,
    X: np.ndarray,
    y: np.ndarray,
    video_ids: list[str],
) -> None:
    """Save combined features, labels, and video IDs."""
    np.save(
        FEATURES_ROOT / f"{split_name}_features.npy",
        X,
    )

    np.save(
        FEATURES_ROOT / f"{split_name}_labels.npy",
        y,
    )

    with (
        FEATURES_ROOT / f"{split_name}_video_ids.json"
    ).open("w", encoding="utf-8") as file:
        json.dump(video_ids, file, indent=2)


def main() -> None:
    print("=" * 70)
    print("BUILDING COMBINED FEATURE DATASETS")
    print("=" * 70)

    all_classes = set()

    for split_name in SPLITS:
        X, y, video_ids = load_split(split_name)

        save_split(
            split_name=split_name,
            X=X,
            y=y,
            video_ids=video_ids,
        )

        all_classes.update(y.tolist())

        print(f"\nSplit: {split_name}")
        print(f"  X shape : {X.shape}")
        print(f"  y shape : {y.shape}")
        print(f"  Classes : {len(set(y.tolist()))}")

    class_names = sorted(all_classes)

    with (
        FEATURES_ROOT / "class_names.json"
    ).open("w", encoding="utf-8") as file:
        json.dump(class_names, file, indent=2)

    print("\nClasses:")

    for index, class_name in enumerate(class_names):
        print(f"  {index}: {class_name}")

    print("\nPASS: Combined feature datasets created successfully.")


if __name__ == "__main__":
    main()