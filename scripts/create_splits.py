from pathlib import Path
import re
import shutil

import pandas as pd
from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_VIDEO_DIR = PROJECT_ROOT / "data" / "raw_videos"
SPLIT_DIR = PROJECT_ROOT / "data" / "splits"
METADATA_DIR = PROJECT_ROOT / "metadata"
CLASSES_FILE = METADATA_DIR / "classes.txt"

RANDOM_STATE = 42
VIDEO_EXTENSIONS = {".avi", ".mp4", ".mov", ".mkv"}
GROUP_PATTERN = re.compile(r"_g(\d+)_")


def load_classes() -> list[str]:
    """Read class names from metadata/classes.txt."""
    if not CLASSES_FILE.exists():
        raise FileNotFoundError(f"Missing classes file: {CLASSES_FILE}")

    classes = [
        line.strip()
        for line in CLASSES_FILE.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    if not classes:
        raise ValueError("metadata/classes.txt is empty.")

    return classes


def collect_videos(classes: list[str]) -> pd.DataFrame:
    """Create a table containing every video and its class."""
    records: list[dict[str, str]] = []

    for class_name in classes:
        class_dir = RAW_VIDEO_DIR / class_name

        if not class_dir.exists():
            raise FileNotFoundError(f"Missing class directory: {class_dir}")

        videos = sorted(
            path
            for path in class_dir.iterdir()
            if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
        )

        if not videos:
            raise ValueError(f"No videos found for class: {class_name}")

        for video_path in videos:
            match = GROUP_PATTERN.search(video_path.stem)
            if match is None:
                raise ValueError(
                    "Could not determine the UCF101 recording group for "
                    f"{video_path.name}"
                )

            records.append(
                {
                    "video_path": str(video_path),
                    "filename": video_path.name,
                    "class_name": class_name,
                    "group_id": match.group(1),
                }
            )

    return pd.DataFrame(records)


def assign_splits(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Create an approximately 70/15/15 split at recording-group level.

    UCF101 clips that share a class and group ID (for example,
    v_Archery_g01_c01 and v_Archery_g01_c02) remain in one split.
    This prevents clips from the same recording group leaking into
    validation or test data.
    """
    split_dataframes: list[pd.DataFrame] = []

    for class_index, (class_name, class_df) in enumerate(
        dataframe.groupby("class_name", sort=True)
    ):
        group_ids = sorted(class_df["group_id"].unique())

        if len(group_ids) < 7:
            raise ValueError(
                f"{class_name} has only {len(group_ids)} recording groups; "
                "at least 7 are required for a train/validation/test split."
            )

        train_groups, temporary_groups = train_test_split(
            group_ids,
            test_size=0.30,
            random_state=RANDOM_STATE + class_index,
            shuffle=True,
        )
        val_groups, test_groups = train_test_split(
            temporary_groups,
            test_size=0.50,
            random_state=RANDOM_STATE + class_index,
            shuffle=True,
        )

        split_by_group = {
            group_id: "train" for group_id in train_groups
        }
        split_by_group.update({group_id: "val" for group_id in val_groups})
        split_by_group.update({group_id: "test" for group_id in test_groups})

        class_split_df = class_df.copy()
        class_split_df["split"] = class_split_df["group_id"].map(
            split_by_group
        )
        split_dataframes.append(class_split_df)

    return pd.concat(split_dataframes, ignore_index=True)


def clear_derived_split_data() -> None:
    """Remove old copied split videos before recreating them."""
    if SPLIT_DIR.exists():
        shutil.rmtree(SPLIT_DIR)


def copy_split_videos(split_dataframe: pd.DataFrame) -> None:
    """Copy videos into data/splits/<split>/<class_name>/."""
    for row in split_dataframe.itertuples(index=False):
        source_path = Path(row.video_path)
        destination_dir = SPLIT_DIR / row.split / row.class_name
        destination_dir.mkdir(parents=True, exist_ok=True)

        destination_path = destination_dir / row.filename
        shutil.copy2(source_path, destination_path)


def print_summary(split_dataframe: pd.DataFrame) -> None:
    """Print the number of videos in every class and split."""
    summary = pd.crosstab(
        split_dataframe["class_name"],
        split_dataframe["split"],
    )

    ordered_columns = [
        column
        for column in ["train", "val", "test"]
        if column in summary.columns
    ]

    print("\nSplit summary:")
    print(summary[ordered_columns])

    print("\nTotal videos per split:")
    print(split_dataframe["split"].value_counts())


def main() -> None:
    classes = load_classes()
    dataframe = collect_videos(classes)
    split_dataframe = assign_splits(dataframe)

    clear_derived_split_data()
    METADATA_DIR.mkdir(parents=True, exist_ok=True)
    split_dataframe.to_csv(
        METADATA_DIR / "video_splits.csv",
        index=False,
    )

    copy_split_videos(split_dataframe)
    print_summary(split_dataframe)

    print("\nCreated:")
    print(f"  {METADATA_DIR / 'video_splits.csv'}")
    print(f"  {SPLIT_DIR}")


if __name__ == "__main__":
    main()
