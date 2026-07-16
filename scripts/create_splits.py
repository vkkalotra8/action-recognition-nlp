from pathlib import Path
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
            records.append(
                {
                    "video_path": str(video_path),
                    "filename": video_path.name,
                    "class_name": class_name,
                }
            )

    return pd.DataFrame(records)


def assign_splits(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Create a 70/15/15 stratified split.

    First: 70% train, 30% temporary.
    Second: divide temporary data equally into validation and test.
    """
    train_df, temporary_df = train_test_split(
        dataframe,
        test_size=0.30,
        random_state=RANDOM_STATE,
        shuffle=True,
        stratify=dataframe["class_name"],
    )

    val_df, test_df = train_test_split(
        temporary_df,
        test_size=0.50,
        random_state=RANDOM_STATE,
        shuffle=True,
        stratify=temporary_df["class_name"],
    )

    train_df = train_df.copy()
    val_df = val_df.copy()
    test_df = test_df.copy()

    train_df["split"] = "train"
    val_df["split"] = "val"
    test_df["split"] = "test"

    return pd.concat([train_df, val_df, test_df], ignore_index=True)


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