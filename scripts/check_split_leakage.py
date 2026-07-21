from pathlib import Path
import re


# Location containing train, val and test frame folders
frames_root = Path("data/frames")

# The three dataset splits
split_names = ["train", "val", "test"]

# This dictionary will store the groups found in each split
groups_by_split = {}


for split_name in split_names:
    split_path = frames_root / split_name
    groups = set()

    # Check that the split folder exists
    if not split_path.exists():
        print(f"ERROR: Folder not found: {split_path}")
        continue

    # Visit every action class, such as Archery and Basketball
    for class_folder in split_path.iterdir():
        if not class_folder.is_dir():
            continue

        # Visit every video folder inside the action class
        for video_folder in class_folder.iterdir():
            if not video_folder.is_dir():
                continue

            # Example video name:
            # v_Archery_g01_c01
            #
            # This extracts the group number "01".
            match = re.search(r"_g(\d+)_", video_folder.name)

            if match:
                group_number = match.group(1)

                # Include the class name because Archery g01 and
                # Basketball g01 are different groups.
                group_identity = (
                    class_folder.name,
                    group_number
                )

                groups.add(group_identity)
            else:
                print(
                    "WARNING: Could not find a group number in:",
                    video_folder.name
                )

    groups_by_split[split_name] = groups

    print(
        f"{split_name}: "
        f"{len(groups)} unique class-group combinations"
    )


print("\nChecking for group leakage...\n")

split_pairs = [
    ("train", "val"),
    ("train", "test"),
    ("val", "test"),
]

leakage_found = False


for first_split, second_split in split_pairs:
    first_groups = groups_by_split.get(first_split, set())
    second_groups = groups_by_split.get(second_split, set())

    overlapping_groups = first_groups & second_groups

    print(
        f"{first_split} vs {second_split}: "
        f"{len(overlapping_groups)} overlapping groups"
    )

    if overlapping_groups:
        leakage_found = True

        for class_name, group_number in sorted(overlapping_groups):
            print(
                f"  Leakage: {class_name}, group g{group_number}"
            )


print()

if leakage_found:
    print("WARNING: Group leakage was detected.")
    print(
        "Some recording groups appear in more than one dataset split."
    )
else:
    print("SUCCESS: No group leakage was detected.")