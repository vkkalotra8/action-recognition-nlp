from pathlib import Path
import json


PROJECT_ROOT = Path(__file__).resolve().parents[1]

CLASSES_FILE = PROJECT_ROOT / "metadata" / "classes.txt"
DESCRIPTIONS_FILE = (
    PROJECT_ROOT
    / "metadata"
    / "action_descriptions.json"
)


def main():

    classes = {
        line.strip()
        for line in CLASSES_FILE.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    }

    with open(
        DESCRIPTIONS_FILE,
        "r",
        encoding="utf-8"
    ) as file:
        descriptions = json.load(file)

    description_classes = set(descriptions.keys())

    missing = classes - description_classes
    extra = description_classes - classes

    empty = []

    for cls, texts in descriptions.items():
        if len(texts) == 0:
            empty.append(cls)

    print("=" * 50)
    print("DESCRIPTION VERIFICATION")
    print("=" * 50)

    print(f"Classes in classes.txt : {len(classes)}")
    print(f"Description entries    : {len(descriptions)}")

    print()

    print("Missing descriptions:")
    print(missing if missing else "None")

    print()

    print("Extra descriptions:")
    print(extra if extra else "None")

    print()

    print("Empty descriptions:")
    print(empty if empty else "None")

    print()

    if (
        len(missing) == 0
        and len(extra) == 0
        and len(empty) == 0
    ):
        print("SUCCESS")
        print(
            "Every action class has "
            "valid descriptions."
        )
    else:
        print("Please fix the problems above.")


if __name__ == "__main__":
    main()