from pathlib import Path
from PIL import Image

DATASETS = {
    "jawline": Path("data/splits/jawline"),
    "hairline": Path("data/splits/hairline")
}

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def check_dataset(name, root):
    print(f"\nChecking {name} dataset")
    print("-" * 40)

    total = 0

    if not root.exists():
        print(f"Missing folder: {root}")
        return

    for split in ["train", "val", "test"]:
        split_path = root / split

        if not split_path.exists():
            print(f"Missing split: {split_path}")
            continue

        for class_dir in sorted(split_path.iterdir()):
            if not class_dir.is_dir():
                continue

            files = [
                p for p in class_dir.iterdir()
                if p.suffix.lower() in VALID_EXTENSIONS
            ]

            valid = 0
            invalid = 0

            for file_path in files:
                try:
                    with Image.open(file_path) as image:
                        image.verify()
                    valid += 1
                except Exception:
                    invalid += 1

            total += valid

            print(
                f"{split:5s} | "
                f"{class_dir.name:15s} | "
                f"valid={valid:4d} | invalid={invalid:4d}"
            )

    print(f"Total valid images: {total}")


for dataset_name, dataset_path in DATASETS.items():
    check_dataset(dataset_name, dataset_path)