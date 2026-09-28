import random
import shutil
from pathlib import Path

SEED = 42
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp"
}

DATASETS = {
    "jawline": {
        "raw": Path("data/raw/jawline"),
        "output": Path("data/splits/jawline")
    },
    "hairline": {
        "raw": Path("data/raw/hairline"),
        "output": Path("data/splits/hairline")
    }
}

random.seed(SEED)


def get_images(folder):
    return [
        path for path in folder.rglob("*")
        if path.is_file()
        and path.suffix.lower() in IMAGE_EXTENSIONS
    ]


def create_split_folders(output_dir, class_name):
    for split in ["train", "val", "test"]:
        folder = output_dir / split / class_name
        folder.mkdir(parents=True, exist_ok=True)


def split_class_images(images):
    random.shuffle(images)

    total = len(images)
    train_end = int(total * TRAIN_RATIO)
    val_end = train_end + int(total * VAL_RATIO)

    train_images = images[:train_end]
    val_images = images[train_end:val_end]
    test_images = images[val_end:]

    return train_images, val_images, test_images


def copy_images(images, destination, class_name, split):
    for number, source in enumerate(images, start=1):
        extension = source.suffix.lower()
        filename = f"{class_name}_{split}_{number:05d}{extension}"
        target = destination / split / class_name / filename
        shutil.copy2(source, target)


def process_dataset(dataset_name, config):
    raw_dir = config["raw"]
    output_dir = config["output"]

    if not raw_dir.exists():
        print(f"Missing raw directory: {raw_dir}")
        return

    class_dirs = [
        directory for directory in raw_dir.iterdir()
        if directory.is_dir()
        and directory.name.lower() != "excluded"
    ]

    if not class_dirs:
        print(f"No class folders found in {raw_dir}")
        return

    print(f"\nProcessing {dataset_name}")
    print("-" * 40)

    for class_dir in sorted(class_dirs):
        class_name = class_dir.name
        images = get_images(class_dir)

        if not images:
            print(
                f"{class_name}: no images found"
            )
            continue

        create_split_folders(
            output_dir,
            class_name
        )

        train_images, val_images, test_images = (
            split_class_images(images)
        )

        copy_images(
            train_images,
            output_dir,
            class_name,
            "train"
        )

        copy_images(
            val_images,
            output_dir,
            class_name,
            "val"
        )

        copy_images(
            test_images,
            output_dir,
            class_name,
            "test"
        )

        print(
            f"{class_name}: "
            f"total={len(images)}, "
            f"train={len(train_images)}, "
            f"val={len(val_images)}, "
            f"test={len(test_images)}"
        )


for name, config in DATASETS.items():
    process_dataset(name, config)

print("\nDataset splitting completed.")