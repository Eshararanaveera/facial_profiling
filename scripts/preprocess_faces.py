import cv2
from pathlib import Path

IMAGE_SIZE = 224

def resize_and_save(input_path: str, output_path: str):
    image = cv2.imread(input_path)

    if image is None:
        return False

    image = cv2.resize(
        image,
        (IMAGE_SIZE, IMAGE_SIZE),
        interpolation=cv2.INTER_AREA
    )

    Path(output_path).parent.mkdir(
        parents=True,
        exist_ok=True
    )

    cv2.imwrite(output_path, image)
    return True