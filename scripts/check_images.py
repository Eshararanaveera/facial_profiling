import cv2
import numpy as np
from pathlib import Path

MIN_WIDTH = 640
MIN_HEIGHT = 640
MIN_BLUR_SCORE = 100.0
MIN_BRIGHTNESS = 70.0
MAX_BRIGHTNESS = 190.0
MIN_CONTRAST = 30.0

def image_quality_metrics(image_path: str) -> dict:
    image = cv2.imread(image_path)

    if image is None:
        return {
            "accepted": False,
            "reason": "invalid_image"
        }

    height, width = image.shape[:2]

    if width < MIN_WIDTH or height < MIN_HEIGHT:
        return {
            "accepted": False,
            "reason": "low_resolution",
            "width": width,
            "height": height
        }

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    blur_score = float(
        cv2.Laplacian(gray, cv2.CV_64F).var()
    )

    brightness = float(gray.mean())
    contrast = float(gray.std())

    accepted = (
        blur_score >= MIN_BLUR_SCORE
        and MIN_BRIGHTNESS <= brightness <= MAX_BRIGHTNESS
        and contrast >= MIN_CONTRAST
    )

    return {
        "accepted": accepted,
        "width": width,
        "height": height,
        "blur_score": round(blur_score, 2),
        "brightness": round(brightness, 2),
        "contrast": round(contrast, 2),
        "reason": "accepted" if accepted else "quality_failed"
    }

if __name__ == "__main__":
    sample = "data/raw/jawline/sharp/sample.jpg"
    print(image_quality_metrics(sample))