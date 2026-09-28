import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms
from torchvision.models import efficientnet_b0


def load_model(model_path, device):
    checkpoint = torch.load(
        model_path,
        map_location=device
    )

    class_names = checkpoint["class_names"]

    model = efficientnet_b0(
        weights=None
    )

    in_features = model.classifier[1].in_features

    model.classifier = torch.nn.Sequential(
        torch.nn.Dropout(p=0.30),
        torch.nn.Linear(
            in_features,
            len(class_names)
        )
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.to(device)
    model.eval()

    threshold = checkpoint.get(
        "confidence_threshold",
        0.60
    )

    return model, class_names, threshold


def predict(model, class_names, threshold, image_path, device):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    image = Image.open(
        image_path
    ).convert("RGB")

    tensor = transform(
        image
    ).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probabilities = torch.softmax(
            logits,
            dim=1
        )[0]

    best_index = int(
        torch.argmax(probabilities)
    )

    best_label = class_names[best_index]
    confidence = float(
        probabilities[best_index]
    )

    probability_map = {
        label: round(
            float(probability),
            4
        )
        for label, probability in zip(
            class_names,
            probabilities
        )
    }

    if confidence < threshold:
        final_label = "uncertain"
        message = (
            "Please upload a clearer, frontal image "
            "with the relevant facial region visible."
        )
    else:
        final_label = best_label
        message = "Prediction accepted."

    return {
        "label": final_label,
        "confidence": round(confidence, 4),
        "confidence_percent": round(
            confidence * 100,
            2
        ),
        "class_probabilities": probability_map,
        "message": message
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--task",
        choices=["jawline", "hairline"],
        required=True
    )

    parser.add_argument(
        "--model",
        required=True
    )

    parser.add_argument(
        "--image",
        required=True
    )

    parser.add_argument(
        "--output",
        default=None
    )

    args = parser.parse_args()

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    model, class_names, threshold = load_model(
        args.model,
        device
    )

    result = predict(
        model,
        class_names,
        threshold,
        args.image,
        device
    )

    payload = {
        "task": args.task,
        "model": Path(args.model).name,
        "image": Path(args.image).name,
        "device": device,
        "result": result
    }

    print(json.dumps(payload, indent=2))

    if args.output:
        with open(
            args.output,
            "w",
            encoding="utf-8"
        ) as file:
            json.dump(payload, file, indent=2)


if __name__ == "__main__":
    main()