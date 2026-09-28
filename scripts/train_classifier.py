import argparse
import copy
import json
import os
import random
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score
)

from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import (
    EfficientNet_B0_Weights,
    efficientnet_b0
)


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_transforms():
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomResizedCrop(
            size=224,
            scale=(0.90, 1.0),
            ratio=(0.95, 1.05)
        ),
        transforms.RandomRotation(5),
        transforms.ColorJitter(
            brightness=0.15,
            contrast=0.15,
            saturation=0.08
        ),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    return train_transform, eval_transform


def calculate_class_weights(dataset):
    targets = np.array(dataset.targets)
    counts = np.bincount(targets)

    weights = len(targets) / (
        len(counts) * counts
    )

    return torch.tensor(
        weights,
        dtype=torch.float32
    )


def build_model(num_classes, device):
    model = efficientnet_b0(
        weights=EfficientNet_B0_Weights.DEFAULT
    )

    in_features = model.classifier[1].in_features

    model.classifier = nn.Sequential(
        nn.Dropout(p=0.30),
        nn.Linear(in_features, num_classes)
    )

    return model.to(device)


def run_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device,
    training
):
    if training:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    true_labels = []
    predicted_labels = []

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        if training:
            optimizer.zero_grad()

        with torch.set_grad_enabled(training):
            outputs = model(images)
            loss = criterion(outputs, labels)

            if training:
                loss.backward()
                optimizer.step()

        total_loss += loss.item() * images.size(0)

        predictions = torch.argmax(outputs, dim=1)

        true_labels.extend(labels.detach().cpu().numpy())
        predicted_labels.extend(
            predictions.detach().cpu().numpy()
        )

    average_loss = total_loss / len(loader.dataset)

    accuracy = accuracy_score(
        true_labels,
        predicted_labels
    )

    macro_f1 = f1_score(
        true_labels,
        predicted_labels,
        average="macro",
        zero_division=0
    )

    return {
        "loss": average_loss,
        "accuracy": accuracy,
        "macro_f1": macro_f1
    }


def evaluate_model(
    model,
    loader,
    class_names,
    device,
    output_dir,
    task_name
):
    model.eval()

    true_labels = []
    predicted_labels = []
    confidence_values = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)

            outputs = model(images)
            probabilities = torch.softmax(outputs, dim=1)

            confidence, predictions = torch.max(
                probabilities,
                dim=1
            )

            true_labels.extend(labels.numpy())
            predicted_labels.extend(
                predictions.cpu().numpy()
            )
            confidence_values.extend(
                confidence.cpu().numpy()
            )

    accuracy = accuracy_score(
        true_labels,
        predicted_labels
    )

    precision = precision_score(
        true_labels,
        predicted_labels,
        average="macro",
        zero_division=0
    )

    recall = recall_score(
        true_labels,
        predicted_labels,
        average="macro",
        zero_division=0
    )

    macro_f1 = f1_score(
        true_labels,
        predicted_labels,
        average="macro",
        zero_division=0
    )

    report = classification_report(
        true_labels,
        predicted_labels,
        target_names=class_names,
        output_dict=True,
        zero_division=0
    )

    matrix = confusion_matrix(
        true_labels,
        predicted_labels
    )

    mean_confidence = float(
        np.mean(confidence_values)
    )

    report_text = classification_report(
        true_labels,
        predicted_labels,
        target_names=class_names,
        zero_division=0
    )

    print(f"\n{task_name} TEST REPORT")
    print("=" * 60)
    print(report_text)
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro precision: {precision:.4f}")
    print(f"Macro recall: {recall:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")
    print(f"Mean confidence: {mean_confidence:.4f}")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(
        output_dir / "classification_report.json",
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            {
                "task": task_name,
                "accuracy": accuracy,
                "macro_precision": precision,
                "macro_recall": recall,
                "macro_f1": macro_f1,
                "mean_confidence": mean_confidence,
                "class_report": report
            },
            file,
            indent=2
        )

    plt.figure(figsize=(7, 6))

    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Greys",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=False
    )

    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.title(f"{task_name} Confusion Matrix")
    plt.tight_layout()

    plt.savefig(
        output_dir / "confusion_matrix.png",
        dpi=300
    )

    plt.close()

    return {
        "accuracy": accuracy,
        "macro_precision": precision,
        "macro_recall": recall,
        "macro_f1": macro_f1,
        "mean_confidence": mean_confidence
    }


def plot_training_history(history, output_dir, task_name):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    epochs = range(1, len(history["train_loss"]) + 1)

    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.plot(
        epochs,
        history["train_loss"],
        label="Train loss"
    )
    plt.plot(
        epochs,
        history["val_loss"],
        label="Validation loss"
    )
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(
        epochs,
        history["train_macro_f1"],
        label="Train macro F1"
    )
    plt.plot(
        epochs,
        history["val_macro_f1"],
        label="Validation macro F1"
    )
    plt.xlabel("Epoch")
    plt.ylabel("Macro F1")
    plt.title("Macro F1")
    plt.legend()

    plt.suptitle(f"{task_name} Training History")
    plt.tight_layout()

    plt.savefig(
        output_dir / "training_history.png",
        dpi=300
    )

    plt.close()


def train_task(
    task_name,
    data_dir,
    model_dir,
    output_dir,
    epochs=20,
    batch_size=16,
    learning_rate=0.001,
    seed=42
):
    set_seed(seed)

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(f"\nDevice: {device}")
    print(f"Task: {task_name}")
    print(f"Data: {data_dir}")

    train_transform, eval_transform = build_transforms()

    train_dataset = datasets.ImageFolder(
        Path(data_dir) / "train",
        transform=train_transform
    )

    val_dataset = datasets.ImageFolder(
        Path(data_dir) / "val",
        transform=eval_transform
    )

    test_dataset = datasets.ImageFolder(
        Path(data_dir) / "test",
        transform=eval_transform
    )

    if train_dataset.classes != val_dataset.classes:
        raise ValueError(
            "Train and validation classes do not match."
        )

    if train_dataset.classes != test_dataset.classes:
        raise ValueError(
            "Train and test classes do not match."
        )

    class_names = train_dataset.classes
    num_classes = len(class_names)

    print(f"Classes: {class_names}")
    print(f"Training images: {len(train_dataset)}")
    print(f"Validation images: {len(val_dataset)}")
    print(f"Test images: {len(test_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    model = build_model(
        num_classes,
        device
    )

    class_weights = calculate_class_weights(
        train_dataset
    ).to(device)

    criterion = nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=0.05
    )

    optimizer = optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=1e-4
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=2
    )

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_macro_f1": [],
        "val_macro_f1": []
    }

    best_val_f1 = -1.0
    best_weights = copy.deepcopy(
        model.state_dict()
    )

    for epoch in range(epochs):
        start = time.time()

        train_metrics = run_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            training=True
        )

        val_metrics = run_epoch(
            model,
            val_loader,
            criterion,
            optimizer,
            device,
            training=False
        )

        scheduler.step(
            val_metrics["macro_f1"]
        )

        history["train_loss"].append(
            train_metrics["loss"]
        )

        history["val_loss"].append(
            val_metrics["loss"]
        )

        history["train_macro_f1"].append(
            train_metrics["macro_f1"]
        )

        history["val_macro_f1"].append(
            val_metrics["macro_f1"]
        )

        print(
            f"Epoch {epoch + 1:02d}/{epochs} | "
            f"Train loss={train_metrics['loss']:.4f} | "
            f"Train F1={train_metrics['macro_f1']:.4f} | "
            f"Val loss={val_metrics['loss']:.4f} | "
            f"Val F1={val_metrics['macro_f1']:.4f} | "
            f"Time={time.time() - start:.1f}s"
        )

        if val_metrics["macro_f1"] > best_val_f1:
            best_val_f1 = val_metrics["macro_f1"]
            best_weights = copy.deepcopy(
                model.state_dict()
            )

    model.load_state_dict(best_weights)

    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    model_path = (
        model_dir / f"{task_name}_efficientnet_b0.pth"
    )

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "class_names": class_names,
            "task": task_name,
            "input_size": 224,
            "confidence_threshold": 0.60,
            "best_validation_macro_f1": best_val_f1
        },
        model_path
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_training_history(
        history,
        output_dir,
        task_name
    )

    test_metrics = evaluate_model(
        model,
        test_loader,
        class_names,
        device,
        output_dir,
        task_name
    )

    summary = {
        "task": task_name,
        "device": device,
        "classes": class_names,
        "best_validation_macro_f1": best_val_f1,
        "test_metrics": test_metrics,
        "model_path": str(model_path)
    }

    with open(
        output_dir / "summary.json",
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(summary, file, indent=2)

    print(f"\nSaved model: {model_path}")
    print(f"Saved results: {output_dir}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--task",
        choices=["jawline", "hairline"],
        required=True
    )

    parser.add_argument(
        "--data",
        required=True
    )

    parser.add_argument(
        "--model-dir",
        required=True
    )

    parser.add_argument(
        "--output-dir",
        required=True
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=20
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=16
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.001
    )

    args = parser.parse_args()

    train_task(
        task_name=args.task,
        data_dir=args.data,
        model_dir=args.model_dir,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate
    )


if __name__ == "__main__":
    main()