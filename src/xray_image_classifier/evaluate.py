"""
Evaluation script for Chest X-ray classification using MONAI.

This script:
- Loads a trained model from the models/ folder
- Evaluates it on the test dataset
- Prints the classification accuracy

Run:
    uv run python evaluate.py --data-dir data/processed/test
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch
import typer
from monai.data import DataLoader, Dataset
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Resized,
    ScaleIntensityd,
)

from model import XRayClassifier

# -----------------------------------------------------------------------------
# CLI app
# -----------------------------------------------------------------------------
app = typer.Typer(help="Evaluate Chest X-ray classifier")

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Device
# -----------------------------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


# -----------------------------------------------------------------------------
# Dataset utilities (same as train.py)
# -----------------------------------------------------------------------------
def build_dataloader(
    data_dir: Path,
    batch_size: int,
    shuffle: bool = False,  # no shuffle for evaluation
) -> DataLoader:
    """
    Build a MONAI DataLoader from an ImageFolder-style directory.
    """
    images, labels = [], []
    class_names = sorted(p.name for p in data_dir.iterdir() if p.is_dir())
    class_to_idx = {cls: idx for idx, cls in enumerate(class_names)}

    for cls, idx in class_to_idx.items():
        for img_path in (data_dir / cls).glob("*"):
            images.append(str(img_path))
            labels.append(idx)

    if len(images) == 0:
        raise RuntimeError(f"No images found in {data_dir}")

    transforms = Compose(
        [
            LoadImaged(keys="image"),
            EnsureChannelFirstd(keys="image"),
            Resized(keys="image", spatial_size=(224, 224)),
            ScaleIntensityd(keys="image"),
        ]
    )

    dataset = Dataset(
        data=[{"image": img, "label": lbl} for img, lbl in zip(images, labels)],
        transform=transforms,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=4,
        pin_memory=torch.cuda.is_available(),
    )


# -----------------------------------------------------------------------------
# Evaluation command
# -----------------------------------------------------------------------------
@app.command()
def evaluate(
    data_dir: Path = typer.Option(
        ...,
        "--data-dir",
        exists=True,
        file_okay=False,
        help="Test dataset directory",
    ),
    backbone: str = typer.Option(
        "densenet121",
        "--backbone",
        help="Model backbone (densenet121 or efficientnet-b0)",
    ),
    batch_size: int = typer.Option(
        16,
        "--batch-size",
        help="Batch size for evaluation",
    ),
) -> None:
    """
    Evaluate a trained X-ray classifier on the test set.
    """
    model_path = Path("models") / f"xray_classifier_{backbone}_finetuned.pth"
    LOGGER.info("Starting evaluation")
    LOGGER.info("Device: %s", DEVICE)
    LOGGER.info("Loading model from %s", model_path)

    # -------------------------------------------------------------------------
    # Data
    # -------------------------------------------------------------------------
    test_loader = build_dataloader(data_dir, batch_size)
    LOGGER.info("Test batches: %d", len(test_loader))

    # -------------------------------------------------------------------------
    # Model
    # -------------------------------------------------------------------------
    model = XRayClassifier(
        num_classes=4,
        in_channels=3,
        backbone=backbone,
        pretrained=False,
    ).to(DEVICE)

    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()

    # -------------------------------------------------------------------------
    # Evaluation loop
    # -------------------------------------------------------------------------
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in test_loader:
            x = batch["image"].to(DEVICE)
            y = batch["label"].to(DEVICE)

            logits = model(x)
            preds = logits.argmax(dim=1)

            correct += (preds == y).sum().item()
            total += y.size(0)

    accuracy = correct / total

    LOGGER.info("===================================")
    LOGGER.info("Test accuracy: %.4f", accuracy)
    LOGGER.info("Correct: %d / %d", correct, total)
    LOGGER.info("===================================")


# -----------------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    app()
