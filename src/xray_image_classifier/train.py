"""
Training script for Chest X-ray classification using MONAI.

This script trains a multi-class classifier (COVID, Pneumonia,
Normal, Tuberculosis) using a MONAI backbone and logs metrics
using Weights & Biases.

Run:
    uv run python train.py train --epochs 10
"""

from __future__ import annotations

import logging
from pathlib import Path
import matplotlib.pyplot as plt

import torch
from torch.profiler import profile, record_function, ProfilerActivity, tensorboard_trace_handler
from torch.utils.tensorboard import SummaryWriter
import typer
import wandb
from monai.data import DataLoader, Dataset
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Resized,
    ScaleIntensityd,
)
from torch.nn import CrossEntropyLoss
from torch.optim import Adam

from model import XRayClassifier

# -----------------------------------------------------------------------------
# CLI app
# -----------------------------------------------------------------------------
app = typer.Typer(help="Train Chest X-ray classifier")

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Device
# -----------------------------------------------------------------------------
DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)


# -----------------------------------------------------------------------------
# Output directories
# -----------------------------------------------------------------------------
MODELS_DIR = Path("models")
FIGURES_DIR = Path("reports/figures")
MODELS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


# -----------------------------------------------------------------------------
# Dataset utilities
# -----------------------------------------------------------------------------
def build_dataloader(data_dir: Path, batch_size: int, shuffle: bool = True) -> DataLoader:
    """
    Build a MONAI DataLoader from an ImageFolder-style directory.

    Expected structure:
        data_dir/
            class_1/
            class_2/
            ...
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
# Training command
# -----------------------------------------------------------------------------
@app.command()
def train(
    data_dir: Path = typer.Option(
        ...,
        "--data-dir",
        exists=True,
        file_okay=False,
        help="Dataset root (ImageFolder-style)",
    ),
    backbone: str = typer.Option(
        "densenet121",
        "--backbone",
        help="Model backbone: densenet121 or efficientnet-b0",
    ),
    batch_size: int = 16,
    epochs_phase1: int = 5,
    epochs_phase2: int = 5,
) -> None:
    """
    Train the X-ray classifier using a two-phase fine-tuning strategy.
    """
    LOGGER.info("Starting training")
    LOGGER.info("Device: %s", DEVICE)

    train_loader = build_dataloader(data_dir, batch_size)
    LOGGER.info("Training batches: %d", len(train_loader))

    # Initialize TorchProfiler
    tb_logdir = Path("reports/tensorboard") / backbone
    tb_logdir.mkdir(parents=True, exist_ok=True)

    PROFILE_STEPS = 20

    # ---------------- W&B init ----------------
    wandb.init(
        project="xray-classification",
        config={
            "batch_size": batch_size,
            "epochs_phase1": epochs_phase1,
            "epochs_phase2": epochs_phase2,
            "backbone": backbone,
            "num_classes": 4,
            "device": str(DEVICE),
        },
    )

    model = XRayClassifier(
        num_classes=4,
        in_channels=3,
        backbone= backbone,
        pretrained=True,
    ).to(DEVICE)

    loss_fn = torch.nn.CrossEntropyLoss()

    train_loss: list[float] = []
    train_acc: list[float] = []

    # -------------------------------------------------------------------------
    # Phase 1: Feature extraction
    # -------------------------------------------------------------------------
    LOGGER.info("Phase 1: Training classifier head")
    model.freeze_backbone()

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-3,
    )

    step = 0

    with profile(
        activities=[
            ProfilerActivity.CPU,
            ProfilerActivity.CUDA if torch.cuda.is_available() else ProfilerActivity.CPU,
        ],
        schedule=torch.profiler.schedule(
            wait=1,
            warmup=1,
            active=3,
            repeat=1,
        ),
        on_trace_ready=tensorboard_trace_handler(
            tb_logdir
        ),
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    ) as prof:

        for epoch in range(epochs_phase1):
            model.train()
            epoch_loss, epoch_acc = 0.0, 0.0

            for batch in train_loader:
                x = batch["image"].to(DEVICE)
                y = batch["label"].to(DEVICE)

                optimizer.zero_grad()

                with record_function("forward"):
                    logits = model(x)
                    loss = loss_fn(logits, y)

                with record_function("backward"):
                    loss.backward()
                    optimizer.step()

                acc = (logits.argmax(dim=1) == y).float().mean().item()
                epoch_loss += loss.item()
                epoch_acc += acc

                prof.step()
                step += 1

                 # profile only a few batches
                if epoch == 0 and step >= 5:
                    break

            epoch_loss /= len(train_loader)
            epoch_acc /= len(train_loader)

            train_loss.append(epoch_loss)
            train_acc.append(epoch_acc)

            LOGGER.info(
                "[Phase 1] Epoch %d/%d | loss=%.4f | acc=%.4f",
                epoch + 1,
                epochs_phase1,
                epoch_loss,
                epoch_acc,
            )

            wandb.log(
                {
                    "phase": "classifier",
                    "epoch": epoch + 1,
                    "train/loss": epoch_loss,
                    "train/accuracy": epoch_acc,
                }
            )


    # -------------------------------------------------------------------------
    # Phase 2: Light fine-tuning
    # -------------------------------------------------------------------------
    LOGGER.info("Phase 2: Fine-tuning last DenseNet block")
    model.unfreeze_for_finetuning()

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-4,
    )

    for epoch in range(epochs_phase2):
        model.train()
        epoch_loss, epoch_acc = 0.0, 0.0

        for batch in train_loader:
            x = batch["image"].to(DEVICE)
            y = batch["label"].to(DEVICE)

            optimizer.zero_grad()
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            optimizer.step()

            acc = (logits.argmax(dim=1) == y).float().mean().item()
            epoch_loss += loss.item()
            epoch_acc += acc

        epoch_loss /= len(train_loader)
        epoch_acc /= len(train_loader)

        train_loss.append(epoch_loss)
        train_acc.append(epoch_acc)

        LOGGER.info(
            "[Phase 2] Epoch %d/%d | loss=%.4f | acc=%.4f",
            epoch + 1,
            epochs_phase2,
            epoch_loss,
            epoch_acc,
        )

        wandb.log(
            {
                "phase": "finetune",
                "epoch": epochs_phase1 + epoch + 1,
                "train/loss": epoch_loss,
                "train/accuracy": epoch_acc,
            }
        )

    # -------------------------------------------------------------------------
    # Save model
    # -------------------------------------------------------------------------
    model_path = MODELS_DIR / f"xray_classifier_{backbone}_finetuned.pth"
    torch.save(model.state_dict(), model_path)

    artifact = wandb.Artifact(
        name="xray_classifier",
        type="model",
        metadata={
            "backbone": wandb.config.backbone,
            "num_classes": 4,
        },
    )
    artifact.add_file(str(model_path))
    wandb.log_artifact(artifact)

    LOGGER.info("Model saved to %s", model_path)

    # -------------------------------------------------------------------------
    # Save training curves
    # -------------------------------------------------------------------------
    fig, axs = plt.subplots(1, 2, figsize=(14, 5))

    axs[0].plot(train_loss)
    axs[0].set_title("Training Loss")
    axs[0].set_xlabel("Epoch")
    axs[0].set_ylabel("Loss")

    axs[1].plot(train_acc)
    axs[1].set_title("Training Accuracy")
    axs[1].set_xlabel("Epoch")
    axs[1].set_ylabel("Accuracy")

    fig.tight_layout()
    fig_path = FIGURES_DIR / f"training_curves_{backbone}.png"
    fig.savefig(fig_path)
    plt.close(fig)
    wandb.save(str(fig_path))

    LOGGER.info("Training curves saved to %s", fig_path)
    wandb.finish()
    LOGGER.info("Training completed successfully")

    writer.close()


# -----------------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    app()