"""
Training script for Chest X-ray classification using MONAI + Hydra.

- Dataset is FIXED (ImageFolder-style)
- Hydra controls ONLY model & training hyperparameters
- Supports DenseNet121 & EfficientNet-B0
- Optional profiling
- Logs to Weights & Biases
"""

from __future__ import annotations

import logging
from pathlib import Path
import matplotlib.pyplot as plt

import torch
import hydra
from omegaconf import DictConfig
import wandb

from torch.profiler import (
    profile,
    record_function,
    ProfilerActivity,
    tensorboard_trace_handler,
)

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
# Paths
# -----------------------------------------------------------------------------
DATA_DIR = Path("data/processed/train")
MODELS_DIR = Path("models")
FIGURES_DIR = Path("reports/figures")
TB_DIR = Path("reports/tensorboard")

MODELS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
TB_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
def build_dataloader(data_dir: Path, batch_size: int) -> DataLoader:
    images, labels = [], []

    class_names = sorted(p.name for p in data_dir.iterdir() if p.is_dir())
    class_to_idx = {cls: i for i, cls in enumerate(class_names)}

    for cls, idx in class_to_idx.items():
        for img in (data_dir / cls).glob("*"):
            images.append(str(img))
            labels.append(idx)

    transforms = Compose(
        [
            LoadImaged(keys="image"),
            EnsureChannelFirstd(keys="image"),
            Resized(keys="image", spatial_size=(224, 224)),
            ScaleIntensityd(keys="image"),
        ]
    )

    dataset = Dataset(
        data=[{"image": x, "label": y} for x, y in zip(images, labels)],
        transform=transforms,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=torch.cuda.is_available(),
    )

def build_optimizer(cfg, model):
    if cfg.optimizer.name == "adam":
        return torch.optim.Adam(
            model.parameters(),
            lr=cfg.optimizer.lr,
            weight_decay=cfg.optimizer.weight_decay,
        )

    if cfg.optimizer.name == "adamw":
        return torch.optim.AdamW(
            model.parameters(),
            lr=cfg.optimizer.lr,
            weight_decay=cfg.optimizer.weight_decay,
        )

    if cfg.optimizer.name == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=cfg.optimizer.lr,
            momentum=cfg.optimizer.momentum,
            weight_decay=cfg.optimizer.weight_decay,
        )

    raise ValueError(f"Unknown optimizer {cfg.optimizer.name}")

# -----------------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------------
@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def train(cfg: DictConfig) -> None:
    LOGGER.info("Device: %s", DEVICE)
    LOGGER.info("Backbone: %s", cfg.model.backbone)

    train_loader = build_dataloader(
        DATA_DIR,
        batch_size=cfg.training.batch_size,
    )

    model = XRayClassifier(
        num_classes=cfg.model.num_classes,
        in_channels=cfg.model.in_channels,
        backbone=cfg.model.backbone,
        pretrained=cfg.model.pretrained,
    ).to(DEVICE)

    loss_fn = torch.nn.CrossEntropyLoss()

    # ---------------- W&B ----------------
    wandb.init(
        project="xray-classification",
        config={
            **cfg.model,
            **cfg.training,
            "device": str(DEVICE),
        },
    )

    # -------------------------------------------------------------------------
    # PROFILING ONLY (no training)
    # -------------------------------------------------------------------------
    if cfg.training.profile:
        LOGGER.warning("Running PROFILING ONLY")

        model.freeze_backbone()

        #optimizer = torch.optim.Adam(
        #    filter(lambda p: p.requires_grad, model.parameters()),
        #    lr=cfg.training.lr_phase1,
        #)

        optimizer = build_optimizer(cfg, model)

        model.train()

        with profile(
            activities=[ProfilerActivity.CPU],
            schedule=torch.profiler.schedule(
                wait=1,
                warmup=1,
                active=3,
                repeat=1,
            ),
            on_trace_ready=tensorboard_trace_handler(
                TB_DIR / cfg.model.backbone
            ),
            record_shapes=False,
            profile_memory=False,
            with_stack=False,
        ) as prof:

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

                prof.step()
                break

        LOGGER.info("Profiling finished.")
        return

    # -------------------------------------------------------------------------
    # Phase 1 — Train classifier
    # -------------------------------------------------------------------------
    LOGGER.info("Phase 1: Training classifier head")
    model.freeze_backbone()

    #optimizer = torch.optim.Adam(
    #    filter(lambda p: p.requires_grad, model.parameters()),
    #    lr=cfg.training.lr_phase1,
    #)

    optimizer = build_optimizer(cfg, model)

    train_loss, train_acc = [], []

    for epoch in range(cfg.training.epochs_phase1):
        model.train()
        loss_sum, acc_sum = 0.0, 0.0

        for batch in train_loader:
            x = batch["image"].to(DEVICE)
            y = batch["label"].to(DEVICE)

            optimizer.zero_grad()
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            optimizer.step()

            acc = (logits.argmax(1) == y).float().mean().item()
            loss_sum += loss.item()
            acc_sum += acc

        loss_epoch = loss_sum / len(train_loader)
        acc_epoch = acc_sum / len(train_loader)

        train_loss.append(loss_epoch)
        train_acc.append(acc_epoch)

        LOGGER.info(
            "[Phase 1] Epoch %d/%d | loss=%.4f | acc=%.4f",
            epoch + 1,
            cfg.training.epochs_phase1,
            loss_epoch,
            acc_epoch,
        )

        wandb.log(
            {
                "phase": "classifier",
                "epoch": epoch + 1,
                "loss": loss_epoch,
                "accuracy": acc_epoch,
            }
        )

    # -------------------------------------------------------------------------
    # Phase 2 — Fine-tuning
    # -------------------------------------------------------------------------
    LOGGER.info("Phase 2: Fine-tuning (%s)", cfg.model.backbone)
    model.unfreeze_for_finetuning()

    #optimizer = torch.optim.Adam(
    #    filter(lambda p: p.requires_grad, model.parameters()),
    #    lr=cfg.training.lr_phase2,
    #)

    optimizer = build_optimizer(cfg, model)

    for epoch in range(cfg.training.epochs_phase2):
        model.train()
        loss_sum, acc_sum = 0.0, 0.0

        for batch in train_loader:
            x = batch["image"].to(DEVICE)
            y = batch["label"].to(DEVICE)

            optimizer.zero_grad()
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            optimizer.step()

            acc = (logits.argmax(1) == y).float().mean().item()
            loss_sum += loss.item()
            acc_sum += acc

        loss_epoch = loss_sum / len(train_loader)
        acc_epoch = acc_sum / len(train_loader)

        train_loss.append(loss_epoch)
        train_acc.append(acc_epoch)

        LOGGER.info(
            "[Phase 2] Epoch %d/%d | loss=%.4f | acc=%.4f",
            epoch + 1,
            cfg.training.epochs_phase2,
            loss_epoch,
            acc_epoch,
        )

        wandb.log(
            {
                "phase": "finetune",
                "epoch": cfg.training.epochs_phase1 + epoch + 1,
                "loss": loss_epoch,
                "accuracy": acc_epoch,
            }
        )

    # -------------------------------------------------------------------------
    # Save model
    # -------------------------------------------------------------------------
    model_path = MODELS_DIR / f"xray_classifier_{cfg.model.backbone}_finetuned.pth"
    torch.save(model.state_dict(), model_path)

    wandb.log_artifact(
        wandb.Artifact(
            name="xray_classifier",
            type="model",
        ).add_file(str(model_path))
    )

    # -------------------------------------------------------------------------
    # Curves
    # -------------------------------------------------------------------------
    fig, axs = plt.subplots(1, 2, figsize=(14, 5))
    axs[0].plot(train_loss)
    axs[0].set_title("Loss")
    axs[1].plot(train_acc)
    axs[1].set_title("Accuracy")

    fig_path = FIGURES_DIR / f"training_curves_{cfg.model.backbone}.png"
    fig.savefig(fig_path)
    plt.close(fig)

    wandb.save(str(fig_path))
    wandb.finish()

    LOGGER.info("Training finished successfully")

# -----------------------------------------------------------------------------
if __name__ == "__main__":
    train()