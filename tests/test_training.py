from pathlib import Path
from typing import Dict

from PIL import Image
import pytest
import torch

import xray_image_classifier.train as train_module


def _write_rgb_image(path: Path, size: tuple[int, int] = (32, 32)) -> None:
    """Create a small RGB image on disk for dataset tests."""
    image = Image.new("RGB", size, color=(0, 0, 0))
    image.save(path)


def test_build_dataloader_raises_when_empty(tmp_path: Path) -> None:
    """Fail fast when the dataset directory contains no images."""
    with pytest.raises(RuntimeError, match="No images found"):
        train_module.build_dataloader(tmp_path, batch_size=2, shuffle=False)


def test_build_dataloader_yields_expected_batch(tmp_path: Path) -> None:
    """Build a DataLoader that returns resized tensors and labels."""
    data_dir = tmp_path / "train"
    class_a = data_dir / "covid"
    class_b = data_dir / "normal"
    class_a.mkdir(parents=True)
    class_b.mkdir(parents=True)

    _write_rgb_image(class_a / "a.png")
    _write_rgb_image(class_b / "b.png")

    loader = train_module.build_dataloader(data_dir, batch_size=2, shuffle=False)
    batch: Dict[str, torch.Tensor] = next(iter(loader))

    assert len(loader) == 1
    assert set(batch.keys()) == {"image", "label"}
    assert batch["image"].shape == (2, 3, 224, 224)
    assert batch["label"].dtype == torch.long
    assert sorted(batch["label"].tolist()) == [0, 1]
