# tests/test_data.py
from pathlib import Path
from PIL import Image
import pytest

from xray_image_classifier.data import preprocess_image, TARGET_SIZE, preprocess_split, preprocess


def test_preprocess_image(tmp_path: Path):
    # Arrange
    input_img = tmp_path / "input.png"
    output_img = tmp_path / "output.png"

    # Create a fake grayscale image
    img = Image.new("L", (512, 512))  # 1 channel
    img.save(input_img)

    # Act
    _, _, oc, nw, nh, nc = preprocess_image(input_img, output_img)

    # Assert
    assert oc == 1
    assert (nw, nh) == TARGET_SIZE
    assert nc == 3
    assert output_img.exists()


def test_preprocess_split_creates_outputs(tmp_path: Path):
    # Arrange
    input_dir = tmp_path / "train"
    output_dir = tmp_path / "processed" / "train"
    class_dir = input_dir / "normal"

    class_dir.mkdir(parents=True)

    # Valid image
    img = Image.new("RGB", (300, 300))
    img_path = class_dir / "img1.png"
    img.save(img_path)

    # Invalid file
    (class_dir / "notes.txt").write_text("not an image")

    # Act
    preprocess_split(input_dir, output_dir)

    # Assert
    processed_img = output_dir / "normal" / "img1.png"
    assert processed_img.exists()


def test_preprocess_raises_on_missing_split(tmp_path: Path):
    raw_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed"

    # Only create train split
    (raw_dir / "train").mkdir(parents=True)

    with pytest.raises(FileNotFoundError):
        preprocess(raw_dir=raw_dir, processed_dir=processed_dir)


def test_corrupt_image_is_skipped(tmp_path: Path):
    input_dir = tmp_path / "train" / "normal"
    output_dir = tmp_path / "processed" / "train" / "normal"

    input_dir.mkdir(parents=True)

    # Corrupt image file
    bad_img = input_dir / "bad.png"
    bad_img.write_bytes(b"not an image")

    # Should not crash
    preprocess_split(input_dir.parent, output_dir.parent)

    assert not (output_dir / "bad.png").exists()
