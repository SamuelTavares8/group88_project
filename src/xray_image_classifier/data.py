from pathlib import Path
from typing import Iterable

import typer
from .data import app as data_app
from PIL import Image

app = typer.Typer(help="xray_image_classifier CLI")
app.add_typer(data_app, name="data")

SPLITS: Iterable[str] = ("train", "val", "test")
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg")
TARGET_SIZE = (224, 224)


def preprocess_image(
    img_path: Path,
    output_path: Path,
) -> tuple[int, int, int, int, int, int]:
    """
    Open, inspect, normalize format, resize, and save an image.

    Returns
    -------
    (orig_w, orig_h, orig_c, new_w, new_h, new_c)
    """
    with Image.open(img_path) as img:
        # original properties
        orig_w, orig_h = img.size
        orig_c = len(img.getbands())

        # normalization
        img = img.convert("RGB")
        img = img.resize(TARGET_SIZE)

        new_w, new_h = img.size
        new_c = len(img.getbands())

        img.save(output_path)

    return orig_w, orig_h, orig_c, new_w, new_h, new_c


def preprocess_split(
    input_split_dir: Path,
    output_split_dir: Path,
    print_example: bool = False,
) -> None:
    """Preprocess a single split (train/val/test)."""
    printed_example = False

    for class_dir in input_split_dir.iterdir():
        if not class_dir.is_dir():
            continue

        output_class_dir = output_split_dir / class_dir.name
        output_class_dir.mkdir(parents=True, exist_ok=True)

        for img_path in class_dir.iterdir():
            if img_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue

            output_img_path = output_class_dir / img_path.name

            try:
                (
                    ow,
                    oh,
                    oc,
                    nw,
                    nh,
                    nc,
                ) = preprocess_image(img_path, output_img_path)

                if print_example and not printed_example:
                    print(
                        f"Example image ({input_split_dir.name}): "
                        f"before = {ow}x{oh}, channels={oc} | "
                        f"after = {nw}x{nh}, channels={nc}"
                    )
                    printed_example = True

            except Exception as exc:
                print(f"Skipping {img_path}: {exc}")


@app.command()
def preprocess(
    raw_dir: Path = Path("data/raw"),
    processed_dir: Path = Path("data/processed"),
) -> None:
    """
    Preprocess X-ray dataset.

    Assumes:
    data/raw/{train,val,test}/{class_name}/*.png|jpg
    """
    print("Starting preprocessing...")

    for split in SPLITS:
        input_split_dir = raw_dir / split
        output_split_dir = processed_dir / split

        if not input_split_dir.exists():
            raise FileNotFoundError(f"Missing split: {input_split_dir}")

        print(f"\nProcessing split: {split}")
        preprocess_split(
            input_split_dir,
            output_split_dir,
            print_example=True,
        )

    print("\nPreprocessing complete.")
    print(f"Processed data saved to: {processed_dir}")


if __name__ == "__main__":
    app()