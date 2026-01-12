"""
Chest X-ray classification models using MONAI.

This module defines neural network architectures for multi-class
classification of chest X-ray images, targeting the following classes:

- Pneumonia
- COVID-19
- Normal
- Tuberculosis

The models defined here are intentionally decoupled from any training
or data-loading logic, following good software engineering practices.

Author: Your Name
"""

from __future__ import annotations
from monai.networks.nets import DenseNet121, EfficientNetBN
from torch import nn
import torch


class XRayClassifier(nn.Module):
    """
    Chest X-ray image classifier based on a MONAI backbone.

    This model is designed for 2D chest X-ray images and outputs logits
    for multi-class classification.

    Parameters
    ----------
    num_classes : int
        Number of output classes.
    in_channels : int, optional
        Number of input channels. Use 1 for grayscale X-rays,
        3 if images are RGB. 
    backbone : str, optional
        Backbone architecture to use. Supported:
        {"densenet121", "efficientnet-b0"}.
    pretrained : bool, optional
        Whether to initialize the backbone with pretrained weights.
        Default is True.
    """

    def __init__(
        self,
        num_classes: int,
        in_channels: int = 3,
        backbone: str = "densenet121",
        pretrained: bool = True,
    ) -> None:
        super().__init__()

        self.num_classes = num_classes
        self.in_channels = in_channels
        self.backbone_name = backbone.lower()

        self.model = self._build_backbone(
            backbone=self.backbone_name,
            in_channels=in_channels,
            num_classes=num_classes,
            pretrained=pretrained,
        )

    @staticmethod
    def _build_backbone(
        backbone: str,
        in_channels: int,
        num_classes: int,
        pretrained: bool,
    ) -> nn.Module:
        """
        Instantiate a MONAI backbone network.

        Raises
        ------
        ValueError
            If an unsupported backbone is requested.
        """
        if backbone == "densenet121":
            return DenseNet121(
                spatial_dims=2,
                in_channels=in_channels,
                out_channels=num_classes,
                pretrained=pretrained,
            )

        if backbone == "efficientnet-b0":
            return EfficientNetBN(
                model_name="efficientnet-b0",
                spatial_dims=2,
                in_channels=in_channels,
                num_classes=num_classes,
                pretrained=pretrained,
            )

        raise ValueError(
            f"Unsupported backbone '{backbone}'. "
            "Choose from {'densenet121', 'efficientnet-b0'}."
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (B, C, H, W).

        Returns
        -------
        torch.Tensor
            Logits of shape (B, num_classes).
        """
        return self.model(x)

    def freeze_backbone(self) -> None:
        """
        Freeze all backbone parameters.
        Keeps the classification head trainable.

        Safe for:
        - DenseNet121
        - EfficientNet-B0
        """
        # Freeze everything
        for param in self.model.parameters():
            param.requires_grad = False

        # Always unfreeze classifier head
        if self.backbone_name == "densenet121":
            for name, param in self.model.named_parameters():
                if name.startswith("class_layers"):
                    param.requires_grad = True

        elif self.backbone_name.startswith("efficientnet"):
            for name, param in self.model.named_parameters():
                if name.startswith("_fc"):
                    param.requires_grad = True

    def unfreeze_for_finetuning(self) -> None:
        """
        Light fine-tuning strategy depending on backbone.

        DenseNet121:
            - Unfreeze last dense block (features.denseblock4)
        EfficientNet-B0:
            - Unfreeze last MBConv stage (_blocks.6)
            - Unfreeze conv head
        """
        if self.backbone_name == "densenet121":
            for name, param in self.model.named_parameters():
                if name.startswith("features.denseblock4"):
                    param.requires_grad = True

        elif self.backbone_name.startswith("efficientnet"):
            for name, param in self.model.named_parameters():
                if (
                    name.startswith("_blocks.6")
                    or name.startswith("_conv_head")
                    or name.startswith("_fc")
                ):
                    param.requires_grad = True
        




if __name__ == "__tmain__":
    """
    Simple sanity check for the XRayClassifier model.

    This block allows the file to be run directly in order to:
    - Print the model architecture
    - Count the number of trainable parameters
    - Perform a forward pass with a dummy input
    """

    import torch

    num_classes = 4
    in_channels = 3
    image_size = 224

    model = XRayClassifier(
        num_classes=num_classes,
        in_channels=in_channels,
        backbone="densenet121",
        pretrained=False,
    )

    num_params = sum(p.numel() for p in model.parameters())
    num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\nTotal parameters: {num_params:,}")
    print(f"Trainable parameters: {num_trainable:,}")

    input = torch.randn(1, in_channels, image_size, image_size)
    output = model(input)

    print(f"Output shape: {tuple(output.shape)}")

    # Freeze backbone and count trainable parameters again
    model.freeze_backbone()

    print("\n=== After freeze_backbone() ===")
    total = 0
    for name, p in model.model.named_parameters():
        if p.requires_grad:
            print("TRAINABLE:", name)
            total += p.numel()
    print(f"Total trainable parameters: {total:,}")

    # Unfreeze and count trainable parameters again
    model.unfreeze_for_finetuning()

    print("\n=== After unfreeze_for_finetuning() ===")
    total = 0
    for name, p in model.model.named_parameters():
        if p.requires_grad:
            print("TRAINABLE:", name)
            total += p.numel()
    print(f"Total trainable parameters: {total:,}")

