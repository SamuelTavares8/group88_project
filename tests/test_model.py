from typing import Dict, List, Tuple

import pytest
import torch
from torch import nn

from xray_image_classifier.model import XRayClassifier


def _params_by_prefix(
    named_params: Dict[str, nn.Parameter],
    prefixes: Tuple[str, ...],
) -> List[nn.Parameter]:
    """Return parameters whose names start with any of the given prefixes."""
    return [param for name, param in named_params.items() if name.startswith(prefixes)]


@pytest.mark.parametrize("backbone", ["densenet121", "efficientnet-b0"])
def test_forward_shape(backbone: str) -> None:
    """Ensure the model returns logits with shape (batch, num_classes)."""
    model = XRayClassifier(
        num_classes=4,
        in_channels=3,
        backbone=backbone,
        pretrained=False,
    )
    model.eval()

    x = torch.randn(2, 3, 64, 64)
    y = model(x)

    assert y.shape == (2, 4)


def test_invalid_backbone_raises() -> None:
    """Reject unsupported backbones with a clear ValueError."""
    with pytest.raises(ValueError, match="Unsupported backbone"):
        XRayClassifier(num_classes=4, backbone="not-a-backbone", pretrained=False)


def test_freeze_backbone_densenet121() -> None:
    """Freeze all but the DenseNet classifier head."""
    model = XRayClassifier(
        num_classes=4,
        in_channels=3,
        backbone="densenet121",
        pretrained=False,
    )
    model.freeze_backbone()

    named_params = dict(model.model.named_parameters())
    head_params = _params_by_prefix(named_params, ("class_layers",))
    frozen_params = [p for name, p in named_params.items() if not name.startswith(("class_layers",))]

    assert head_params
    assert all(param.requires_grad for param in head_params)
    assert frozen_params
    assert all(not param.requires_grad for param in frozen_params)


def test_freeze_backbone_efficientnet() -> None:
    """Freeze all but the EfficientNet classifier head."""
    model = XRayClassifier(
        num_classes=4,
        in_channels=3,
        backbone="efficientnet-b0",
        pretrained=False,
    )
    model.freeze_backbone()

    named_params = dict(model.model.named_parameters())
    head_params = _params_by_prefix(named_params, ("_fc",))
    frozen_params = [p for name, p in named_params.items() if not name.startswith(("_fc",))]

    assert head_params
    assert all(param.requires_grad for param in head_params)
    assert frozen_params
    assert all(not param.requires_grad for param in frozen_params)


def test_unfreeze_for_finetuning_densenet121() -> None:
    """Unfreeze the last DenseNet block for light fine-tuning."""
    model = XRayClassifier(
        num_classes=4,
        in_channels=3,
        backbone="densenet121",
        pretrained=False,
    )
    model.freeze_backbone()
    model.unfreeze_for_finetuning()

    named_params = dict(model.model.named_parameters())
    finetune_params = _params_by_prefix(named_params, ("features.denseblock4",))
    other_params = [
        p for name, p in named_params.items() if not name.startswith(("features.denseblock4", "class_layers"))
    ]

    assert finetune_params
    assert all(param.requires_grad for param in finetune_params)
    assert other_params
    assert all(not param.requires_grad for param in other_params)


def test_unfreeze_for_finetuning_efficientnet() -> None:
    """Unfreeze the EfficientNet tail layers for light fine-tuning."""
    model = XRayClassifier(
        num_classes=4,
        in_channels=3,
        backbone="efficientnet-b0",
        pretrained=False,
    )
    model.freeze_backbone()
    model.unfreeze_for_finetuning()

    named_params = dict(model.model.named_parameters())
    finetune_prefixes = ("_blocks.6", "_conv_head", "_fc")
    finetune_params = _params_by_prefix(named_params, finetune_prefixes)
    other_params = [p for name, p in named_params.items() if not name.startswith(finetune_prefixes)]

    assert finetune_params
    assert all(param.requires_grad for param in finetune_params)
    assert other_params
    assert all(not param.requires_grad for param in other_params)
