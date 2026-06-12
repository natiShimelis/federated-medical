import torch.nn as nn
from torchvision.models import resnet18


class ResNet18Binary(nn.Module):
    """ResNet18 adapted for 28x28 single-channel binary classification.

    Standard ResNet18 was designed for 224x224 ImageNet images: the opening conv uses
    a 7x7 kernel with stride 2 and is immediately followed by a stride-2 maxpool,
    shrinking the input by 4x before the residual blocks start. On 28x28 inputs that
    leaves a 7x7 feature map going into layer1, and by layer4 the spatial size has
    collapsed to 1x1 — the AdaptiveAvgPool then has nothing meaningful to pool.

    Three changes to fix this:
      - conv1 replaced with a 3x3 kernel at stride 1 to preserve spatial resolution
      - maxpool replaced with Identity so the 28x28 map flows into layer1 untouched
      - the 1000-class fc head replaced with a single logit for BCEWithLogitsLoss
    """

    def __init__(self, pretrained: bool = False):
        super().__init__()
        weights = "IMAGENET1K_V1" if pretrained else None
        base = resnet18(weights=weights)

        base.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        base.maxpool = nn.Identity()
        base.fc = nn.Linear(512, 1)

        self._model = base

    def forward(self, x):
        return self._model(x)
