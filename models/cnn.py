import torch.nn as nn


class SimpleCNN(nn.Module):
    """Lightweight 3-layer CNN for binary pneumonia classification on 28x28 grayscale images.

    Kept intentionally small so each federated client can train a full epoch quickly on CPU.
    Outputs a single logit — pair with BCEWithLogitsLoss during training.
    """

    def __init__(self):
        super().__init__()
        # three conv blocks with doubling channel counts; BatchNorm stabilises training
        # with the small, potentially imbalanced local datasets each client sees
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),          # 28 -> 14

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),          # 14 -> 7

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        # AdaptiveAvgPool makes the classifier independent of whatever spatial size comes in
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 1),
        )

    def forward(self, x):
        return self.classifier(self.features(x))
