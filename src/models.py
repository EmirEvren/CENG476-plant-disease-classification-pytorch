import torch
from torch import nn


class BaselineCNN(nn.Module):
    def __init__(self, num_classes=38, dropout_rate=0.4):
        super().__init__()

        self.features = nn.Sequential(
            # 3 x 224 x 224 -> 32 x 112 x 112
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),

            # 32 x 112 x 112 -> 64 x 56 x 56
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),

            # 64 x 56 x 56 -> 128 x 28 x 28
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),

            # 128 x 28 x 28 -> 256 x 14 x 14
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )

        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(256, num_classes),
        )

    def forward(self, images):
        features = self.features(images)
        features = self.pool(features)
        logits = self.classifier(features)

        return logits


if __name__ == "__main__":
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model = BaselineCNN(num_classes=38).to(device)

    dummy_images = torch.randn(
        2,
        3,
        224,
        224,
        device=device,
    )

    model.eval()

    with torch.inference_mode():
        outputs = model(dummy_images)

    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    print("=" * 50)
    print("BASELINE CNN KONTROLÜ")
    print("=" * 50)
    print("Cihaz:", device)
    print("Girdi şekli:", dummy_images.shape)
    print("Çıktı şekli:", outputs.shape)
    print("Eğitilebilir parametre:", f"{trainable_parameters:,}")