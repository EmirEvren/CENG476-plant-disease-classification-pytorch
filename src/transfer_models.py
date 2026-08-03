import torch
from torch import nn
from torchvision.models import (
    ResNet18_Weights,
    resnet18,
)


def create_resnet18_transfer(
    num_classes=38,
    dropout_rate=0.3,
    pretrained=True,
):
    weights = (
        ResNet18_Weights.DEFAULT
        if pretrained
        else None
    )
    model = resnet18(weights=weights)
    input_features = model.fc.in_features

    model.fc = nn.Sequential(
        nn.Dropout(p=dropout_rate),
        nn.Linear(
            input_features,
            num_classes,
        ),
    )

    return model


def split_resnet18_parameters(model):
    backbone_parameters = []
    classifier_parameters = []

    for parameter_name, parameter in (
        model.named_parameters()
    ):
        if parameter_name.startswith("fc."):
            classifier_parameters.append(parameter)
        else:
            backbone_parameters.append(parameter)

    if not backbone_parameters:
        raise RuntimeError(
            "ResNet18 backbone parametreleri bulunamadı."
        )

    if not classifier_parameters:
        raise RuntimeError(
            "ResNet18 classifier parametreleri bulunamadı."
        )

    return (
        backbone_parameters,
        classifier_parameters,
    )


if __name__ == "__main__":
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model = create_resnet18_transfer(
        num_classes=38,
        dropout_rate=0.3,
        pretrained=True,
    ).to(device)

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

    total_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
    )
    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    (
        backbone_parameters,
        classifier_parameters,
    ) = split_resnet18_parameters(model)
    backbone_parameter_count = sum(
        parameter.numel()
        for parameter in backbone_parameters
    )
    classifier_parameter_count = sum(
        parameter.numel()
        for parameter in classifier_parameters
    )

    print("=" * 60)
    print("RESNET18 TRANSFER LEARNING KONTROLÜ")
    print("=" * 60)
    print("Cihaz:", device)
    print(
        "Ön-eğitimli ağırlıklar:",
        "ResNet18_Weights.DEFAULT",
    )
    print("Girdi şekli:", dummy_images.shape)
    print("Çıktı şekli:", outputs.shape)
    print(
        "Toplam parametre:",
        f"{total_parameters:,}",
    )
    print(
        "Eğitilebilir parametre:",
        f"{trainable_parameters:,}",
    )
    print(
        "Backbone parametresi:",
        f"{backbone_parameter_count:,}",
    )
    print(
        "Classifier parametresi:",
        f"{classifier_parameter_count:,}",
    )
    print("Dropout:", 0.3)
    print("Tüm katmanlar fine-tuning için açık.")
