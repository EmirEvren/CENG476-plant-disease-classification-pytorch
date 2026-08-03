import torch
from torch import nn
from torchvision.models import (
    EfficientNet_B0_Weights,
    efficientnet_b0,
)


def create_efficientnet_b0_transfer(
    num_classes=38,
    dropout_rate=0.3,
    pretrained=True,
):
    weights = (
        EfficientNet_B0_Weights.DEFAULT
        if pretrained
        else None
    )
    model = efficientnet_b0(weights=weights)
    input_features = (
        model.classifier[1].in_features
    )

    model.classifier = nn.Sequential(
        nn.Dropout(
            p=dropout_rate,
            inplace=True,
        ),
        nn.Linear(
            input_features,
            num_classes,
        ),
    )

    return model


def split_efficientnet_b0_parameters(model):
    backbone_parameters = []
    classifier_parameters = []

    for parameter_name, parameter in (
        model.named_parameters()
    ):
        if parameter_name.startswith(
            "classifier."
        ):
            classifier_parameters.append(parameter)
        else:
            backbone_parameters.append(parameter)

    if not backbone_parameters:
        raise RuntimeError(
            "EfficientNet-B0 backbone parametreleri "
            "bulunamadı."
        )

    if not classifier_parameters:
        raise RuntimeError(
            "EfficientNet-B0 classifier parametreleri "
            "bulunamadı."
        )

    return (
        backbone_parameters,
        classifier_parameters,
    )


if __name__ == "__main__":
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model = create_efficientnet_b0_transfer(
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
    ) = split_efficientnet_b0_parameters(model)
    backbone_parameter_count = sum(
        parameter.numel()
        for parameter in backbone_parameters
    )
    classifier_parameter_count = sum(
        parameter.numel()
        for parameter in classifier_parameters
    )

    print("=" * 60)
    print("EFFICIENTNET-B0 TRANSFER LEARNING KONTROLÜ")
    print("=" * 60)
    print("Cihaz:", device)
    print(
        "Ön-eğitimli ağırlıklar:",
        "EfficientNet_B0_Weights.DEFAULT",
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
