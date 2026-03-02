import torch
import torch.nn as nn
import torchvision.models as models


class ResNetFeatureExtractor(nn.Module):
    """
    返回 stride=8 的特征图（layer3 输出），适合 PatchCore。
    out_channels 默认为 1024（resnet50 layer3）。
    """
    def __init__(self, name="resnet50", pretrained=True, out_layer="layer3"):
        super().__init__()
        assert name in ["resnet18", "resnet34", "resnet50", "resnet101"], "Unsupported resnet name"
        assert out_layer in ["layer2", "layer3", "layer4"], "out_layer must be layer2/layer3/layer4"

        net = getattr(models, name)(weights="DEFAULT" if pretrained else None)

        self.stem = nn.Sequential(
            net.conv1, net.bn1, net.relu, net.maxpool
        )
        self.layer1 = net.layer1
        self.layer2 = net.layer2
        self.layer3 = net.layer3
        self.layer4 = net.layer4

        self.out_layer = out_layer

        # 输出通道数
        if name in ["resnet18", "resnet34"]:
            ch = {"layer2": 128, "layer3": 256, "layer4": 512}
        else:
            ch = {"layer2": 512, "layer3": 1024, "layer4": 2048}
        self.out_channels = ch[out_layer]

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        if self.out_layer == "layer2":
            return x
        x = self.layer3(x)
        if self.out_layer == "layer3":
            return x
        x = self.layer4(x)
        return x