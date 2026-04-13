import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class GaussianPatchCoreBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

        self.conv1 = nn.Conv2d(
            8, 64, kernel_size=7, stride=2, padding=3, bias=False
        )

        # RGB 通道沿用预训练
        self.conv1.weight.data[:, :3] = resnet.conv1.weight.data

        # geometry 通道用 RGB 卷积核均值初始化，而不是 0
        rgb_mean_kernel = resnet.conv1.weight.data.mean(dim=1, keepdim=True)  # [64,1,7,7]
        self.conv1.weight.data[:, 3:] = rgb_mean_kernel.repeat(1, 5, 1, 1)

        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3

        # PatchCore 常用的局部聚合
        self.local_agg = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        f2 = self.layer2(x)
        f3 = self.layer3(f2)

        f3 = F.interpolate(f3, size=f2.shape[-2:], mode="bilinear", align_corners=False)
        feat = torch.cat([f2, f3], dim=1)
        feat = self.local_agg(feat)
        return feat