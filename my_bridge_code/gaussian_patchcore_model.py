import torch.nn as nn
import torchvision.models as models


class GaussianPatchCoreBackbone(nn.Module):

    def __init__(self):

        super().__init__()

        resnet = models.resnet18(pretrained=True)

        self.conv1 = nn.Conv2d(
            8,
            64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False
        )

        self.conv1.weight.data[:, :3] = resnet.conv1.weight.data
        self.conv1.weight.data[:, 3:] = 0

        self.encoder = nn.Sequential(
            self.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3
        )

    def forward(self, x):

        return self.encoder(x)