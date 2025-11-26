import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights


# ---------------------------
# SPP Block (안정 버전)
# ---------------------------
class SPP(nn.Module):
    def __init__(self, in_channels):
        super(SPP, self).__init__()
        self.pool1 = nn.MaxPool2d(kernel_size=5, stride=1, padding=2)
        self.pool2 = nn.MaxPool2d(kernel_size=9, stride=1, padding=4)
        self.pool3 = nn.MaxPool2d(kernel_size=13, stride=1, padding=6)

        # 채널을 줄여주기 위한 1x1 conv
        self.reduce = nn.Conv2d(in_channels, 512, kernel_size=1)

    def forward(self, x):
        x = self.reduce(x)  # → (batch, 512, H, W)

        p1 = self.pool1(x)
        p2 = self.pool2(x)
        p3 = self.pool3(x)

        # concat → (512*4 = 2048 channels)
        return torch.cat([x, p1, p2, p3], dim=1)


# ---------------------------
# YOLO Detection Head
# ---------------------------
class YOLOHead(nn.Module):
    def __init__(self):
        super(YOLOHead, self).__init__()
        self.conv1 = nn.Conv2d(2048, 1024, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(1024)

        self.conv2 = nn.Conv2d(1024, 512, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(512)

        self.conv3 = nn.Conv2d(512, 30, kernel_size=1)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.conv3(x)

        # YOLO는 14x14로 맞춰야 함
        x = F.adaptive_avg_pool2d(x, (14, 14))

        return x.permute(0, 2, 3, 1)


# ---------------------------
# Backbone (ResNet50)
# ---------------------------
class ResNetBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        # pretrained resnet
        net = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)

        # small object 개선: conv1 stride 2 → 1
        net.conv1.stride = (1, 1)

        # layer4 output 채널 2048
        self.backbone = nn.Sequential(
            net.conv1,
            net.bn1,
            net.relu,
            net.maxpool,
            net.layer1,
            net.layer2,
            net.layer3,
            net.layer4,
        )

        # SPP 추가
        self.spp = SPP(in_channels=2048)

    def forward(self, x):
        x = self.backbone(x)
        x = self.spp(x)
        return x


# ---------------------------
# Full YOLOv1 model
# ---------------------------
class resnet50_yolo(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = ResNetBackbone()
        self.head = YOLOHead()

    def forward(self, x):
        x = self.backbone(x)
        x = self.head(x)
        return x


def resnet50():
    return resnet50_yolo()