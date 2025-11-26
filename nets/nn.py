import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50


# -------------------------------------------------------
# 1) SPP Block (Spatial Pyramid Pooling)
# -------------------------------------------------------
class SPP(nn.Module):
    def __init__(self):
        super(SPP, self).__init__()
        self.pool1 = nn.MaxPool2d(kernel_size=5, stride=1, padding=2)
        self.pool2 = nn.MaxPool2d(kernel_size=9, stride=1, padding=4)
        self.pool3 = nn.MaxPool2d(kernel_size=13, stride=1, padding=6)

    def forward(self, x):
        p1 = self.pool1(x)
        p2 = self.pool2(x)
        p3 = self.pool3(x)
        return torch.cat([x, p1, p2, p3], dim=1)  # channel concat


# -------------------------------------------------------
# 2) YOLO Head
# -------------------------------------------------------
class YOLOHead(nn.Module):
    def __init__(self):
        super(YOLOHead, self).__init__()
        self.conv1 = nn.Conv2d(2048*4, 1024, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(1024)

        self.conv2 = nn.Conv2d(1024, 512, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(512)

        self.conv3 = nn.Conv2d(512, 30, kernel_size=1)  # 2 box * 5 + 20 class = 30

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.conv3(x)
        x = F.adaptive_avg_pool2d(x, (14, 14))
        return x.permute(0, 2, 3, 1)  # (batch, 14, 14, 30)


# -------------------------------------------------------
# 3) ResNet50 Backbone (stride reduced)
# -------------------------------------------------------
class ResNetBackbone(nn.Module):
    def __init__(self):
        super(ResNetBackbone, self).__init__()
        net = resnet50(weights='IMAGENET1K_V1')

        # ---- Modify stride for small object performance ----
        # conv1: stride 2 → 1
        net.conv1.stride = (1, 1)

        # layer3: first block stride 2 → 1
        net.layer3[0].conv1.stride = (1, 1)
        net.layer3[0].downsample[0].stride = (1, 1)

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

        self.spp = SPP()

    def forward(self, x):
        x = self.backbone(x)  # (batch, 2048, H, W)
        x = self.spp(x)       # (batch, 2048*4, H, W)
        return x


# -------------------------------------------------------
# 4) Full YOLOv1 Model
# -------------------------------------------------------
class resnet50_yolo(nn.Module):
    def __init__(self):
        super(resnet50_yolo, self).__init__()
        self.backbone = ResNetBackbone()
        self.head = YOLOHead()

    def forward(self, x):
        x = self.backbone(x)
        x = self.head(x)
        return x


# alias to match existing import
def resnet50():
    return resnet50_yolo()