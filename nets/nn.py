import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo

# ResNet50 weight URL (공식 ImageNet)
RESNET50_URL = 'https://download.pytorch.org/models/resnet50-19c8e357.pth'


# ---------------------------
# Basic Bottleneck Block
# ---------------------------
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        return self.relu(out)


# ---------------------------
# SPP Module
# ---------------------------
class SPP(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.pool1 = nn.MaxPool2d(5, 1, 2)
        self.pool2 = nn.MaxPool2d(9, 1, 4)
        self.pool3 = nn.MaxPool2d(13, 1, 6)

        self.reduce = nn.Conv2d(in_channels, 512, kernel_size=1)

    def forward(self, x):
        x = self.reduce(x)
        p1 = self.pool1(x)
        p2 = self.pool2(x)
        p3 = self.pool3(x)
        return torch.cat([x, p1, p2, p3], 1)    # 512*4 = 2048 채널


# ---------------------------
# YOLO Head
# ---------------------------
class YOLOHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(2048, 1024, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(1024)

        self.conv2 = nn.Conv2d(1024, 512, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(512)

        self.out = nn.Conv2d(512, 30, 1)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.out(x)

        # YOLO output: 14×14×30 고정
        x = F.adaptive_avg_pool2d(x, (14, 14))

        return x.permute(0, 2, 3, 1)    # (B, 14, 14, 30)


# ---------------------------
# Full ResNet Backbone + SPP + YOLO Head
# ---------------------------
class ResNet50_YOLO(nn.Module):
    def __init__(self):
        super().__init__()
        self.inplanes = 64

        # ======== conv1 stride=1 로 변경 (small object 개선) ========
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=1, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # ResNet layers
        self.layer1 = self._make_layer(Bottleneck, 64, 3)
        self.layer2 = self._make_layer(Bottleneck, 128, 4, stride=2)
        self.layer3 = self._make_layer(Bottleneck, 256, 6, stride=2)
        self.layer4 = self._make_layer(Bottleneck, 512, 3, stride=2)

        # SPP
        self.spp = SPP(in_channels=2048)

        # Detection head
        self.head = YOLOHead()

        # ------------- Pretrained weight load -------------
        self._load_pretrained_resnet()


    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None

        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion)
            )

        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion

        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    # =====================================================
    # Pretrained ResNet50 weights 로딩 (구버전 호환)
    # =====================================================
    def _load_pretrained_resnet(self):
        print("Loading pretrained ResNet50 weights (model_zoo)...")
        pretrained_dict = model_zoo.load_url(RESNET50_URL)

        model_dict = self.state_dict()

        # fc 제외한 부분만 로드
        pretrained_dict = {k: v for k, v in pretrained_dict.items()
                           if k in model_dict and not k.startswith("fc")}

        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict, strict=False)


    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.spp(x)
        x = self.head(x)
        return x


# main.py에서 import하는 이름
def resnet50():
    return ResNet50_YOLO()