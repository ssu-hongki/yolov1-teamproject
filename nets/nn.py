import torch
import torch.nn as nn
import torch.nn.functional as F
import math

##############################################
# 1. CBAM-lite (간단한 채널 + 공간 주의)
##############################################
class CBAMLite(nn.Module):
    def __init__(self, channels):
        super().__init__()
        # 채널 주의: Global Avg Pool + 1x1 Conv
        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.Sigmoid()
        )
        # 공간 주의: 1x1 Conv
        self.spatial_att = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # 채널 attention
        ca = self.channel_att(x)
        x = x * ca

        # 공간 attention
        sa = self.spatial_att(torch.mean(x, dim=1, keepdim=True))
        x = x * sa
        return x


##############################################
# 2. Basic ResNet Bottleneck
##############################################
def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, 3, stride=stride, padding=1, bias=False)

class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(planes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.conv3 = nn.Conv2d(planes, planes*4, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes*4)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        return self.relu(out)


##############################################
# 3. DetNet Neck (교수 기본 코드 유지)
##############################################
class DetNet(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1, block_type='A'):
        super().__init__()

        self.conv1 = nn.Conv2d(in_planes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(
            planes, planes,
            3,
            stride=stride,
            padding=2,
            dilation=2,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(planes)

        self.conv3 = nn.Conv2d(planes, planes, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)

        if stride != 1 or in_planes != planes or block_type == 'B':
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, planes, 1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )
        else:
            self.downsample = nn.Identity()

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.downsample(x)
        return F.relu(out)


##############################################
# 4. Improved YOLO Head
##############################################
class YOLOHead(nn.Module):
    def __init__(self, in_channels, out_channels=30):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, 256, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(256)

        # dilated conv → 작은 객체 정보 강화
        self.dilated = nn.Conv2d(256, 256, 3, padding=2, dilation=2)
        self.bn2 = nn.BatchNorm2d(256)

        self.conv_out = nn.Conv2d(256, out_channels, 1)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.dilated(x)))
        x = torch.sigmoid(self.conv_out(x))
        return x


##############################################
# 5. Final ResNet + CBAM + DetNet + YOLO Head
##############################################
class ResNet_CBAM(nn.Module):
    def __init__(self, block, layers):
        super().__init__()
        self.in_planes = 64

        # stem
        self.conv1 = nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)

        # backbone
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.cbam1 = CBAMLite(256)   # ★ layer1 뒤에 CBAM-lite 추가

        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        # neck (DetNet)
        self.detnet = nn.Sequential(
            DetNet(2048, 256, block_type='B'),
            DetNet(256, 256, block_type='A'),
            DetNet(256, 256, block_type='A'),
        )

        # head
        self.head = YOLOHead(256, 30)

    def _make_layer(self, block, planes, num_blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_planes, planes * block.expansion, 1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = [block(self.in_planes, planes, stride, downsample)]
        self.in_planes = planes * block.expansion
        for _ in range(1, num_blocks):
            layers.append(block(self.in_planes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        # stem
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))

        # backbone + CBAM
        x = self.layer1(x);  x = self.cbam1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # DetNet
        x = self.detnet(x)

        # YOLO head
        x = self.head(x)
        x = x.permute(0, 2, 3, 1)  # (N,14,14,30)
        return x


##############################################
# 6. Entry point for main.py
##############################################
def resnet50(pretrained=True, **kwargs):
    return ResNet_CBAM(Bottleneck, [3, 4, 6, 3])