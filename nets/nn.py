import torch
import torch.nn as nn
import torch.nn.functional as F

###############################################
# 1. LKA Attention (안정화 버전)
###############################################
class LKA(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dw = nn.Conv2d(dim, dim, kernel_size=5, padding=2, groups=dim)
        self.dilated = nn.Conv2d(dim, dim, kernel_size=7, padding=9, dilation=3, groups=dim)
        self.pw = nn.Conv2d(dim, dim, kernel_size=1)

    def forward(self, x):
        attn = self.dw(x)
        attn = self.dilated(attn)
        attn = self.pw(attn)
        return x * attn


###############################################
# 2. 기본 ResNet Bottleneck 블록
###############################################
class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(planes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.conv3 = nn.Conv2d(planes, planes * 4, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        idn = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            idn = self.downsample(x)

        out += idn
        return self.relu(out)


###############################################
# 3. DetNet Neck 블록 (원본 유지)
###############################################
class DetNet(nn.Module):
    def __init__(self, in_planes, planes, block_type='A'):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(
            planes, planes,
            3, padding=2, dilation=2, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)

        self.conv3 = nn.Conv2d(planes, planes, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)

        if block_type == 'B':
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, planes, 1, bias=False),
                nn.BatchNorm2d(planes)
            )
        else:
            self.downsample = nn.Identity()

    def forward(self, x):
        idn = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.downsample(idn)
        return F.relu(out)


###############################################
# 4. YOLO Head (NaN 방지 버전)
###############################################
class YOLOHead(nn.Module):
    def __init__(self, in_channels, out_channels=30):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 256, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(256)

        self.conv2 = nn.Conv2d(256, 256, 3, padding=2, dilation=2)
        self.bn2 = nn.BatchNorm2d(256)

        self.conv3 = nn.Conv2d(256, out_channels, 1)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.conv3(x)   # raw output (activation은 나중에 손실에서)
        return x


###############################################
# 5. Final Model
###############################################
class ResNet_LKA(nn.Module):
    def __init__(self, block, layers):
        super().__init__()
        self.in_planes = 64

        # stem
        self.conv1 = nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)

        # backbone + LKA
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.lka1 = LKA(256)

        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.lka2 = LKA(512)

        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.lka3 = LKA(1024)

        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.lka4 = LKA(2048)

        # neck
        self.detnet = nn.Sequential(
            DetNet(2048, 256, block_type='B'),
            DetNet(256, 256),
            DetNet(256, 256),
        )

        # head
        self.head = YOLOHead(256, out_channels=30)

    def _make_layer(self, block, planes, num_blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_planes, planes * block.expansion, 1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion)
            )

        layers = [block(self.in_planes, planes, stride, downsample)]
        self.in_planes = planes * block.expansion

        for _ in range(1, num_blocks):
            layers.append(block(self.in_planes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))

        x = self.layer1(x); x = self.lka1(x)
        x = self.layer2(x); x = self.lka2(x)
        x = self.layer3(x); x = self.lka3(x)
        x = self.layer4(x); x = self.lka4(x)

        x = self.detnet(x)
        x = self.head(x)

        return x.permute(0, 2, 3, 1)


###############################################
def resnet50(pretrained=False, **kwargs):
    return ResNet_LKA(Bottleneck, [3,4,6,3])