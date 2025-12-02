import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torch.utils.model_zoo as model_zoo

resnet50_url = "https://download.pytorch.org/models/resnet50-19c8e357.pth"


##############################################
# 3×3 Conv helper
##############################################
def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(
        in_planes, out_planes, 3,
        stride=stride, padding=1, bias=False
    )


##############################################
# BasicBlock
##############################################
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        idt = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        if self.downsample is not None:
            idt = self.downsample(x)

        return self.relu(out + idt)


##############################################
# Bottleneck
##############################################
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=stride,
            padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)

        self.conv3 = nn.Conv2d(planes, planes * 4, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        idt = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            idt = self.downsample(x)

        return self.relu(out + idt)


##############################################
# DetNet (C5 유지)
##############################################
class DetNet(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, block_type="A"):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(
            planes, planes,
            3, stride=stride,
            padding=2, dilation=2, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)

        self.conv3 = nn.Conv2d(planes, planes, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)

        if stride != 1 or in_planes != planes or block_type == "B":
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, planes, 1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )
        else:
            self.downsample = nn.Identity()

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        return F.relu(out + self.downsample(x))


##############################################
# Full FPN + Head
##############################################
class ResNet(nn.Module):
    def __init__(self, block, layers):
        super().__init__()
        self.in_planes = 64

        # Stem
        self.conv1 = nn.Conv2d(
            3, 64, 7, stride=2, padding=3, bias=False
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, 2, 1)

        # Backbone
        self.layer1 = self._make_layer(block, 64, layers[0])      
        self.layer2 = self._make_layer(block, 128, layers[1], 2)
        self.layer3 = self._make_layer(block, 256, layers[2], 2)  
        self.layer4 = self._make_layer(block, 512, layers[3], 2)

        # DetNet C5
        self.layer5 = self._make_detnet_layer(2048)

        # -----------------------------
        # FPN FULL
        # -----------------------------
        # reduce channels for P3/P4/P5
        self.c3_reduce = nn.Conv2d(1024, 256, 1, bias=False)
        self.c4_reduce = nn.Conv2d(2048, 256, 1, bias=False)
        self.c5_reduce = nn.Conv2d(256, 256, 1, bias=False)

        # smoothing after add
        self.smooth3 = nn.Conv2d(256, 256, 3, padding=1)
        self.smooth4 = nn.Conv2d(256, 256, 3, padding=1)
        self.smooth5 = nn.Conv2d(256, 256, 3, padding=1)

        # downsample P3 → 14×14
        self.p3_down = nn.Conv2d(256, 256, 3, stride=2, padding=1)

        # -----------------------------
        # Detection Head
        # -----------------------------
        self.head_conv1 = nn.Conv2d(512, 512, 3, padding=1)
        self.head_bn1 = nn.BatchNorm2d(512)

        self.head_conv2 = nn.Conv2d(512, 256, 3, padding=1)
        self.head_bn2 = nn.BatchNorm2d(256)

        self.head_dil = nn.Conv2d(256, 256, 3, padding=2, dilation=2)
        self.head_bn3 = nn.BatchNorm2d(256)

        self.conv_end = nn.Conv2d(256, 30, 1)
        self.bn_end = nn.BatchNorm2d(30)

    ##################################
    # Layer builder
    ##################################
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None

        if stride != 1 or self.in_planes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.in_planes, planes * block.expansion,
                    1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(planes * block.expansion)
            )

        layers = [
            block(self.in_planes, planes, stride, downsample)
        ]
        self.in_planes = planes * block.expansion

        for _ in range(1, blocks):
            layers.append(block(self.in_planes, planes))

        return nn.Sequential(*layers)

    ##################################
    # DetNet
    ##################################
    def _make_detnet_layer(self, in_channels):
        return nn.Sequential(
            DetNet(in_channels, 256, block_type="B"),
            DetNet(256, 256, block_type="A"),
            DetNet(256, 256, block_type="A")
        )

    ##################################
    # Forward
    ##################################
    def forward(self, x):
        # Stem
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)

        # Backbone
        c3 = self.layer3(self.layer2(self.layer1(x)))   # 28×28, 1024ch
        c4 = self.layer4(c3)                            # 14×14, 2048ch
        c5 = self.layer5(c4)                            # 14×14, 256ch

        # FPN
        p5 = self.smooth5(self.c5_reduce(c5))

        p4 = self.c4_reduce(c4) + F.interpolate(p5, size=c4.shape[-2:], mode='nearest')
        p4 = self.smooth4(p4)

        p3 = self.c3_reduce(c3) + F.interpolate(p4, size=c3.shape[-2:], mode='nearest')
        p3 = self.smooth3(p3)

        # P3 → 14×14
        p3_down = self.p3_down(p3)

        # concat P5 + p3_down
        out = torch.cat([p5, p3_down], dim=1)  # (B, 512, 14, 14)

        # HEAD
        out = F.relu(self.head_bn1(self.head_conv1(out)))
        out = F.relu(self.head_bn2(self.head_conv2(out)))
        out = F.relu(self.head_bn3(self.head_dil(out)))

        out = torch.sigmoid(self.bn_end(self.conv_end(out)))

        return out.permute(0, 2, 3, 1)


##############################################
# resnet50 entry
##############################################
def resnet50(pretrained=True, **kwargs):
    model = ResNet(Bottleneck, [3,4,6,3])

    if pretrained:
        state_dict = model_zoo.load_url(resnet50_url)
        model_dict = model.state_dict()

        for k, v in state_dict.items():
            if k in model_dict and not k.startswith("fc"):
                model_dict[k] = v

        model.load_state_dict(model_dict, strict=False)

    return model