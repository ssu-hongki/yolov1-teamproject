import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torch.utils.model_zoo as model_zoo

resnet50_url = "https://download.pytorch.org/models/resnet50-19c8e357.pth"


##############################################
# 3x3 Conv helper
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
# DetNet block (dilated C5)
##############################################
class DetNet(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, block_type="A"):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(
            planes, planes,
            kernel_size=3, stride=stride,
            dilation=2, padding=2,
            bias=False
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
        out = out + self.downsample(x)
        return F.relu(out)


##############################################
# Main Model: ResNet + DetNet + fusion + head
##############################################
class ResNet(nn.Module):
    def __init__(self, block, layers):
        super().__init__()
        self.in_planes = 64

        # Stem
        self.conv1 = nn.Conv2d(
            3, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, 2, 1)

        # Backbone
        self.layer1 = self._make_layer(block, 64, layers[0])        # 112 → 112
        self.layer2 = self._make_layer(block, 128, layers[1], 2)    # 112 → 56
        self.layer3 = self._make_layer(block, 256, layers[2], 2)    # 56 → 28
        self.layer4 = self._make_layer(block, 512, layers[3], 2)    # 28 → 14

        # DetNet C5 유지
        self.layer5 = self._make_detnet_layer(2048)                # 14×14 → 256ch

        # -------- Fusion 개선 (C3 + DetNet C5) --------
        self.fuse_reduce = nn.Conv2d(1024, 256, kernel_size=1, bias=False)
        self.fuse_bn = nn.BatchNorm2d(256)

        # stride=2 conv downsample (더 좋음)
        self.fuse_down = nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1)

        # concat 후 feature alignment conv
        self.fuse_align = nn.Conv2d(512, 512, 3, padding=1, bias=False)
        self.fuse_align_bn = nn.BatchNorm2d(512)

        # -------- 개선된 Detection Head --------
        self.head_conv1 = nn.Conv2d(512, 512, 3, padding=1, bias=False)
        self.head_bn1 = nn.BatchNorm2d(512)

        self.head_conv2 = nn.Conv2d(512, 256, 3, padding=1, bias=False)
        self.head_bn2 = nn.BatchNorm2d(256)

        # 추가: 256 → 256 conv
        self.head_conv3 = nn.Conv2d(256, 256, 3, padding=1, bias=False)
        self.head_bn3 = nn.BatchNorm2d(256)

        # 추가: dilated conv
        self.head_dilated = nn.Conv2d(256, 256, 3, padding=2, dilation=2, bias=False)
        self.head_bn_dil = nn.BatchNorm2d(256)

        # final head
        self.conv_end = nn.Conv2d(256, 30, kernel_size=1, bias=False)
        self.bn_end = nn.BatchNorm2d(30)

        # Init
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    ##############################################
    # ResNet layer maker
    ##############################################
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None

        if stride != 1 or self.in_planes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.in_planes, planes * block.expansion,
                    kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = [
            block(self.in_planes, planes, stride, downsample)
        ]
        self.in_planes = planes * block.expansion

        for _ in range(1, blocks):
            layers.append(block(self.in_planes, planes))

        return nn.Sequential(*layers)

    ##############################################
    # DetNet
    ##############################################
    def _make_detnet_layer(self, in_channels):
        return nn.Sequential(
            DetNet(in_channels, 256, block_type="B"),
            DetNet(256, 256, block_type="A"),
            DetNet(256, 256, block_type="A"),
        )

    ##############################################
    # Forward
    ##############################################
    def forward(self, x):
        # ---- stem ----
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)

        # ---- backbone ----
        x = self.layer1(x)
        x = self.layer2(x)
        c3 = self.layer3(x)  # 28×28, 1024ch
        x = self.layer4(c3)  # 14×14, 2048ch

        # ---- DetNet C5 ----
        c5 = self.layer5(x)  # (B,256,14,14)

        # ---- Fusion ----
        p3 = F.relu(self.fuse_bn(self.fuse_reduce(c3)))
        p3 = self.fuse_down(p3)  # 28→14

        x = torch.cat([c5, p3], dim=1)  # (B,512,14,14)
        x = F.relu(self.fuse_align_bn(self.fuse_align(x)))

        # ---- Detection head ----
        x = F.relu(self.head_bn1(self.head_conv1(x)))
        x = F.relu(self.head_bn2(self.head_conv2(x)))
        x = F.relu(self.head_bn3(self.head_conv3(x)))
        x = F.relu(self.head_bn_dil(self.head_dilated(x)))

        x = torch.sigmoid(self.bn_end(self.conv_end(x)))

        return x.permute(0, 2, 3, 1)


##############################################
# model factory
##############################################
def resnet50(pretrained=False, **kwargs):
    model = ResNet(Bottleneck, [3, 4, 6, 3])

    if pretrained:
        state_dict = model_zoo.load_url(resnet50_url)
        model_dict = model.state_dict()

        for k, v in state_dict.items():
            if k in model_dict and not k.startswith("fc"):
                model_dict[k] = v

        model.load_state_dict(model_dict)

    return model


if __name__ == "__main__":
    m = resnet50()
    a = torch.randn(1, 3, 448, 448)
    print(m(a).shape)