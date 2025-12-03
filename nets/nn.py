import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torch.utils.model_zoo as model_zoo

# -------------------------------------------------------------
#  resnet50 wrapper
# -------------------------------------------------------------
resnet50_url = "https://download.pytorch.org/models/resnet50-19c8e357.pth"


# -------------------------------------------------------------
#  기본 함수: 3x3 Convolution
# -------------------------------------------------------------
def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes, out_planes,
        kernel_size=3, stride=stride,
        padding=1, bias=False
    )


# -------------------------------------------------------------
#  ResNet BasicBlock
# -------------------------------------------------------------
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
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        if self.downsample is not None:
            residual = self.downsample(x)

        out = self.relu(out + residual)
        return out


# -------------------------------------------------------------
#  ResNet Bottleneck
# -------------------------------------------------------------
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()

        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=stride,
            padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)

        self.conv3 = nn.Conv2d(
            planes, planes * 4, kernel_size=1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(planes * 4)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            residual = self.downsample(x)

        out = self.relu(out + residual)
        return out


# -------------------------------------------------------------
#  DetNet block (C5 dilation 유지)
# -------------------------------------------------------------
class DetNet(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, block_type="A"):
        super().__init__()

        # 1x1
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        # 3x3 dilated conv
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3,
            stride=stride, padding=2, dilation=2,
            bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)

        # 1x1
        self.conv3 = nn.Conv2d(
            planes, planes * self.expansion,
            kernel_size=1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)

        # Downsample if needed
        self.downsample = nn.Sequential()
        if stride != 1 or in_planes != planes or block_type == "B":
            self.downsample = nn.Sequential(
                nn.Conv2d(
                    in_planes, planes * self.expansion,
                    kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(planes * self.expansion)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        out = F.relu(out + self.downsample(x))
        return out


# -------------------------------------------------------------
#  ResNet + DetNet + FPN-style Multi-layer Fusion (C2, C3, C5)
# -------------------------------------------------------------
class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000):
        super().__init__()

        self.in_planes = 64

        # -------- Stem --------
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7,
                               stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # -------- Backbone --------
        # conv2_x
        self.layer1 = self._make_layer(block, 64, layers[0])         # 112×112
        # conv3_x
        self.layer2 = self._make_layer(block, 128, layers[1], 2)     # 56×56 (C2)
        # conv4_x
        self.layer3 = self._make_layer(block, 256, layers[2], 2)     # 28×28 (C3)
        # conv5_x
        self.layer4 = self._make_layer(block, 512, layers[3], 2)     # 14×14

        # -------- DetNet C5 --------
        self.layer5 = self._make_detnet_layer(2048)                  # 14×14 → (B,256,14,14)

        # -------- FPN-style Fusion: C3(28x28), C2(56x56) --------
        # C3: 1024ch @ 28x28 → 256ch @ 14x14
        self.fuse3_reduce = nn.Conv2d(1024, 256, kernel_size=1, bias=False)
        self.fuse3_bn = nn.BatchNorm2d(256)
        self.fuse3_pool = nn.MaxPool2d(2, 2)  # 28 -> 14

        # C2: 512ch @ 56x56 → 256ch @ 14x14
        self.fuse2_reduce = nn.Conv2d(512, 256, kernel_size=1, bias=False)
        self.fuse2_bn = nn.BatchNorm2d(256)
        self.fuse2_pool1 = nn.MaxPool2d(2, 2)  # 56 -> 28
        self.fuse2_pool2 = nn.MaxPool2d(2, 2)  # 28 -> 14

        # -------- Detection Head --------
        # 입력 채널: 256(C5) + 256(P3) + 256(P2) = 768
        self.head_conv1 = nn.Conv2d(768, 512, kernel_size=3, padding=1, bias=False)
        self.head_bn1 = nn.BatchNorm2d(512)

        self.head_conv2 = nn.Conv2d(512, 256, kernel_size=3, padding=1, bias=False)
        self.head_bn2 = nn.BatchNorm2d(256)

        self.conv_end = nn.Conv2d(256, 30, kernel_size=1, bias=False)
        self.bn_end = nn.BatchNorm2d(30)

        # -------- Weight Initialization --------
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1.0)
                m.bias.data.zero_()

    # ---------------------------------------------------------
    # Make ResNet Layer
    # ---------------------------------------------------------
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None

        if stride != 1 or self.in_planes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.in_planes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = [block(self.in_planes, planes, stride, downsample)]
        self.in_planes = planes * block.expansion

        for _ in range(1, blocks):
            layers.append(block(self.in_planes, planes))

        return nn.Sequential(*layers)

    # ---------------------------------------------------------
    # Make DetNet Layer
    # ---------------------------------------------------------
    def _make_detnet_layer(self, in_channels):
        return nn.Sequential(
            DetNet(in_channels, 256, block_type="B"),
            DetNet(256, 256, block_type="A"),
            DetNet(256, 256, block_type="A"),
        )

    # ---------------------------------------------------------
    # Forward Pass
    # ---------------------------------------------------------
    def forward(self, x):
        # Stem
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)

        # Backbone
        x = self.layer1(x)
        c2 = self.layer2(x)    # 56x56, 512ch
        c3 = self.layer3(c2)   # 28x28, 1024ch
        x = self.layer4(c3)    # 14x14, 2048ch

        # DetNet C5
        c5 = self.layer5(x)    # 14x14, 256ch

        # ---- FPN-style small-object fusion ----
        # from C3
        p3 = F.relu(self.fuse3_bn(self.fuse3_reduce(c3)))  # (B,256,28,28)
        p3 = self.fuse3_pool(p3)                           # (B,256,14,14)

        # from C2
        p2 = F.relu(self.fuse2_bn(self.fuse2_reduce(c2)))  # (B,256,56,56)
        p2 = self.fuse2_pool1(p2)                          # 56 -> 28
        p2 = self.fuse2_pool2(p2)                          # 28 -> 14

        # concat C5 + P3 + P2
        x = torch.cat([c5, p3, p2], dim=1)                 # (B,768,14,14)

        # ---- Detection head ----
        x = F.relu(self.head_bn1(self.head_conv1(x)))      # (B,512,14,14)
        x = F.relu(self.head_bn2(self.head_conv2(x)))      # (B,256,14,14)

        x = torch.sigmoid(self.bn_end(self.conv_end(x)))   # (B,30,14,14)

        # (B, 30, 14, 14) → (B, 14, 14, 30)
        return x.permute(0, 2, 3, 1)


def resnet50(pretrained=False, **kwargs):
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)

    if pretrained:
        state_dict = model_zoo.load_url(resnet50_url)
        model_dict = model.state_dict()

        # backbone(conv1, layer1~4)만 ImageNet 프리트레인 로딩
        for k, v in state_dict.items():
            if k in model_dict and not k.startswith("fc"):
                model_dict[k] = v

        model.load_state_dict(model_dict)

    return model


# -------------------------------------------------------------
#  Test
# -------------------------------------------------------------
if __name__ == "__main__":
    a = torch.randn(2, 3, 448, 448)
    m = resnet50()
    out = m(a)
    print(out.shape)     # expected: (2, 14, 14, 30)
