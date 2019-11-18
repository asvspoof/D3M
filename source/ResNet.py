import torch

from torch import nn


def conv3x3(in_channels, out_channels, stride=1, groups=1, dilation=1):
    """3x3 conv with padding"""
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=dilation, groups=groups,
                     bias=False, dilation=dilation)


def conv1x1(in_channels, out_channels, stride=1):
    """1x1 conv"""
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)


class Residual_Basicblock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None, groups=1,
                 dilation=1, norm_layer=None):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1:
            raise ValueError('BasicBock only supports groups=1')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1

        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = norm_layer(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = norm_layer(out_channels)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(identity)

        out += identity
        out = self.relu(out)

        return out


# implementation of SENet
class SELayer(nn.Module):
    def __init__(self, channel, reductratio=16):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d((1, 1))
        self.excitation = nn.Sequential(
            nn.Linear(channel, channel // reductratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reductratio, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        z = self.squeeze(x).squeeze()
        s = self.excitation(z).view(b, c, 1, 1)
        return x * s


class SEBasicblock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None, groups=1,
                 dilation=1, norm_layer=None, reductratio=8):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1:
            raise ValueError('BasicBock only supports groups=1')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1

        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = norm_layer(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = norm_layer(out_channels)
        self.se = SELayer(out_channels, reductratio)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)

        if self.downsample is not None:
            identity = self.downsample(identity)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes, zero_init_residual=False,
                 groups=1, replace_stride_with_dilation=None, norm_layer=None):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.input_channels = 1
        self.in_channels = 16
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]

        if len(replace_stride_with_dilation) != 3:
            raise ValueError('replace_stride_with_dilation should be None'
                             'or a 3-element tuple, got {}'.format(replace_stride_with_dilation))
        self.groups = groups
        self.conv1 = nn.Conv2d(self.input_channels, self.in_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = norm_layer(self.in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 16, layers[0])
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 128, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        # Global average pooling
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # x = nn.avg_pool2d(x, x.size()[2:]) or torch.mean , all of them are ok
        self.fc = nn.Sequential(
            nn.Linear(128 * block.expansion, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes)
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Residual_Basicblock):
                    nn.init.constant_(m.bn2.weight, 0)
                # elif isinstance(m, BasicBlock):
                #     nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, out_channels, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.in_channels, out_channels * block.expansion, stride),
                norm_layer(out_channels * block.expansion)
            )
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample, self.groups,
                            previous_dilation, norm_layer))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels,
                                groups=self.groups, dilation=self.dilation, norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        # x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)

        # LL = nn.LogSoftmax(dim=1)(x)
        # score = LL[:, 1]-LL[:, 0]
        return x


def _resnet(arch, block, layers, pretrained, progress=True, **kwargs):
    model = ResNet(block, layers, **kwargs)
    if pretrained:
        pass
    # state_dict = load_state_dict_from_url(model_urls[arch],
    #                                        progress = progress)
    # model.load_state_dict(state_dict)
    return model


def DKU_ResNet(num_classes):
    return _resnet('DKU_ResNet', Residual_Basicblock, [3, 4, 6, 3], pretrained=False, num_classes=num_classes)


def DKU_SEResNet(num_classes):
    return _resnet('DKU_SEResNet', SEBasicblock, [3, 4, 6, 3], pretrained=False, num_classes=num_classes)
