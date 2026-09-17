import torch
import torch.nn as nn
import math

def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1, norm_layer=None):
        padding = (kernel_size - 1) // 2
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False),
            norm_layer(out_planes),
            nn.ReLU6(inplace=True)
        )


class SqueezeExcitation(nn.Module):
    def __init__(self, input_channels, squeeze_channels, activation=nn.ReLU, scale_activation=nn.Hardsigmoid):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(input_channels, squeeze_channels, 1)
        self.fc2 = nn.Conv2d(squeeze_channels, input_channels, 1)
        self.activation = activation()
        self.scale_activation = scale_activation()

    def forward(self, x):
        scale = self.avgpool(x)
        scale = self.fc1(scale)
        scale = self.activation(scale)
        scale = self.fc2(scale)
        return x * self.scale_activation(scale)


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio, kernel_size, skip_connection, norm_layer=None, se_layer=None):
        super(InvertedResidual, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = skip_connection and stride == 1 and inp == oup

        layers = []
        if expand_ratio != 1:
            # pw
            layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1, norm_layer=norm_layer))
        layers.extend([
            # dw
            ConvBNReLU(hidden_dim, hidden_dim, kernel_size=kernel_size, stride=stride, groups=hidden_dim, norm_layer=norm_layer),
        ])
        if se_layer:
            layers.append(se_layer(hidden_dim, _make_divisible(inp // 4, 8)))
        layers.extend([
            # pw-linear
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            norm_layer(oup),
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class EfficientNet(nn.Module):
    def __init__(self, num_classes=10, width_mult=1.0, depth_mult=1.0, inverted_residual_setting=None, round_nearest=8, block=None, norm_layer=None, se_layer=None):
        super(EfficientNet, self).__init__()

        if block is None:
            block = InvertedResidual

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        if se_layer is None:
            se_layer = SqueezeExcitation

        input_channel = 32
        last_channel = 1280

        if inverted_residual_setting is None:
            inverted_residual_setting = [
                # t, c, n, s, k
                [1, 16, 1, 1, 3],  # 0
                [6, 24, 2, 2, 3],  # 1
                [6, 40, 2, 2, 5],  # 2
                [6, 80, 3, 2, 3],  # 3
                [6, 112, 3, 1, 5], # 4
                [6, 192, 4, 2, 5], # 5
                [6, 320, 1, 1, 3], # 6
            ]

        # building first layer
        input_channel = _make_divisible(input_channel * width_mult, round_nearest)
        self.last_channel = _make_divisible(last_channel * width_mult, round_nearest)
        features = [ConvBNReLU(3, input_channel, stride=2, norm_layer=norm_layer)]

        # building inverted residual blocks
        for t, c, n, s, k in inverted_residual_setting:
            output_channel = _make_divisible(c * width_mult, round_nearest)
            num_blocks = int(math.ceil(n * depth_mult))
            for i in range(num_blocks):
                stride = s if i == 0 else 1
                features.append(block(input_channel, output_channel, stride, expand_ratio=t, kernel_size=k, skip_connection=True, norm_layer=norm_layer, se_layer=se_layer))
                input_channel = output_channel

        # building last several layers
        features.append(ConvBNReLU(input_channel, self.last_channel, kernel_size=1, norm_layer=norm_layer))
        # make it nn.Sequential
        self.features = nn.Sequential(*features)

        # building classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.last_channel, num_classes),
        )

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def _forward_impl(self, x):
        x = self.features(x)
        x = x.mean([2, 3])
        x = self.classifier(x)
        return x

    def forward(self, x):
        return self._forward_impl(x)

def efficientnet_b0(pretrained=False, progress=True, **kwargs):
    return EfficientNet(width_mult=1.0, depth_mult=1.0, **kwargs)

def efficientnet_b1(pretrained=False, progress=True, **kwargs):
    return EfficientNet(width_mult=1.0, depth_mult=1.1, **kwargs)

def efficientnet_b2(pretrained=False, progress=True, **kwargs):
    return EfficientNet(width_mult=1.1, depth_mult=1.2, **kwargs)

def efficientnet_b3(pretrained=False, progress=True, **kwargs):
    return EfficientNet(width_mult=1.2, depth_mult=1.4, **kwargs)

def efficientnet_b4(pretrained=False, progress=True, **kwargs):
    return EfficientNet(width_mult=1.4, depth_mult=1.8, **kwargs)

def efficientnet_b5(pretrained=False, progress=True, **kwargs):
    return EfficientNet(width_mult=1.6, depth_mult=2.2, **kwargs)

def efficientnet_b6(pretrained=False, progress=True, **kwargs):
    return EfficientNet(width_mult=1.8, depth_mult=2.6, **kwargs)

def efficientnet_b7(pretrained=False, progress=True, **kwargs):
    return EfficientNet(width_mult=2.0, depth_mult=3.1, **kwargs)


