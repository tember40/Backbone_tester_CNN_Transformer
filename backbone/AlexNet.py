import torch
import torch.nn as nn

class AlexNet(nn.Module):
    def __init__(self, features, num_classes=10, init_weights=True):
        super(AlexNet, self).__init__()
        self.features = features
        # self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.avgpool = nn.AdaptiveAvgPool2d((4, 4))
        # self.classifier = nn.Sequential(
        #     nn.Linear(256 * 6 * 6, 4096),
        self.classifier = nn.Sequential(
            nn.Linear(256 * 4 * 4, 4096),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, num_classes),
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

def make_layers(cfg, batch_norm=False):
    """
    AlexNet 전용 파서.
    항목 형태:
      ('C', out, k, s, p[, groups])  -> Conv2d + (BN) + ReLU
      ('M', k, s)                    -> MaxPool2d
      'LRN'                          -> LocalResponseNorm(size=5, alpha=1e-4, beta=0.75, k=2)
    """
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'LRN':
            layers += [nn.LocalResponseNorm(5, alpha=1e-4, beta=0.75, k=2.0)]
        elif isinstance(v, tuple) and v[0] == 'M':
            _, k, s = v
            layers += [nn.MaxPool2d(kernel_size=k, stride=s)]
        elif isinstance(v, tuple) and v[0] == 'C':
            # ('C', out, k, s, p[, groups])
            _, out_c, k, s, p, *rest = v
            groups = rest[0] if rest else 1
            conv2d = nn.Conv2d(in_channels, out_c, kernel_size=k, stride=s, padding=p, groups=groups, bias=True)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(out_c), nn.ReLU(True)]
            else:
                layers += [conv2d, nn.ReLU(True)]
            in_channels = out_c
        else:
            raise ValueError(f"Unsupported cfg item: {v}")
    return nn.Sequential(*layers)

# 원본 AlexNet 구조(단일 GPU 구현; groups=1). 필요 시 conv2/4/5에 groups=2로 바꿔도 됨.
# alexnet_cfg = [
#     ('C', 64, 11, 4, 2), 'LRN', ('M', 3, 2),
#     ('C', 192, 5, 1, 2), 'LRN', ('M', 3, 2),
#     ('C', 384, 3, 1, 1),
#     ('C', 256, 3, 1, 1),
#     ('C', 256, 3, 1, 1),
#     ('M', 3, 2),
# ]

# CIFAR-10 
alexnet_cfg = [
    ('C', 64,  3, 1, 1), 'LRN', ('M', 2, 2),
    ('C', 192, 5, 1, 2), 'LRN', ('M', 2, 2),
    ('C', 384, 3, 1, 1),
    ('C', 256, 3, 1, 1),
    ('C', 256, 3, 1, 1),
    ('M', 2, 2),
]

def _alexnet(arch, cfg, batch_norm, pretrained, progress, **kwargs):
    if pretrained:
        kwargs['init_weights'] = False
    model = AlexNet(make_layers(cfg, batch_norm=batch_norm), **kwargs)
    # if pretrained:
    #     state_dict = load_state_dict_from_url(model_urls[arch], progress=progress)
    #     model.load_state_dict(state_dict)
    return model

def alexnet(pretrained=False, progress=True, **kwargs):
    # batch_norm=False가 기본. BN 변형을 원하면 alexnet_bn 사용.
    return _alexnet('alexnet', alexnet_cfg, False, pretrained, progress, **kwargs)

def alexnet_bn(pretrained=False, progress=True, **kwargs):
    # AlexNet에 BN을 추가한 변형
    return _alexnet('alexnet_bn', alexnet_cfg, True, pretrained, progress, **kwargs)
