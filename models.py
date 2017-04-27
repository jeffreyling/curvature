import torch
import torch.nn as nn
import torch.legacy

# cannot figure out number of filters
class AlexNet(nn.Module):
    def __init__(self, num_classes=10):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3),
            torch.legacy.SpatialCrossMapLRN(5, alpha=1, beta=0.5),
            nn.Conv2d(64, 192, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3),
            torch.legacy.SpatialCrossMapLRN(5, alpha=1, beta=0.5)
        )
        self.classifier = nn.Sequential(
            nn.Linear(192 * 16 * 16, 384),
            nn.ReLU(inplace=True),
            nn.Linear(384, 192),
            nn.ReLU(inplace=True),
            nn.Linear(192, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        return x

class MLP(nn.Module):
    def __init__(self, num_classes=10):
        super(MLP, self).__init__()
        self.linear = nn.Linear(3 * 28 * 28, 512)
        self.classifier = nn.Linear(512, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.linear(x.view(-1, 3*28*28)))
        return self.classifier(x)



class ConvModule(nn.Module):
    def __init__(self, C_in, C, **kwargs):
        super(ConvModule, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(C_in, C, bias=True, **kwargs),
            nn.BatchNorm2d(C, affine=False),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.net(x)

class InceptionModule(nn.Module):
    def __init__(self, C_in, Ch1, Ch3):
        super(InceptionModule, self).__init__()
        self.conv1 = ConvModule(C_in, Ch1, kernel_size=1, stride=1)
        self.conv3 = ConvModule(C_in, Ch3, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        conv1 = self.conv1(x)
        conv3 = self.conv3(x)
        return torch.cat( (conv1, conv3), 1 )

class DownsampleModule(nn.Module):
    def __init__(self, C_in, Ch3):
        super(DownsampleModule, self).__init__()
        self.conv = ConvModule(C_in, Ch3, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        conv = self.conv(x)
        pool = self.pool(x)
        return torch.cat( (conv, pool), 1 )


class Inception(nn.Module):
    def __init__(self, num_classes=10):
        super(Inception, self).__init__()
        self.net = nn.Sequential(
                ConvModule(3, 96, kernel_size=3, stride=1, padding=1),
                InceptionModule(96, 32, 32),
                InceptionModule(32 + 32, 32, 48),
                DownsampleModule(32 + 48, 80),
                InceptionModule(80 + 80, 112, 48),
                InceptionModule(112 + 48, 96, 64),
                InceptionModule(96 + 64, 80, 80),
                InceptionModule(80 + 80, 48, 96),
                DownsampleModule(48 + 96, 96),
                InceptionModule(96 + 144, 176, 160),
                InceptionModule(176 + 160, 176, 160),
                )

        self.avg_pool = nn.AvgPool2d((7, 7)) # global
        self.classifier = nn.Linear(176 + 160, num_classes)

    def forward(self, x):
        features = self.net(x)
        features = self.avg_pool(features)
        return self.classifier(features.view(features.size(0), 336))
