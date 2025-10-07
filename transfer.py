import torch
import torch.nn as nn


class transfer(nn.Module):
    def __init__(self):
        super(transfer, self).__init__()

        self.conv1 = nn.Conv2d(1024, 1024, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(1024, 512, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU(inplace=True)

        self.c = 512
        self.concat = lambda x: torch.cat(x, dim=1)

        self.channel_weights_conv = nn.Conv2d(512, self.c, kernel_size=1)

        self.concat_x2 = lambda x: torch.cat(x, dim=1)

        self.conv4 = nn.Conv2d(1024, 1024, kernel_size=3, padding=1)
        self.relu4 = nn.ReLU(inplace=True)
        self.conv5 = nn.Conv2d(1024, 512, kernel_size=3, padding=1)
        self.relu5 = nn.ReLU(inplace=True)
        self.conv6 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.relu6 = nn.ReLU(inplace=True)

    def forward(self, opt, sar):

        x1 = self.concat([opt, sar])

        x1 = self.relu1(self.conv1(x1))
        x1 = self.relu2(self.conv2(x1))
        x1 = self.relu3(self.conv3(x1))

        channel_weights = self.channel_weights_conv(x1)
        x2 = sar * channel_weights

        x3 = self.concat_x2([opt, x2])

        x3 = self.relu4(self.conv1(x3))
        x3 = self.relu5(self.conv2(x3))
        x3 = self.relu6(self.conv3(x3))

        return x3


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
        nn.Linear(channel, channel // reduction, bias=False),
        nn.ReLU(inplace=True),
        nn.Linear(channel // reduction, channel, bias=False),
        nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)
