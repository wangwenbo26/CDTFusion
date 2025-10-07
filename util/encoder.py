import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU(inplace=True)

        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.downsample = None

    def forward(self, x):
        identity = x

        out = self.relu1(self.conv1(x))
        out = self.conv2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu2(out)

        return out

class EncoderVi(nn.Module):
    def __init__(self):
        super(EncoderVi, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU(inplace=True)

        self.res_block1 = ResidualBlock(64, 64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.res_block2 = ResidualBlock(64, 128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.res_block3 = ResidualBlock(128, 256)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.res_block4 = ResidualBlock(256, 512)

    def forward(self, x):
        x = self.relu1(self.conv1(x))

        x = self.res_block1(x)
        x = self.pool1(x)

        x = self.res_block2(x)
        x = self.pool2(x)

        x = self.res_block3(x)
        x = self.pool3(x)

        x = self.res_block4(x)

        return x

class EncoderIr(nn.Module):
    def __init__(self):
        super(EncoderIr, self).__init__()

        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU(inplace=True)

        self.res_block1 = ResidualBlock(64, 64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.res_block2 = ResidualBlock(64, 128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.res_block3 = ResidualBlock(128, 256)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.res_block4 = ResidualBlock(256, 512)

    def forward(self, x):
        x = self.relu1(self.conv1(x))

        x = self.res_block1(x)
        x = self.pool1(x)

        x = self.res_block2(x)
        x = self.pool2(x)

        x = self.res_block3(x)
        x = self.pool3(x)

        x = self.res_block4(x)

        return x
