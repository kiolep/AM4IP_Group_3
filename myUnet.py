import torch
import torch.nn as nn



class DownDouble(nn.Module):
    def __init__(self, input_channels):
        super(DownDouble, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, input_channels * 2, kernel_size=3, padding=1)#, kernel_size=11, padding=5)
        self.conv2 = nn.Conv2d(input_channels * 2, input_channels * 2, kernel_size=3, padding=1)
        self.activation = nn.LeakyReLU(0.2)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        # self.batchnorm = nn.BatchNorm2d(input_channels * 2, momentum=0.8)
        
    def forward(self, x):
        x = self.conv1(x)
        # x = self.batchnorm(x)
        x = self.activation(x)
        
        x = self.conv2(x)
        # x = self.batchnorm(x)
        x = self.activation(x)
        
        x = self.maxpool(x)
        return x


class UpDouble(nn.Module):
    def __init__(self, input_channels):
        super(UpDouble, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv1 = nn.Conv2d(input_channels, input_channels // 2, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(input_channels, input_channels // 2, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(input_channels // 2, input_channels // 2, kernel_size=3, padding=1)
        # self.batchnorm = nn.BatchNorm2d(input_channels // 2, momentum=0.8)
        self.activation = nn.ReLU()

    def forward(self, x, skip_con_x):
        x = self.upsample(x)
        x = self.conv1(x)
        x = torch.cat([x, skip_con_x], axis=1)

        x = self.conv2(x)
        # x = self.batchnorm(x)
        x = self.activation(x)

        x = self.conv3(x)
        # x = self.batchnorm(x)
        x = self.activation(x)

        return x

class FeatureMapBlock(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(FeatureMapBlock, self).__init__()
        self.conv = nn.Conv2d(input_channels, output_channels, kernel_size=1)

    def forward(self, x):
        x = self.conv(x)
        return x


class UNet(nn.Module):
    def __init__(self, input_channels=1, output_channels=1, hidden_channels=64, depth=2):
        super(UNet, self).__init__()
        self.depth = depth
        self.upfeature = FeatureMapBlock(input_channels, hidden_channels)
        self.contract = []
        self.expand = []
        for level in range(depth):
            self.contract.append(DownDouble(hidden_channels * (2 ** level)))
            self.expand.append(UpDouble(hidden_channels * (2 ** (depth - level))))
        self.contracts = nn.Sequential(*self.contract)
        self.expands = nn.Sequential(*self.expand)
        self.downfeature = FeatureMapBlock(hidden_channels, output_channels)

    def forward(self, x):
        xenc = []
        x = self.upfeature(x)
        xenc.append(x)
        for level in range(self.depth):
            x = self.contract[level](x)
            xenc.append(x)
        for level in range(self.depth):
            x = self.expand[level](x, xenc[self.depth - level - 1])
        xn = self.downfeature(x)
        return xn