import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

class UpConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, up_in=None, up_out=None):
        super().__init__()
        self.upsample = nn.ConvTranspose2d(up_in or in_channels, up_out or out_channels, kernel_size=2, stride=2)
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, up_input, skip_input):
        up_sampled = self.upsample(up_input)
        combined = torch.cat([up_sampled, skip_input], dim=1)
        return self.conv_block(combined)

class UnetResnet34(nn.Module):
    def __init__(self, output_classes=30):
        super().__init__()
        base_model = torchvision.models.resnet34(pretrained=True)

        self.initial_layers = nn.Sequential(*list(base_model.children())[:3])
        self.pool = list(base_model.children())[3]
        self.enc1 = base_model.layer1
        self.enc2 = base_model.layer2
        self.enc3 = base_model.layer3
        self.enc4 = base_model.layer4

        self.bottleneck = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.dec1 = UpConvBlock(1024, 512)
        self.dec2 = UpConvBlock(512, 256)
        self.dec3 = UpConvBlock(256, 128)
        self.dec4 = UpConvBlock(128, 64)
        self.dec5 = UpConvBlock(128, 64, up_in=64, up_out=64)

        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2),
            nn.Conv2d(64, output_classes, kernel_size=3, padding=1, bias=False)
        )

    def forward(self, input_data):
        enc1_out = self.initial_layers(input_data)
        pooled = self.pool(enc1_out)
        enc2_out = self.enc1(pooled)
        enc3_out = self.enc2(enc2_out)
        enc4_out = self.enc3(enc3_out)
        enc5_out = self.enc4(enc4_out)

        bottleneck_out = self.bottleneck(enc5_out)

        dec1_out = self.dec1(bottleneck_out, enc5_out)
        dec2_out = self.dec2(dec1_out, enc4_out)
        dec3_out = self.dec3(dec2_out, enc3_out)
        dec4_out = self.dec4(dec3_out, enc2_out)
        dec5_out = self.dec5(dec4_out, enc1_out)

        return self.final_layer(dec5_out)
