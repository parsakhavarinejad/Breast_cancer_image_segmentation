import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):

    def __init__(self, input_channel, out_channel):
        super().__init__()
        self.conv2d_1 = nn.Conv2d(input_channel, out_channel, kernel_size=3, padding='same')
        self.batchnorm_1 = nn.BatchNorm2d(out_channel)
        self.gelu_1 = nn.GELU()

        self.conv2d_2 = nn.Conv2d(input_channel, out_channel, kernel_size=3, padding='same')
        self.batchnorm_2 = nn.BatchNorm2d(out_channel)
        self.gelu_2 = nn.GELU()

    def forward(self, x):
        x = self.conv2d_1(x)
        x = self.batchnorm_1(x)
        x = self.gelu_1(x)

        x = self.conv2d_2(x)
        x = self.batchnorm_2(x)
        x = self.gelu_2(x)

        return x

class Encoder(nn.Module):

    def __init__(self, input_channel, out_channel):
        super().__init__()
        self.conv2d_1 = ConvBlock(input_channel, out_channel)
        self.maxpool = nn.MaxPool2d((2,2))

    def forward(self, x):
        x = self.conv2d_1(x)
        p = self.maxpool(x)

        return x, p

class Decoder(nn.Module):

    def __init__(self, input_channel, output_channel):
        super().__init__()
        self.conv_t = nn.ConvTranspose2d(input_channel, output_channel, stride=2, kernel_size=3)
        self.conv2d_1 = ConvBlock(input_channel, output_channel)

    def forward(self, x, skip):
        x = self.conv_t(x)
        x = torch.cat([x, skip], dim=1)
        x = self.conv2d_1(x)

        return x

class Unet(nn.Module):

    def __init__(self, input_chennel=3):
        super().__init__()
        self.encoder_1 = Encoder(input_chennel, 64)
        self.encoder_2 = Encoder(64, 128)
        self.encoder_3 = Encoder(128, 256)
        self.encoder_4 = Encoder(256, 512)

        self.conv_block = ConvBlock(512, 1024)

        self.decoder_1 = Decoder(1024, 512)
        self.decoder_2 = Decoder(512, 256)
        self.decoder_3 = Decoder(256, 128)
        self.decoder_4 = Decoder(128, 64)

        self.cls = nn.Conv2d(64, 1, kernel_size=1, padding=0)

    def forward(self, x):

        """ ------ Encoder ------"""
        x1, p1 = self.encoder_1(x)
        x2, p2 = self.encoder_2(p1)
        x3, p3 = self.encoder_3(p2)
        x4, p4 = self.encoder_4(p3)

        """ ------ BottleNeck ------"""
        x6 = self.conv_block(p4)

        """ ------ Decoder ------"""
        x7 = self.decoder_1(x6, x4)
        x8 = self.decoder_1(x7, x3)
        x9 = self.decoder_1(x8, x2)
        x10 = self.decoder_1(x9, x1)

        """ ------ Final Layer ------"""
        x_final = self.cls(x10)

        return x_final

