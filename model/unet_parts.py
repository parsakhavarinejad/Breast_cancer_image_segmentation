import torch.nn as nn
import torch

class ConvBlock(nn.Module):
    def __init__(self, input_channel, out_channel):
        super(ConvBlock, self).__init__()
        self.conv2d_1 = nn.Conv2d(input_channel, out_channel, kernel_size=3, padding=1)
        self.batchnorm_1 = nn.BatchNorm2d(out_channel)
        self.gelu_1 = nn.ReLU()

        self.conv2d_2 = nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1)
        self.batchnorm_2 = nn.BatchNorm2d(out_channel)
        self.gelu_2 = nn.ReLU()

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
        super(Encoder, self).__init__()
        self.conv2d_1 = ConvBlock(input_channel, out_channel)
        self.maxpool = nn.MaxPool2d((2,2))
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x):
        x = self.conv2d_1(x)
        p = self.maxpool(x)
        p = self.dropout(p)

        return x, p

class Decoder(nn.Module):
    def __init__(self, input_channel, output_channel):
        super(Decoder, self).__init__()
        self.conv_t = nn.ConvTranspose2d(input_channel, output_channel, stride=2, kernel_size=2)
        self.conv2d_1 = ConvBlock(output_channel*2, output_channel)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x, skip):
        x = self.conv_t(x)
        x = torch.cat([x, skip], dim=1)
        x = self.dropout(x)
        x = self.conv2d_1(x)

        return x