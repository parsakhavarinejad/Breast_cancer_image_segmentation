from .unet_parts import ConvBlock, Encoder, Decoder

class Unet(nn.Module):

    def __init__(self, input_channel=1):
        super().__init__()
        self.encoder_1 = Encoder(input_channel, 64)
        self.encoder_2 = Encoder(64, 128)
        self.encoder_3 = Encoder(128, 256)
        self.encoder_4 = Encoder(256, 512)

        self.conv_block = ConvBlock(512, 1024)

        self.decoder_1 = Decoder(1024, 512)
        self.decoder_2 = Decoder(512, 256)
        self.decoder_3 = Decoder(256, 128)
        self.decoder_4 = Decoder(128, 64)

        self.cls = nn.Conv2d(64, 1, kernel_size=1, padding=0)
        self.relu = nn.Sigmoid() 

    def forward(self, x):

        """ ------ Encoder ------"""
        x1, p1 = self.encoder_1(x)
        x2, p2 = self.encoder_2(p1)
        x3, p3 = self.encoder_3(p2)
        x4, p4 = self.encoder_4(p3)

        """ ------ BottleNeck ------"""
        x5 = self.conv_block(p4)

        """ ------ Decoder ------"""
        x6 = self.decoder_1(x5, x4)
        x7 = self.decoder_2(x6, x3)
        x8 = self.decoder_3(x7, x2)
        x9 = self.decoder_4(x8, x1)
        
        """ ------ Final Layer ------"""
        x_final = self.cls(x9)
        x_final = self.relu(x_final)

        return x_final
