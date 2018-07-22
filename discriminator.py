import torch
from blocks import ConvBlock
from layers.gan import MiniBatchStddev


class DiscriminatorBlock(torch.nn.Module):
    def __init__(self, in_ch,
                 out_ch,
                 downsampling='stride'):
        self.conv = ConvBlock(in_ch,
                              out_ch,
                              sampling='same',
                              normalization='instance')
        self.down = ConvBlock(out_ch,
                              out_ch,
                              sampling=downsampling,
                              normalization='instance')
        super().__init__()

    def forward(self, x):
        x = self.conv(x)
        x = self.down(x)
        return x


class LastDiscriminatorBlock(torch.nn.Module):
    def __init__(self, in_ch,
                 out_ch,
                 downsampling='stride'):
        self.mb_stddev = MiniBatchStddev()
        self.conv = ConvBlock(in_ch,
                              out_ch,
                              sampling='same',
                              normalization='instance')
        self.down = ConvBlock(out_ch,
                              out_ch,
                              kernel_size=4,
                              padding='valid',
                              sampling=downsampling,
                              normalization='instance')
        super().__init__()

    def forward(self, x):
        x = self.mb_stddev(x)
        x = self.conv(x)
        x = self.down(x)
        return x
