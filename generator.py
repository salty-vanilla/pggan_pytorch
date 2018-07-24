import torch
from blocks import ConvBlock


class FirstGeneratorBlock(torch.nn.Module):
    def __init__(self, input_dim,
                 out_ch):
        super().__init__()
        self.input_dim = input_dim
        self.conv1 = ConvBlock(input_dim,
                               out_ch,
                               sampling='same',
                               normalization='pixel')
        self.conv2 = ConvBlock(out_ch,
                               out_ch,
                               sampling='same',
                               normalization='pixel')

    def forward(self, x):
        x = x.view(-1, self.input_dim, 1, 1)
        x = torch.nn.UpsamplingNearest2d(4)(x)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class GeneratorBlock(torch.nn.Module):
    def __init__(self, in_ch,
                 out_ch,
                 upsampling='upsampling'):
        super().__init__()
        self.up = ConvBlock(in_ch,
                            out_ch,
                            sampling=upsampling)
        self.conv1 = ConvBlock(out_ch,
                               out_ch,
                               sampling='same',
                               normalization='pixel')
        self.conv2 = ConvBlock(out_ch,
                               out_ch,
                               sampling='same',
                               normalization='pixel')

    def forward(self, x):
        x = self.up(x)
        x = self.conv1(x)
        x = self.conv2(x)
        return x
