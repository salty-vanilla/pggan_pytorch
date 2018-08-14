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
                            in_ch,
                            sampling=upsampling)
        self.conv1 = ConvBlock(in_ch,
                               in_ch,
                               sampling='same',
                               normalization='pixel')
        self.conv2 = ConvBlock(in_ch,
                               out_ch,
                               sampling='same',
                               normalization='pixel')

    def forward(self, x):
        x = self.up(x)
        x = self.conv1(x)
        x = self.conv2(x)
        return x

class ToRGB(torch.nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.conv = ConvBlock(in_ch, 
                              3,
                              sampling='same',
                              normalization=None,
                              activation=torch.nn.Tanh())

    def forward(self, x):
        return self.conv(x)


class Generator(torch.nn.Module):
    def __init__(self, input_dim,
                 nb_growing=8,
                 upsampling='subpixel'):
        super().__init__()
        self.filters = [512, 512, 512, 512, 256, 256, 128, 64, 32][:nb_growing]
        self.upsampling = upsampling

        self.blocks = []
        self.to_rgbs = []

        for i in range(nb_growing):
            if i == 0:
                self.blocks.append(FirstGeneratorBlock(input_dim,
                                                       self.filters[i]))
                self.to_rgbs.append(ToRGB(self.filters[i]))
            else:
                self.blocks.append(GeneratorBlock(self.filters[i-1],
                                                  self.filters[i],
                                                  upsampling))
                self.to_rgbs.append(ToRGB(self.filters[i]))
            print(self.blocks[-1])
    
    def forward(self, x,
                growing_step):
        x = torch.Tensor(x)
        for i in range(growing_step+1):
            x = self.blocks[i](x)
        # print(self.to_rgbs[growing_step])
        x = self.to_rgbs[growing_step](x)
        return x


if __name__ == '__main__':
    import numpy as np
    g = Generator(10, 5)
    _x = torch.Tensor(np.random.normal(size=(3, 10)))
    for i in range(5):
        print(i)
        print(_x.size())
        __x = g.forward(_x, i)
        print(__x.size())
