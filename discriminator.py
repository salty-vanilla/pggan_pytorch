import torch
from blocks import ConvBlock
from layers.gan import MiniBatchStddev


class DiscriminatorBlock(torch.nn.Module):
    def __init__(self, in_ch,
                 out_ch,
                 downsampling='stride'):
        super().__init__()
        self.conv = ConvBlock(in_ch,
                              out_ch,
                              sampling='same',
                              normalization='instance')
        self.down = ConvBlock(out_ch,
                              out_ch,
                              sampling=downsampling,
                              normalization='instance')

    def forward(self, x):
        x = self.conv(x)
        x = self.down(x)
        return x


class LastDiscriminatorBlock(torch.nn.Module):
    def __init__(self, in_ch,
                 out_ch):
        super().__init__()
        self.mb_stddev = MiniBatchStddev()
        self.conv = ConvBlock(in_ch+1,
                              out_ch,
                              sampling='same',
                              normalization='instance')
        self.down = ConvBlock(out_ch,
                              out_ch,
                              kernel_size=4,
                              padding='valid',
                              normalization=None)
        self.norm = torch.nn.LayerNorm(out_ch)

    def forward(self, x):
        x = self.mb_stddev(x)
        x = self.conv(x)
        x = self.down(x)
        x = x.view(x.size()[0], -1)
        x = self.norm(x)
        x = torch.nn.LeakyReLU()(x)
        return x


class FromRGB(torch.nn.Module):
    def __init__(self, out_ch):
        super().__init__()
        self.conv = ConvBlock(3,
                              out_ch,
                              sampling='same',
                              normalization='instance',
                              activation=torch.nn.Tanh())

    def forward(self, x):
        return self.conv(x)


class Discriminator(torch.nn.Module):
    def __init__(self, nb_growing=8,
                 downsampling='avg_pool'):
        super().__init__()
        self.filters = [512, 512, 512, 512, 256, 256, 128, 64, 32][:nb_growing]
        self.resolutions = [(2**(2+i), 2**(2+i)) for i in range(nb_growing)]
        self.downsampling = downsampling

        self.blocks = []
        self.from_rgbs = []

        for i in range(nb_growing):
            if i == 0:
                self.blocks.append(LastDiscriminatorBlock(self.filters[i],
                                                          self.filters[i]))
            else:
                self.blocks.append(DiscriminatorBlock(self.filters[i],
                                                      self.filters[i-1],
                                                      self.downsampling))
            self.from_rgbs.append(FromRGB(self.filters[i]))
            self.add_module('discriminator_block_%d' % i, self.blocks[-1])
            self.add_module('from_rgb_%d' % i, self.from_rgbs[-1])

        self.dense = torch.nn.Linear(self.filters[0], 1)

    def forward(self, x,
                growing_step,
                alpha=1.):
        is_resl = alpha != 1. and growing_step != 0
        if is_resl:
            _x = torch.nn.functional.avg_pool2d(x, kernel_size=2, stride=2)
            _x = self.from_rgbs[growing_step-1](_x)

        x = self.from_rgbs[growing_step](x)
        for i in range(growing_step+1)[::-1]:
            x = self.blocks[i](x)

            if is_resl and i == growing_step:
                x = alpha*x + (1.-alpha)*_x

        x = self.dense(x)
        return x


if __name__ == '__main__':
    import numpy as np
    d = Discriminator(nb_growing=8, downsampling='max_pool')
    d.cuda()
    for i in range(8):
        _x = np.random.normal(size=(4, 3, 4*(2**i), 4*(2**i)))
        o = d(torch.Tensor(_x).cuda(), i, 0.5)
        print(o.size())
