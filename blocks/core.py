import torch
from layers.normalization import PixelNorm, LayerNorm
from layers.conv import SubpixelConv


class ConvBlock(torch.nn.Module):
    def __init__(self, in_ch,
                 out_ch,
                 kernel_size=3,
                 sampling='same',
                 padding='same',
                 normalization=None,
                 activation=torch.nn.LeakyReLU(),
                 dropout_rate=0.0):
        assert sampling in ['deconv', 'subpixel', 'upsampling',
                            'stride', 'max_pool', 'avg_pool',
                            'same']
        assert normalization in ['batch', 'layer', 'pixel', 'instance', None]
        super().__init__()

        self.net = torch.nn.Sequential()

        if sampling == 'upsampling':
            self.net.add_module('up', torch.nn.UpsamplingNearest2d(2))

        padding = kernel_size // 2 if padding == 'same' else 0
        if sampling in ['same', 'max_pool', 'avg_pool']:
            conv = torch.nn.Conv2d(in_ch,
                                   out_ch,
                                   kernel_size,
                                   stride=1,
                                   padding=padding)
        elif sampling == 'stride':
            conv = torch.nn.Conv2d(in_ch,
                                   out_ch,
                                   kernel_size,
                                   stride=2,
                                   padding=padding)
        elif sampling == 'deconv':
            conv = torch.nn.ConvTranspose2d(in_ch,
                                            out_ch,
                                            kernel_size,
                                            stride=2,
                                            padding=padding)
        elif sampling == 'subpixel':
            conv = SubpixelConv(in_ch,
                                rate=2)
        else:
            raise ValueError
        self.net.add_module('conv', conv)

        if normalization is not None:
            if normalization == 'batch':
                norm = torch.nn.BatchNorm2d(out_ch)
            elif normalization == 'layer':
                norm = LayerNorm(out_ch)
            elif normalization == 'pixel':
                norm = PixelNorm()
            elif normalization == 'instance':
                norm = torch.nn.InstanceNorm2d(out_ch)
            else:
                raise ValueError
            self.net.add_module('norm', norm)

        if activation is not None:
            self.net.add_module('act', activation)

        if dropout_rate != 0.0:
            self.net.add_module('dropout', torch.nn.Dropout2d(dropout_rate))

        if sampling == 'max_pool':
            self.net.add_module('pool', torch.nn.MaxPool2d(2, 2))
        elif sampling == 'avg_pool':
            self.net.add_module('pool', torch.nn.AvgPool2d(2, 2))
        else:
            pass

    def forward(self, x):
        return self.net(x)
