import torch


class SubpixelConv(torch.nn.Module):
    def __init__(self, in_ch,
                 out_ch=None,
                 kernel_size=3,
                 rate=2):
        if out_ch is None:
            out_ch = in_ch * (rate**2)
        else:
            out_ch = out_ch * (rate**2)

        self.conv = torch.nn.Conv2d(in_ch,
                                    out_ch,
                                    kernel_size,
                                    padding=kernel_size//2)
        self.ps = torch.nn.PixelShuffle(rate)
        super().__init__()

    def forward(self, x):
        x = self.conv(x)
        x = self.ps(x)
        return x
