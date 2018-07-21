import torch


class PixelNorm(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.eps = 1e-8

    def forward(self, x):
        norm = torch.mean(x**2,
                          axis=1,
                          keepdim=True)
        return x / (norm + self.eps) ** 0.5
