import torch
import numpy as np


def _tile(x, dim, n_tile):
    init_dim = x.size(dim)
    repeat_idx = [1] * x.dim()
    repeat_idx[dim] = n_tile
    x = x.repeat(*repeat_idx)
    if isinstance(x, torch.Tensor):
        order_index = torch.LongTensor(np.concatenate([init_dim*np.arange(n_tile) + i
                                                       for i in range(init_dim)]))
    else:
        order_index = torch.cuda.LongTensor(np.concatenate([init_dim*np.arange(n_tile) + i
                                                            for i in range(init_dim)]))
    return torch.index_select(x, dim, order_index)


class MiniBatchStddev(torch.nn.Module):
    def __init__(self, group_size=1, **kwargs):
        super().__init__()
        self.group_size = group_size
        self.eps = 1e-8

    def forward(self, x):
        y = x.clone()
        bs, c, h, w = list(y.size())
        y = y.view(self.group_size, bs//self.group_size, c, h, w)
        y -= torch.mean(y, dim=0, keepdim=True)
        y = torch.mean(y ** 2, dim=0)
        y = torch.sqrt(y + self.eps)
        y = torch.mean(y.view(-1, c*h*w), dim=-1, keepdim=True)
        y = y.unsqueeze(-1)
        y = y.unsqueeze(-1)
        y = _tile(y, 0, self.group_size)
        y = _tile(y, 3, w)
        y = _tile(y, 2, h)
        return torch.cat((x, y), dim=1)


if __name__ == '__main__':
    import numpy as np
    np.random.seed(32)
    o = MiniBatchStddev(2)(np.random.normal(size=(4, 1, 2, 2)))
    print(o.size())
