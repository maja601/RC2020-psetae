import torch
import torch.nn as nn
import numpy as np

from models.mlps import MLP1, MLP2


class SpatialEncoder(nn.Module):
    """
    First part of the presented architecture.
    Yields to a spatio-spectral embedding at time t
    """

    def __init__(self, device):
        super(SpatialEncoder, self).__init__()
        self.device = device
        self.mlp1 = MLP1()
        self.mlp2 = MLP2()

    def forward(self, x, geom, pixels_in_parcel, mask):  # x: [batch_size x set_size x seq_len x channels:10]
        # mask: [batch_size x seq_len x set_size]
        x = x.permute(0, 3, 1, 2)    # from [batch_size x seq_len x channels:10 x set_size] to usual
        batch_size, set_size, seq_len, channels = x.shape

        x = x.permute(0, 2, 1, 3).contiguous().view(-1, set_size,
                                                    channels)  # [batch_size * seq_len x set_size x hidden_state:10]
        mask = mask.contiguous().view(-1, set_size)            # [batch_size * seq_len x set_size]
        # geom = geom.unsqueeze(1).repeat(1, seq_len, 1).contiguous().view(-1, 4)
        geom = geom.contiguous().view(-1, 4)
        mlp1_output = self.mlp1(x)                              # [batch_size * seq_len x set_size x hidden_state:64]
        # mlp1_output = mlp1_output.permute(2, 0, 1)              # [hidden_state:64 x batch_size * seq_len x set_size]

        # Mean and std from garnot
        pooling = 'mean_std'
        masked_mean_std = torch.cat([pooling_methods[n](mlp1_output, mask) for n in pooling.split('_')], dim=1)

        pooled = torch.cat((masked_mean_std, geom), dim=1)  # [batch_size * seq_len x hidden_state:132]
        pooled = pooled.contiguous().view(batch_size, seq_len, -1)  # [batch_size x seq_len x hidden_state:132]
        pooled = pooled.type('torch.FloatTensor')
        mlp2_output = self.mlp2(pooled)                             # [batch_size x seq_len x hidden_state:128]
        return mlp2_output


def masked_mean(x, mask):                       # mask: [batch_size * seq_len x set_size]
    out = x.permute((2, 0, 1))                  # [channels:64 x batch_size * seq_len x set_size]
    out = out * mask                            # [channels:64 x batch_size * seq_len x set_size]
    out = out.sum(dim=-1) / mask.sum(dim=-1)    # [channels:64 x batch_size * seq_len]
    out = out.permute((1, 0))                   # [batch_size * seq_len x channels:64]
    return out


def masked_std(x, mask):    # [batch_size * seq_len x set_size x hidden_state:64]
    m = masked_mean(x, mask)

    out = x.permute((1, 0, 2))  # [set_size x batch_size * seq_len x hidden_state:64]
    out = out - m
    out = out.permute((2, 1, 0))

    out = out * mask
    d = mask.sum(dim=-1)
    d[d == 1] = 2

    out = (out ** 2).sum(dim=-1) / (d - 1)
    out = torch.sqrt(out + 10e-32)  # To ensure differentiability
    out = out.permute(1, 0)
    return out


def maximum(x, mask):
    return x.max(dim=-1)[0].squeeze()


def minimum(x, mask):
    return x.min(dim=-1)[0].squeeze()


pooling_methods = {
    'mean': masked_mean,
    'std': masked_std,
    'max': maximum,
    'min': minimum
}