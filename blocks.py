import math
import torch
from torch import nn
from diffusion_utils import device
from einops import rearrange


class PositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super(PositionEmbeddings, self).__init__()
        self.dim = dim

    def forward(self, time):
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=1)
        return embeddings.to(device)


class resnet_block(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(resnet_block, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1, bias=False, device=device),
            nn.BatchNorm2d(out_channel),
            nn.SiLU(),
            nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1, bias=False, device=device),
            nn.BatchNorm2d(out_channel),
        )
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=1, padding=0, bias=False, device=device)

    def forward(self, x):
        out = self.layer(x)
        x = self.conv(x)
        return out + x


class ResUnit(nn.Module):
    def __init__(self, in_channel, out_channel, emb_channel):
        super(ResUnit, self).__init__()
        self.time_embedding = nn.Sequential(
            PositionEmbeddings(dim=emb_channel // 4),
            nn.Linear(emb_channel // 4, emb_channel),
            nn.SiLU(),
            nn.Linear(emb_channel, emb_channel),
        )
        self.layer = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1, bias=False, device=device),
            nn.BatchNorm2d(out_channel),
            nn.SiLU(),
            resnet_block(out_channel, out_channel),
            nn.SiLU(),
            nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1, bias=False, device=device),
            nn.BatchNorm2d(out_channel),
            nn.SiLU(),
        )

    def forward(self, x, t):
        t_embedding = self.time_embedding(t)
        t_embedding = rearrange(t_embedding, 'b c -> b c 1 1')
        x = x + t_embedding
        return self.layer(x).to(device)


class SpacialEnhanced(nn.Module):
    def __init__(self, channel):
        super(SpacialEnhanced, self).__init__()
        self.u = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=1),
            nn.Sigmoid(),
            nn.Conv2d(channel, channel, kernel_size=1),
            nn.Sigmoid(),
        )
        self.Conv = nn.Conv2d(channel, channel, kernel_size=3, padding=1)

    def forward(self, x):
        u = self.u(x)
        fm = self.Conv(x)
        out1 = u * fm
        out2 = (1 - u) * x
        out = out1 + out2
        return out.to(device)


class ChannelEnhanced(nn.Module):
    def __init__(self, channel):
        super(ChannelEnhanced, self).__init__()
        self.mp = nn.AdaptiveMaxPool2d(1)
        self.h = nn.Sequential(
            nn.Linear(channel, channel // 2, bias=False),
            nn.Sigmoid(),
            nn.Linear(channel // 2, channel, bias=False),
            nn.Sigmoid(),
        )
        self.Conv = nn.Conv2d(channel, channel, kernel_size=3, padding=1)

    def forward(self, x):
        b, c, h, w = x.shape
        mp = self.mp(x).reshape(b, c)
        h = self.h(mp)
        h = h.reshape(b, c, 1, 1)
        fm = self.Conv(x)
        out1 = h * fm
        out2 = (1 - h) * x
        out = out1 + out2
        return out.to(device)


class CEM(nn.Module):
    def __init__(self, in_channel, out_channel, emb_channel):
        super(CEM, self).__init__()
        self.ru = ResUnit(in_channel, out_channel, emb_channel)
        self.se = SpacialEnhanced(out_channel)
        self.ce = ChannelEnhanced(out_channel)

    def forward(self, x, t):
        x1 = self.ru(x, t)
        x2 = self.se(x1)
        x3 = self.ce(x2)
        return x3
