import math
import torch
from torch import nn
from blocks import CEM, ResUnit
from diffusion_utils import device


class down(nn.Module):
    def __init__(self, channel):
        super(down, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=3, stride=2, padding=1, bias=False, device=device),
            nn.BatchNorm2d(channel),
            nn.SiLU()
        )

    def forward(self, x):
        return self.layer(x)


class up(nn.Module):
    def __init__(self, channel):
        super(up, self).__init__()
        self.layer = nn.Sequential(
            nn.ConvTranspose2d(channel, channel // 2, kernel_size=4, stride=2, padding=1, device=device),
            nn.BatchNorm2d(channel // 2),
            nn.SiLU(),
        )

    def forward(self, x, feature_map):
        x = self.layer(x)
        up = torch.concat([x, feature_map], dim=1)
        return up


class NEU(nn.Module):
    def __init__(self, in_channel=2, out_channel=1):
        super(NEU, self).__init__()
        self.init_conv = nn.Conv2d(in_channel, 64, kernel_size=3, stride=1, padding=1, bias=False, device=device)
        self.c1 = CEM(64, 64, 64)
        self.d1 = down(64)
        self.c2 = CEM(64, 128, 64)
        self.d2 = down(128)
        self.c3 = CEM(128, 256, 128)
        self.d3 = down(256)
        self.c4 = CEM(256, 512, 256)
        self.d4 = down(512)
        self.c5 = CEM(512, 1024, 512)
        self.u1 = up(1024)
        self.c6 = ResUnit(1024, 512, 1024)
        self.u2 = up(512)
        self.c7 = ResUnit(512, 256, 512)
        self.u3 = up(256)
        self.c8 = ResUnit(256, 128, 256)
        self.u4 = up(128)
        self.c9 = ResUnit(128, 64, 128)
        self.final_conv = nn.Conv2d(64, out_channel, kernel_size=3, padding=1, bias=False, device=device)

    def forward(self, x_t, t):
        x_t = x_t.to(device)
        init = self.init_conv(x_t)
        R1 = self.c1(init, t)
        D1 = self.d1(R1)
        R2 = self.c2(D1, t)
        D2 = self.d2(R2)
        R3 = self.c3(D2, t)
        D3 = self.d3(R3)
        R4 = self.c4(D3, t)
        D4 = self.d4(R4)
        R5 = self.c5(D4, t)
        U1 = self.u1(R5, R4)
        R6 = self.c6(U1, t)
        U2 = self.u2(R6, R3)
        R7 = self.c7(U2, t)
        U3 = self.u3(R7, R2)
        R8 = self.c8(U3, t)
        U4 = self.u4(R8, R1)
        R9 = self.c9(U4, t)
        out = self.final_conv(R9)
        return out

