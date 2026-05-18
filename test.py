import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm
import torch.nn.functional as F
import numpy as np
import cv2


class BlockFFT(nn.Module):
    def __init__(self, dim, h, w):
        super().__init__()
        self.complex_weight = nn.Parameter(torch.randn(dim, h, w // 2 + 1, 2, dtype=torch.float32) * 0.02)

    def forward(self, x):
        B, C, H, W = x.shape
        x_fre = torch.fft.rfft2(x, dim=(2, 3), norm='ortho')
        x = x_fre * torch.view_as_complex(self.complex_weight)
        x = torch.fft.irfft2(x, s=(H, W), dim=(2, 3), norm='ortho')
        x = x.reshape(B, C, H, W)

        return x, x_fre


if __name__ == "__main__":
    device = 'cuda:5'
    input = torch.randn(1, 3, 256, 256).to(device)
    blockFFT = BlockFFT(dim=3, h=256, w=256).to(device)
    x, x_fre = blockFFT(input)
    x_fre = F.interpolate(x_fre, 256)
    print(x_fre.shape)
    print(x.shape)