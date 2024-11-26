import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm
import torch.nn.functional as F
import numpy as np
import cv2

seq = nn.Sequential


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        try:
            m.weight.data.normal_(0.0, 0.02)
        except:
            pass
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def conv2d(*args, **kwargs):
    return spectral_norm(nn.Conv2d(*args, **kwargs))


def convTranspose2d(*args, **kwargs):
    return spectral_norm(nn.ConvTranspose2d(*args, **kwargs))


def batchNorm2d(*args, **kwargs):
    return nn.BatchNorm2d(*args, **kwargs)


def linear(*args, **kwargs):
    return spectral_norm(nn.Linear(*args, **kwargs))


class PixelNorm(nn.Module):
    def forward(self, input):
        return input * torch.rsqrt(torch.mean(input ** 2, dim=1, keepdim=True) + 1e-8)


class Reshape(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.target_shape = shape

    def forward(self, feat):
        batch = feat.shape[0]
        return feat.view(batch, *self.target_shape)


class GLU(nn.Module):
    def forward(self, x):
        nc = x.size(1)
        assert nc % 2 == 0, 'channels dont divide 2!'
        nc = int(nc / 2)
        return x[:, :nc] * torch.sigmoid(x[:, nc:])


class NoiseInjection(nn.Module):
    def __init__(self):
        super().__init__()

        self.weight = nn.Parameter(torch.zeros(1), requires_grad=True)

    def forward(self, feat, noise=None):
        if noise is None:
            batch, _, height, width = feat.shape
            noise = torch.randn(batch, 1, height, width).to(feat.device)

        return feat + self.weight * noise


class Swish(nn.Module):
    def forward(self, feat):
        return feat * torch.sigmoid(feat)


class SEBlock(nn.Module):
    def __init__(self, ch_in, ch_out):
        super().__init__()

        self.main = nn.Sequential(nn.AdaptiveAvgPool2d(4),
                                  conv2d(ch_in, ch_out, 4, 1, 0, bias=False), Swish(),
                                  conv2d(ch_out, ch_out, 1, 1, 0, bias=False), nn.Sigmoid())

    def forward(self, feat_small, feat_big):
        return feat_big * self.main(feat_small)


class InitLayer(nn.Module):
    def __init__(self, nz, channel):
        super().__init__()

        self.init = nn.Sequential(
            convTranspose2d(nz, channel * 2, 4, 1, 0, bias=False),
            batchNorm2d(channel * 2), GLU())

    def forward(self, noise):
        noise = noise.view(noise.shape[0], -1, 1, 1)
        return self.init(noise)


def UpBlock(in_planes, out_planes):
    block = nn.Sequential(
        nn.Upsample(scale_factor=2, mode='nearest'),
        conv2d(in_planes, out_planes * 2, 3, 1, 1, bias=False),
        # convTranspose2d(in_planes, out_planes*2, 4, 2, 1, bias=False),
        batchNorm2d(out_planes * 2), GLU())
    return block


def UpBlockComp(in_planes, out_planes):
    block = nn.Sequential(
        nn.Upsample(scale_factor=2, mode='nearest'),

        conv2d(in_planes, out_planes * 2, 3, 1, 1, bias=False),
        # convTranspose2d(in_planes, out_planes*2, 4, 2, 1, bias=False),
        NoiseInjection(),
        batchNorm2d(out_planes * 2), GLU(),
        conv2d(out_planes, out_planes * 2, 3, 1, 1, bias=False),
        NoiseInjection(),
        batchNorm2d(out_planes * 2), GLU()
    )
    return block


class BlockFFT(nn.Module):
    def __init__(self, dim, h, w, groups=8):
        super().__init__()
        # self.proj = WeightStandardizedConv2d(dim, dim_out, 3, padding = 1)
        self.complex_weight = nn.Parameter(torch.randn(dim, h, w // 2 + 1, 2, dtype=torch.float32) * 0.02)
        # self.complex_weight = nn.Parameter(torch.normal(mean=0, std=0.01, size=(dim, h, w // 2 + 1, 2), dtype=torch.float32))
        # self.norm = nn.GroupNorm(groups, dim)
        # self.act = nn.SiLU()

    def forward(self, x, scale_shift=None):
        B, C, H, W = x.shape
        x = torch.fft.rfft2(x, dim=(2, 3), norm='ortho')
        x = x * torch.view_as_complex(self.complex_weight)
        x = torch.fft.irfft2(x, s=(H, W), dim=(2, 3), norm='ortho')
        x = x.reshape(B, C, H, W)

        return x


# class Generator(nn.Module):
#     def __init__(self, ngf=64, nz=100, nc=3, im_size=1024):
#         super(Generator, self).__init__()
#
#         nfc_multi = {4: 16, 8: 8, 16: 4, 32: 2, 64: 2, 128: 1, 256: 0.5, 512: 0.25, 1024: 0.125}
#         nfc = {}
#         for k, v in nfc_multi.items():
#             nfc[k] = int(v * ngf)
#
#         self.im_size = im_size
#
#         self.init = InitLayer(nz, channel=nfc[4])
#
#         self.feat_8 = UpBlockComp(nfc[4], nfc[8])
#         self.feat_16 = UpBlock(nfc[8], nfc[16])
#         self.feat_32 = UpBlockComp(nfc[16], nfc[32])
#         self.feat_64 = UpBlock(nfc[32], nfc[64])
#         self.feat_128 = UpBlockComp(nfc[64], nfc[128])
#         self.feat_256 = UpBlock(nfc[128], nfc[256])
#
#         self.to_128 = conv2d(nfc[128], nc, 1, 1, 0, bias=False)
#         self.to_big = conv2d(nfc[im_size], nc, 3, 1, 1, bias=False)
#
#         self.feat_8_output = conv2d(nfc[8], nc, 1, 1, 0, bias=False)
#         self.feat_16_output = conv2d(nfc[16], nc, 1, 1, 0, bias=False)
#         self.feat_32_output = conv2d(nfc[32], nc, 1, 1, 0, bias=False)
#         self.feat_64_output = conv2d(nfc[64], nc, 1, 1, 0, bias=False)
#         self.feat_128_output = conv2d(nfc[128], nc, 1, 1, 0, bias=False)
#         self.feat_256_output = conv2d(nfc[256], nc, 1, 1, 0, bias=False)
#
#         self.feat_16_residual_fequency_output = conv2d(nfc[16], nc, 1, 1, 0, bias=False)
#         self.feat_32_residual_fequency_output = conv2d(nfc[32], nc, 1, 1, 0, bias=False)
#         self.feat_64_residual_fequency_output = conv2d(nfc[64], nc, 1, 1, 0, bias=False)
#         self.feat_128_residual_fequency_output = conv2d(nfc[128], nc, 1, 1, 0, bias=False)
#         self.feat_256_residual_fequency_output = conv2d(nfc[256], nc, 3, 1, 1, bias=False)
#
#         # residual_frequency_obtain
#         self.get_feat_16_residual_frequency = nn.Sequential(
#             nn.Upsample(scale_factor=2, mode='nearest'),
#             conv2d(nfc[8], nfc[8] * 2, 3, 1, 1, bias=False),
#             # convTranspose2d(in_planes, out_planes*2, 4, 2, 1, bias=False),
#             batchNorm2d(nfc[8] * 2),
#             GLU(),
#             BlockFFT(nfc[8], 16, 16),
#             conv2d(nfc[8], nfc[16], 3, 1, 1, bias=False),
#             batchNorm2d(nfc[16])
#         )
#
#         self.get_feat_32_residual_frequency = nn.Sequential(
#             nn.Upsample(scale_factor=2, mode='nearest'),
#             conv2d(nfc[16], nfc[16] * 2, 3, 1, 1, bias=False),
#             # convTranspose2d(in_planes, out_planes*2, 4, 2, 1, bias=False),
#             batchNorm2d(nfc[16] * 2),
#             GLU(),
#             BlockFFT(nfc[16], 32, 32),
#             conv2d(nfc[16], nfc[32], 3, 1, 1, bias=False),
#             batchNorm2d(nfc[32])
#         )
#
#         self.get_feat_64_residual_frequency = nn.Sequential(
#             nn.Upsample(scale_factor=2, mode='nearest'),
#             conv2d(nfc[32], nfc[32] * 2, 3, 1, 1, bias=False),
#             # convTranspose2d(in_planes, out_planes*2, 4, 2, 1, bias=False),
#             batchNorm2d(nfc[32] * 2),
#             GLU(),
#             BlockFFT(nfc[32], 64, 64),
#             conv2d(nfc[32], nfc[64], 3, 1, 1, bias=False),
#             batchNorm2d(nfc[64])
#         )
#
#         self.get_feat_128_residual_frequency = nn.Sequential(
#             nn.Upsample(scale_factor=2, mode='nearest'),
#             conv2d(nfc[64], nfc[64] * 2, 3, 1, 1, bias=False),
#             # convTranspose2d(in_planes, out_planes*2, 4, 2, 1, bias=False),
#             batchNorm2d(nfc[64] * 2),
#             GLU(),
#             BlockFFT(nfc[64], 128, 128),
#             conv2d(nfc[64], nfc[128], 3, 1, 1, bias=False),
#             batchNorm2d(nfc[128])
#         )
#
#         self.get_feat_256_residual_frequency = nn.Sequential(
#             nn.Upsample(scale_factor=2, mode='nearest'),
#             conv2d(nfc[128], nfc[128] * 2, 3, 1, 1, bias=False),
#             # convTranspose2d(in_planes, out_planes*2, 4, 2, 1, bias=False),
#             batchNorm2d(nfc[128] * 2),
#             GLU(),
#             BlockFFT(nfc[128], 256, 256),
#             conv2d(nfc[128], nfc[256], 3, 1, 1, bias=False),
#             batchNorm2d(nfc[256])
#         )
#
#         if im_size > 256:
#             self.get_feat_512_residual_frequency = nn.Sequential(
#                 nn.Upsample(scale_factor=2, mode='nearest'),
#                 conv2d(nfc[256], nfc[256] * 2, 3, 1, 1, bias=False),
#                 # convTranspose2d(in_planes, out_planes*2, 4, 2, 1, bias=False),
#                 batchNorm2d(nfc[256] * 2),
#                 GLU(),
#                 BlockFFT(nfc[256], 512, 512),
#                 conv2d(nfc[256], nfc[512], 3, 1, 1, bias=False),
#                 batchNorm2d(nfc[512])
#             )
#             self.feat_512 = UpBlockComp(nfc[256], nfc[512])
#             self.feat_512_residual_fequency_output = conv2d(nfc[512], nc, 3, 1, 1, bias=False)
#             self.feat_512_output = conv2d(nfc[512], nc, 3, 1, 1, bias=False)
#         if im_size > 512:
#             self.get_feat_1024_residual_frequency = nn.Sequential(
#                 nn.Upsample(scale_factor=2, mode='nearest'),
#                 conv2d(nfc[512], nfc[512] * 2, 3, 1, 1, bias=False),
#                 # convTranspose2d(in_planes, out_planes*2, 4, 2, 1, bias=False),
#                 batchNorm2d(nfc[512] * 2),
#                 GLU(),
#                 BlockFFT(nfc[512], 1024, 1024),
#                 conv2d(nfc[512], nfc[1024], 3, 1, 1, bias=False),
#                 batchNorm2d(nfc[1024])
#             )
#             self.feat_1024 = UpBlock(nfc[512], nfc[1024])
#             self.feat_1024_residual_fequency_output = conv2d(nfc[1024], nc, 3, 1, 1, bias=False)
#             self.feat_1024_output = conv2d(nfc[1024], nc, 3, 1, 1, bias=False)
#
#     def forward(self, input, skips=False):
#         if skips:
#             feat_4 = self.init(input)  # input: b*256, feat_4: b*1024*4*4
#             feat_8 = self.feat_8(feat_4)  # feat_4: b*1024*4*4, feat_8: b*512*8*8
#
#             feat_16 = self.feat_16(feat_8)  # feat_8: b*512*8*8, feat_16: b*256*16*16
#             feat_16_residual_frequency = self.get_feat_16_residual_frequency(feat_8)
#
#             feat_32 = self.feat_32(feat_16)  # feat_16: b*256*16*16, feat_32: b*128*32*32
#             feat_32_residual_frequency = self.get_feat_32_residual_frequency(feat_16_residual_frequency)
#
#             feat_64 = self.feat_64(feat_32)
#             feat_64_residual_frequency = self.get_feat_64_residual_frequency(feat_32_residual_frequency)
#
#             feat_128 = self.feat_128(feat_64)
#             feat_128_residual_frequency = self.get_feat_128_residual_frequency(feat_64_residual_frequency)
#
#             # feat_64 = self.se_64(feat_4, self.feat_64(feat_32)) # feat_32: b*128*32*32, feat_64: b*128*64*64
#             feat_256 = self.feat_256(feat_128)
#             feat_256_residual_frequency = self.get_feat_256_residual_frequency(feat_128_residual_frequency)
#             complete_feat_256 = feat_256 + feat_256_residual_frequency
#
#             # feat_128 = self.se_128(feat_8, self.feat_128(feat_64)) # feat_64: b*128*64*64, feat_128: b*64*128*128
#             # feat_256 = self.se_256(feat_16, self.feat_256(feat_128)) # feat_128: b*642*128*128, feat_256: b*32*256*256
#             # here only obtain 8x8, 16x16 and 32x32 to do alignment, the left to align residual frequencies
#             if self.im_size == 256:
#                 return self.to_big(complete_feat_256)
#
#             feat_512 = self.feat_512(feat_256)
#             feat_512_residual_frequency = self.get_feat_512_residual_frequency(feat_256_residual_frequency)
#             complete_feat_512 = feat_512 + feat_512_residual_frequency
#             if self.im_size == 512:
#                 return self.to_big(complete_feat_512)
#
#             feat_1024 = self.feat_1024(feat_512)  # feat_512: b*16*512*512, feat_1024: b*8*1024*1024
#             feat_1024_residual_frequency = self.get_feat_1024_residual_frequency(feat_512_residual_frequency)
#             complete_feat_1024 = feat_1024 + feat_1024_residual_frequency
#
#             im_1024 = torch.tanh(self.to_big(complete_feat_1024))
#
#             return im_1024
#
#         else:
#             feat_4 = self.init(input) # input: b*256, feat_4: b*1024*4*4
#             feat_8 = self.feat_8(feat_4) # feat_4: b*1024*4*4, feat_8: b*512*8*8
#             feat_8_alignment = self.feat_8_output(feat_8)
#
#             feat_16 = self.feat_16(feat_8) # feat_8: b*512*8*8, feat_16: b*256*16*16
#             feat_16_residual_frequency = self.get_feat_16_residual_frequency(feat_8)
#             feat_16_residual_frequency_alignment = self.feat_16_residual_fequency_output(feat_16_residual_frequency)
#             feat_16_alignment = self.feat_16_output(feat_16)
#             complete_feat_16 = feat_16 + feat_16_residual_frequency
#             complete_feat_16_output = self.feat_16_output(complete_feat_16)
#
#             feat_32 = self.feat_32(feat_16) # feat_16: b*256*16*16, feat_32: b*128*32*32
#             feat_32_residual_frequency = self.get_feat_32_residual_frequency(feat_16_residual_frequency)
#             feat_32_residual_frequency_alignment = self.feat_32_residual_fequency_output(feat_32_residual_frequency)
#             feat_32_alignment = self.feat_32_output(feat_32)
#             complete_feat_32 = feat_32 + feat_32_residual_frequency
#             complete_feat_32_output = self.feat_32_output(complete_feat_32)
#
#             feat_64 = self.feat_64(feat_32)
#             feat_64_residual_frequency = self.get_feat_64_residual_frequency(feat_32_residual_frequency)
#             feat_64_residual_frequency_alignment = self.feat_64_residual_fequency_output(feat_64_residual_frequency)
#             feat_64_alignment = self.feat_64_output(feat_64)
#             complete_feat_64 = feat_64 + feat_64_residual_frequency
#             complete_feat_64_output = self.feat_64_output(complete_feat_64)
#
#             feat_128 = self.feat_128(feat_64)
#             feat_128_residual_frequency = self.get_feat_128_residual_frequency(feat_64_residual_frequency)
#             feat_128_residual_frequency_alignment = self.feat_128_residual_fequency_output(feat_128_residual_frequency)
#             feat_128_alignment = self.feat_128_output(feat_128)
#             complete_feat_128 = feat_128 + feat_128_residual_frequency
#
#             # feat_64 = self.se_64(feat_4, self.feat_64(feat_32)) # feat_32: b*128*32*32, feat_64: b*128*64*64
#             feat_256 = self.feat_256(feat_128)
#             feat_256_residual_frequency = self.get_feat_256_residual_frequency(feat_128_residual_frequency)
#             feat_256_residual_frequency_alignment = self.feat_256_residual_fequency_output(feat_256_residual_frequency)
#             feat_256_alignment = self.feat_256_output(feat_256)
#             complete_feat_256 = feat_256 + feat_256_residual_frequency
#             complete_feat_256_output = self.feat_256_output(complete_feat_256)
#
#             # feat_128 = self.se_128(feat_8, self.feat_128(feat_64)) # feat_64: b*128*64*64, feat_128: b*64*128*128
#             # feat_256 = self.se_256(feat_16, self.feat_256(feat_128)) # feat_128: b*642*128*128, feat_256: b*32*256*256
#             # here only obtain 8x8, 16x16 and 32x32 to do alignment, the left to align residual frequencies
#             if self.im_size == 256:
#
#                 return [self.to_big(complete_feat_256), self.to_128(complete_feat_128)], [feat_256_residual_frequency_alignment,
#                         feat_128_residual_frequency_alignment, feat_64_residual_frequency_alignment, feat_32_residual_frequency_alignment, feat_16_residual_frequency_alignment, feat_8_alignment], [feat_64_alignment, feat_32_alignment, feat_16_alignment]
#
#             feat_512 = self.feat_512(feat_256)
#             feat_512_residual_frequency = self.get_feat_512_residual_frequency(feat_256_residual_frequency)
#             feat_512_residual_frequency_alignment = self.feat_512_residual_fequency_output(feat_512_residual_frequency)
#             feat_512_alignment = self.feat_512_output(feat_512)
#             complete_feat_512 = feat_512 + feat_512_residual_frequency
#             complete_feat_512_output = self.feat_512_output(complete_feat_512)
#
#             # feat_512 = self.se_512(feat_32, self.feat_512(feat_256)) # feat_256: b*32*256*256, feat_512: b*16*512*512
#             if self.im_size == 512:
#                 return [self.to_big(complete_feat_512), complete_feat_256_output, self.to_128(complete_feat_128)], [feat_512_residual_frequency_alignment, feat_256_residual_frequency_alignment,
#                         feat_128_residual_frequency_alignment, feat_64_residual_frequency_alignment, feat_32_residual_frequency_alignment, feat_16_residual_frequency_alignment, feat_8_alignment], [feat_64_alignment, feat_32_alignment, feat_16_alignment]
#
#             feat_1024 = self.feat_1024(feat_512) # feat_512: b*16*512*512, feat_1024: b*8*1024*1024
#             feat_1024_residual_frequency = self.get_feat_1024_residual_frequency(feat_512_residual_frequency)
#             feat_1024_residual_frequency_alignment = self.feat_1024_residual_fequency_output(feat_1024_residual_frequency)
#             feat_1024_alignment = self.feat_1024_output(feat_1024)
#             complete_feat_1024 = feat_1024 + feat_1024_residual_frequency
#
#             im_128 = torch.tanh(self.to_128(complete_feat_128))
#             im_1024 = torch.tanh(self.to_big(complete_feat_1024))
#
#             return [im_1024, torch.tanh(complete_feat_512_output), torch.tanh(complete_feat_256_output), im_128], [feat_1024_residual_frequency_alignment, feat_512_residual_frequency_alignment, feat_256_residual_frequency_alignment,
#                         feat_128_residual_frequency_alignment, feat_64_residual_frequency_alignment, feat_32_residual_frequency_alignment, feat_16_residual_frequency_alignment, feat_8_alignment], [feat_64_alignment, feat_32_alignment, feat_16_alignment]


# class Generator(nn.Module):
#     def __init__(self, ngf=64, nz=100, nc=3, im_size=1024, kernel_size=5, num_high=7):
#         super(Generator, self).__init__()
#
#         nfc_multi = {4: 16, 8: 8, 16: 4, 32: 2, 64: 2, 128: 1, 256: 0.5, 512: 0.25, 1024: 0.125}
#         nfc = {}
#         for k, v in nfc_multi.items():
#             nfc[k] = int(v * ngf)
#
#         self.kernel = self.gauss_kernel(kernel_size, nc)
#         self.num_high = num_high
#
#         self.im_size = im_size
#
#         self.init = InitLayer(nz, channel=nfc[4])
#
#         self.feat_8 = UpBlockComp(nfc[4], nfc[8])
#         self.feat_16 = UpBlock(nfc[8], nfc[16])
#         self.feat_32 = UpBlockComp(nfc[16], nfc[32])
#         self.feat_32_output = conv2d(nfc[32], nc, 1, 1, 0, bias=False)
#
#         self.feat_64_residual_fequency_output = conv2d(nfc[64], nc, 1, 1, 0, bias=False)
#         self.feat_128_residual_fequency_output = conv2d(nfc[128], nc, 1, 1, 0, bias=False)
#         self.feat_256_residual_fequency_output = conv2d(nfc[256], nc, 3, 1, 1, bias=False)
#
#         # residual_frequency_obtain
#         self.get_feat_32_residual_frequency = UpBlockComp(nfc[16], nfc[32])
#         self.get_feat_64_residual_frequency = UpBlock(nfc[32], nfc[64])
#         self.get_feat_128_residual_frequency = UpBlockComp(nfc[64], nfc[128])
#         self.get_feat_256_residual_frequency = UpBlock(nfc[128], nfc[256])
#
#         if im_size > 256:
#             self.get_feat_512_residual_frequency = UpBlockComp(nfc[256], nfc[512])
#             self.feat_512_residual_fequency_output = conv2d(nfc[512], nc, 3, 1, 1, bias=False)
#             self.feat_512_output = conv2d(nfc[512], nc, 3, 1, 1, bias=False)
#         if im_size > 512:
#             self.get_feat_1024_residual_frequency = UpBlock(nfc[512], nfc[1024])
#             self.feat_1024_residual_fequency_output = conv2d(nfc[1024], nc, 3, 1, 1, bias=False)
#             self.feat_1024_output = conv2d(nfc[1024], nc, 3, 1, 1, bias=False)
#
#     def downsample(self, x):
#         return x[:, :, ::2, ::2]
#
#     def pyramid_down(self, x):
#         return self.downsample(self.conv_gauss(x, self.kernel))
#
#     def gauss_kernel(self, kernel_size, channels):
#         kernel = cv2.getGaussianKernel(kernel_size, 0).dot(
#             cv2.getGaussianKernel(kernel_size, 0).T)
#         kernel = torch.FloatTensor(kernel).unsqueeze(0).repeat(
#             channels, 1, 1, 1)
#         kernel = torch.nn.Parameter(data=kernel, requires_grad=False)
#         return kernel
#
#     def conv_gauss(self, x, kernel):
#         n_channels, _, kw, kh = kernel.shape
#         kernel_cuda = kernel.to(x.device)
#
#         x = torch.nn.functional.pad(x, (kw // 2, kh // 2, kw // 2, kh // 2),
#                                     mode='reflect')  # replicate    # reflect
#         x = torch.nn.functional.conv2d(x, kernel_cuda, groups=n_channels)
#
#         return x
#
#     def upsample(self, x):
#         up = torch.zeros((x.size(0), x.size(1), x.size(2) * 2, x.size(3) * 2),
#                          device=x.device)
#         up[:, :, ::2, ::2] = x * 4
#
#         return self.conv_gauss(up, self.kernel)
#
#     def pyramid_decom(self, img):
#         # self.kernel = self.kernel.to(img.device)
#         current = img
#         pyr = []
#         subtrahend = []
#         minuends = []
#         minuends.append(current)
#         for _ in range(self.num_high):
#             down = self.pyramid_down(current)
#             minuends.append(down)
#             up = self.upsample(down)
#             diff = current - up
#             pyr.append(diff)
#             subtrahend.append(up)
#             current = down
#         pyr.append(current)
#         return pyr, subtrahend, minuends
#
#     def pyramid_recons(self, pyr):
#         image = pyr[0]
#         for level in pyr[1:]:
#             up = self.upsample(image)
#             image = up + level
#         return image
#
#     def forward(self, input, skips=False):
#         if skips:
#             feat_4 = self.init(input)  # input: b*256, feat_4: b*1024*4*4
#             feat_8 = self.feat_8(feat_4)  # feat_4: b*1024*4*4, feat_8: b*512*8*8
#             feat_16 = self.feat_16(feat_8)
#             feat_32 = self.feat_32(feat_16)
#             # feat_8_alignment = self.feat_8_output(feat_8)
#
#             # feat_16_alignment = self.upsample(feat_8_alignment)
#             # feat_16_residual_frequency_alignment = self.feat_16_residual_fequency_output(feat_16_residual_frequency)
#             # complete_feat_16 = feat_16_residual_frequency_alignment + feat_16_alignment
#
#             feat_32_alignment = self.feat_32_output(feat_32)
#             feat_32_residual_frequency = self.get_feat_32_residual_frequency(feat_16)
#
#             feat_64_alignment = self.upsample(feat_32_alignment)
#             feat_64_residual_frequency = self.get_feat_64_residual_frequency(feat_32_residual_frequency)
#             feat_64_residual_frequency_alignment = self.feat_64_residual_fequency_output(feat_64_residual_frequency)
#             complete_feat_64 = feat_64_alignment + feat_64_residual_frequency_alignment
#
#             feat_128_alignment = self.upsample(complete_feat_64)
#             feat_128_residual_frequency = self.get_feat_128_residual_frequency(feat_64_residual_frequency)
#             feat_128_residual_frequency_alignment = self.feat_128_residual_fequency_output(feat_128_residual_frequency)
#             complete_feat_128 = feat_128_alignment + feat_128_residual_frequency_alignment
#
#             feat_256_alignment = self.upsample(complete_feat_128)
#             feat_256_residual_frequency = self.get_feat_256_residual_frequency(feat_128_residual_frequency)
#             feat_256_residual_frequency_alignment = self.feat_256_residual_fequency_output(feat_256_residual_frequency)
#             complete_feat_256 = feat_256_alignment + feat_256_residual_frequency_alignment
#
#             if self.im_size == 256:
#                 return complete_feat_256
#
#             feat_512_alignment = self.upsample(complete_feat_256)
#             feat_512_residual_frequency = self.get_feat_512_residual_frequency(feat_256_residual_frequency)
#             feat_512_residual_frequency_alignment = self.feat_512_residual_fequency_output(feat_512_residual_frequency)
#             complete_feat_512 = feat_512_alignment + feat_512_residual_frequency_alignment
#
#             if self.im_size == 512:
#                 return complete_feat_512
#
#             feat_1024_alignment = self.upsample(complete_feat_512)
#             feat_1024_residual_frequency = self.get_feat_1024_residual_frequency(feat_512_residual_frequency)
#             feat_1024_residual_frequency_alignment = self.feat_1024_residual_fequency_output(
#                 feat_1024_residual_frequency)
#
#             complete_feat_1024 = feat_1024_alignment + feat_1024_residual_frequency_alignment
#
#             im_128 = torch.tanh(complete_feat_128)
#             im_1024 = torch.tanh(complete_feat_1024)
#
#             return im_1024
#         else:
#             feat_4 = self.init(input) # input: b*256, feat_4: b*1024*4*4
#             feat_8 = self.feat_8(feat_4) # feat_4: b*1024*4*4, feat_8: b*512*8*8
#             feat_16 = self.feat_16(feat_8)
#             feat_32 = self.feat_32(feat_16)
#             # feat_8_alignment = self.feat_8_output(feat_8)
#
#             # feat_16_alignment = self.upsample(feat_8_alignment)
#             # feat_16_residual_frequency_alignment = self.feat_16_residual_fequency_output(feat_16_residual_frequency)
#             # complete_feat_16 = feat_16_residual_frequency_alignment + feat_16_alignment
#
#             feat_32_alignment = self.feat_32_output(feat_32)
#             feat_32_residual_frequency = self.get_feat_32_residual_frequency(feat_16)
#
#             feat_64_alignment = self.upsample(feat_32_alignment)
#             feat_64_residual_frequency = self.get_feat_64_residual_frequency(feat_32_residual_frequency)
#             feat_64_residual_frequency_alignment = self.feat_64_residual_fequency_output(feat_64_residual_frequency)
#             complete_feat_64 = feat_64_alignment + feat_64_residual_frequency_alignment
#
#             feat_128_alignment = self.upsample(complete_feat_64)
#             feat_128_residual_frequency = self.get_feat_128_residual_frequency(feat_64_residual_frequency)
#             feat_128_residual_frequency_alignment = self.feat_128_residual_fequency_output(feat_128_residual_frequency)
#             complete_feat_128 = feat_128_alignment + feat_128_residual_frequency_alignment
#
#             feat_256_alignment = self.upsample(complete_feat_128)
#             feat_256_residual_frequency = self.get_feat_256_residual_frequency(feat_128_residual_frequency)
#             feat_256_residual_frequency_alignment = self.feat_256_residual_fequency_output(feat_256_residual_frequency)
#             complete_feat_256 = feat_256_alignment + feat_256_residual_frequency_alignment
#
#             if self.im_size == 256:
#                 return [complete_feat_256, complete_feat_128], feat_32_alignment
#
#             feat_512_alignment = self.upsample(complete_feat_256)
#             feat_512_residual_frequency = self.get_feat_512_residual_frequency(feat_256_residual_frequency)
#             feat_512_residual_frequency_alignment = self.feat_512_residual_fequency_output(feat_512_residual_frequency)
#             complete_feat_512 = feat_512_alignment + feat_512_residual_frequency_alignment
#
#             if self.im_size == 512:
#                 return [complete_feat_512, complete_feat_256, complete_feat_128], feat_32_alignment
#
#             feat_1024_alignment = self.upsample(complete_feat_512)
#             feat_1024_residual_frequency = self.get_feat_1024_residual_frequency(feat_512_residual_frequency)
#             feat_1024_residual_frequency_alignment = self.feat_1024_residual_fequency_output(feat_1024_residual_frequency)
#
#             complete_feat_1024 = feat_1024_alignment + feat_1024_residual_frequency_alignment
#
#             im_128 = torch.tanh(complete_feat_128)
#             im_1024 = torch.tanh(complete_feat_1024)
#
#             return [im_1024, torch.tanh(complete_feat_512), torch.tanh(complete_feat_256), im_128], feat_32_alignment


#
# class Generator(nn.Module):
#     def __init__(self, ngf=64, nz=100, nc=3, im_size=1024, kernel_size=5, num_high=7):
#         super(Generator, self).__init__()
#
#         nfc_multi = {4: 16, 8: 8, 16: 4, 32: 2, 64: 2, 128: 1, 256: 0.5, 512: 0.25, 1024: 0.125}
#         nfc = {}
#         for k, v in nfc_multi.items():
#             nfc[k] = int(v * ngf)
#
#         self.kernel = self.gauss_kernel(kernel_size, nc)
#         self.num_high = num_high
#
#         self.im_size = im_size
#
#         self.init = InitLayer(nz, channel=nfc[4])
#
#         self.feat_8 = UpBlockComp(nfc[4], nfc[8])
#         self.feat_16 = UpBlock(nfc[8], nfc[16])
#         self.feat_32 = UpBlockComp(nfc[16], nfc[32])
#         self.feat_32_output = conv2d(nfc[32], nc, 1, 1, 0, bias=False)
#
#         self.feat_64_residual_fequency_output = conv2d(nfc[64], nc, 1, 1, 0, bias=False)
#         self.feat_128_residual_fequency_output = conv2d(nfc[128], nc, 1, 1, 0, bias=False)
#         self.feat_256_residual_fequency_output = conv2d(nfc[256], nc, 3, 1, 1, bias=False)
#
#         # residual_frequency_obtain
#         self.get_feat_32_residual_frequency = UpBlockComp(nfc[16], nfc[32])
#         self.get_feat_64_residual_frequency = UpBlock(nfc[32], nfc[64])
#         self.get_feat_128_residual_frequency = UpBlockComp(nfc[64], nfc[128])
#         self.get_feat_256_residual_frequency = UpBlock(nfc[128], nfc[256])
#
#         if im_size > 256:
#             self.get_feat_512_residual_frequency = UpBlockComp(nfc[256], nfc[512])
#             self.feat_512_residual_fequency_output = conv2d(nfc[512], nc, 3, 1, 1, bias=False)
#             self.feat_512_output = conv2d(nfc[512], nc, 3, 1, 1, bias=False)
#         if im_size > 512:
#             self.get_feat_1024_residual_frequency = UpBlock(nfc[512], nfc[1024])
#             self.feat_1024_residual_fequency_output = conv2d(nfc[1024], nc, 3, 1, 1, bias=False)
#             self.feat_1024_output = conv2d(nfc[1024], nc, 3, 1, 1, bias=False)
#
#     def downsample(self, x):
#         return x[:, :, ::2, ::2]
#
#     def pyramid_down(self, x):
#         return self.downsample(self.conv_gauss(x, self.kernel))
#
#     def gauss_kernel(self, kernel_size, channels):
#         kernel = cv2.getGaussianKernel(kernel_size, 0).dot(
#             cv2.getGaussianKernel(kernel_size, 0).T)
#         kernel = torch.FloatTensor(kernel).unsqueeze(0).repeat(
#             channels, 1, 1, 1)
#         kernel = torch.nn.Parameter(data=kernel, requires_grad=False)
#         return kernel
#
#     def conv_gauss(self, x, kernel):
#         n_channels, _, kw, kh = kernel.shape
#         kernel_cuda = kernel.to(x.device)
#
#         x = torch.nn.functional.pad(x, (kw // 2, kh // 2, kw // 2, kh // 2),
#                                     mode='reflect')  # replicate    # reflect
#         x = torch.nn.functional.conv2d(x, kernel_cuda, groups=n_channels)
#
#         return x
#
#     def upsample(self, x):
#         up = torch.zeros((x.size(0), x.size(1), x.size(2) * 2, x.size(3) * 2),
#                          device=x.device)
#         up[:, :, ::2, ::2] = x * 4
#
#         return self.conv_gauss(up, self.kernel)
#
#     def pyramid_decom(self, img):
#         # self.kernel = self.kernel.to(img.device)
#         current = img
#         pyr = []
#         subtrahend = []
#         minuends = []
#         minuends.append(current)
#         for _ in range(self.num_high):
#             down = self.pyramid_down(current)
#             minuends.append(down)
#             up = self.upsample(down)
#             diff = current - up
#             pyr.append(diff)
#             subtrahend.append(up)
#             current = down
#         pyr.append(current)
#         return pyr, subtrahend, minuends
#
#     def pyramid_recons(self, pyr):
#         image = pyr[0]
#         for level in pyr[1:]:
#             up = self.upsample(image)
#             image = up + level
#         return image
#
#     def forward(self, input, skips=False):
#         if skips:
#             feat_4 = self.init(input)  # input: b*256, feat_4: b*1024*4*4
#             feat_8 = self.feat_8(feat_4)  # feat_4: b*1024*4*4, feat_8: b*512*8*8
#             feat_16 = self.feat_16(feat_8)
#             feat_32 = self.feat_32(feat_16)
#             # feat_8_alignment = self.feat_8_output(feat_8)
#
#             # feat_16_alignment = self.upsample(feat_8_alignment)
#             # feat_16_residual_frequency_alignment = self.feat_16_residual_fequency_output(feat_16_residual_frequency)
#             # complete_feat_16 = feat_16_residual_frequency_alignment + feat_16_alignment
#
#             feat_32_alignment = self.feat_32_output(feat_32)
#             feat_32_residual_frequency = self.get_feat_32_residual_frequency(feat_16)
#
#             feat_64_alignment = self.upsample(feat_32_alignment)
#             feat_64_residual_frequency = self.get_feat_64_residual_frequency(feat_32_residual_frequency)
#             feat_64_residual_frequency_alignment = self.feat_64_residual_fequency_output(feat_64_residual_frequency)
#             complete_feat_64 = feat_64_alignment + feat_64_residual_frequency_alignment
#
#             feat_128_alignment = self.upsample(complete_feat_64)
#             feat_128_residual_frequency = self.get_feat_128_residual_frequency(feat_64_residual_frequency)
#             feat_128_residual_frequency_alignment = self.feat_128_residual_fequency_output(feat_128_residual_frequency)
#             complete_feat_128 = feat_128_alignment + feat_128_residual_frequency_alignment
#
#             feat_256_alignment = self.upsample(complete_feat_128)
#             feat_256_residual_frequency = self.get_feat_256_residual_frequency(feat_128_residual_frequency)
#             feat_256_residual_frequency_alignment = self.feat_256_residual_fequency_output(feat_256_residual_frequency)
#             complete_feat_256 = feat_256_alignment + feat_256_residual_frequency_alignment
#
#             if self.im_size == 256:
#                 return complete_feat_256
#
#             feat_512_alignment = self.upsample(complete_feat_256)
#             feat_512_residual_frequency = self.get_feat_512_residual_frequency(feat_256_residual_frequency)
#             feat_512_residual_frequency_alignment = self.feat_512_residual_fequency_output(feat_512_residual_frequency)
#             complete_feat_512 = feat_512_alignment + feat_512_residual_frequency_alignment
#
#             if self.im_size == 512:
#                 return complete_feat_512
#
#             feat_1024_alignment = self.upsample(complete_feat_512)
#             feat_1024_residual_frequency = self.get_feat_1024_residual_frequency(feat_512_residual_frequency)
#             feat_1024_residual_frequency_alignment = self.feat_1024_residual_fequency_output(
#                 feat_1024_residual_frequency)
#
#             complete_feat_1024 = feat_1024_alignment + feat_1024_residual_frequency_alignment
#
#             im_1024 = torch.tanh(complete_feat_1024)
#
#             return im_1024
#         else:
#             feat_4 = self.init(input) # input: b*256, feat_4: b*1024*4*4
#             feat_8 = self.feat_8(feat_4) # feat_4: b*1024*4*4, feat_8: b*512*8*8
#             feat_16 = self.feat_16(feat_8)
#             feat_32 = self.feat_32(feat_16)
#
#             feat_32_alignment = self.feat_32_output(feat_32)
#             feat_32_residual_frequency = self.get_feat_32_residual_frequency(feat_16)
#
#             feat_64_alignment = self.upsample(feat_32_alignment)
#             feat_64_residual_frequency = self.get_feat_64_residual_frequency(feat_32_residual_frequency)
#             feat_64_residual_frequency_alignment = self.feat_64_residual_fequency_output(feat_64_residual_frequency)
#             complete_feat_64 = feat_64_alignment + feat_64_residual_frequency_alignment
#
#             feat_128_alignment = self.upsample(complete_feat_64)
#             feat_128_residual_frequency = self.get_feat_128_residual_frequency(feat_64_residual_frequency)
#             feat_128_residual_frequency_alignment = self.feat_128_residual_fequency_output(feat_128_residual_frequency)
#             complete_feat_128 = feat_128_alignment + feat_128_residual_frequency_alignment
#
#             feat_256_alignment = self.upsample(complete_feat_128)
#             feat_256_residual_frequency = self.get_feat_256_residual_frequency(feat_128_residual_frequency)
#             feat_256_residual_frequency_alignment = self.feat_256_residual_fequency_output(feat_256_residual_frequency)
#             complete_feat_256 = feat_256_alignment + feat_256_residual_frequency_alignment
#
#             if self.im_size == 256:
#                 return complete_feat_256, feat_32_alignment
#
#             feat_512_alignment = self.upsample(complete_feat_256)
#             feat_512_residual_frequency = self.get_feat_512_residual_frequency(feat_256_residual_frequency)
#             feat_512_residual_frequency_alignment = self.feat_512_residual_fequency_output(feat_512_residual_frequency)
#             complete_feat_512 = feat_512_alignment + feat_512_residual_frequency_alignment
#
#             if self.im_size == 512:
#                 return complete_feat_512, feat_32_alignment
#
#             feat_1024_alignment = self.upsample(complete_feat_512)
#             feat_1024_residual_frequency = self.get_feat_1024_residual_frequency(feat_512_residual_frequency)
#             feat_1024_residual_frequency_alignment = self.feat_1024_residual_fequency_output(feat_1024_residual_frequency)
#
#             complete_feat_1024 = feat_1024_alignment + feat_1024_residual_frequency_alignment
#
#             im_1024 = torch.tanh(complete_feat_1024)
#
#             return im_1024, feat_32_alignment


# num_high=3
class Generator3(nn.Module):
    def __init__(self, ngf=64, nz=100, nc=3, im_size=1024, kernel_size=3, num_high=7):
        super(Generator3, self).__init__()

        nfc_multi = {4: 16, 8: 8, 16: 4, 32: 2, 64: 2, 128: 1, 256: 0.5, 512: 0.25, 1024: 0.125}
        nfc = {}
        for k, v in nfc_multi.items():
            nfc[k] = int(v * ngf)

        self.kernel = self.gauss_kernel(kernel_size, nc)
        self.num_high = num_high

        self.im_size = im_size

        self.init = InitLayer(nz, channel=nfc[4])

        self.feat_8 = UpBlockComp(nfc[4], nfc[8])
        self.feat_16 = UpBlock(nfc[8], nfc[16])
        self.feat_32 = UpBlockComp(nfc[16], nfc[32])

        self.to_32 = conv2d(nfc[32], nc, 1, 1, 0, bias=False)
        self.feat_64_residual_fequency_output = conv2d(nfc[64], nc, 1, 1, 0, bias=False)
        self.feat_128_residual_fequency_output = conv2d(nfc[128], nc, 1, 1, 0, bias=False)
        self.feat_256_residual_fequency_output = conv2d(nfc[256], nc, 3, 1, 1, bias=False)

        # residual_frequency_obtain
        self.get_feat_32_residual_frequency = UpBlockComp(nfc[16], nfc[32])
        self.get_feat_64_residual_frequency = UpBlock(nfc[32], nfc[64])
        self.get_feat_128_residual_frequency = UpBlockComp(nfc[64], nfc[128])
        self.get_feat_256_residual_frequency = UpBlock(nfc[128], nfc[256])

        if im_size > 256:
            self.get_feat_512_residual_frequency = UpBlockComp(nfc[256], nfc[512])
            self.feat_512_residual_fequency_output = conv2d(nfc[512], nc, 3, 1, 1, bias=False)
            self.feat_512_output = conv2d(nfc[512], nc, 3, 1, 1, bias=False)
        if im_size > 512:
            self.get_feat_1024_residual_frequency = UpBlock(nfc[512], nfc[1024])
            self.feat_1024_residual_fequency_output = conv2d(nfc[1024], nc, 3, 1, 1, bias=False)
            self.feat_1024_output = conv2d(nfc[1024], nc, 3, 1, 1, bias=False)

    def downsample(self, x):
        return x[:, :, ::2, ::2]

    def pyramid_down(self, x):
        return self.downsample(self.conv_gauss(x, self.kernel))

    def gauss_kernel(self, kernel_size, channels):
        kernel = cv2.getGaussianKernel(kernel_size, 0).dot(
            cv2.getGaussianKernel(kernel_size, 0).T)
        kernel = torch.FloatTensor(kernel).unsqueeze(0).repeat(
            channels, 1, 1, 1)
        kernel = torch.nn.Parameter(data=kernel, requires_grad=False)
        return kernel

    def conv_gauss(self, x, kernel):
        n_channels, _, kw, kh = kernel.shape
        kernel_cuda = kernel.to(x.device)

        x = torch.nn.functional.pad(x, (kw // 2, kh // 2, kw // 2, kh // 2),
                                    mode='reflect')  # replicate    # reflect
        x = torch.nn.functional.conv2d(x, kernel_cuda, groups=n_channels)

        return x

    def upsample(self, x):
        up = torch.zeros((x.size(0), x.size(1), x.size(2) * 2, x.size(3) * 2),
                         device=x.device)
        up[:, :, ::2, ::2] = x * 4

        return self.conv_gauss(up, self.kernel)

    def pyramid_decom(self, img):
        # self.kernel = self.kernel.to(img.device)
        current = img
        pyr = []
        subtrahend = []
        minuends = []
        minuends.append(current)
        for _ in range(self.num_high):
            down = self.pyramid_down(current)
            minuends.append(down)
            up = self.upsample(down)
            diff = current - up
            pyr.append(diff)
            subtrahend.append(up)
            current = down
        pyr.append(current)
        return pyr, subtrahend, minuends

    def pyramid_recons(self, pyr):
        image = pyr[0]
        for level in pyr[1:]:
            up = self.upsample(image)
            image = up + level
        return image

    def forward(self, input, skips=False):
        if skips:
            feat_4 = self.init(input)  # input: b*256, feat_4: b*1024*4*4
            feat_8 = self.feat_8(feat_4)  # feat_4: b*1024*4*4, feat_8: b*512*8*8
            feat_16 = self.feat_16(feat_8)
            feat_32 = self.feat_32(feat_16)

            feat_32_residual_frequency = self.get_feat_32_residual_frequency(feat_16)
            feat_32_alignment = self.to_32(feat_32)

            feat_64_alignment = self.upsample(feat_32_alignment)
            feat_64_residual_frequency = self.get_feat_64_residual_frequency(feat_32_residual_frequency)
            feat_64_residual_frequency_alignment = self.feat_64_residual_fequency_output(feat_64_residual_frequency)
            complete_feat_64 = feat_64_alignment + feat_64_residual_frequency_alignment

            feat_128_alignment = self.upsample(complete_feat_64)
            feat_128_residual_frequency = self.get_feat_128_residual_frequency(feat_64_residual_frequency)
            feat_128_residual_frequency_alignment = self.feat_128_residual_fequency_output(feat_128_residual_frequency)
            complete_feat_128 = feat_128_alignment + feat_128_residual_frequency_alignment

            feat_256_alignment = self.upsample(complete_feat_128)
            feat_256_residual_frequency = self.get_feat_256_residual_frequency(feat_128_residual_frequency)
            feat_256_residual_frequency_alignment = self.feat_256_residual_fequency_output(feat_256_residual_frequency)
            complete_feat_256 = feat_256_alignment + feat_256_residual_frequency_alignment

            if self.im_size == 256:
                return complete_feat_256

            feat_512_alignment = self.upsample(complete_feat_256)
            feat_512_residual_frequency = self.get_feat_512_residual_frequency(feat_256_residual_frequency)
            feat_512_residual_frequency_alignment = self.feat_512_residual_fequency_output(feat_512_residual_frequency)
            complete_feat_512 = feat_512_alignment + feat_512_residual_frequency_alignment

            if self.im_size == 512:
                return complete_feat_512

            feat_1024_alignment = self.upsample(complete_feat_512)
            feat_1024_residual_frequency = self.get_feat_1024_residual_frequency(feat_512_residual_frequency)
            feat_1024_residual_frequency_alignment = self.feat_1024_residual_fequency_output(
                feat_1024_residual_frequency)

            complete_feat_1024 = feat_1024_alignment + feat_1024_residual_frequency_alignment

            im_1024 = torch.tanh(complete_feat_1024)

            return im_1024
        else:
            feat_4 = self.init(input) # input: b*256, feat_4: b*1024*4*4
            feat_8 = self.feat_8(feat_4) # feat_4: b*1024*4*4, feat_8: b*512*8*8
            feat_16 = self.feat_16(feat_8)
            feat_32 = self.feat_32(feat_16)

            feat_32_residual_frequency = self.get_feat_32_residual_frequency(feat_16)
            feat_32_alignment = self.to_32(feat_32)

            feat_64_alignment = self.upsample(feat_32_alignment)
            feat_64_residual_frequency = self.get_feat_64_residual_frequency(feat_32_residual_frequency)
            feat_64_residual_frequency_alignment = self.feat_64_residual_fequency_output(feat_64_residual_frequency)
            complete_feat_64 = feat_64_alignment + feat_64_residual_frequency_alignment

            feat_128_alignment = self.upsample(complete_feat_64)
            feat_128_residual_frequency = self.get_feat_128_residual_frequency(feat_64_residual_frequency)
            feat_128_residual_frequency_alignment = self.feat_128_residual_fequency_output(feat_128_residual_frequency)
            complete_feat_128 = feat_128_alignment + feat_128_residual_frequency_alignment

            feat_256_alignment = self.upsample(complete_feat_128)
            feat_256_residual_frequency = self.get_feat_256_residual_frequency(feat_128_residual_frequency)
            feat_256_residual_frequency_alignment = self.feat_256_residual_fequency_output(feat_256_residual_frequency)
            complete_feat_256 = feat_256_alignment + feat_256_residual_frequency_alignment

            if self.im_size == 256:
                return [complete_feat_256, complete_feat_128, complete_feat_64, feat_32_alignment], [feat_256_residual_frequency_alignment, feat_128_residual_frequency_alignment, feat_64_residual_frequency_alignment, feat_32_alignment]

            feat_512_alignment = self.upsample(complete_feat_256)
            feat_512_residual_frequency = self.get_feat_512_residual_frequency(feat_256_residual_frequency)
            feat_512_residual_frequency_alignment = self.feat_512_residual_fequency_output(feat_512_residual_frequency)
            complete_feat_512 = feat_512_alignment + feat_512_residual_frequency_alignment

            if self.im_size == 512:
                return [complete_feat_512, complete_feat_256, complete_feat_128, complete_feat_64, feat_32_alignment], [feat_512_residual_frequency_alignment, feat_256_residual_frequency_alignment, feat_128_residual_frequency_alignment, feat_64_residual_frequency_alignment, feat_32_alignment]

            feat_1024_alignment = self.upsample(complete_feat_512)
            feat_1024_residual_frequency = self.get_feat_1024_residual_frequency(feat_512_residual_frequency)
            feat_1024_residual_frequency_alignment = self.feat_1024_residual_fequency_output(feat_1024_residual_frequency)

            complete_feat_1024 = feat_1024_alignment + feat_1024_residual_frequency_alignment

            im_1024 = torch.tanh(complete_feat_1024)

            return [im_1024, complete_feat_512, complete_feat_256, complete_feat_128, complete_feat_64, feat_32_alignment], [feat_1024_residual_frequency_alignment, feat_512_residual_frequency_alignment, feat_256_residual_frequency_alignment, feat_128_residual_frequency_alignment, feat_64_residual_frequency_alignment, feat_32_alignment]

# num_high=4
class Generator4(nn.Module):
    def __init__(self, ngf=64, nz=100, nc=3, im_size=1024, kernel_size=3, num_high=7):
        super(Generator4, self).__init__()

        nfc_multi = {4: 16, 8: 8, 16: 4, 32: 2, 64: 2, 128: 1, 256: 0.5, 512: 0.25, 1024: 0.125}
        nfc = {}
        for k, v in nfc_multi.items():
            nfc[k] = int(v * ngf)

        self.kernel = self.gauss_kernel(kernel_size, nc)
        self.num_high = num_high

        self.im_size = im_size

        self.init = InitLayer(nz, channel=nfc[4])

        self.feat_8 = UpBlockComp(nfc[4], nfc[8])
        self.feat_16 = UpBlock(nfc[8], nfc[16])

        self.to_16 = conv2d(nfc[16], nc, 1, 1, 0, bias=False)
        self.feat_32_residual_fequency_output = conv2d(nfc[32], nc, 1, 1, 0, bias=False)
        self.feat_64_residual_fequency_output = conv2d(nfc[64], nc, 1, 1, 0, bias=False)
        self.feat_128_residual_fequency_output = conv2d(nfc[128], nc, 1, 1, 0, bias=False)
        self.feat_256_residual_fequency_output = conv2d(nfc[256], nc, 3, 1, 1, bias=False)

        # residual_frequency_obtain
        self.get_feat_16_residual_frequency = UpBlock(nfc[8], nfc[16])
        self.get_feat_32_residual_frequency = UpBlockComp(nfc[16], nfc[32])
        self.get_feat_64_residual_frequency = UpBlock(nfc[32], nfc[64])
        self.get_feat_128_residual_frequency = UpBlockComp(nfc[64], nfc[128])
        self.get_feat_256_residual_frequency = UpBlock(nfc[128], nfc[256])

        if im_size > 256:
            self.get_feat_512_residual_frequency = UpBlockComp(nfc[256], nfc[512])
            self.feat_512_residual_fequency_output = conv2d(nfc[512], nc, 3, 1, 1, bias=False)
            self.feat_512_output = conv2d(nfc[512], nc, 3, 1, 1, bias=False)
        if im_size > 512:
            self.get_feat_1024_residual_frequency = UpBlock(nfc[512], nfc[1024])
            self.feat_1024_residual_fequency_output = conv2d(nfc[1024], nc, 3, 1, 1, bias=False)
            self.feat_1024_output = conv2d(nfc[1024], nc, 3, 1, 1, bias=False)

    def downsample(self, x):
        return x[:, :, ::2, ::2]

    def pyramid_down(self, x):
        return self.downsample(self.conv_gauss(x, self.kernel))

    def gauss_kernel(self, kernel_size, channels):
        kernel = cv2.getGaussianKernel(kernel_size, 0).dot(
            cv2.getGaussianKernel(kernel_size, 0).T)
        kernel = torch.FloatTensor(kernel).unsqueeze(0).repeat(
            channels, 1, 1, 1)
        kernel = torch.nn.Parameter(data=kernel, requires_grad=False)
        return kernel

    def conv_gauss(self, x, kernel):
        n_channels, _, kw, kh = kernel.shape
        kernel_cuda = kernel.to(x.device)

        x = torch.nn.functional.pad(x, (kw // 2, kh // 2, kw // 2, kh // 2),
                                    mode='reflect')  # replicate    # reflect
        x = torch.nn.functional.conv2d(x, kernel_cuda, groups=n_channels)

        return x

    def upsample(self, x):
        up = torch.zeros((x.size(0), x.size(1), x.size(2) * 2, x.size(3) * 2),
                         device=x.device)
        up[:, :, ::2, ::2] = x * 4

        return self.conv_gauss(up, self.kernel)

    def pyramid_decom(self, img):
        # self.kernel = self.kernel.to(img.device)
        current = img
        pyr = []
        subtrahend = []
        minuends = []
        minuends.append(current)
        for _ in range(self.num_high):
            down = self.pyramid_down(current)
            minuends.append(down)
            up = self.upsample(down)
            diff = current - up
            pyr.append(diff)
            subtrahend.append(up)
            current = down
        pyr.append(current)
        return pyr, subtrahend, minuends

    def pyramid_recons(self, pyr):
        image = pyr[0]
        for level in pyr[1:]:
            up = self.upsample(image)
            image = up + level
        return image

    def forward(self, input, skips=False):
        if skips:
            feat_4 = self.init(input)  # input: b*256, feat_4: b*1024*4*4
            feat_8 = self.feat_8(feat_4)  # feat_4: b*1024*4*4, feat_8: b*512*8*8
            feat_16 = self.feat_16(feat_8)

            feat_16_residual_frequency = self.get_feat_16_residual_frequency(feat_8)
            feat_16_alignment = self.to_16(feat_16)

            feat_32_alignment = self.upsample(feat_16_alignment)
            feat_32_residual_frequency = self.get_feat_32_residual_frequency(feat_16_residual_frequency)
            feat_32_residual_frequency_alignment = self.feat_32_residual_fequency_output(feat_32_residual_frequency)
            complete_feat_32 = feat_32_alignment + feat_32_residual_frequency_alignment

            feat_64_alignment = self.upsample(complete_feat_32)
            feat_64_residual_frequency = self.get_feat_64_residual_frequency(feat_32_residual_frequency)
            feat_64_residual_frequency_alignment = self.feat_64_residual_fequency_output(feat_64_residual_frequency)
            complete_feat_64 = feat_64_alignment + feat_64_residual_frequency_alignment

            feat_128_alignment = self.upsample(complete_feat_64)
            feat_128_residual_frequency = self.get_feat_128_residual_frequency(feat_64_residual_frequency)
            feat_128_residual_frequency_alignment = self.feat_128_residual_fequency_output(feat_128_residual_frequency)
            complete_feat_128 = feat_128_alignment + feat_128_residual_frequency_alignment

            feat_256_alignment = self.upsample(complete_feat_128)
            feat_256_residual_frequency = self.get_feat_256_residual_frequency(feat_128_residual_frequency)
            feat_256_residual_frequency_alignment = self.feat_256_residual_fequency_output(feat_256_residual_frequency)
            complete_feat_256 = feat_256_alignment + feat_256_residual_frequency_alignment

            if self.im_size == 256:
                return complete_feat_256

            feat_512_alignment = self.upsample(complete_feat_256)
            feat_512_residual_frequency = self.get_feat_512_residual_frequency(feat_256_residual_frequency)
            feat_512_residual_frequency_alignment = self.feat_512_residual_fequency_output(feat_512_residual_frequency)
            complete_feat_512 = feat_512_alignment + feat_512_residual_frequency_alignment

            if self.im_size == 512:
                return complete_feat_512

            feat_1024_alignment = self.upsample(complete_feat_512)
            feat_1024_residual_frequency = self.get_feat_1024_residual_frequency(feat_512_residual_frequency)
            feat_1024_residual_frequency_alignment = self.feat_1024_residual_fequency_output(
                feat_1024_residual_frequency)

            complete_feat_1024 = feat_1024_alignment + feat_1024_residual_frequency_alignment

            im_1024 = torch.tanh(complete_feat_1024)

            return im_1024
        else:
            feat_4 = self.init(input) # input: b*256, feat_4: b*1024*4*4
            feat_8 = self.feat_8(feat_4) # feat_4: b*1024*4*4, feat_8: b*512*8*8
            feat_16 = self.feat_16(feat_8)

            feat_16_residual_frequency = self.get_feat_16_residual_frequency(feat_8)
            feat_16_alignment = self.to_16(feat_16)

            feat_32_alignment = self.upsample(feat_16_alignment)
            feat_32_residual_frequency = self.get_feat_32_residual_frequency(feat_16_residual_frequency)
            feat_32_residual_frequency_alignment = self.feat_32_residual_fequency_output(feat_32_residual_frequency)
            complete_feat_32 = feat_32_alignment + feat_32_residual_frequency_alignment

            feat_64_alignment = self.upsample(complete_feat_32)
            feat_64_residual_frequency = self.get_feat_64_residual_frequency(feat_32_residual_frequency)
            feat_64_residual_frequency_alignment = self.feat_64_residual_fequency_output(feat_64_residual_frequency)
            complete_feat_64 = feat_64_alignment + feat_64_residual_frequency_alignment

            feat_128_alignment = self.upsample(complete_feat_64)
            feat_128_residual_frequency = self.get_feat_128_residual_frequency(feat_64_residual_frequency)
            feat_128_residual_frequency_alignment = self.feat_128_residual_fequency_output(feat_128_residual_frequency)
            complete_feat_128 = feat_128_alignment + feat_128_residual_frequency_alignment

            feat_256_alignment = self.upsample(complete_feat_128)
            feat_256_residual_frequency = self.get_feat_256_residual_frequency(feat_128_residual_frequency)
            feat_256_residual_frequency_alignment = self.feat_256_residual_fequency_output(feat_256_residual_frequency)
            complete_feat_256 = feat_256_alignment + feat_256_residual_frequency_alignment

            if self.im_size == 256:
                return [complete_feat_256, complete_feat_128, complete_feat_64, complete_feat_32, feat_16_alignment], [feat_256_residual_frequency_alignment, feat_128_residual_frequency_alignment, feat_64_residual_frequency_alignment, feat_32_residual_frequency_alignment, feat_16_alignment]

            feat_512_alignment = self.upsample(complete_feat_256)
            feat_512_residual_frequency = self.get_feat_512_residual_frequency(feat_256_residual_frequency)
            feat_512_residual_frequency_alignment = self.feat_512_residual_fequency_output(feat_512_residual_frequency)
            complete_feat_512 = feat_512_alignment + feat_512_residual_frequency_alignment

            if self.im_size == 512:
                return [complete_feat_512, complete_feat_256, complete_feat_128, complete_feat_64, complete_feat_32, feat_16_alignment], [feat_512_residual_frequency_alignment, feat_256_residual_frequency_alignment, feat_128_residual_frequency_alignment, feat_64_residual_frequency_alignment, feat_32_residual_frequency_alignment, feat_16_alignment]

            feat_1024_alignment = self.upsample(complete_feat_512)
            feat_1024_residual_frequency = self.get_feat_1024_residual_frequency(feat_512_residual_frequency)
            feat_1024_residual_frequency_alignment = self.feat_1024_residual_fequency_output(feat_1024_residual_frequency)

            complete_feat_1024 = feat_1024_alignment + feat_1024_residual_frequency_alignment

            im_1024 = torch.tanh(complete_feat_1024)

            return [im_1024, complete_feat_512, complete_feat_256, complete_feat_128, complete_feat_64, complete_feat_32, feat_16_alignment], [feat_1024_residual_frequency_alignment, feat_512_residual_frequency_alignment, feat_256_residual_frequency_alignment, feat_128_residual_frequency_alignment, feat_64_residual_frequency_alignment, feat_32_residual_frequency_alignment, feat_16_alignment]


# num_high=5
class Generator5(nn.Module):
    def __init__(self, ngf=64, nz=100, nc=3, im_size=1024, kernel_size=5, num_high=7):
        super(Generator5, self).__init__()

        nfc_multi = {4: 16, 8: 8, 16: 4, 32: 2, 64: 2, 128: 1, 256: 0.5, 512: 0.25, 1024: 0.125}
        nfc = {}
        for k, v in nfc_multi.items():
            nfc[k] = int(v * ngf)

        self.kernel = self.gauss_kernel(kernel_size, nc)
        self.num_high = num_high

        self.im_size = im_size

        self.init = InitLayer(nz, channel=nfc[4])

        self.feat_8 = UpBlockComp(nfc[4], nfc[8])
        self.feat_16 = UpBlock(nfc[8], nfc[16])
        self.to_8 = conv2d(nfc[8], nc, 1, 1, 0, bias=False)

        self.feat_16_residual_fequency_output = conv2d(nfc[16], nc, 1, 1, 0, bias=False)
        self.feat_32_residual_fequency_output = conv2d(nfc[32], nc, 1, 1, 0, bias=False)
        self.feat_64_residual_fequency_output = conv2d(nfc[64], nc, 1, 1, 0, bias=False)
        self.feat_128_residual_fequency_output = conv2d(nfc[128], nc, 1, 1, 0, bias=False)
        self.feat_256_residual_fequency_output = conv2d(nfc[256], nc, 3, 1, 1, bias=False)

        # residual_frequency_obtain
        self.get_feat_8_residual_frequency = UpBlockComp(nfc[4], nfc[8])
        self.get_feat_16_residual_frequency = UpBlock(nfc[8], nfc[16])
        self.get_feat_32_residual_frequency = UpBlockComp(nfc[16], nfc[32])
        self.get_feat_64_residual_frequency = UpBlock(nfc[32], nfc[64])
        self.get_feat_128_residual_frequency = UpBlockComp(nfc[64], nfc[128])
        self.get_feat_256_residual_frequency = UpBlock(nfc[128], nfc[256])

        if im_size > 256:
            self.get_feat_512_residual_frequency = UpBlockComp(nfc[256], nfc[512])
            self.feat_512_residual_fequency_output = conv2d(nfc[512], nc, 3, 1, 1, bias=False)
            self.feat_512_output = conv2d(nfc[512], nc, 3, 1, 1, bias=False)
        if im_size > 512:
            self.get_feat_1024_residual_frequency = UpBlock(nfc[512], nfc[1024])
            self.feat_1024_residual_fequency_output = conv2d(nfc[1024], nc, 3, 1, 1, bias=False)
            self.feat_1024_output = conv2d(nfc[1024], nc, 3, 1, 1, bias=False)

    def downsample(self, x):
        return x[:, :, ::2, ::2]

    def pyramid_down(self, x):
        return self.downsample(self.conv_gauss(x, self.kernel))

    def gauss_kernel(self, kernel_size, channels):
        kernel = cv2.getGaussianKernel(kernel_size, 0).dot(
            cv2.getGaussianKernel(kernel_size, 0).T)
        kernel = torch.FloatTensor(kernel).unsqueeze(0).repeat(
            channels, 1, 1, 1)
        kernel = torch.nn.Parameter(data=kernel, requires_grad=False)
        return kernel

    def conv_gauss(self, x, kernel):
        n_channels, _, kw, kh = kernel.shape
        kernel_cuda = kernel.to(x.device)

        x = torch.nn.functional.pad(x, (kw // 2, kh // 2, kw // 2, kh // 2),
                                    mode='reflect')  # replicate    # reflect
        x = torch.nn.functional.conv2d(x, kernel_cuda, groups=n_channels)

        return x

    def upsample(self, x):
        up = torch.zeros((x.size(0), x.size(1), x.size(2) * 2, x.size(3) * 2),
                         device=x.device)
        up[:, :, ::2, ::2] = x * 4

        return self.conv_gauss(up, self.kernel)

    def pyramid_decom(self, img):
        # self.kernel = self.kernel.to(img.device)
        current = img
        pyr = []
        subtrahend = []
        minuends = []
        minuends.append(current)
        for _ in range(self.num_high):
            down = self.pyramid_down(current)
            minuends.append(down)
            up = self.upsample(down)
            diff = current - up
            pyr.append(diff)
            subtrahend.append(up)
            current = down
        pyr.append(current)
        return pyr, subtrahend, minuends

    def pyramid_recons(self, pyr):
        image = pyr[0]
        for level in pyr[1:]:
            up = self.upsample(image)
            image = up + level
        return image

    def forward(self, input, skips=False):
        if skips:
            feat_4 = self.init(input)  # input: b*256, feat_4: b*1024*4*4
            feat_8 = self.feat_8(feat_4)  # feat_4: b*1024*4*4, feat_8: b*512*8*8
            feat_8_residual_frequency = self.get_feat_8_residual_frequency(feat_4)
            feat_8_alignment = self.to_8(feat_8)

            feat_16_alignment = self.upsample(feat_8_alignment)
            feat_16_residual_frequency = self.get_feat_16_residual_frequency(feat_8_residual_frequency)
            feat_16_residual_frequency_alignment = self.feat_16_residual_fequency_output(feat_16_residual_frequency)
            complete_feat_16 = feat_16_alignment + feat_16_residual_frequency_alignment

            feat_32_alignment = self.upsample(complete_feat_16)
            feat_32_residual_frequency = self.get_feat_32_residual_frequency(feat_16_residual_frequency)
            feat_32_residual_frequency_alignment = self.feat_32_residual_fequency_output(feat_32_residual_frequency)
            complete_feat_32 = feat_32_alignment + feat_32_residual_frequency_alignment

            feat_64_alignment = self.upsample(complete_feat_32)
            feat_64_residual_frequency = self.get_feat_64_residual_frequency(feat_32_residual_frequency)
            feat_64_residual_frequency_alignment = self.feat_64_residual_fequency_output(feat_64_residual_frequency)
            complete_feat_64 = feat_64_alignment + feat_64_residual_frequency_alignment

            feat_128_alignment = self.upsample(complete_feat_64)
            feat_128_residual_frequency = self.get_feat_128_residual_frequency(feat_64_residual_frequency)
            feat_128_residual_frequency_alignment = self.feat_128_residual_fequency_output(feat_128_residual_frequency)
            complete_feat_128 = feat_128_alignment + feat_128_residual_frequency_alignment

            feat_256_alignment = self.upsample(complete_feat_128)
            feat_256_residual_frequency = self.get_feat_256_residual_frequency(feat_128_residual_frequency)
            feat_256_residual_frequency_alignment = self.feat_256_residual_fequency_output(feat_256_residual_frequency)
            complete_feat_256 = feat_256_alignment + feat_256_residual_frequency_alignment

            if self.im_size == 256:
                return complete_feat_256

            feat_512_alignment = self.upsample(complete_feat_256)
            feat_512_residual_frequency = self.get_feat_512_residual_frequency(feat_256_residual_frequency)
            feat_512_residual_frequency_alignment = self.feat_512_residual_fequency_output(feat_512_residual_frequency)
            complete_feat_512 = feat_512_alignment + feat_512_residual_frequency_alignment

            if self.im_size == 512:
                return complete_feat_512

            feat_1024_alignment = self.upsample(complete_feat_512)
            feat_1024_residual_frequency = self.get_feat_1024_residual_frequency(feat_512_residual_frequency)
            feat_1024_residual_frequency_alignment = self.feat_1024_residual_fequency_output(
                feat_1024_residual_frequency)

            complete_feat_1024 = feat_1024_alignment + feat_1024_residual_frequency_alignment

            im_1024 = torch.tanh(complete_feat_1024)

            return im_1024
        else:
            feat_4 = self.init(input) # input: b*256, feat_4: b*1024*4*4
            feat_8 = self.feat_8(feat_4) # feat_4: b*1024*4*4, feat_8: b*512*8*8
            feat_8_residual_frequency = self.get_feat_8_residual_frequency(feat_4)
            feat_8_alignment = self.to_8(feat_8)

            feat_16_alignment = self.upsample(feat_8_alignment)
            feat_16_residual_frequency = self.get_feat_16_residual_frequency(feat_8_residual_frequency)
            feat_16_residual_frequency_alignment = self.feat_16_residual_fequency_output(feat_16_residual_frequency)
            complete_feat_16 = feat_16_alignment + feat_16_residual_frequency_alignment


            feat_32_alignment = self.upsample(complete_feat_16)
            feat_32_residual_frequency = self.get_feat_32_residual_frequency(feat_16_residual_frequency)
            feat_32_residual_frequency_alignment = self.feat_32_residual_fequency_output(feat_32_residual_frequency)
            complete_feat_32 = feat_32_alignment + feat_32_residual_frequency_alignment

            feat_64_alignment = self.upsample(complete_feat_32)
            feat_64_residual_frequency = self.get_feat_64_residual_frequency(feat_32_residual_frequency)
            feat_64_residual_frequency_alignment = self.feat_64_residual_fequency_output(feat_64_residual_frequency)
            complete_feat_64 = feat_64_alignment + feat_64_residual_frequency_alignment

            feat_128_alignment = self.upsample(complete_feat_64)
            feat_128_residual_frequency = self.get_feat_128_residual_frequency(feat_64_residual_frequency)
            feat_128_residual_frequency_alignment = self.feat_128_residual_fequency_output(feat_128_residual_frequency)
            complete_feat_128 = feat_128_alignment + feat_128_residual_frequency_alignment

            feat_256_alignment = self.upsample(complete_feat_128)
            feat_256_residual_frequency = self.get_feat_256_residual_frequency(feat_128_residual_frequency)
            feat_256_residual_frequency_alignment = self.feat_256_residual_fequency_output(feat_256_residual_frequency)
            complete_feat_256 = feat_256_alignment + feat_256_residual_frequency_alignment

            if self.im_size == 256:
                return [complete_feat_256, complete_feat_128, complete_feat_64, complete_feat_32, complete_feat_16, feat_8_alignment], [feat_256_residual_frequency_alignment, feat_128_residual_frequency_alignment, feat_64_residual_frequency_alignment, feat_32_residual_frequency_alignment, feat_16_residual_frequency_alignment, feat_8_alignment]

            feat_512_alignment = self.upsample(complete_feat_256)
            feat_512_residual_frequency = self.get_feat_512_residual_frequency(feat_256_residual_frequency)
            feat_512_residual_frequency_alignment = self.feat_512_residual_fequency_output(feat_512_residual_frequency)
            complete_feat_512 = feat_512_alignment + feat_512_residual_frequency_alignment

            if self.im_size == 512:
                return [complete_feat_512, complete_feat_256, complete_feat_128, complete_feat_64, complete_feat_32, complete_feat_16, feat_8_alignment], [feat_512_residual_frequency_alignment, feat_256_residual_frequency_alignment, feat_128_residual_frequency_alignment, feat_64_residual_frequency_alignment, feat_32_residual_frequency_alignment, feat_16_residual_frequency_alignment, feat_8_alignment]

            feat_1024_alignment = self.upsample(complete_feat_512)
            feat_1024_residual_frequency = self.get_feat_1024_residual_frequency(feat_512_residual_frequency)
            feat_1024_residual_frequency_alignment = self.feat_1024_residual_fequency_output(feat_1024_residual_frequency)

            complete_feat_1024 = feat_1024_alignment + feat_1024_residual_frequency_alignment

            im_1024 = torch.tanh(complete_feat_1024)

            return [im_1024, complete_feat_512, complete_feat_256, complete_feat_128, complete_feat_64, complete_feat_32, complete_feat_16, feat_8_alignment], [feat_1024_residual_frequency_alignment, feat_512_residual_frequency_alignment, feat_256_residual_frequency_alignment, feat_128_residual_frequency_alignment, feat_64_residual_frequency_alignment, feat_32_residual_frequency_alignment, feat_16_residual_frequency_alignment, feat_8_alignment]


# num_high=6
class Generator6(nn.Module):
    def __init__(self, ngf=64, nz=100, nc=3, im_size=1024, kernel_size=5, num_high=7):
        super(Generator6, self).__init__()

        nfc_multi = {4: 16, 8: 8, 16: 4, 32: 2, 64: 2, 128: 1, 256: 0.5, 512: 0.25, 1024: 0.125}
        nfc = {}
        for k, v in nfc_multi.items():
            nfc[k] = int(v * ngf)

        self.kernel = self.gauss_kernel(kernel_size, nc)
        self.num_high = num_high

        self.im_size = im_size

        self.init = InitLayer(nz, channel=nfc[4])

        self.to_4 = conv2d(nfc[4], nc, 1, 1, 0, bias=False)

        self.feat_8_residual_fequency_output = conv2d(nfc[8], nc, 1, 1, 0, bias=False)
        self.feat_16_residual_fequency_output = conv2d(nfc[16], nc, 1, 1, 0, bias=False)
        self.feat_32_residual_fequency_output = conv2d(nfc[32], nc, 1, 1, 0, bias=False)
        self.feat_64_residual_fequency_output = conv2d(nfc[64], nc, 1, 1, 0, bias=False)
        self.feat_128_residual_fequency_output = conv2d(nfc[128], nc, 1, 1, 0, bias=False)
        self.feat_256_residual_fequency_output = conv2d(nfc[256], nc, 3, 1, 1, bias=False)

        # residual_frequency_obtain
        self.get_feat_4_residual_frequency = InitLayer(nz, channel=nfc[4])
        self.get_feat_8_residual_frequency = UpBlockComp(nfc[4], nfc[8])
        self.get_feat_16_residual_frequency = UpBlock(nfc[8], nfc[16])
        self.get_feat_32_residual_frequency = UpBlockComp(nfc[16], nfc[32])
        self.get_feat_64_residual_frequency = UpBlock(nfc[32], nfc[64])
        self.get_feat_128_residual_frequency = UpBlockComp(nfc[64], nfc[128])
        self.get_feat_256_residual_frequency = UpBlock(nfc[128], nfc[256])

        if im_size > 256:
            self.get_feat_512_residual_frequency = UpBlockComp(nfc[256], nfc[512])
            self.feat_512_residual_fequency_output = conv2d(nfc[512], nc, 3, 1, 1, bias=False)
            self.feat_512_output = conv2d(nfc[512], nc, 3, 1, 1, bias=False)
        if im_size > 512:
            self.get_feat_1024_residual_frequency = UpBlock(nfc[512], nfc[1024])
            self.feat_1024_residual_fequency_output = conv2d(nfc[1024], nc, 3, 1, 1, bias=False)
            self.feat_1024_output = conv2d(nfc[1024], nc, 3, 1, 1, bias=False)

    def downsample(self, x):
        return x[:, :, ::2, ::2]

    def pyramid_down(self, x):
        return self.downsample(self.conv_gauss(x, self.kernel))

    def gauss_kernel(self, kernel_size, channels):
        kernel = cv2.getGaussianKernel(kernel_size, 0).dot(
            cv2.getGaussianKernel(kernel_size, 0).T)
        kernel = torch.FloatTensor(kernel).unsqueeze(0).repeat(
            channels, 1, 1, 1)
        kernel = torch.nn.Parameter(data=kernel, requires_grad=False)
        return kernel

    def conv_gauss(self, x, kernel):
        n_channels, _, kw, kh = kernel.shape
        kernel_cuda = kernel.to(x.device)

        x = torch.nn.functional.pad(x, (kw // 2, kh // 2, kw // 2, kh // 2),
                                    mode='reflect')  # replicate    # reflect
        x = torch.nn.functional.conv2d(x, kernel_cuda, groups=n_channels)

        return x

    def upsample(self, x):
        up = torch.zeros((x.size(0), x.size(1), x.size(2) * 2, x.size(3) * 2),
                         device=x.device)
        up[:, :, ::2, ::2] = x * 4

        return self.conv_gauss(up, self.kernel)

    def pyramid_decom(self, img):
        # self.kernel = self.kernel.to(img.device)
        current = img
        pyr = []
        subtrahend = []
        minuends = []
        minuends.append(current)
        for _ in range(self.num_high):
            down = self.pyramid_down(current)
            minuends.append(down)
            up = self.upsample(down)
            diff = current - up
            pyr.append(diff)
            subtrahend.append(up)
            current = down
        pyr.append(current)
        return pyr, subtrahend, minuends

    def pyramid_recons(self, pyr):
        image = pyr[0]
        for level in pyr[1:]:
            up = self.upsample(image)
            image = up + level
        return image

    def forward(self, input, skips=False):
        if skips:
            feat_4 = self.init(input)  # input: b*256, feat_4: b*1024*4*4
            feat_4_alignment = self.to_4(feat_4)
            feat_4_residual_frequency = self.get_feat_4_residual_frequency(input)

            feat_8_alignment = self.upsample(feat_4_alignment)
            feat_8_residual_frequency = self.get_feat_8_residual_frequency(feat_4_residual_frequency)
            feat_8_residual_frequency_alignment = self.feat_8_residual_fequency_output(feat_8_residual_frequency)
            complete_feat_8 = feat_8_alignment + feat_8_residual_frequency_alignment

            feat_16_alignment = self.upsample(complete_feat_8)
            feat_16_residual_frequency = self.get_feat_16_residual_frequency(feat_8_residual_frequency)
            feat_16_residual_frequency_alignment = self.feat_16_residual_fequency_output(feat_16_residual_frequency)
            complete_feat_16 = feat_16_alignment + feat_16_residual_frequency_alignment

            feat_32_alignment = self.upsample(complete_feat_16)
            feat_32_residual_frequency = self.get_feat_32_residual_frequency(feat_16_residual_frequency)
            feat_32_residual_frequency_alignment = self.feat_32_residual_fequency_output(feat_32_residual_frequency)
            complete_feat_32 = feat_32_alignment + feat_32_residual_frequency_alignment

            feat_64_alignment = self.upsample(complete_feat_32)
            feat_64_residual_frequency = self.get_feat_64_residual_frequency(feat_32_residual_frequency)
            feat_64_residual_frequency_alignment = self.feat_64_residual_fequency_output(feat_64_residual_frequency)
            complete_feat_64 = feat_64_alignment + feat_64_residual_frequency_alignment

            feat_128_alignment = self.upsample(complete_feat_64)
            feat_128_residual_frequency = self.get_feat_128_residual_frequency(feat_64_residual_frequency)
            feat_128_residual_frequency_alignment = self.feat_128_residual_fequency_output(feat_128_residual_frequency)
            complete_feat_128 = feat_128_alignment + feat_128_residual_frequency_alignment

            feat_256_alignment = self.upsample(complete_feat_128)
            feat_256_residual_frequency = self.get_feat_256_residual_frequency(feat_128_residual_frequency)
            feat_256_residual_frequency_alignment = self.feat_256_residual_fequency_output(feat_256_residual_frequency)
            complete_feat_256 = feat_256_alignment + feat_256_residual_frequency_alignment

            if self.im_size == 256:
                return complete_feat_256

            feat_512_alignment = self.upsample(complete_feat_256)
            feat_512_residual_frequency = self.get_feat_512_residual_frequency(feat_256_residual_frequency)
            feat_512_residual_frequency_alignment = self.feat_512_residual_fequency_output(feat_512_residual_frequency)
            complete_feat_512 = feat_512_alignment + feat_512_residual_frequency_alignment

            if self.im_size == 512:
                return complete_feat_512

            feat_1024_alignment = self.upsample(complete_feat_512)
            feat_1024_residual_frequency = self.get_feat_1024_residual_frequency(feat_512_residual_frequency)
            feat_1024_residual_frequency_alignment = self.feat_1024_residual_fequency_output(
                feat_1024_residual_frequency)

            complete_feat_1024 = feat_1024_alignment + feat_1024_residual_frequency_alignment

            im_1024 = torch.tanh(complete_feat_1024)

            return im_1024
        else:
            feat_4 = self.init(input) # input: b*256, feat_4: b*1024*4*4
            feat_4_alignment = self.to_4(feat_4)
            feat_4_residual_frequency = self.get_feat_4_residual_frequency(input)

            feat_8_alignment = self.upsample(feat_4_alignment)
            feat_8_residual_frequency = self.get_feat_8_residual_frequency(feat_4_residual_frequency)
            feat_8_residual_frequency_alignment = self.feat_8_residual_fequency_output(feat_8_residual_frequency)
            complete_feat_8 = feat_8_alignment + feat_8_residual_frequency_alignment

            feat_16_alignment = self.upsample(complete_feat_8)
            feat_16_residual_frequency = self.get_feat_16_residual_frequency(feat_8_residual_frequency)
            feat_16_residual_frequency_alignment = self.feat_16_residual_fequency_output(feat_16_residual_frequency)
            complete_feat_16 = feat_16_alignment + feat_16_residual_frequency_alignment

            feat_32_alignment = self.upsample(complete_feat_16)
            feat_32_residual_frequency = self.get_feat_32_residual_frequency(feat_16_residual_frequency)
            feat_32_residual_frequency_alignment = self.feat_32_residual_fequency_output(feat_32_residual_frequency)
            complete_feat_32 = feat_32_alignment + feat_32_residual_frequency_alignment

            feat_64_alignment = self.upsample(complete_feat_32)
            feat_64_residual_frequency = self.get_feat_64_residual_frequency(feat_32_residual_frequency)
            feat_64_residual_frequency_alignment = self.feat_64_residual_fequency_output(feat_64_residual_frequency)
            complete_feat_64 = feat_64_alignment + feat_64_residual_frequency_alignment

            feat_128_alignment = self.upsample(complete_feat_64)
            feat_128_residual_frequency = self.get_feat_128_residual_frequency(feat_64_residual_frequency)
            feat_128_residual_frequency_alignment = self.feat_128_residual_fequency_output(feat_128_residual_frequency)
            complete_feat_128 = feat_128_alignment + feat_128_residual_frequency_alignment

            feat_256_alignment = self.upsample(complete_feat_128)
            feat_256_residual_frequency = self.get_feat_256_residual_frequency(feat_128_residual_frequency)
            feat_256_residual_frequency_alignment = self.feat_256_residual_fequency_output(feat_256_residual_frequency)
            complete_feat_256 = feat_256_alignment + feat_256_residual_frequency_alignment

            if self.im_size == 256:
                return [complete_feat_256, complete_feat_128, complete_feat_64, complete_feat_32, complete_feat_16, complete_feat_8, feat_4_alignment], [feat_256_residual_frequency_alignment, feat_128_residual_frequency_alignment, feat_64_residual_frequency_alignment, feat_32_residual_frequency_alignment, feat_16_residual_frequency_alignment, feat_8_residual_frequency_alignment, feat_4_alignment]

            feat_512_alignment = self.upsample(complete_feat_256)
            feat_512_residual_frequency = self.get_feat_512_residual_frequency(feat_256_residual_frequency)
            feat_512_residual_frequency_alignment = self.feat_512_residual_fequency_output(feat_512_residual_frequency)
            complete_feat_512 = feat_512_alignment + feat_512_residual_frequency_alignment

            if self.im_size == 512:
                return [complete_feat_512, complete_feat_256, complete_feat_128, complete_feat_64, complete_feat_32, complete_feat_16, complete_feat_8, feat_4_alignment], [feat_512_residual_frequency_alignment, feat_256_residual_frequency_alignment, feat_128_residual_frequency_alignment, feat_64_residual_frequency_alignment, feat_32_residual_frequency_alignment, feat_16_residual_frequency_alignment, feat_8_residual_frequency_alignment, feat_4_alignment]

            feat_1024_alignment = self.upsample(complete_feat_512)
            feat_1024_residual_frequency = self.get_feat_1024_residual_frequency(feat_512_residual_frequency)
            feat_1024_residual_frequency_alignment = self.feat_1024_residual_fequency_output(feat_1024_residual_frequency)

            complete_feat_1024 = feat_1024_alignment + feat_1024_residual_frequency_alignment

            im_1024 = torch.tanh(complete_feat_1024)

            return [im_1024, complete_feat_512, complete_feat_256, complete_feat_128, complete_feat_64, complete_feat_32, complete_feat_16, complete_feat_8, feat_4_alignment], [feat_1024_residual_frequency_alignment, feat_512_residual_frequency_alignment, feat_256_residual_frequency_alignment, feat_128_residual_frequency_alignment, feat_64_residual_frequency_alignment, feat_32_residual_frequency_alignment, feat_16_residual_frequency_alignment, feat_8_residual_frequency_alignment, feat_4_alignment]


if __name__ == "__main__":
    device = 'cuda:0'
    nz = 256
    generator = Generator3(ngf=64, nz=256, im_size=1024).to(device)
    fixed_noise = torch.FloatTensor(1, nz).normal_(0, 1).to(device)
    output = generator(fixed_noise, skips=None)


