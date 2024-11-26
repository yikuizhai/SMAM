import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm
import cv2
import torch.nn.functional as F
import numpy as np
import random


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


class Swish(nn.Module):
    def forward(self, feat):
        return feat * torch.sigmoid(feat)


class GLU(nn.Module):
    def forward(self, x):
        nc = x.size(1)
        assert nc % 2 == 0, 'channels dont divide 2!'
        nc = int(nc/2)
        return x[:, :nc] * torch.sigmoid(x[:, nc:])


class SEBlock(nn.Module):
    def __init__(self, ch_in, ch_out):
        super().__init__()

        self.main = nn.Sequential(  nn.AdaptiveAvgPool2d(4),
                                    conv2d(ch_in, ch_out, 4, 1, 0, bias=False), Swish(),
                                    conv2d(ch_out, ch_out, 1, 1, 0, bias=False), nn.Sigmoid() )

    def forward(self, feat_small, feat_big):
        return feat_big * self.main(feat_small)


class DownBlock(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(DownBlock, self).__init__()

        self.main = nn.Sequential(
            conv2d(in_planes, out_planes, 4, 2, 1, bias=False),
            batchNorm2d(out_planes), nn.LeakyReLU(0.2, inplace=True),
            )

    def forward(self, feat):
        return self.main(feat)


class DownBlockComp(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(DownBlockComp, self).__init__()

        self.main = nn.Sequential(
            conv2d(in_planes, out_planes, 4, 2, 1, bias=False),
            batchNorm2d(out_planes), nn.LeakyReLU(0.2, inplace=True),
            conv2d(out_planes, out_planes, 3, 1, 1, bias=False),
            batchNorm2d(out_planes), nn.LeakyReLU(0.2)
            )

        self.direct = nn.Sequential(
            nn.AvgPool2d(2, 2),
            conv2d(in_planes, out_planes, 1, 1, 0, bias=False),
            batchNorm2d(out_planes), nn.LeakyReLU(0.2))

    def forward(self, feat):
        return (self.main(feat) + self.direct(feat)) / 2


class Discriminator(nn.Module):
    def __init__(self, ndf=64, nc=3, im_size=512, kernel_size=5, num_high=4):
        super(Discriminator, self).__init__()
        self.ndf = ndf
        self.im_size = im_size
        self.resolution_change_list = []
        if im_size == 256:
            assert num_high > 2
        elif im_size == 512:
            assert num_high > 3
        elif im_size == 1024:
            assert num_high > 4
        for resolution_size in range(num_high + 1):
            self.resolution_change_list.append(int(im_size / (2**resolution_size)))
        self.flag_add = {"256": 0, "512": 1, "1024": 2}

        self.cut_off = {"6": 4, "5": 3, "4": 2, "3": 1}
        self.kernel = self.gauss_kernel(kernel_size, nc)
        self.num_high = num_high

        # 用于正则化的权重
        self.l2_lambda = 1e-4

        nfc_multi = {4 :16, 8 :16, 16 :8, 32 :4, 64 :2, 128 :1, 256 :0.5, 512 :0.25, 1024 :0.125}
        nfc = {}
        for k, v in nfc_multi.items():
            nfc[k] = int(v*ndf)

        if im_size == 1024:
            self.down_from_big = nn.Sequential(
                conv2d(nc, nfc[1024], 4, 2, 1, bias=False),
                nn.LeakyReLU(0.2, inplace=True),
                conv2d(nfc[1024], nfc[512], 4, 2, 1, bias=False),
                batchNorm2d(nfc[512]),
                nn.LeakyReLU(0.2, inplace=True))
        elif im_size == 512:
            self.down_from_big = nn.Sequential(
                conv2d(nc, nfc[512], 4, 2, 1, bias=False),
                nn.LeakyReLU(0.2, inplace=True) )
        elif im_size == 256:
            self.down_from_big = nn.Sequential(
                conv2d(nc, nfc[512], 3, 1, 1, bias=False),
                nn.LeakyReLU(0.2, inplace=True) )

        self.down_4 = DownBlockComp(nfc[512], nfc[256])
        self.down_8 = DownBlockComp(nfc[256], nfc[128])
        self.down_16 = DownBlockComp(nfc[128], nfc[64])
        self.down_32 = DownBlockComp(nfc[64],  nfc[32])
        self.down_64 = DownBlockComp(nfc[32],  nfc[16])

        self.rf_big = nn.Sequential(
            conv2d(nfc[16] , nfc[8], 1, 1, 0, bias=False),
            batchNorm2d(nfc[8]), nn.LeakyReLU(0.2, inplace=True),
            conv2d(nfc[8], 1, 4, 1, 0, bias=False))

        self.se_2_16 = SEBlock(nfc[512], nfc[64])
        self.se_4_32 = SEBlock(nfc[256], nfc[32])
        self.se_8_64 = SEBlock(nfc[128], nfc[16])

        self.down_from_small = nn.Sequential(
            conv2d(nc, nfc[256], 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            DownBlock(nfc[256],  nfc[128]),
            DownBlock(nfc[128],  nfc[64]),
            DownBlock(nfc[64],  nfc[32]))

        self.rf_small = conv2d(nfc[32], 1, 4, 1, 0, bias=False)

        self.decoder_big = SimpleDecoder(nfc[16], nc)
        self.decoder_part = SimpleDecoder(nfc[32], nc)
        self.decoder_small = SimpleDecoder(nfc[32], nc)

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

    def regularization(self):
    # 正则化项是所有权重参数的平方的和
        l2_reg = sum(param.pow(2).sum() for param in self.parameters() if param.requires_grad)
        return self.l2_lambda * l2_reg

    def forward(self, imgs, label, part=None):
        # first decompose the real images or generated images
        pyrs = None
        if label=='real':
            pyrs, subtrahends, minuends = self.pyramid_decom(imgs)
        else:
            minuends = imgs
        rf_list = []
        reconstruct_big_img_list = []
        reconstruct_feat_32_img_list = []
        for every_batch_big_imgs in minuends[:-(self.num_high - self.flag_add[str(self.im_size)])]:
            feat_2 = self.down_from_big(every_batch_big_imgs)
            feat_4 = self.down_4(feat_2)
            feat_8 = self.down_8(feat_4)

            feat_16 = self.down_16(feat_8)
            feat_16 = self.se_2_16(feat_2, feat_16)

            feat_32 = self.down_32(feat_16)
            feat_32 = self.se_4_32(feat_4, feat_32)

            reconstruct_feat_32_img_list.append(feat_32)

            feat_last = self.down_64(feat_32)
            feat_last = self.se_8_64(feat_8, feat_last)

            reconstruct_big_img_list.append(feat_last)

            rf_0 = self.rf_big(feat_last).view(-1)
            rf_list.append(rf_0)

        reconstruct_small_img_list = []
        for every_batch_small_imgs in minuends[-(self.num_high - self.flag_add[str(self.im_size)]):-(self.cut_off[str(self.num_high)] - self.flag_add[str(self.im_size)])]:
            feat_small = self.down_from_small(every_batch_small_imgs)
            reconstruct_small_img_list.append(feat_small)
            rf_1 = self.rf_small(feat_small).view(-1)
            rf_list.append(rf_1)

        if label=='real':
            rec_img_big = self.decoder_big(reconstruct_big_img_list[0])
            rec_img_small = self.decoder_small(reconstruct_small_img_list[0])

            assert part is not None
            rec_img_part = None
            if part==0:
                rec_img_part = self.decoder_part(reconstruct_feat_32_img_list[0][: ,: ,:8 ,:8])
            if part==1:
                rec_img_part = self.decoder_part(reconstruct_feat_32_img_list[0][: ,: ,:8 ,8:])
            if part==2:
                rec_img_part = self.decoder_part(reconstruct_feat_32_img_list[0][: ,: ,8: ,:8])
            if part==3:
                rec_img_part = self.decoder_part(reconstruct_feat_32_img_list[0][: ,: ,8: ,8:])

            return torch.cat(rf_list), [rec_img_big, rec_img_small, rec_img_part], pyrs

        return torch.cat(rf_list)


class Discriminator4(nn.Module):
    def __init__(self, ndf=64, nc=3, im_size=512, kernel_size=5, num_high=4):
        super(Discriminator4, self).__init__()
        self.ndf = ndf
        self.im_size = im_size

        self.kernel = self.gauss_kernel(kernel_size, nc)
        self.num_high = num_high

        # 用于正则化的权重
        self.l2_lambda = 1e-4

        nfc_multi = {4 :16, 8 :16, 16 :8, 32 :4, 64 :2, 128 :1, 256 :0.5, 512 :0.25, 1024 :0.125}
        nfc = {}
        for k, v in nfc_multi.items():
            nfc[k] = int(v*ndf)

        if im_size == 1024:
            self.down_from_big = nn.Sequential(
                conv2d(nc, nfc[1024], 4, 2, 1, bias=False),
                nn.LeakyReLU(0.2, inplace=True),
                conv2d(nfc[1024], nfc[512], 4, 2, 1, bias=False),
                batchNorm2d(nfc[512]),
                nn.LeakyReLU(0.2, inplace=True))
        elif im_size == 512:
            self.down_from_big = nn.Sequential(
                conv2d(nc, nfc[512], 4, 2, 1, bias=False),
                nn.LeakyReLU(0.2, inplace=True) )
        elif im_size == 256:
            self.down_from_big = nn.Sequential(
                conv2d(nc, nfc[512], 3, 1, 1, bias=False),
                nn.LeakyReLU(0.2, inplace=True) )

        self.down_4 = DownBlockComp(nfc[512], nfc[256])
        self.down_8 = DownBlockComp(nfc[256], nfc[128])
        self.down_16 = DownBlockComp(nfc[128], nfc[64])
        self.down_32 = DownBlockComp(nfc[64],  nfc[32])
        self.down_64 = DownBlockComp(nfc[32],  nfc[16])

        self.rf_big = nn.Sequential(
            conv2d(nfc[16] , nfc[8], 1, 1, 0, bias=False),
            batchNorm2d(nfc[8]), nn.LeakyReLU(0.2, inplace=True),
            conv2d(nfc[8], 1, 4, 1, 0, bias=False))

        self.se_2_16 = SEBlock(nfc[512], nfc[64])
        self.se_4_32 = SEBlock(nfc[256], nfc[32])
        self.se_8_64 = SEBlock(nfc[128], nfc[16])

        self.down_from_small = nn.Sequential(
            conv2d(nc, nfc[256], 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            DownBlock(nfc[256],  nfc[128]),
            DownBlock(nfc[128],  nfc[64]),
            DownBlock(nfc[64],  nfc[32]))

        self.rf_small = conv2d(nfc[32], 1, 4, 1, 0, bias=False)

        self.decoder_big = SimpleDecoder(nfc[16], nc)
        self.decoder_part = SimpleDecoder(nfc[32], nc)
        self.decoder_small = SimpleDecoder(nfc[32], nc)

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

    def regularization(self):
    # 正则化项是所有权重参数的平方的和
        l2_reg = sum(param.pow(2).sum() for param in self.parameters() if param.requires_grad)
        return self.l2_lambda * l2_reg

    def forward(self, imgs, label, part=None):
        # first decompose the real images or generated images
        pyrs = None
        if label=='real':
            pyrs, subtrahends, minuends = self.pyramid_decom(imgs)
        else:
            minuends = imgs
        rf_list = []
        reconstruct_big_img_list = []
        reconstruct_feat_32_img_list = []
        for every_batch_big_imgs in minuends[:-5]:
            feat_2 = self.down_from_big(every_batch_big_imgs)
            feat_4 = self.down_4(feat_2)
            feat_8 = self.down_8(feat_4)

            feat_16 = self.down_16(feat_8)
            feat_16 = self.se_2_16(feat_2, feat_16)

            feat_32 = self.down_32(feat_16)
            feat_32 = self.se_4_32(feat_4, feat_32)

            reconstruct_feat_32_img_list.append(feat_32)

            feat_last = self.down_64(feat_32)
            feat_last = self.se_8_64(feat_8, feat_last)

            reconstruct_big_img_list.append(feat_last)

            rf_0 = self.rf_big(feat_last).view(-1)
            rf_list.append(rf_0)

        reconstruct_small_img_list = []
        for every_batch_small_imgs in minuends[-5:-3]:
            feat_small = self.down_from_small(every_batch_small_imgs)
            reconstruct_small_img_list.append(feat_small)
            rf_1 = self.rf_small(feat_small).view(-1)
            rf_list.append(rf_1)


        if label=='real':
            rec_img_big = self.decoder_big(reconstruct_big_img_list[0])
            rec_img_small = self.decoder_small(reconstruct_small_img_list[0])

            assert part is not None
            rec_img_part = None
            if part==0:
                rec_img_part = self.decoder_part(reconstruct_feat_32_img_list[0][: ,: ,:8 ,:8])
            if part==1:
                rec_img_part = self.decoder_part(reconstruct_feat_32_img_list[0][: ,: ,:8 ,8:])
            if part==2:
                rec_img_part = self.decoder_part(reconstruct_feat_32_img_list[0][: ,: ,8: ,:8])
            if part==3:
                rec_img_part = self.decoder_part(reconstruct_feat_32_img_list[0][: ,: ,8: ,8:])

            return torch.cat(rf_list), [rec_img_big, rec_img_small, rec_img_part], pyrs

        return torch.cat(rf_list)

class Discriminator5(nn.Module):
    def __init__(self, ndf=64, nc=3, im_size=512, kernel_size=5, num_high=4):
        super(Discriminator5, self).__init__()
        self.ndf = ndf
        self.im_size = im_size

        self.kernel = self.gauss_kernel(kernel_size, nc)
        self.num_high = num_high

        # 用于正则化的权重
        self.l2_lambda = 1e-4

        nfc_multi = {4 :16, 8 :16, 16 :8, 32 :4, 64 :2, 128 :1, 256 :0.5, 512 :0.25, 1024 :0.125}
        nfc = {}
        for k, v in nfc_multi.items():
            nfc[k] = int(v*ndf)

        if im_size == 1024:
            self.down_from_big = nn.Sequential(
                conv2d(nc, nfc[1024], 4, 2, 1, bias=False),
                nn.LeakyReLU(0.2, inplace=True),
                conv2d(nfc[1024], nfc[512], 4, 2, 1, bias=False),
                batchNorm2d(nfc[512]),
                nn.LeakyReLU(0.2, inplace=True))
        elif im_size == 512:
            self.down_from_big = nn.Sequential(
                conv2d(nc, nfc[512], 4, 2, 1, bias=False),
                nn.LeakyReLU(0.2, inplace=True) )
        elif im_size == 256:
            self.down_from_big = nn.Sequential(
                conv2d(nc, nfc[512], 3, 1, 1, bias=False),
                nn.LeakyReLU(0.2, inplace=True) )

        self.down_4 = DownBlockComp(nfc[512], nfc[256])
        self.down_8 = DownBlockComp(nfc[256], nfc[128])
        self.down_16 = DownBlockComp(nfc[128], nfc[64])
        self.down_32 = DownBlockComp(nfc[64],  nfc[32])
        self.down_64 = DownBlockComp(nfc[32],  nfc[16])

        self.rf_big = nn.Sequential(
            conv2d(nfc[16] , nfc[8], 1, 1, 0, bias=False),
            batchNorm2d(nfc[8]), nn.LeakyReLU(0.2, inplace=True),
            conv2d(nfc[8], 1, 4, 1, 0, bias=False))

        self.se_2_16 = SEBlock(nfc[512], nfc[64])
        self.se_4_32 = SEBlock(nfc[256], nfc[32])
        self.se_8_64 = SEBlock(nfc[128], nfc[16])

        self.down_from_small = nn.Sequential(
            conv2d(nc, nfc[256], 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            DownBlock(nfc[256],  nfc[128]),
            DownBlock(nfc[128],  nfc[64]),
            DownBlock(nfc[64],  nfc[32]))

        self.rf_small = conv2d(nfc[32], 1, 4, 1, 0, bias=False)

        self.decoder_big = SimpleDecoder(nfc[16], nc)
        self.decoder_part = SimpleDecoder(nfc[32], nc)
        self.decoder_small = SimpleDecoder(nfc[32], nc)

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

    def regularization(self):
    # 正则化项是所有权重参数的平方的和
        l2_reg = sum(param.pow(2).sum() for param in self.parameters() if param.requires_grad)
        return self.l2_lambda * l2_reg

    def forward(self, imgs, label, part=None):
        # first decompose the real images or generated images
        pyrs = None
        if label=='real':
            pyrs, subtrahends, minuends = self.pyramid_decom(imgs)
        else:
            minuends = imgs
        rf_list = []
        reconstruct_big_img_list = []
        reconstruct_feat_32_img_list = []
        for every_batch_big_imgs in minuends[:-5]:
            feat_2 = self.down_from_big(every_batch_big_imgs)
            feat_4 = self.down_4(feat_2)
            feat_8 = self.down_8(feat_4)

            feat_16 = self.down_16(feat_8)
            feat_16 = self.se_2_16(feat_2, feat_16)

            feat_32 = self.down_32(feat_16)
            feat_32 = self.se_4_32(feat_4, feat_32)

            reconstruct_feat_32_img_list.append(feat_32)

            feat_last = self.down_64(feat_32)
            feat_last = self.se_8_64(feat_8, feat_last)

            reconstruct_big_img_list.append(feat_last)

            rf_0 = self.rf_big(feat_last).view(-1)
            rf_list.append(rf_0)

        reconstruct_small_img_list = []
        for every_batch_small_imgs in minuends[-5:-3]:
            feat_small = self.down_from_small(every_batch_small_imgs)
            reconstruct_small_img_list.append(feat_small)
            rf_1 = self.rf_small(feat_small).view(-1)
            rf_list.append(rf_1)


        if label=='real':
            rec_img_big = self.decoder_big(reconstruct_big_img_list[0])
            rec_img_small = self.decoder_small(reconstruct_small_img_list[0])

            assert part is not None
            rec_img_part = None
            if part==0:
                rec_img_part = self.decoder_part(reconstruct_feat_32_img_list[0][: ,: ,:8 ,:8])
            if part==1:
                rec_img_part = self.decoder_part(reconstruct_feat_32_img_list[0][: ,: ,:8 ,8:])
            if part==2:
                rec_img_part = self.decoder_part(reconstruct_feat_32_img_list[0][: ,: ,8: ,:8])
            if part==3:
                rec_img_part = self.decoder_part(reconstruct_feat_32_img_list[0][: ,: ,8: ,8:])

            return torch.cat(rf_list), [rec_img_big, rec_img_small, rec_img_part], pyrs

        return torch.cat(rf_list)


class Discriminator6(nn.Module):
    def __init__(self, ndf=64, nc=3, im_size=512, kernel_size=5, num_high=4):
        super(Discriminator6, self).__init__()
        self.ndf = ndf
        self.im_size = im_size

        self.kernel = self.gauss_kernel(kernel_size, nc)
        self.num_high = num_high

        # 用于正则化的权重
        self.l2_lambda = 1e-4

        nfc_multi = {4 :16, 8 :16, 16 :8, 32 :4, 64 :2, 128 :1, 256 :0.5, 512 :0.25, 1024 :0.125}
        nfc = {}
        for k, v in nfc_multi.items():
            nfc[k] = int(v*ndf)

        if im_size == 1024:
            self.down_from_big = nn.Sequential(
                conv2d(nc, nfc[1024], 4, 2, 1, bias=False),
                nn.LeakyReLU(0.2, inplace=True),
                conv2d(nfc[1024], nfc[512], 4, 2, 1, bias=False),
                batchNorm2d(nfc[512]),
                nn.LeakyReLU(0.2, inplace=True))
        elif im_size == 512:
            self.down_from_big = nn.Sequential(
                conv2d(nc, nfc[512], 4, 2, 1, bias=False),
                nn.LeakyReLU(0.2, inplace=True) )
        elif im_size == 256:
            self.down_from_big = nn.Sequential(
                conv2d(nc, nfc[512], 3, 1, 1, bias=False),
                nn.LeakyReLU(0.2, inplace=True) )

        self.down_4 = DownBlockComp(nfc[512], nfc[256])
        self.down_8 = DownBlockComp(nfc[256], nfc[128])
        self.down_16 = DownBlockComp(nfc[128], nfc[64])
        self.down_32 = DownBlockComp(nfc[64],  nfc[32])
        self.down_64 = DownBlockComp(nfc[32],  nfc[16])

        self.rf_big = nn.Sequential(
            conv2d(nfc[16] , nfc[8], 1, 1, 0, bias=False),
            batchNorm2d(nfc[8]), nn.LeakyReLU(0.2, inplace=True),
            conv2d(nfc[8], 1, 4, 1, 0, bias=False))

        self.se_2_16 = SEBlock(nfc[512], nfc[64])
        self.se_4_32 = SEBlock(nfc[256], nfc[32])
        self.se_8_64 = SEBlock(nfc[128], nfc[16])

        self.down_from_small = nn.Sequential(
            conv2d(nc, nfc[256], 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            DownBlock(nfc[256],  nfc[128]),
            DownBlock(nfc[128],  nfc[64]),
            DownBlock(nfc[64],  nfc[32]))

        self.rf_small = conv2d(nfc[32], 1, 4, 1, 0, bias=False)

        self.decoder_big = SimpleDecoder(nfc[16], nc)
        self.decoder_part = SimpleDecoder(nfc[32], nc)
        self.decoder_small = SimpleDecoder(nfc[32], nc)

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

    def regularization(self):
    # 正则化项是所有权重参数的平方的和
        l2_reg = sum(param.pow(2).sum() for param in self.parameters() if param.requires_grad)
        return self.l2_lambda * l2_reg

    def forward(self, imgs, label, part=None):
        # first decompose the real images or generated images
        pyrs = None
        if label=='real':
            pyrs, subtrahends, minuends = self.pyramid_decom(imgs)
        else:
            minuends = imgs
        rf_list = []
        reconstruct_big_img_list = []
        reconstruct_feat_32_img_list = []
        for every_batch_big_imgs in minuends[:-6]:
            feat_2 = self.down_from_big(every_batch_big_imgs)
            feat_4 = self.down_4(feat_2)
            feat_8 = self.down_8(feat_4)

            feat_16 = self.down_16(feat_8)
            feat_16 = self.se_2_16(feat_2, feat_16)

            feat_32 = self.down_32(feat_16)
            feat_32 = self.se_4_32(feat_4, feat_32)

            reconstruct_feat_32_img_list.append(feat_32)

            feat_last = self.down_64(feat_32)
            feat_last = self.se_8_64(feat_8, feat_last)

            reconstruct_big_img_list.append(feat_last)

            rf_0 = self.rf_big(feat_last).view(-1)
            rf_list.append(rf_0)

        reconstruct_small_img_list = []
        for every_batch_small_imgs in minuends[-6:-4]:
            feat_small = self.down_from_small(every_batch_small_imgs)
            reconstruct_small_img_list.append(feat_small)
            rf_1 = self.rf_small(feat_small).view(-1)
            rf_list.append(rf_1)


        if label=='real':
            rec_img_big = self.decoder_big(reconstruct_big_img_list[0])
            rec_img_small = self.decoder_small(reconstruct_small_img_list[0])

            assert part is not None
            rec_img_part = None
            if part==0:
                rec_img_part = self.decoder_part(reconstruct_feat_32_img_list[0][: ,: ,:8 ,:8])
            if part==1:
                rec_img_part = self.decoder_part(reconstruct_feat_32_img_list[0][: ,: ,:8 ,8:])
            if part==2:
                rec_img_part = self.decoder_part(reconstruct_feat_32_img_list[0][: ,: ,8: ,:8])
            if part==3:
                rec_img_part = self.decoder_part(reconstruct_feat_32_img_list[0][: ,: ,8: ,8:])

            return torch.cat(rf_list), [rec_img_big, rec_img_small, rec_img_part], pyrs

        return torch.cat(rf_list)

class SimpleDecoder(nn.Module):
    """docstring for CAN_SimpleDecoder"""
    def __init__(self, nfc_in=64, nc=3):
        super(SimpleDecoder, self).__init__()

        nfc_multi = {4 :16, 8 :8, 16 :4, 32 :2, 64 :2, 128 :1, 256 :0.5, 512 :0.25, 1024 :0.125}
        nfc = {}
        for k, v in nfc_multi.items():
            nfc[k] = int( v *32)

        def upBlock(in_planes, out_planes):
            block = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='nearest'),
                conv2d(in_planes, out_planes *2, 3, 1, 1, bias=False),
                batchNorm2d(out_planes *2), GLU())
            return block

        self.main = nn.Sequential(  nn.AdaptiveAvgPool2d(8),
                                    upBlock(nfc_in, nfc[16]) ,
                                    upBlock(nfc[16], nfc[32]),
                                    upBlock(nfc[32], nfc[64]),
                                    upBlock(nfc[64], nfc[128]),
                                    conv2d(nfc[128], nc, 3, 1, 1, bias=False),
                                    nn.Tanh() )

    def forward(self, input):
        # input shape: c x 4 x 4
        return self.main(input)