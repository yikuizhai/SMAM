import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
from PIL import Image
import numpy as np
from torchvision import transforms
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from tqdm import tqdm
from torchvision import utils as vutils


class Lap_Pyramid_Conv(nn.Module):
    def __init__(self, num_high=3, kernel_size=5, channels=3):
        super().__init__()

        self.num_high = num_high
        self.kernel = self.gauss_kernel(kernel_size, channels)

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

    def downsample(self, x):
        return x[:, :, ::2, ::2]

    def pyramid_down(self, x):
        return self.downsample(self.conv_gauss(x, self.kernel))

    def upsample(self, x):
        up = torch.zeros((x.size(0), x.size(1), x.size(2) * 2, x.size(3) * 2),
                         device=x.device)
        up[:, :, ::2, ::2] = x * 4

        return self.conv_gauss(up, self.kernel)

    # def pyramid_decom(self, img):
    #     # self.kernel = self.kernel.to(img.device)
    #     current = img
    #     pyr = []
    #     for _ in range(self.num_high):
    #         down = self.pyramid_down(current)
    #         up = self.upsample(down)
    #         diff = current - up
    #         pyr.append(diff)
    #         current = down
    #     pyr.append(current)
    #     return pyr

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


if __name__ == "__main__":
    num_high = 5
    gauss_kernel = 5
    device = 'cuda:0'
    lap_pyramid = Lap_Pyramid_Conv(num_high, gauss_kernel)

    # img_PIL = Image.open('/media/zhihao/F05CC6255CC5E706/dataset/few-shot-image-datasets/few-shot-images/100-shot-grumpy_cat/img/6.jpg')

    imgs_list = os.listdir('/home/longzhihao/dataset/few_shot_images_datasets/100-shot-grumpy_cat/img')
    save_dir = '/home/longzhihao/dataset/few_shot_images_datasets/100-shot-grumpy_cat_theoreticcl/img'

    save_intermedian_pyrs_imgs = './Laplacian_Pyramid_decomposed_pyrs_imgs'
    os.makedirs(save_intermedian_pyrs_imgs, exist_ok=True)
    save_intermedian_subtrahend_imgs = './Laplacian_Pyramid_decomposed_subtrahend_imgs'
    os.makedirs(save_intermedian_subtrahend_imgs, exist_ok=True)
    save_intermedian_minuends_imgs = './Laplacian_Pyramid_decomposed_minuends_imgs'
    os.makedirs(save_intermedian_minuends_imgs, exist_ok=True)

    os.makedirs(save_dir, exist_ok=True)

    data_transform = transforms.Compose([
        transforms.Resize((256, 256)),  # 调整图像大小为256x256
        transforms.ToTensor(),  # 将图像转换为张量
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
          # 标准化图像张量
    ])
    solo_normalization = transforms.Compose([transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
    trans_PILImage = transforms.ToPILImage()
    for every_img in tqdm(imgs_list):
        img_PIL = Image.open(os.path.join('/home/longzhihao/dataset/few_shot_images_datasets/100-shot-grumpy_cat/img', every_img))
        tensor_img = data_transform(img_PIL)

        pyrs, subtrahend, minuends = lap_pyramid.pyramid_decom(img=tensor_img.unsqueeze(0).to(device))

        # for i in pyrs:
        #     print(i.shape)

        trans_pyrs = []
        for i in range(num_high + 1):
            trans_pyr = pyrs[-1 - i]
            trans_img = trans_pyr.add(1).mul(0.5)
            trans_pil_img = trans_PILImage(trans_img.squeeze(0))
            trans_pil_img.save(os.path.join(save_intermedian_pyrs_imgs, every_img.split('.')[0] + '_' + str(i) + '.jpg'))
            trans_pyrs.append(trans_pyr)

        for j, img in enumerate(subtrahend):
            img =img.add(1).mul(0.5)
            img = trans_PILImage(img.squeeze(0))
            img.save(
                os.path.join(save_intermedian_subtrahend_imgs, every_img.split('.')[0] + '_' + str(j) + '.jpg'))

        for k, img in enumerate(minuends):
            img =img.add(1).mul(0.5)
            img = trans_PILImage(img.squeeze(0))
            img.save(
                os.path.join(save_intermedian_minuends_imgs, every_img.split('.')[0] + '_' + str(k) + '.jpg'))

        out = lap_pyramid.pyramid_recons(trans_pyrs)

       #  resiudal_high_frequency = tensor_img.unsqueeze(0).to(device) - out

        # final_out = resiudal_high_frequency + out

        final_img = out.add(1).mul(0.5)
        pil_img = trans_PILImage(final_img.squeeze(0))

        pil_img.save(os.path.join(save_dir, every_img))



