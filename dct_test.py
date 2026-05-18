import random
from torchvision import utils as vutils
import torch
from torch import fft
from torch.fft import fft2, ifft2
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
import os
from PIL import Image
from torchvision import transforms

device = "cuda:0"

data_transform = transforms.Compose([
        transforms.Resize((1024, 1024)),  # 调整图像大小为256x256
        transforms.ToTensor(),  # 将图像转换为张量
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
          # 标准化图像张量
    ])
imgs_list = os.listdir('/home/longzhihao/dataset/few_shot_images_datasets/FFHQ-2000/img')
img_PIL = Image.open(os.path.join('/home/longzhihao/dataset/few_shot_images_datasets/FFHQ-2000/img', random.choice(imgs_list)))
x = data_transform(img_PIL)
x = x.unsqueeze(0)

#x = x.add(1).mul(0.5)
print(x)
# # 计算DCT
# dct_matrix = fft.fft(x, 2 * x.size(-1))  # 对图像进行FFT变换
# dct_matrix *= np.sqrt(1.0 / x.size(-1))  # 归一化
# print(dct_matrix.shape)
# dct_matrix = fft.ifft(dct_matrix, 2 * x.size(-1)).real  # 进行IFFT变换以获得实数部分
# 执行FFT
#fft_image = fft2(x)

# 将结果转换到双精度并归一化
#fft_image = torch.abs(fft2(x))
#dct_matrix = (fft_image / fft_image.max()) * 255
# # 执行傅里叶变换
# complex_tensor = fft.fft2(x)

# 进行傅里叶变换
fre = torch.fft.fftn(x, dim=(-2, -1))  # 在图像的最后两个维度上执行傅里叶变换
fre_SHIFT = torch.fft.fftshift(fre)
print(fre_SHIFT)
fre_m = torch.abs(fre)  # 幅度谱，求模得到
fre_p = torch.angle(fre)  # 相位谱，求相角得到

# 把相位设为常数
constant = torch.mean(fre_m)
fre_ = constant * torch.exp(1j * fre_p)  # 把幅度谱和相位谱再合并为复数形式的频域图数据
img_onlyphase = torch.abs(torch.fft.ifftn(fre_, dim=(-2, -1)))  # 还原为空间域图像

# 把振幅设为常数
constant = torch.mean(fre_p)
fre_ = fre_m * torch.exp(1j * constant)
print(fre_.shape)
img_onlymagnitude = torch.abs(torch.fft.ifftn(fre_, dim=(-2, -1)))


# # 进行DCT变换
# dct_matrix = fft.fftshift(complex_tensor)  # 将DC分量移到频谱中心
# dct_matrix = torch.abs(dct_matrix)  # 计算频谱的幅度
print(img_onlymagnitude.shape)
print(img_onlyphase.shape)

vutils.save_image(torch.cat([x, img_onlymagnitude, img_onlyphase]).add(1).mul(0.5), 'result.jpg')
