import torch
import torch.nn.functional as F
from torchvision import transforms
from tqdm import tqdm
import os
from PIL import Image


# 自定义的创建拉普拉斯金字塔的函数
def create_laplacian_pyramid(image, max_levels=5):
    current_img = image
    pyramid = []

    for level in range(max_levels):
        # 使用金字塔形状的卷积进行下采样
        down_img = F.avg_pool2d(current_img, 2)
        pyramid.append(current_img - F.interpolate(down_img, scale_factor=2, mode='nearest'))
        current_img = down_img
    pyramid.append(current_img)
    return pyramid


# 自定义的还原拉普拉斯金字塔的函数
def reconstruct_laplacian_pyramid(pyramid, original_size):
    current_img = pyramid[0]
    for level in range(1, len(pyramid)):
        expand = F.interpolate(current_img, scale_factor=2, mode='nearest')
        current_img = expand + pyramid[level]

    return current_img


# 示例：创建并还原拉普拉斯金字塔
image = torch.randn(1, 3, 64, 64)  # 假设输入图像大小为64x64



if __name__ == "__main__":
    num_high = 5
    gauss_kernel = 5
    device = 'cuda:0'

    # img_PIL = Image.open('/media/zhihao/F05CC6255CC5E706/dataset/few-shot-image-datasets/few-shot-images/100-shot-grumpy_cat/img/6.jpg')

    imgs_list = os.listdir('/home/longzhihao/dataset/few_shot_images_datasets/100-shot-grumpy_cat/img')
    save_dir = '/home/longzhihao/dataset/few_shot_images_datasets/100-shot-grumpy_cat_theoreticcl/img'

    os.makedirs(save_dir, exist_ok=True)

    data_transform = transforms.Compose([
        transforms.Resize((256, 256)),  # 调整图像大小为256x256
        transforms.ToTensor(),  # 将图像转换为张量
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
          # 标准化图像张量
    ])
    solo_normalization = transforms.Compose([transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])

    for every_img in tqdm(imgs_list):
        img_PIL = Image.open(os.path.join('/home/longzhihao/dataset/few_shot_images_datasets/100-shot-grumpy_cat/img', every_img))
        tensor_img = data_transform(img_PIL)

        # pyrs = lap_pyramid.pyramid_decom(img=tensor_img.unsqueeze(0).to(device))

        # laplacian_pyramid = create_laplacian_pyramid(tensor_img.unsqueeze(0).to(device), max_levels=5)
        # trans_pyrs = []
        # for i in range(num_high + 1):
        #     trans_pyr = laplacian_pyramid[-1 - i]
        #     trans_pyrs.append(trans_pyr)
        #
        # reconstructed_image = reconstruct_laplacian_pyramid(trans_pyrs, original_size=(256, 256))

        # for i in pyrs:
        #     print(i.shape)


        #
        # out = lap_pyramid.pyramid_recons(trans_pyrs)

       #  resiudal_high_frequency = tensor_img.unsqueeze(0).to(device) - out

        # final_out = resiudal_high_frequency + out
        trans_PILImage = transforms.ToPILImage()
        final_img = tensor_img.unsqueeze(0).add(1).mul(0.5)
        pil_img = trans_PILImage(final_img.squeeze(0))

        pil_img.save(os.path.join(save_dir, every_img))