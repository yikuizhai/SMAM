import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader
from torchvision import transforms
from torchvision import utils as vutils
import cv2
import numpy as np
import argparse
import random
from tqdm import tqdm
import os
from tensorboardX import SummaryWriter

from models import weights_init
from generator import Generator3
from discriminator import Discriminator
from operation import copy_G_params, load_params, get_dir
from operation import ImageFolder, InfiniteSamplerWrapper
from diffaug import DiffAugment

policy = 'color,translation'
import lpips

percept = lpips.PerceptualLoss(model='net-lin', net='vgg', use_gpu=True)


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


def crop_image_by_part(image, part):
    hw = image.shape[2] // 2
    if part == 0:
        return image[:, :, :hw, :hw]
    if part == 1:
        return image[:, :, :hw, hw:]
    if part == 2:
        return image[:, :, hw:, :hw]
    if part == 3:
        return image[:, :, hw:, hw:]


# def train_d(net, data, label="real"):
#     """Train function of discriminator"""
#     if label=="real":
#         part = random.randint(0, 3)
#         pred = net(data, label, part=part)
#         err = F.relu(torch.rand_like(pred) * 0.2 + 0.8 - pred).mean()
#         return err
#     else:
#         pred = net(data, label)
#         err = F.relu(torch.rand_like(pred) * 0.2 + 0.8 + pred).mean()
#         return err

def train_d(net, data, label="real"):
    """Train function of discriminator"""
    if label == "real":
        part = random.randint(0, 3)
        pred, [rec_all, rec_small, rec_part], pyrs = net(data, label, part=part)
        err = F.relu(torch.rand_like(pred) * 0.2 + 0.8 - pred).mean() + \
              percept(rec_all, F.interpolate(data, rec_all.shape[2])).sum() + \
              percept(rec_small, F.interpolate(data, rec_small.shape[2])).sum() + \
              percept(rec_part, F.interpolate(crop_image_by_part(data, part), rec_part.shape[2])).sum()
        return err, rec_all, rec_small, rec_part, pyrs
    else:
        pred = net(data, label)
        err = F.relu(torch.rand_like(pred) * 0.2 + 0.8 + pred).mean()
        return err


def train(args):
    data_root = args.path
    total_iterations = args.iter
    checkpoint = args.ckpt
    batch_size = args.batch_size
    im_size = args.im_size
    ndf = 64
    ngf = 64
    nz = 256
    nlr = 0.0002
    nbeta1 = 0.5
    lambda_alignment = 1.5
    lambda_d_alignment = 0.5
    num_high_choose = {'256': 3, '512': 4, '1024': 5}
    num_high = num_high_choose[str(im_size)]
    gauss_kernel = 3
    use_cuda = True
    multi_gpu = False
    dataloader_workers = 4
    current_iteration = 0
    save_interval = 100
    saved_model_folder, saved_image_folder, saved_freimage_folder = get_dir(args)
    tb_writer = SummaryWriter(os.path.join(saved_freimage_folder, 'logs'))
    vision_tag = ['D-loss', 'D-loss-real', 'D-loss-fake', 'G-loss']

    device = torch.device("cpu")
    if use_cuda:
        device = torch.device("cuda:%d" % args.cuda)

    transform_list = [
        transforms.Resize((int(im_size), int(im_size))),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ]
    trans = transforms.Compose(transform_list)

    if 'lmdb' in data_root:
        from operation import MultiResolutionDataset
        dataset = MultiResolutionDataset(data_root, trans, 1024)
    else:
        dataset = ImageFolder(root=data_root, transform=trans)

    dataloader = iter(DataLoader(dataset, batch_size=batch_size, shuffle=False,
                                 sampler=InfiniteSamplerWrapper(dataset), num_workers=dataloader_workers,
                                 pin_memory=True))
    '''
    loader = MultiEpochsDataLoader(dataset, batch_size=batch_size, 
                               shuffle=True, num_workers=dataloader_workers, 
                               pin_memory=True)
    dataloader = CudaDataLoader(loader, 'cuda')
    '''

    lap_pyramid = Lap_Pyramid_Conv(num_high, gauss_kernel)
    # from model_s import Generator, Discriminator
    netG = Generator3(ngf=ngf, nz=nz, im_size=im_size, kernel_size=gauss_kernel, num_high=num_high)
    netG.apply(weights_init)

    netD = Discriminator(ndf=ndf, im_size=im_size, kernel_size=gauss_kernel, num_high=num_high)
    netD.apply(weights_init)

    netG.to(device)
    netD.to(device)

    avg_param_G = copy_G_params(netG)

    fixed_noise = torch.FloatTensor(8, nz).normal_(0, 1).to(device)
    criteria = torch.nn.L1Loss()

    optimizerG = optim.Adam(netG.parameters(), lr=nlr, betas=(nbeta1, 0.999))
    optimizerD = optim.Adam(netD.parameters(), lr=nlr, betas=(nbeta1, 0.999))

    if checkpoint != 'None':
        ckpt = torch.load(checkpoint)
        netG.load_state_dict(ckpt['g'])
        netD.load_state_dict(ckpt['d'])
        avg_param_G = ckpt['g_ema']
        optimizerG.load_state_dict(ckpt['opt_g'])
        optimizerD.load_state_dict(ckpt['opt_d'])
        current_iteration = int(checkpoint.split('_')[-1].split('.')[0])
        del ckpt

    if multi_gpu:
        netG = nn.DataParallel(netG.to(device))
        netD = nn.DataParallel(netD.to(device))

    for iteration in tqdm(range(current_iteration, total_iterations + 1)):
        real_image = next(dataloader)
        real_image = real_image.to(device)

        current_batch_size = real_image.size(0)
        noise = torch.Tensor(current_batch_size, nz).normal_(0, 1).to(device)

        fake_images, residual_frequency_alignment = netG.forward(noise, skips=None)

        real_images = DiffAugment(real_image, policy=policy)

        fake_images = [DiffAugment(fake, policy=policy) for fake in fake_images]

        ## 2. train Discriminator
        netD.zero_grad()
        do_Dr1 = (iteration % 10 == 0)
        err_dr, rec_img_all, rec_img_small, rec_img_part, pyrs_real = train_d(netD, real_images, label="real")

        err_df = train_d(netD, [fi.detach() for fi in fake_images], label="fake")
        # loss_Dr1 = 0
        # if do_Dr1:
        #     loss_Dr1 = netD.regularization()
        loss = err_df + err_dr
        loss.backward()
        optimizerD.step()

        ## 3. train Generator
        netG.zero_grad()
        fake_images2, residual_frequency_alignment2 = netG.forward(noise, skips=None)
        fake_images2 = [DiffAugment(fake, policy=policy) for fake in fake_images2]
        loss_alignment = 0
        for fake, real in zip(residual_frequency_alignment2, pyrs_real):
            loss_alignment += criteria(fake, real)
        # loss_alignment = criteria(pyrs[-1], residual_frequency_alignment2)

        pred_g = netD.forward(fake_images2, "fake")
        err_g = -pred_g.mean() * lambda_d_alignment + loss_alignment * lambda_alignment

        err_g.backward()
        optimizerG.step()

        for p, avg_p in zip(netG.parameters(), avg_param_G):
            avg_p.mul_(0.999).add_(0.001 * p.data)

        for tag, value in zip(vision_tag, [err_df + err_dr, err_df, err_dr, err_g]):
            tb_writer.add_scalars(tag, {'train': value}, iteration)

        if iteration % 100 == 0:
            print("GAN: loss d: %.5f    loss g: %.5f" % (err_dr, -err_g.item()))

        if iteration % (save_interval * 10) == 0:
            backup_para = copy_G_params(netG)
            load_params(netG, avg_param_G)
            with torch.no_grad():
                vutils.save_image(netG.forward(fixed_noise, skips=True).add(1).mul(0.5),
                                  saved_image_folder + '/%d.jpg' % iteration, nrow=4)
                vutils.save_image(torch.cat([
                    F.interpolate(real_image, 128),
                    rec_img_all, rec_img_small,
                    rec_img_part, F.interpolate(residual_frequency_alignment2[-1], 128)
                ]).add(1).mul(0.5), saved_image_folder + '/rec_%d.jpg' % iteration)
            load_params(netG, backup_para)

        if iteration % (save_interval * 50) == 0 or iteration == total_iterations:
            backup_para = copy_G_params(netG)
            load_params(netG, avg_param_G)
            torch.save({'g': netG.state_dict(), 'd': netD.state_dict()}, saved_model_folder + '/%d.pth' % iteration)
            load_params(netG, backup_para)
            torch.save({'g': netG.state_dict(),
                        'd': netD.state_dict(),
                        'g_ema': avg_param_G,
                        'opt_g': optimizerG.state_dict(),
                        'opt_d': optimizerD.state_dict()}, saved_model_folder + '/all_%d.pth' % iteration)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='region gan')

    parser.add_argument('--path', type=str,
                        default='/dssg/home/zn_lzhx/PytorchPro/few-shot-images/moongate/img',
                        help='path of resource dataset, should be a folder that has one or many sub image folders inside')
    parser.add_argument('--cuda', type=int, default=2, help='index of gpu to use')
    parser.add_argument('--name', type=str, default='test20240410_latent_adversarial_all', help='experiment name')
    parser.add_argument('--iter', type=int, default=100000, help='number of iterations')
    parser.add_argument('--start_iter', type=int, default=0, help='the iteration to start training')
    parser.add_argument('--batch_size', type=int, default=8, help='mini batch number of images')
    parser.add_argument('--im_size', type=int, default=512, help='image resolution')
    parser.add_argument('--ckpt', type=str,
                        default='None',
                        help='checkpoint weight path if have one')

    args = parser.parse_args()
    print(args)

    train(args)
