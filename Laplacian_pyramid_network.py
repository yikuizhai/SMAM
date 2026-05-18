

class Lap_Pyramid_Network(nn.Module):
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