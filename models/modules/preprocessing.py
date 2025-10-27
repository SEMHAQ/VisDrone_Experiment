import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms


class ImageEnhancement(nn.Module):
    """
    图像增强预处理模块
    针对VisDrone数据集的图像质量问题设计
    """

    def __init__(self):
        super(ImageEnhancement, self).__init__()

    def forward(self, x):
        """
        x: 输入图像张量 [B, C, H, W]
        返回: 增强后的图像张量
        """
        # 这里实现图像增强逻辑
        # 由于在数据加载阶段实现更合适，这里主要定义接口
        return x


class AdaptiveCLAHE:
    """
    自适应对比度受限直方图均衡化
    用于改善图像对比度
    """

    def __init__(self, clip_limit=2.0, grid_size=(8, 8)):
        self.clip_limit = clip_limit
        self.grid_size = grid_size

    def __call__(self, image):
        if isinstance(image, torch.Tensor):
            # 转换为numpy进行处理
            image_np = image.permute(1, 2, 0).numpy() * 255
            image_np = image_np.astype(np.uint8)
        else:
            image_np = image

        # 转换为LAB颜色空间
        lab = cv2.cvtColor(image_np, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)

        # 对L通道应用CLAHE
        clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=self.grid_size)
        l_enhanced = clahe.apply(l)

        # 合并通道并转换回RGB
        lab_enhanced = cv2.merge([l_enhanced, a, b])
        enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2RGB)

        if isinstance(image, torch.Tensor):
            # 转换回torch张量
            enhanced = torch.from_numpy(enhanced).float() / 255.0
            enhanced = enhanced.permute(2, 0, 1)

        return enhanced


class MotionDeblur:
    """
    运动去模糊处理
    针对无人机运动模糊问题
    """

    def __init__(self, kernel_size=15, angle=0, strength=1.0):
        self.kernel_size = kernel_size
        self.angle = angle
        self.strength = strength

    def __call__(self, image):
        if isinstance(image, torch.Tensor):
            image_np = image.permute(1, 2, 0).numpy() * 255
            image_np = image_np.astype(np.uint8)
        else:
            image_np = image

        # 创建运动模糊核
        kernel = self._create_motion_kernel()

        # 应用维纳滤波去模糊
        deblurred = self._wiener_filter(image_np, kernel)

        if isinstance(image, torch.Tensor):
            deblurred = torch.from_numpy(deblurred).float() / 255.0
            deblurred = deblurred.permute(2, 0, 1)

        return deblurred

    def _create_motion_kernel(self):
        """创建运动模糊核"""
        kernel = np.zeros((self.kernel_size, self.kernel_size))
        kernel[int((self.kernel_size - 1) / 2), :] = np.ones(self.kernel_size)
        kernel = kernel / self.kernel_size

        # 旋转核以匹配运动角度
        from scipy import ndimage
        kernel = ndimage.rotate(kernel, self.angle, reshape=False)

        return kernel

    def _wiener_filter(self, img, kernel, K=0.01):
        """维纳滤波去模糊"""
        from scipy.signal import convolve2d

        # 对每个通道分别处理
        result = np.zeros_like(img)
        for i in range(3):
            channel = img[:, :, i]

            # 计算频域滤波
            kernel_padded = np.zeros_like(channel)
            kh, kw = kernel.shape
            kernel_padded[:kh, :kw] = kernel

            # 维纳滤波
            H = np.fft.fft2(kernel_padded)
            G = np.fft.fft2(channel)
            F_est = np.conj(H) / (np.abs(H) ** 2 + K) * G
            f_est = np.fft.ifft2(F_est)

            result[:, :, i] = np.real(f_est)

        return np.clip(result, 0, 255).astype(np.uint8)