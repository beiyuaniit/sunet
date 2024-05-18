import cv2
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

'''
r_unet只验证v4版
v4:x 1
'''


class DoubleConv(nn.Sequential):
    # in_channels:输入特征层的channel
    # out_channels:通过 DoubleConv 后的输出特征层的 channel
    # mid_channels:过第一个卷积层后的输出特征图的 channel
    def __init__(self, in_channels, out_channels, mid_channels=None):
        if mid_channels is None:
            mid_channels = out_channels
        super(DoubleConv, self).__init__(
            # 接收输入通道数为 in_channels，输出通道数为 mid_channels，使用 3x3 的卷积核
            # 通过设置 padding=1 来保持特征图尺寸不变，同时禁用了偏置项（bias）
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            # 针对 mid_channels 通道数的 nn.BatchNorm2d 层，用于规范化输出特征
            nn.BatchNorm2d(mid_channels),
            # 添加 ReLU 激活函数，inplace=True 表示在原张量上直接进行激活操作，节省内存
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )


# 下采样+两个卷积
class Down(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__(
            nn.MaxPool2d(2, stride=2),  # 一个下采样
            DoubleConv(in_channels, out_channels)  # 两个卷积
        )


# 上采样
class Up(nn.Module):
    # in_channels是concat 拼接之后的 channels。对应 Up 模块中第一个卷积的输入 channels
    def __init__(self, in_channels, out_channels, bilinear=True):
        super(Up, self).__init__()
        if bilinear:
            # 双线插值
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            # 转置卷积
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)

        # 为了确保要 concat 拼接的 x1 和 x2 的宽高相同
        # 因为当最初输入的特征图的宽高不是 16 的整数倍时，在下采样后需要进行取整，再进行上采样后可能会出现 尺寸对不上 的问题
        # [N, C, H, W]
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]
        # padding_left, padding_right, padding_top, padding_bottom
        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2,
                        diff_y // 2, diff_y - diff_y // 2])

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


# 输出卷积层
class OutConv(nn.Sequential):
    # 卷积核个数为包含背景的分类类别个数 num_classes
    def __init__(self, in_channels, num_classes):
        super(OutConv, self).__init__(
            nn.Conv2d(in_channels, num_classes, kernel_size=1)
        )


def Roberts(img: torch.Tensor) -> torch.Tensor:
    # 确保在CPU上执行OpenCV操作
    img_np = img.permute(0, 1, 2, 3).detach().cpu().numpy().astype(np.float32)

    # 初始化存储Roberts边缘强度的张量
    roberts_xy_torch = torch.empty(img.shape, dtype=torch.float32, device=img.device)

    roberts_x = np.array([[1, 0], [0, -1]], dtype=np.float32)
    roberts_y = np.array([[0, 1], [-1, 0]], dtype=np.float32)

    for b in range(img.shape[0]):  # 遍历批次
        for c in range(img.shape[1]):  # 遍历通道
            single_channel_img = img_np[b, c, :, :]
            # 应用Roberts算子
            roberts_x_res = cv2.filter2D(single_channel_img, cv2.CV_32F, roberts_x)
            roberts_y_res = cv2.filter2D(single_channel_img, cv2.CV_32F, roberts_y)
            # 合并结果
            roberts_xy = cv2.absdiff(roberts_x_res, roberts_y_res)
            # 将结果复制到roberts_xy_torch对应的批次、通道、高度、宽度位置
            roberts_xy_torch[b, c, :, :] = torch.from_numpy(roberts_xy).to(device=img.device)

    return roberts_xy_torch


class UNetWithRoberts(nn.Module):
    def __init__(self,
                 in_channels: int = 1,
                 num_classes: int = 2,
                 bilinear: bool = True,
                 base_c: int = 64):
        super(UNetWithRoberts, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.bilinear = bilinear

        self.in_conv = DoubleConv(in_channels, base_c)
        self.down1 = Down(base_c * 2, base_c * 2)
        self.down2 = Down(base_c * 2, base_c * 4)
        self.down3 = Down(base_c * 4, base_c * 8)
        factor = 2 if bilinear else 1
        self.down4 = Down(base_c * 8, base_c * 16 // factor)

        self.up1 = Up(base_c * 16, base_c * 8 // factor, bilinear)
        self.up2 = Up(base_c * 8, base_c * 4 // factor, bilinear)
        self.up3 = Up(base_c * 4, base_c * 2 // factor, bilinear)
        self.up4 = Up(base_c * 2, base_c, bilinear)

        self.out_conv = OutConv(base_c, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 计算原始输入的Roberts梯度
        input_with_roberts = torch.cat((x, Roberts(x)), dim=1)
        x1 = self.in_conv(input_with_roberts)

        x1_with_roberts = torch.cat((x1, Roberts(x1)), dim=1)
        x2 = self.down1(x1_with_roberts)

        x2_with_roberts = torch.cat((x2, Roberts(x2)), dim=1)
        x3 = self.down2(x2)

        x3_with_roberts = torch.cat((x3, Roberts(x3)), dim=1)
        x4 = self.down3(x3)

        x4_with_roberts = torch.cat((x4, Roberts(x4)), dim=1)
        x5 = self.down4(x4)

        y1 = self.up1(x5, x4)
        y2 = self.up2(y1, x3)
        y3 = self.up3(y2, x2)
        y4 = self.up4(y3, x1)

        logits = self.out_conv(y4)

        return {"out": logits}
