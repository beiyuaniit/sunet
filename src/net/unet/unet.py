from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F


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


# 下采样
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


class UNet(nn.Module):
    # in_channels: 输入特征图的通道数，彩色图片为 3 ，灰度图片为 1
    # num_classes: 包含背景的分类类别个数
    # bilinear: 表示是否使用双线性插值法
    # base_c: 基础通道数 channel
    def __init__(self,
                 in_channels: int = 1,
                 num_classes: int = 2,
                 bilinear: bool = True,
                 base_c: int = 64):
        super(UNet, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.bilinear = bilinear

        self.in_conv = DoubleConv(in_channels, base_c)
        self.down1 = Down(base_c, base_c * 2)
        self.down2 = Down(base_c * 2, base_c * 4)
        self.down3 = Down(base_c * 4, base_c * 8)
        # factor因子，用于控制上采样过程中特征图的通道数，这个因子的值取决于是否使用双线性插值的 bilinear 标志
        # factor 因子的引入主要是为了在上采用过程中控制特征图的大小和复杂度，以适应不同的任务需求和计算资源限制
        # 当 bilinear=True 时，factor = 2，说明在上采样过程中，特征图的通道数减半，当 base_c 为 64 时，上采样后特征图的通道数为 32
        # 当 bilinear=False 时，factor = 1，说明在上采样过程中，特征图的通道数不变，当 base_c 为 64 时，上采样后特征图的通道数为 64
        factor = 2 if bilinear else 1
        self.down4 = Down(base_c * 8, base_c * 16 // factor)


        self.up1 = Up(base_c * 16, base_c * 8 // factor, bilinear)
        self.up2 = Up(base_c * 8, base_c * 4 // factor, bilinear)
        self.up3 = Up(base_c * 4, base_c * 2 // factor, bilinear)
        self.up4 = Up(base_c * 2, base_c, bilinear)
        self.out_conv = OutConv(base_c, num_classes)

    # U-Net 的 forward 前向传播函数，它接受一个 torch.Tensor 类型的输入 x ，并返回一个字典类型的输出
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        # train 1
        # x 4 3 480 480
        # x1 4 32 480 480
        # x2 4 64 240 240
        # x3 4 128 120 120
        # x5 4 256 30 30

        # y1 4 128 60 60
        # y2 4 64 120 120
        # y3 4 32 240 240
        # y4 4 32 480 480
        # out 4 2 480 480

        # predict 1
        # x
        # x1
        # x2
        # x4
        # x5

        # x_s
        # x1_s
        # x2_s
        # x3_s
        # x4_s

        # y1
        # y2
        # y3
        # y4
        # out


        x1 = self.in_conv(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4) # y1
        x = self.up2(x, x3) # y2
        x = self.up3(x, x2) # y3
        x = self.up4(x, x1) # y4
        logits = self.out_conv(x) # y5

        return {"out": logits}
