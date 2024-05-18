import torch
import torch.nn as nn


# 构建用于计算 Dice 系数的目标张量
# 参数
# target (torch.Tensor): 输入的目标张量，通常是一个二维或三维张量，其中包含每个像素的类别标签
# num_classes (int): 类别的总数，默认为2
# ignore_index (int): 需要忽略的类别索引，默认为-100，表示在计算 Dice 系数时不考虑此索引对应类别的像素
def build_target(target: torch.Tensor, num_classes: int = 2, ignore_index: int = -100):
    """build target for dice coefficient"""
    dice_target = target.clone()
    # 如果指定了需要忽略的类别索引，找出该类别索引在目标张量中出现的位置，并将这些位置的值设为0
    if ignore_index >= 0:
        ignore_mask = torch.eq(target, ignore_index)
        # 将 ignore_mask 对应位置的值设为0
        dice_target[ignore_mask] = 0
        # [N, H, W] -> [N, H, W, C]
        # 将目标张量转换为 one-hot 编码形式
        dice_target = nn.functional.one_hot(dice_target, num_classes).float()
        # 恢复原值
        dice_target[ignore_mask] = ignore_index
    else:
        dice_target = nn.functional.one_hot(dice_target, num_classes).float()

    # 转换后用于计算 Dice 系数的目标张量，其形状为 [N, C, H, W]，其中 N 是批量大小，C 是类别数，H 和 W 是图像的高度和宽度
    return dice_target.permute(0, 3, 1, 2)

# 计算一个批次内所有图像的平均 Dice 系数，或单个分割掩膜的 Dice 系数
# 参数
# x (torch.Tensor): 输入张量，模型预测的输出，通常是一个四维张量，形状为 [batch_size, num_classes, height, width]。
# target (torch.Tensor): 真实标签张量（GT），也是一个四维张量，形状与输入 x 相同。
# ignore_index (int): 可选参数，指定要忽略的类别索引，默认为 -100，计算 Dice 系数时不会考虑该类别的像素。
# epsilon (float): 添加到分子和分母的小正值，用于防止除以零的情况，提高计算稳定性，默认为 1e-6。
def dice_coeff(x: torch.Tensor, target: torch.Tensor, ignore_index: int = -100, epsilon=1e-6):
    # Average of Dice coefficient for all batches, or for a single mask
    # 计算一个batch中所有图片某个类别的dice_coefficient
    # 初始化 Dice 系数总和
    d = 0.
    # 获取批次大小
    batch_size = x.shape[0]
    # 遍历批次中的每一幅图像
    for i in range(batch_size):
        # 将当前图像的预测输出与真实标签展平为一维张量
        x_i = x[i].reshape(-1)
        t_i = target[i].reshape(-1)
        # 如果存在需要忽略的类别，则筛选出有效的像素
        if ignore_index >= 0:
            # 找出mask中不为ignore_index的区域
            roi_mask = torch.ne(t_i, ignore_index)
            x_i = x_i[roi_mask]
            t_i = t_i[roi_mask]
        # 计算交集的元素个数
        inter = torch.dot(x_i, t_i)
        # 计算预测集合与真实集合元素的总和
        sets_sum = torch.sum(x_i) + torch.sum(t_i)
        # 对于 sets_sum 为 0 的特殊情况，避免除以零错误
        if sets_sum == 0:
            sets_sum = 2 * inter
        # 更新 Dice 系数总和
        d += (2 * inter + epsilon) / (sets_sum + epsilon)
    # 计算并返回整个批次的平均 Dice 系数
    return d / batch_size

# 计算所有类别的 Dice 系数平均值
# 参数:
# x (torch.Tensor): 模型预测输出的张量，通常是一个四维张量，形状为 [batch_size, num_classes, height, width]。
# target (torch.Tensor): 真实标签张量，与输入 x 形状相同。
# ignore_index (int): 可选参数，指定要忽略的类别索引，默认为 -100，在计算 Dice 系数时不会考虑该类别的像素。
# epsilon (float): 添加到分子和分母的小正值，用于防止除以零的情况，提高计算稳定性，默认为 1e-6。
def multiclass_dice_coeff(x: torch.Tensor, target: torch.Tensor, ignore_index: int = -100, epsilon=1e-6):
    """Average of Dice coefficient for all classes"""
    dice = 0.
    # 遍历预测输出的所有通道（即类别）
    for channel in range(x.shape[1]):
        # 对于每个类别，计算其 Dice 系数，并累加到 dice 中
        dice += dice_coeff(x[:, channel, ...], target[:, channel, ...], ignore_index, epsilon)
    # 计算并返回所有类别的 Dice 系数平均值
    return dice / x.shape[1]

# 计算 Dice 损失
# 参数:
# x (torch.Tensor): 模型的原始输出张量，通常是一个四维张量，形状为 [batch_size, num_classes, height, width]。
# target (torch.Tensor): 真实标签张量，与输入 x 形状相同。
# multiclass (bool): 可选参数，决定是否计算多类别 Dice 损失，默认为 False，即计算单类别 Dice 损失。
# ignore_index (int): 可选参数，指定要忽略的类别索引，默认为 -100，在计算 Dice 损失时不会考虑该类别的像素。
def dice_loss(x: torch.Tensor, target: torch.Tensor, multiclass: bool = False, ignore_index: int = -100):
    # Dice loss (objective to minimize) between 0 and 1
    # 将原始输出经过 softmax 函数处理，转换为概率分布。得到每个像素针对每个类别的概率
    x = nn.functional.softmax(x, dim=1)
    # 根据 multiclass 参数选择 Dice 系数计算函数
    fn = multiclass_dice_coeff if multiclass else dice_coeff
    # 计算 Dice 系数，并从 1 中减去以得到 Dice 损失
    # 得到的 Dice 损失值，范围在 0 和 1 之间，值越小表示模型预测效果越好
    return 1 - fn(x, target, ignore_index=ignore_index)
