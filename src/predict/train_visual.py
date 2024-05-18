import matplotlib.pyplot as plt
import numpy as np


# 读取并解析result.txt文件中的train_loss和lr数据
def get_loss_lr(filename):
    train_losses = []
    lrs = []
    with open(filename, 'r') as file:
        lines = file.readlines()
        epoch_data = []  # 临时存储当前epoch的所有数据行
        for line in lines:
            if line.startswith('[epoch:'):  # 新的epoch开始
                if epoch_data:  # 如果已经收集了一组epoch数据
                    # 从epoch_data中解析train_loss和lr
                    train_loss = float(epoch_data[1].split(': ')[1])
                    lr = float(epoch_data[2].split(': ')[1].strip())
                    train_losses.append(train_loss)
                    lrs.append(lr)
                    epoch_data.clear()  # 清空列表准备下一个epoch的数据
                epoch_data.append(line.strip())  # 添加当前行到epoch_data
            else:  # 非epoch开始行，属于当前epoch的数据
                epoch_data.append(line.strip())
        # 处理最后一个epoch的数据（如果有的话）
        if epoch_data:
            train_loss = float(epoch_data[1].split(': ')[1])
            lr = float(epoch_data[2].split(': ')[1].strip())
            train_losses.append(train_loss)
            lrs.append(lr)
    return train_losses, lrs


def loss_lr_show(path):

    # 解析数据
    train_losses, lrs = get_loss_lr(path)

    # 可视化
    epochs = np.arange(1, len(train_losses) + 1)  # epoch从1开始计数

    # 设置绘图
    plt.figure(figsize=(14, 7))

    # Train Loss曲线
    ax0 = plt.subplot(2, 1, 1)
    ax0.plot(epochs, train_losses, label='Train Loss')
    ax0.set_xlabel('Epoch')
    ax0.set_ylabel('Train Loss')
    ax0.legend()

    # Learning Rate曲线
    ax1 = plt.subplot(2, 1, 2)
    ax1.plot(epochs, lrs, label='Learning Rate', color='orange')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Learning Rate')
    ax1.legend()

    # 调整子图间距
    plt.tight_layout()

    plt.savefig("sunt_v4_1_glas_train_loss_lr.png")
    # plt.show()



# 读取并解析result.txt文件中的train_loss和lr数据
def get_dice_iou(filename):
    dice = []
    iou = []
    with open(filename, 'r') as file:
        lines = file.readlines()
        epoch_data = []  # 临时存储当前epoch的所有数据行
        for line in lines:
            if line.startswith('[epoch:'):  # 新的epoch开始
                if epoch_data:  # 如果已经收集了一组epoch数据
                    # 从epoch_data中解析train_loss和lr
                    train_loss = float(epoch_data[3].split(': ')[1])
                    lr = float(epoch_data[7].split(': ')[1].strip())
                    dice.append(train_loss)
                    iou.append(lr)
                    epoch_data.clear()  # 清空列表准备下一个epoch的数据
                epoch_data.append(line.strip())  # 添加当前行到epoch_data
            else:  # 非epoch开始行，属于当前epoch的数据
                epoch_data.append(line.strip())
        # 处理最后一个epoch的数据（如果有的话）
        if epoch_data:
            train_loss = float(epoch_data[3].split(': ')[1])
            lr = float(epoch_data[7].split(': ')[1].strip())
            dice.append(train_loss)
            iou.append(lr)
    return dice, iou


def diec_iou_show(path):

    # 解析数据
    dice, iou = get_dice_iou(path)

    # 可视化
    epochs = np.arange(1, len(dice) + 1)  # epoch从1开始计数

    # 设置绘图
    plt.figure(figsize=(14, 7))

    # Train Loss曲线
    ax0 = plt.subplot(2, 1, 1)
    ax0.plot(epochs, dice, label='Dice')
    ax0.set_xlabel('Epoch')
    ax0.set_ylabel('Dice')
    ax0.legend()

    # Learning Rate曲线
    ax1 = plt.subplot(2, 1, 2)
    ax1.plot(epochs, iou, label='IoU', color='orange')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('IoU')
    ax1.legend()

    # 调整子图间距
    plt.tight_layout()

    plt.savefig("sunt_v4_1_glas_train_dice_iou.png")
    # plt.show()



if __name__ == '__main__':
    path = "../../result/train/sunet/v4_1/result_glas.txt"
    loss_lr_show(path)
    diec_iou_show(path)
