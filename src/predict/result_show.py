import matplotlib.pyplot as plt
import numpy as np


def dice_show():
    # 定义数据集和模型的Dice系数
    dice_results = {
        'DRIVE': (0.79368, 0.78978, 0.78526),
        'GLAS': (0.8867, 0.8818, 0.88004),
        'DSB': (0.84016, 0.84818, 0.87666),
    }

    models = ('U-Net', 'R-UNet', 'S-UNet')
    x = np.arange(len(dice_results.keys()))  # the label locations on x-axis for datasets
    width = 0.2  # the width of the bars
    multiplier = 0

    fig, ax = plt.subplots(figsize=(10, 6), layout='constrained')

    colors = ['tab:blue', 'tab:orange', 'tab:green']

    for model_idx, model in enumerate(models):
        offset = width * multiplier
        rects = ax.bar(x - width / 2 + offset, [val[model_idx] for val in dice_results.values()],
                       width, label=model, color=colors[model_idx])
        ax.bar_label(rects, padding=3, fmt='%.4f')
        multiplier += 1

    # Customizing the plot
    ax.set_ylabel('Dice Coefficient')
    ax.set_title('Dice Coefficients by Model on Different Datasets')
    ax.set_xticks(x, dice_results.keys())
    ax.legend(title="Models", loc='upper center', bbox_to_anchor=(0.5, -0.15), fancybox=True, shadow=True, ncol=3)
    ax.set_ylim(0.7, 0.9)

    plt.xticks(rotation=0)  # Keep x-axis labels horizontal
    plt.tight_layout()  # Adjust layout

    # plt.show()
    plt.savefig("result_dice.png")


def iou_show():
    # 定义数据集和模型的Dice系数
    iou_results = {
        'DRIVE': (0.6586, 0.65346, 0.64708),
        'GLAS': (0.8055, 0.79824, 0.7941),
        'DSB': (0.74838, 0.75502, 0.79162),
    }

    models = ('U-Net', 'R-UNet', 'S-UNet')
    x = np.arange(len(iou_results.keys()))  # the label locations on x-axis for datasets
    width = 0.2  # the width of the bars
    multiplier = 0

    fig, ax = plt.subplots(figsize=(10, 6), layout='constrained')

    colors = ['tab:blue', 'tab:orange', 'tab:green']

    for model_idx, model in enumerate(models):
        offset = width * multiplier
        rects = ax.bar(x - width / 2 + offset, [val[model_idx] for val in iou_results.values()],
                       width, label=model, color=colors[model_idx])
        ax.bar_label(rects, padding=3, fmt='%.4f')
        multiplier += 1

    # Customizing the plot
    ax.set_ylabel('IoU')
    ax.set_title('IoU by Model on Different Datasets')
    ax.set_xticks(x, iou_results.keys())
    ax.legend(title="Models", loc='upper center', bbox_to_anchor=(0.5, -0.15), fancybox=True, shadow=True, ncol=3)
    ax.set_ylim(0.6, 0.85)

    plt.xticks(rotation=0)  # Keep x-axis labels horizontal
    plt.tight_layout()  # Adjust layout

    # plt.show()
    plt.savefig("result_iou.png")


if __name__ == '__main__':
    dice_show()
    iou_show()
