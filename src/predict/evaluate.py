import os
import numpy as np
from PIL import Image

dataset = "glas"
use_net = "sunet"
version = 'v5'


def get_gt_img_suffix():
    if dataset == 'glas':
        return '_anno.bmp'
    if dataset == 'dsb':
        return '.png'
    if dataset == 'drive':
        return '.gif'
    if dataset == 'ph2':
        return '_lesion.bmp'


def calculate_metrics(predict_img_path, gt_img_path, smooth=1e-6):
    predict_mask = np.array(Image.open(predict_img_path).convert("L"), dtype=np.uint8)
    gt_mask = np.array(Image.open(gt_img_path).convert("L"), dtype=np.uint8)

    predict_mask[predict_mask > 0] = 1
    gt_mask[gt_mask > 0] = 1

    assert predict_mask.shape == gt_mask.shape, "Shape mismatch between predicted and ground truth masks"

    intersection = np.logical_and(predict_mask, gt_mask)
    union = np.logical_or(predict_mask, gt_mask)

    iou = np.sum(intersection) / np.sum(union)
    dice = (2 * np.sum(intersection) + smooth) / (np.sum(predict_mask) + np.sum(gt_mask) + smooth)

    return dice, iou


def total_eval(gt_img, predict_img, evaluate_result_root):
    dice_scores = []
    iou_scores = []
    result_txt = os.path.join(evaluate_result_root, dataset + "-result.txt")
    img_count = 0

    gt_img_suffix = get_gt_img_suffix()
    predict_img_suffix = "_predict.png"
    with open(result_txt, 'w') as f:
        for filename in os.listdir(gt_img):
            gt_img_path = os.path.join(gt_img, filename)
            predict_img_path = os.path.join(predict_img,
                                            filename.replace(gt_img_suffix, predict_img_suffix))

            dice, iou = calculate_metrics(predict_img_path, gt_img_path)

            dice_scores.append(dice)
            iou_scores.append(iou)

            f.write("[image: {}]\nDice coefficient: {:.4f}\nIoU: {:.2f}\n\n".format(img_count + 1, dice, iou))
            img_count += 1

        avg_dice = sum(dice_scores) / len(dice_scores)
        avg_iou = sum(iou_scores) / len(iou_scores)

        f.write("Average Dice Coefficient: {:.4f}\nAverage IoU (Jaccard Index): {:.4f}\n".format(avg_dice, avg_iou))

        print(f"Average Dice Coefficient: {avg_dice}")
        print(f"Average IoU (Jaccard Index): {avg_iou}")


def main():
    # os.mkdir(evaluate_result_root) 不会递归创建，好像只能创建一层
    evaluate_result_root = "../../result/evaluate/" + use_net + "/" + version+'/'
    if not os.path.exists(evaluate_result_root):
        os.makedirs(evaluate_result_root)

    gt_img = "../../dataset/" + dataset + "/test/gt_img"
    predict_img = "../../dataset/" + dataset + "/test/predict_img/" + use_net + '/' + version
    total_eval(gt_img, predict_img, evaluate_result_root)


if __name__ == '__main__':
    # todo 这里的IoU为什么是单个？是不是正确的？
    main()
