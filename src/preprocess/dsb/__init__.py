import os
import pandas as pd
import numpy as np
from skimage.io import imread, imsave
from skimage.draw import polygon
import matplotlib.pyplot as plt
from collections import defaultdict


# RLE解码函数
def rle_decode(rle, shape=(1024, 1024)):
    s = rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0::2], s[1::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape).T


# 解析CSV并获取掩码集合
def process_csv_labels(csv_path):
    df_labels = pd.read_csv(csv_path)
    masks_dict = defaultdict(list)

    for index, row in df_labels.iterrows():
        image_id = row['ImageId']
        encoded_pixels = row['EncodedPixels']

        if isinstance(encoded_pixels, str) and encoded_pixels != '':
            mask = rle_decode(encoded_pixels)
            masks_dict[image_id].append(mask)

    return masks_dict


# 合并同一ImageId下的多个掩码
def combine_masks(image_masks):
    height, width = image_masks[0].shape
    combined_mask = np.zeros((height, width), dtype=np.uint8)
    for mask in image_masks:
        combined_mask[mask > 0] = 1
    return combined_mask


# 保存掩码到图像文件
def save_masks_to_images(masks_dict, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for image_id, masks in masks_dict.items():
        combined_mask = combine_masks(masks)
        mask_save_path = os.path.join(output_dir, f'{image_id}_mask.png')
        imsave(mask_save_path, combined_mask.astype(np.uint8), cmap='gray')
        print(f"Mask for Image ID: {image_id} has been saved to {mask_save_path}")


def main():
    # 设置CSV文件路径和结果保存目录
    csv_labels_path = 'stage1_train_labels.csv'
    output_dir = 'result'

    # 处理CSV标签并获取掩码集合
    masks_dict = process_csv_labels(csv_labels_path)

    # 保存每个图像的掩码到结果目录
    save_masks_to_images(masks_dict, output_dir)


if __name__ == "__main__":
    main()