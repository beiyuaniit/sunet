from PIL import Image
import numpy as np
import os

'''
分割前景用白色表示
背景用黑色
'''


def chang_style(data_root):
    original_path = data_root + "/gt_img_original"
    img_names = [i for i in os.listdir(os.path.join(original_path)) if i.endswith(".bmp")]

    save_path = data_root + "/gt_img"
    for name in img_names:
        img_path = original_path + "/" + name
        img = Image.open(img_path)
        img_arr = np.array(img)

        img_arr[img_arr != 0] = 255
        img_arr[img_arr != 255] = 0

        new_img = Image.fromarray(img_arr)
        new_img.save(save_path + "/" + name)


if __name__ == '__main__':
    if not os.path.exists("../../../dataset/glas/validation/gt_img"):
        os.mkdir("../../../dataset/glas/validation/gt_img")

    # train_root = "../glas/train"
    # chang_style(train_root)

    validation_root = "../../../dataset/glas/validation"
    chang_style(validation_root)
