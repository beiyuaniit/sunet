import os
import cv2
import sys
import warnings
import numpy as np
import skimage
from skimage.io import imread
from skimage.transform import resize
from keras.utils import Progbar
from preparation import get_contour, split_overlay_mask_by_contour

data_root = '../../../dataset/data-science-bowl-2018'
TRAIN_PATH = data_root + '/stage1_train/'
TEST_PATH = data_root + '/stage2_test/'
OUTPUT_IMAGES_DIR = '../../../dataset/dsb/train/img'
OUTPUT_MASKS_DIR = '../../../dataset/dsb/train/gt_img'

'''
将data-science-bowl-2018中的每张图片的多个分割图合并为一并
和原图一起保存到Dsb目录中
'''


# Function read train images and mask return as nump array
def get_train_data( IMG_WIDTH = 256, IMG_HEIGHT=256, IMG_CHANNELS=3):
    train_ids = next(os.walk(TRAIN_PATH))[1]
    X_train = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
    Y_train = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool_)
    print('Getting and resizing train images and masks ... ')
    sys.stdout.flush()
    a = Progbar(len(train_ids))
    for n, id_ in enumerate(train_ids):
        path = TRAIN_PATH + id_
        img = imread(path + '/images/' + id_ + '.png')
        if len(img.shape) == 2:
            img = skimage.color.gray2rgb(img)
        img = img[:, :, :IMG_CHANNELS]
        img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
        X_train[n] = img
        masks, masks_counters = [], []
        for mask_file in next(os.walk(path + '/masks/'))[2]:
            mask_ = imread(path + '/masks/' + mask_file)
            masks.append(mask_)
            mask_contour = get_contour(mask_)
            masks_counters.append(mask_contour)
        masks = np.sum(np.array(masks), axis=0)
        masks_counters = np.sum(np.array(masks_counters), axis=0)
        split_masks = split_overlay_mask_by_contour(masks, masks_counters)
        Y_train[n] = np.expand_dims(resize(split_masks, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True),
                                    axis=-1)
        a.update(n)

        # 保存图像和掩模
        img_output_path = os.path.join(OUTPUT_IMAGES_DIR, f"{id_}.png")
        cv2.imwrite(img_output_path, X_train[n])
        mask_output_path = os.path.join(OUTPUT_MASKS_DIR, f"{id_}.png")
        cv2.imwrite(mask_output_path, (Y_train[n] * 255).astype(np.uint8))


# Function to read test images and return as numpy array
def get_test_data(IMG_WIDTH=256, IMG_HEIGHT=256, IMG_CHANNELS=3):
    test_ids = next(os.walk(TEST_PATH))[1]
    X_test = np.zeros((len(test_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
    sizes_test = []
    print('\nGetting and resizing test images ... ')
    sys.stdout.flush()

    b = Progbar(len(test_ids))
    for n, id_ in enumerate(test_ids):
        path = TEST_PATH + id_
        img = imread(path + '/images/' + id_ + '.png')
        if len(img.shape) == 2:
            img = skimage.color.gray2rgb(img)
        img = img[:, :, :IMG_CHANNELS]
        sizes_test.append([img.shape[0], img.shape[1]])
        img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
        X_test[n] = img
        b.update(n)


def main():
    warnings.filterwarnings('ignore', category=UserWarning, module='skimage')
    if not os.path.exists(OUTPUT_IMAGES_DIR):
        os.makedirs(OUTPUT_IMAGES_DIR)
    if not os.path.exists(OUTPUT_MASKS_DIR):
        os.makedirs(OUTPUT_MASKS_DIR)


if __name__ == '__main__':
    main()
    get_train_data()
    # get_test_data()
