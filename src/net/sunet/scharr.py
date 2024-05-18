import cv2
import numpy as np
import torch

def show_images(image):
    cv2.namedWindow('image', cv2.WINDOW_KEEPRATIO)
    cv2.imshow('image', image)
    cv2.waitKey()
    cv2.destroyAllWindows()


def Scharr_img(img):
    # 应用Scharr算子进行边缘检测
    grad_x = cv2.Scharr(img, cv2.CV_64F, 1, 0)  # 在x方向上计算梯度
    grad_y = cv2.Scharr(img, cv2.CV_64F, 0, 1)  # 在y方向上计算梯度

    abs_x = cv2.convertScaleAbs(grad_x)
    abs_y = cv2.convertScaleAbs(grad_y)

    grad_xy = cv2.addWeighted(abs_x, 0.5, abs_y, 0.5, 0)

    return grad_x, grad_y, grad_xy




def show_images_separately(image1, image2, image3, image4):
    cv2.namedWindow('Original Image', cv2.WINDOW_NORMAL)
    cv2.imshow('Original Image', image1)

    cv2.namedWindow('Gradient X', cv2.WINDOW_NORMAL)
    cv2.imshow('Gradient X', image2)

    cv2.namedWindow('Gradient Y', cv2.WINDOW_NORMAL)
    cv2.imshow('Gradient Y', image3)

    cv2.namedWindow('Combined Gradient', cv2.WINDOW_NORMAL)
    cv2.imshow('Combined Gradient', image4)

    cv2.waitKey()
    cv2.destroyAllWindows()


def main():
    img_name = 'dsb_train_2.png'
    dir = 'img/'
    # 因为这里是灰度图读入
    img = cv2.imread(dir+img_name, flags=0)
    re_img = Scharr_img(img)
    # 将四张图像连接成一个大图像
    # top_row = np.hstack((img, re_img[2]))
    # bottom_row = np.hstack((re_img[0], re_img[1]))
    # combined_img = np.vstack((top_row, bottom_row))
    # 好像因为维度问题，原图放在一起就显示不出来
    # show_images(combined_img)
    # show_images(top_row)
    # show_images(bottom_row)

    cv2.imwrite(dir+"scharr_"+img_name,re_img[2])

    # 4张图单独显式
    show_images_separately(img, re_img[0], re_img[1], re_img[2])


if __name__ == '__main__':
    main()
