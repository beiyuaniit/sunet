import cv2
import numpy as np


def roberts_edge_detection(image_path):
    # 2. 定义Roberts算子核
    roberts_x = np.array([[1, 0], [0, -1]], dtype=np.float32)
    roberts_y = np.array([[0, 1], [-1, 0]], dtype=np.float32)

    # 3. 应用Roberts算子
    edges_x = cv2.filter2D(img, -1, roberts_x)
    edges_y = cv2.filter2D(img, -1, roberts_y)

    # 4. 合并结果
    # 取绝对值之和作为边缘强度
    edges = cv2.absdiff(edges_x, edges_y)

    return edges_x, edges_y, edges


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


if __name__ == '__main__':
    img_name = 'testA_1.bmp'
    dir = 'img/'
    img = cv2.imread(dir+img_name, flags=0)

    re_img = roberts_edge_detection(img)

    # cv2.imwrite(dir + "roberts__" + img_name, re_img[2])
    # 4张图单独显式
    show_images_separately(img, re_img[0], re_img[1], re_img[2])
