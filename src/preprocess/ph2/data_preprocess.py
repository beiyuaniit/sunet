





import os
import shutil

# 指定源目录和目标目录
src_dir = '../../../dataset/PH2Dataset'
train_img_dir = '../../../dataset/ph2/train/img'
train_gt_img_dir = '../../../dataset/ph2/train/gt_img'


def move_to_ph2():
    # 遍历源目录下的子文件夹
    for root, dirs, files in os.walk(src_dir):
        # 获取当前子文件夹的名字
        subfolder_name = os.path.basename(root)

        # 构建对应的 Dermoscopic_Image 和 lesion 目录路径
        dermoscopic_images_src = os.path.join(root, subfolder_name + '_Dermoscopic_Image')
        lesion_images_src = os.path.join(root, subfolder_name + '_lesion')

        if os.path.isdir(dermoscopic_images_src):
            for img_file in os.listdir(dermoscopic_images_src):
                if img_file.endswith('.bmp'):
                    src_file_path = os.path.join(dermoscopic_images_src, img_file)
                    dst_file_path = os.path.join(train_img_dir, img_file)
                    shutil.copy2(src_file_path, dst_file_path)

        if os.path.isdir(lesion_images_src):
            for img_file in os.listdir(lesion_images_src):
                if img_file.endswith('.bmp'):
                    src_file_path = os.path.join(lesion_images_src, img_file)
                    dst_file_path = os.path.join(train_gt_img_dir, img_file)
                    shutil.copy2(src_file_path, dst_file_path)


if __name__ == '__main__':
    # 如果目标目录不存在，则创建它们
    if not os.path.exists(train_img_dir):
        os.makedirs(train_img_dir)
    if not os.path.exists(train_gt_img_dir):
        os.makedirs(train_gt_img_dir)
    move_to_ph2()




