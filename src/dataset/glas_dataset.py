import os
from PIL import Image
import numpy as np
from torch.utils.data import Dataset


class GlasDataset(Dataset):
    # root: DRIVE 文件夹所在的根目录
    # train: True 时载入 training 数据集中的数据，train 为 False 时载入 test 数据集中的数据
    # transforms: 定义了针对数据的预处理方式
    def __init__(self, root: str, train: bool, transforms=None):
        super(GlasDataset, self).__init__()
        # 拼接路径
        self.flag = "train" if train else "validation"
        data_root = os.path.join(root, self.flag)
        assert os.path.exists(data_root), f"path '{data_root}' does not exists."
        self.transforms = transforms
        # 获取图片名[列表]
        img_names = [i for i in os.listdir(os.path.join(data_root, "img")) if i.endswith(".bmp")]
        # 获取原图片列表
        self.img_list = [os.path.join(data_root, "img", i) for i in img_names]
        # 获取手动分割图片列表
        self.manual = [os.path.join(data_root, "gt_img", i.split(".")[0] + "_anno.bmp")
                       for i in img_names]
        # check files
        for i in self.manual:
            if os.path.exists(i) is False:
                raise FileNotFoundError(f"file {i} does not exists.")


    def __getitem__(self, idx):
        # 读入图像。RGB彩色，L灰度
        img = Image.open(self.img_list[idx]).convert('RGB')
        manual = Image.open(self.manual[idx]).convert('L')
        # 使前景像素值变为1，背景为0
        manual = np.array(manual) / 255
        # roi_mask = Image.open(self.roi_mask[idx]).convert('L')
        roi_mask = manual.copy()
        roi_mask [roi_mask!=255] = 255 # 全部区域设置为白色。就是都感兴趣
        roi_mask = 255 - np.array(roi_mask)
        #  mask 对应前景区域是 1 ，背景区域是 0 ，不感兴趣的区域是 255。计算损失时就可以将不感兴趣的像素都忽略掉
        mask = np.clip(manual + roi_mask, a_min=0, a_max=255)

        # 这里转回PIL的原因是，transforms中是对PIL数据进行处理
        mask = Image.fromarray(mask)

        # transforms进行相应的预处理
        if self.transforms is not None:
            img, mask = self.transforms(img, mask)

        return img, mask

    def __len__(self):
        return len(self.img_list)

    # 将一批样本数据组合成统一大小的批次（按照最大尺寸），以便送入神经网络进行训练或推理
    @staticmethod
    def collate_fn(batch):
        # 分离图像数据和标签数据
        images, targets = list(zip(*batch))
        # 对图像数据进行处理，不足部分填充 0、255
        batched_imgs = cat_list(images, fill_value=0)
        batched_targets = cat_list(targets, fill_value=255)
        return batched_imgs, batched_targets

    # 定义 cat_list 函数，用于将一组图像数据合并为具有相同形状的批次张量，不足部分填充指定值


def cat_list(images, fill_value=0):
    max_size = tuple(max(s) for s in zip(*[img.shape for img in images]))
    batch_shape = (len(images),) + max_size
    batched_imgs = images[0].new(*batch_shape).fill_(fill_value)
    for img, pad_img in zip(images, batched_imgs):
        pad_img[..., :img.shape[-2], :img.shape[-1]].copy_(img)
    return batched_imgs