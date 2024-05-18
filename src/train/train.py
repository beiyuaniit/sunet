import os
import time
import datetime
import torch
import argparse

from src.dataset.drive_dataset import DriveDataset
from src.net.runt.r_unet import UNetWithRoberts
from src.net.unet import UNet
from src.net.sunet.s_unet import UNetWithScharr
from src.util import transforms as T
from src.util.train_and_eval import train_one_epoch, create_lr_scheduler, evaluate

from src.dataset.glas_dataset import GlasDataset
from src.dataset.dsb_dataset import DsbDataset

dataset = "glas"
use_net = 'sunet'
epochs = 100
version = 'v5'


class SegmentationPresetTrain:
    def __init__(self, base_size, crop_size, hflip_prob=0.5, vflip_prob=0.5,
                 mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        min_size = int(0.5 * base_size)
        max_size = int(1.2 * base_size)

        trans = [T.RandomResize(min_size, max_size)]
        if hflip_prob > 0:
            trans.append(T.RandomHorizontalFlip(hflip_prob))
        if vflip_prob > 0:
            trans.append(T.RandomVerticalFlip(vflip_prob))
        trans.extend([
            T.RandomCrop(crop_size),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])
        self.transforms = T.Compose(trans)

    def __call__(self, img, target):
        return self.transforms(img, target)


class SegmentationPresetEval:
    def __init__(self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.transforms = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])

    def __call__(self, img, target):
        return self.transforms(img, target)


def get_transform(train, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    base_size = 565
    crop_size = 480

    if train:
        return SegmentationPresetTrain(base_size, crop_size, mean=mean, std=std)
    else:
        return SegmentationPresetEval(mean=mean, std=std)


def create_unet_model(num_classes):
    model = UNet(in_channels=3, num_classes=num_classes, base_c=32)
    return model


def create_sunet_model(num_classes):
    # 这里多了一个三通道的边缘图
    model = UNetWithScharr(in_channels=6, num_classes=num_classes, base_c=32)
    return model


def create_runet_model(num_classes):
    # 也是6通道
    model = UNetWithRoberts(in_channels=6, num_classes=num_classes, base_c=32);
    return model


def get_dataset(mean, std):
    if dataset == 'glas':
        train_dataset = GlasDataset(args.data_path,
                                    train=True,
                                    transforms=get_transform(train=True, mean=mean, std=std))

        # 这是验证集啊
        val_dataset = GlasDataset(args.data_path,
                                  train=False,
                                  transforms=get_transform(train=False, mean=mean, std=std))
        return train_dataset, val_dataset
    if dataset == 'dsb':
        train_dataset = DsbDataset(args.data_path,
                                   train=True,
                                   transforms=get_transform(train=True, mean=mean, std=std))

        # 这是验证集啊
        val_dataset = DsbDataset(args.data_path,
                                 train=False,
                                 transforms=get_transform(train=False, mean=mean, std=std))

        return train_dataset, val_dataset
    if dataset == 'drive':
        train_dataset = DriveDataset(args.data_path,
                                   train=True,
                                   transforms=get_transform(train=True, mean=mean, std=std))

        # 这是验证集啊
        val_dataset = DriveDataset(args.data_path,
                                 train=False,
                                 transforms=get_transform(train=False, mean=mean, std=std))

        return train_dataset, val_dataset


def train(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    batch_size = args.batch_size
    # segmentation nun_classes + background
    num_classes = args.num_classes + 1

    # using compute_mean_std.py
    mean = (0.709, 0.381, 0.224)
    std = (0.127, 0.079, 0.043)

    train_dataset, val_dataset = get_dataset(mean, std)
    num_workers = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               num_workers=num_workers,
                                               shuffle=True,
                                               pin_memory=True,
                                               collate_fn=train_dataset.collate_fn)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=1,
                                             num_workers=num_workers,
                                             pin_memory=True,
                                             collate_fn=val_dataset.collate_fn)

    model = ''
    if use_net == 'unet':
        model = create_unet_model(num_classes=num_classes)
    if use_net == 'sunet':
        model = create_sunet_model(num_classes=num_classes)
    if use_net == 'runet':
        model = create_runet_model(num_classes=num_classes)
    model.to(device)

    params_to_optimize = [p for p in model.parameters() if p.requires_grad]

    optimizer = torch.optim.SGD(
        params_to_optimize,
        lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay
    )

    scaler = torch.cuda.amp.GradScaler() if args.amp else None

    # 创建学习率更新策略，这里是每个step更新一次(不是每个epoch)
    lr_scheduler = create_lr_scheduler(optimizer, len(train_loader), args.epochs, warmup=True)

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        args.start_epoch = checkpoint['epoch'] + 1
        if args.amp:
            scaler.load_state_dict(checkpoint["scaler"])

    best_dice = 0.
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        mean_loss, lr = train_one_epoch(model, optimizer, train_loader, device, epoch, num_classes,
                                        lr_scheduler=lr_scheduler, print_freq=args.print_freq, scaler=scaler)

        confmat, dice = evaluate(model, val_loader, device=device, num_classes=num_classes)
        val_info = str(confmat)
        print(val_info)
        print(f"dice coefficient: {dice:.3f}")
        # write into txt
        with open(results_file, "a") as f:
            # 记录每个epoch对应的train_loss、lr以及验证集各指标
            train_info = f"[epoch: {epoch}]\n" \
                         f"train_loss: {mean_loss:.4f}\n" \
                         f"lr: {lr:.6f}\n" \
                         f"dice coefficient: {dice:.3f}\n"
            f.write(train_info + val_info + "\n\n")

        if args.save_best is True:
            if best_dice < dice:
                best_dice = dice
            else:
                continue

        save_file = {"model": model.state_dict(),
                     "optimizer": optimizer.state_dict(),
                     "lr_scheduler": lr_scheduler.state_dict(),
                     "epoch": epoch,
                     "args": args}
        if args.amp:
            save_file["scaler"] = scaler.state_dict()

        # 模型保存的位置
        if args.save_best is True:
            torch.save(save_file, best_model_file)
        else:
            torch.save(save_file, model_dir + "model_{}.pth".format(epoch))

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("training time {}".format(total_time_str))


def parse_args():
    parser = argparse.ArgumentParser(description="pytorch unet training")

    parser.add_argument("--data-path", default="../../dataset/" + dataset, help="dataset root")
    # exclude background
    parser.add_argument("--num-classes", default=1, type=int)
    parser.add_argument("--device", default="cuda", help="training device")
    parser.add_argument("-b", "--batch-size", default=4, type=int)
    parser.add_argument("--epochs", default=epochs, type=int, metavar="N",
                        help="number of total epochs to train")

    parser.add_argument('--lr', default=0.01, type=float, help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    parser.add_argument('--print-freq', default=1, type=int, help='print frequency')
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--save-best', default=True, type=bool, help='only save best dice weights')
    # Mixed precision training parameters
    parser.add_argument("--amp", default=False, type=bool,
                        help="Use torch.cuda.amp for mixed precision training")

    arg_map = parser.parse_args()

    return arg_map


if __name__ == '__main__':
    args = parse_args()

    # 局部全局变量
    # 用来保存训练以及验证过程中信息
    result_dir = "../../result/train/" + use_net + '/' + version + '/'
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    results_file = result_dir + "result_" + dataset + ".txt"

    # 模型
    model_dir = "../../model/" + use_net + '/' + version + '/'
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    best_model_file = model_dir + "best_model_" + dataset + ".pth"

    train(args)
