import os
import glob
import time
import torch
import numpy as np

from torchvision import transforms
from PIL import Image

from src.net.runt.r_unet import UNetWithRoberts
from src.net.unet import UNet
from src.net.sunet.s_unet import UNetWithScharr

dataset = "glas"
use_net = "sunet"
version = 'v5'


def get_img_suffix():
    if dataset == 'glas':
        return '*.bmp'
    if dataset == 'dsb':
        return '*.png'
    if dataset == 'drive':
        return '*.tif'
    if dataset == 'ph2':
        return '*.bmp'


def time_synchronized():
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()


def predict_batch(model_path, input_folder, output_folder, classes=1):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = ''
    if use_net == 'unet':
        model = UNet(in_channels=3, num_classes=classes + 1, base_c=32)
    if use_net == 'sunet':
        model = UNetWithScharr(in_channels=6, num_classes=classes + 1, base_c=32)
    if use_net == 'runet':
        model = UNetWithRoberts(in_channels=6, num_classes=classes + 1, base_c=32)
    model.load_state_dict(torch.load(model_path, map_location='cpu')['model'])
    model.to(device)
    model.eval()

    mean = (0.709, 0.381, 0.224)
    std = (0.127, 0.079, 0.043)
    data_transform = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize(mean=mean, std=std)])

    img_suffix = get_img_suffix()
    for img_filename in glob.glob(os.path.join(input_folder, img_suffix)):
        assert os.path.exists(img_filename), f"image {img_filename} not found."
        original_img = Image.open(img_filename).convert('RGB')
        img = data_transform(original_img)
        img = torch.unsqueeze(img, dim=0).to(device)

        with torch.no_grad():
            _, _, img_height, img_width = img.shape
            init_img = torch.zeros((1, 3, img_height, img_width), device=device)
            model(init_img)

            t_start = time_synchronized()
            output = model(img)  # 1 3 522 775 即可
            t_end = time_synchronized()
            print(f"Inference time for {os.path.basename(img_filename)}: {t_end - t_start:.4f}s")

            prediction = output['out'].argmax(1).squeeze(0).cpu().numpy().astype(np.uint8)
            prediction[prediction == 1] = 255

            mask = Image.fromarray(prediction)

            output_path = os.path.join(output_folder,
                                       os.path.splitext(os.path.basename(img_filename))[0] + '_predict.png')
            mask.save(output_path)


def main():
    # exclude background
    classes = 1

    model_path = "../../model/" + use_net + '/' + version + '/' + 'best_model_' + dataset + '.pth'
    input_folder = "../../dataset/" + dataset + "/test/img"
    output_folder = "../../dataset/" + dataset + "/test/predict_img/" + use_net + '/' + version

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    predict_batch(model_path, input_folder, output_folder, classes)


if __name__ == '__main__':
    main()
