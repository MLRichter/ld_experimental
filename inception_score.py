from torchmetrics.image.inception import InceptionScore
import os
import pathlib
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from pprint import pprint

import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision.models.inception import inception_v3


import torch
import numpy as np
from scipy import linalg
from torch.autograd import Variable
from torch.nn.functional import adaptive_avg_pool2d
import torchvision.transforms as transforms
from tqdm import tqdm

import torch.utils.data
from PIL import Image
from torch.utils import data

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

def to_rgb(x):
    return x.repeat(3, 1, 1) if x.size(0) == 1 else x

parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
#parser.add_argument('path', type=str, nargs=2,
#                    help=('Path to the generated images or '
#                          'to .npz statistic files'))
parser.add_argument('--batch-size', type=int, default=64,
                    help='Batch size to use')
parser.add_argument('-c', '--gpu', default='', type=str,
                    help='GPU to use (leave blank for CPU only)')
parser.add_argument('--path', type=str, default=64)


class MyDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = [os.path.join(image_paths, file)
                            if os.path.exists(os.path.join(image_paths, file)) else os.path.join(image_paths, pathlib.Path(file).with_suffix(".png"))
                            for file in os.listdir(image_paths) ]
        self.transform = transform

    def get_class_label(self, image_name):
        # your method here
        return 0

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        x = Image.open(image_path)
        y = self.get_class_label(image_path.split('/')[-1])
        if self.transform is not None:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.image_paths)


def compute_inception_score(path =  '../coco2017/val2014/', batch_size: int = 256, gpu: bool = True, dims=2048):
    """Calculates the IC of two paths"""
    dataset = MyDataset(image_paths=path, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((512, 512)),
            #transforms.Resize((256, 256)),
            transforms.Lambda(to_rgb)
        ]))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=5, pin_memory=True)
    inception = InceptionScore(normalize=True)
    inception.inception.cuda()
    for batch, _ in tqdm(loader):
        batch = batch.cuda()
        inception.update(batch)
    return inception.compute()



def main():
    paths = [
        #'../coco2017/val2014/',
        "output/df_gan_generated",
        "output/GALIP_generated",
        "output/wuerstchen_generated",
        "output/v3_1B_coco_30k",
        "output/ldm14_generated",
        "output/sd14_generated",
        "output/sd21_generated",
        "output/sdxl_generated",
        "output/df_gan_long_context_generated",
        "output/GALIP_long_context_generated",
        "output/wuerstchen_long_context_generated",
        "output/ldm14_long_context_generated",
        "output/sd14_long_context_generated",
        "output/sd21_long_context_generated",
        "output/sdxl_long_context_generated",
    ]
    result = {}
    for path in paths:
        name = pathlib.Path(path).name
        ic_score = compute_inception_score(path)
        pprint(ic_score)
        result[f"{name} mean"] = [ic_score[0].detach().cpu().item()]
        result[f"{name} std"] = [ic_score[1].detach().cpu().item()]
        pd.DataFrame.from_dict(result).to_csv("./output/fid_long_context_scores.csv", sep=";")


if __name__ == '__main__':
    main()