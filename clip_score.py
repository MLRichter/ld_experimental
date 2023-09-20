from typing import Tuple

from torchmetrics.multimodal import CLIPScore
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

def coco_caption_loader_pyarrow(data_path: str) -> Tuple[int, str]:
    import pandas as pd

    # Read the Parquet file into a pandas DataFrame
    df = pd.read_parquet(data_path, engine="pyarrow")
    content = df['caption'].values
    file_names = df['file_name'].values
    for i, (file_name, caption) in enumerate(zip(file_names, content)):
        yield file_name, caption


def create_valid_ath(root, file):
    if os.path.exists(os.path.join(root, file)) :
        return os.path.join(root, file)
    elif os.path.exists(os.path.join(root, pathlib.Path(file).with_suffix(".png"))):
        return os.path.join(root , pathlib.Path(file).with_suffix(".png"))
    else:
        print("failed to find", file, "in", root)
        return None


class MyDataset(Dataset):
    def __init__(self, image_paths, root, transform=None):
        paths = list(coco_caption_loader_pyarrow(image_paths))
        file_names = [f[0] for f in paths]
        self.file_names = [create_valid_ath(root, f) for f in file_names if create_valid_ath(root, f) is not None]
        self.captions = [f[1] for f in paths]
        self.transform = transform

    def get_class_label(self, image_name):
        # your method here
        return 0

    def __getitem__(self, index):
        x = Image.open(self.file_names[index])
        y = self.captions[index]
        if self.transform is not None:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.file_names)


def compute_clip_score(path =  '../coco2017/val2014/', parquet: str = "../coco2017/long_context_val.parquet", batch_size: int = 256, gpu: bool = True, dims=2048):
    """Calculates the IC of two paths"""
    dataset = MyDataset(image_paths=parquet, root=path, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((512, 512)),
            #transforms.Resize((256, 256)),
            transforms.Lambda(to_rgb)
        ]))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=5, pin_memory=True)
    clip = CLIPScore()
    clip.cuda()
    for batch, captions in tqdm(loader):
        batch = batch.cuda()
        clip.update(batch, text=captions)
    return clip.compute()



def main():
    paths = [

        "output/wuerstchen_generated",
        "output/GALIP_generated",
        '../coco2017/val2014/',
        "output/df_gan_generated",
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
    #parquet_file = "../coco2017/long_context_val.parquet"
    parquet_file = "../coco2017/coco_30k.parquet"
    result = {}
    for path in paths:
        parquet_file = "../coco2017/coco_30k.parquet" if "long_context" not in path else "../coco2017/long_context_val.parquet"
        name = pathlib.Path(path).name
        c_score = compute_clip_score(path, parquet=parquet_file)
        pprint(c_score)
        result[f"{name}"] = [c_score.detach().cpu().item()]
        pd.DataFrame.from_dict(result).to_csv("./output/clip_scores.csv", sep=";")


if __name__ == '__main__':
    main()