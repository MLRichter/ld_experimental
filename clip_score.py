from typing import Tuple

from torchmetrics.multimodal import CLIPScore
import os
import pathlib
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from pprint import pprint

import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision.models.inception import inception_v3

import clip
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
    clip = CLIPScore(model_name_or_path="laion/CLIP-ViT-g-14-laion2B-s12B-b42K")
    clip.cuda()
    pbar = tqdm(loader)
    pbar.set_description(f"CLIP-Score NAN")
    for i, (batch, captions) in enumerate(pbar):
        batch = batch.cuda()
        clip.update(batch, text=captions)
        if i%5 == 0:
            pbar.set_description(f"CLIP-Score {clip.compute().detach().cpu().item()}")
    return clip.compute()

@torch.no_grad()
def compute_clip_score2(path =  '../coco2017/val2014/', parquet: str = "../coco2017/coco_30k.parquet", batch_size: int = 256, gpu: bool = True, dims=2048):
    """Calculates the IC of two paths"""
    dataset = MyDataset(image_paths=parquet, root=path, transform=None)
    #loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=5, pin_memory=True)
    clip_model, preprocess = clip.load("ViT-L/14", device="cuda:0")
    similarities = 0.0
    pbar = tqdm(range(len(dataset)))
    for i in pbar:
        img, caption = dataset[i]
        try:
            image = preprocess(img).unsqueeze(0)
            image_embedding = clip_model.encode_image(image.to("cuda:0"))
            text_embedding = clip_model.encode_text(clip.tokenize(caption).to("cuda:0"))
        except:
            continue
        similarity = (image_embedding / image_embedding.norm(dim=1, keepdim=True) @ (text_embedding / text_embedding.norm(dim=1, keepdim=True)).T).cpu().item()
        similarities += similarity
        pbar.set_postfix({"similarity": similarities / (i+1)})
    return similarities / len(dataset)


def main():
    paths = [
        #"output/wuerstchen_partiprompts_generated",
        #"output/df_gan_partiprompts_generated",
        #"output/GALIP_partiprompts_generated",
        #"output/ldm14_partiprompts_generated",
        #"output/sd14_partiprompts_generated",
        #"output/sd21_partiprompts_generated",
        #"output/sdxl_partiprompts_generated",

        #"./output/wuerstchen_spl5_generated",
        #"./output/wuerstchen_spl10_generated",
        #"./output/wuerstchen_spl20_generated",
        #"./output/wuerstchen_spl40_generated",
        #"./output/wuerstchen_spl80_generated",
        #"./output/wuerstchen_spl160_generated",

        #"output/sd21_1.0_generated",
        #"output/sd21_3.0_generated",
        #"output/sd21_5.0_generated",
        #"output/sd21_7.0_generated",
        #"output/sd21_9.0_generated",

        #"output/wuerstchen_0.5_generated",
        #"output/wuerstchen_1.0_generated",
        #"output/wuerstchen_3.0_generated",
        #"output/wuerstchen_5.0_generated",
        #"output/wuerstchen_7.0_generated",
        #"output/wuerstchen_9.0_generated",

        "output/wuerstchen_generated",
        "output/wuerstchen_no_text_generated",
        "output/wuerstchen_no_prior_text_generated",

        #"output/wuerstchen_generated",
        #"output/df_gan_generated",
        #"output/GALIP_generated",
        #"output/ldm14_generated",
        #"output/sd14_generated",
        #"output/sd21_generated",
        #"output/sdxl_generated",

        #"output/df_gan_long_context_generated",
        #"output/GALIP_long_context_generated",
        #"output/wuerstchen_long_context_generated",
        #"output/ldm14_long_context_generated",
        #"output/sd14_long_context_generated",
        #"output/sd21_long_context_generated",
        #"output/sdxl_long_context_generated",
    ]
    #parquet_file = "../coco2017/long_context_val.parquet"
    parquet_file = "../coco2017/coco_30k.parquet"
    result = {'model': [], "clip-score": []}
    for path in paths:
        parquet_file = "../coco2017/coco_30k.parquet" if "long_context" not in path else "../coco2017/long_context_val.parquet"
        parquet_file = parquet_file if "partiprompts" not in path else "./results/partiprompts.parquet"
        print(parquet_file)
        name = pathlib.Path(path).name
        c_score = compute_clip_score2(path, parquet=parquet_file)
        pprint(c_score)
        result['model'].append(name)
        result["clip-score"].append(c_score)
        pd.DataFrame.from_dict(result).to_csv("./output/clip_appendix_j.csv", sep=";")


if __name__ == '__main__':
    main()