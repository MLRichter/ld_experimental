import argparse
import warnings
from typing import Any, List, Tuple, Union

import PIL.Image
import torch
import torchvision
from torch import nn, optim
from torch.utils.data import DataLoader
from warmup_scheduler import GradualWarmupScheduler
from tqdm import tqdm
import numpy as np
import wandb
import os
import shutil
from transformers import AutoTokenizer, CLIPTextModel, CLIPVisionModel
import time

import webdataset as wds
from webdataset.handlers import warn_and_continue

from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP, DistributedDataParallel
from diffusers import AutoencoderKL, StableDiffusionKDiffusionPipeline

from torchtools.utils import Diffuzz, Diffuzz2
from torchtools.utils.diffusion2 import DDPMSampler

from inferencer import sd14, wuerstchen, ldm14
from ld_model import LDM
from pathlib import Path

from train import tokenizer_text_encoder_factory, prompts2embed
from attrs import define
import numpy as np


def denormalize_image(image, mean, std):
    """
    Denormalize an image by providing the mean and standard deviation.

    Parameters:
        image (torch.Tensor): The normalized image tensor in shape [C, H, W]
        mean (float): The mean used for normalization
        std (float): The standard deviation used for normalization

    Returns:
        torch.Tensor: The denormalized image
    """
    # Apply denormalization
    denormalized_image = (image * std) + mean
    print(torch.max(image), torch.min(image))

    # Clip pixel values to be in [0, 255]
    #denormalized_image = torch.clamp(denormalized_image, 0, 255).byte()

    return denormalized_image


def get_processed_images(dst_path: Path) -> List[str]:
    filenames = os.listdir(dst_path)
    return filenames


def coco_caption_loader_json(data_path: Path) -> Tuple[int, str]:
    import json
    with data_path.open("r") as fp:
        content = json.load(fp)["annotations"]
    for datapoint in tqdm(content):
        yield datapoint["id"], datapoint["caption"]


def coco_caption_loader_pyarrow(data_path: Path) -> Tuple[int, str]:
    import pandas as pd

    # Read the Parquet file into a pandas DataFrame
    df = pd.read_parquet(data_path, engine="pyarrow")
    content = df['caption'].values
    file_names = df['file_name'].values
    for i, (file_name, caption) in enumerate(zip(tqdm(file_names), content)):
        yield file_name, caption


coco_caption_loader = coco_caption_loader_pyarrow


def _poly_save(image: Union[torch.Tensor, PIL.Image.Image], savename: Path) -> None:
    if isinstance(image, torch.Tensor):
        torchvision.utils.save_image(image.cpu(), savename)
    else:
        image.save(savename)


def save_image(image: torch.Tensor, output_path: Path, i: Union[int, List[str]]):
    if isinstance(i, list):
        for idx, filename in enumerate(i):
            _poly_save(image[idx], f'{output_path}/{filename}')
    else:
        _poly_save(image, f'{output_path}/{i}')


def main(
        factory: callable,
        dataset_path: str,
        output_path: str,
        batch_size: int,
        device: str,
        start_index: int,
        num_datapoints: int):

    dataset_path = Path(dataset_path)
    output_path = Path(output_path)
    output_path.mkdir(exist_ok=True, parents=True)
    model = factory(device=device)

    existing_files = get_processed_images(output_path)

    prompts = []
    ids = []
    processed_datapoints = 0
    # this is a wiring fix in order to not deal with the dataset (yet)
    for i, (id, prompt) in enumerate(coco_caption_loader(data_path=dataset_path)):

        if i < start_index:  # Skip until we reach the starting index
            continue

        if processed_datapoints >= num_datapoints:  # Stop after processing num_datapoints
            break

        filename = f"{id}"
        if filename in existing_files:
            print("Found", filename, "skipping...")
            continue
        prompts.append(prompt)
        ids.append(id)

        if len(prompts) == batch_size:
            images = model(prompts, device, batch_size=batch_size)
            save_image(images, output_path=output_path, i=ids)
            processed_datapoints += len(images)
            prompts = []
            ids = []


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("start_index", type=int, help="The starting index of datapoints to process")
    parser.add_argument("num_datapoints", type=int, help="The number of datapoints to process")
    parser.add_argument("factory", type=str, help="The number of datapoints to process")
    args = parser.parse_args()
    weight_path: str = "./models/baseline/exp1.pt"
    if args.factory == "sd14":
        factory = sd14
    elif args.factory == "wuerstchen":
        factory = wuerstchen
    elif args.factory == "ldm14":
        factory = ldm14
    else:
        raise ArithmeticError(f"Unknown factory {args.factory}")

    factory_name = factory.__name__
    print(factory_name)
    dataset_path: str = "../coco2017/coco_30k.parquet"
    output_path: str = f"./output/{factory_name}_generated"
    device = "cuda:0"
    batch_size: int = 16
    main(factory=factory, dataset_path=dataset_path, output_path=output_path, device=device, batch_size=batch_size,
         start_index=args.start_index, num_datapoints=args.num_datapoints)





