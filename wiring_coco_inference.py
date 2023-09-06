import warnings
from typing import Any, List, Tuple

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


@define
class DiffusionModelInferencer:
    generator: LDM
    clip_tokenizer: Any
    clip_model: torch.nn.Module
    vae: torch.nn.Module
    diffuzz: Diffuzz2

    def __call__(self, captions: List[str], device_lang: str = "cpu"):
        with torch.no_grad():
            # clip stuff
            embeds = prompts2embed(captions, self.clip_tokenizer, self.clip_model, device_lang)
            embeds_uncond = prompts2embed([""] * len(captions), self.clip_tokenizer, self.clip_model, device_lang)
            # ---
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                *_, (sampled, _, _) = self.diffuzz.sample(self.generator, {
                    'clip': embeds,
                }, (1, 4, 64, 64), unconditional_inputs={
                    'clip': embeds_uncond,
                }, cfg=7, sample_mode="e")

            magic_norm = 0.18215
            sampled_images = self.vae.decode(sampled / magic_norm).sample.clamp(0, 1)
            sampled_images = denormalize_image(image=sampled_images, mean=0.5, std=0.5)
        return sampled_images


def coco_caption_loader(data_path: Path) -> Tuple[int, str]:
    import json
    with data_path.open("r") as fp:
        content = json.load(fp)["annotations"]
    for datapoint in tqdm(content):
        yield datapoint["id"], datapoint["caption"]


def save_image(image: torch.Tensor, output_path: Path, i: int):
    torchvision.utils.save_image(image.cpu(), f'{output_path}/img_{i}.jpg')


def load_model(weight_path: Path, device: str = "cpu") -> torch.nn.Module:
    generator = LDM(c_hidden=[320, 640, 1280, 1280], nhead=[5, 10, 20, 20], blocks=[[2, 4, 14, 4], [5, 15, 5, 3]],
                    level_config=['CTA', 'CTA', 'CTA', 'CTA']).cuda()
    generator.eval()
    clip_tokenizer, clip_model = tokenizer_text_encoder_factory(device=device)
    vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae").to(device)
    vae.eval().requires_grad_(False)

    checkpoint = torch.load(weight_path, map_location=device)
    generator.load_state_dict(checkpoint['state_dict'])
    diffuzz = Diffuzz2(device=device, scaler=1, clamp_range=(0.0001, 0.9999))
    return DiffusionModelInferencer(generator=generator,
                                    clip_tokenizer=clip_tokenizer,
                                    clip_model=clip_model,
                                    vae=vae,
                                    diffuzz=diffuzz
                                    )


def main(
        weight_path: str,
        dataset_path: str,
        output_path: str,
        device: str):
    weight_path = Path(weight_path)
    dataset_path = Path(dataset_path)
    output_path = Path(output_path)
    model = load_model(weight_path, device=device)

    # this is a wiring fix in order to not deal with the dataset (yet)
    for id, prompt in coco_caption_loader(data_path=dataset_path):
        images = model([prompt], device)

        save_image(images, output_path=output_path, i=id)


if __name__ == '__main__':
    weight_path: str = "./models/baseline/exp1.pt"
    dataset_path: str = "./data/captions_train2014.json"
    output_path: str = "./output/generated"
    device = "cuda"
    main(weight_path=weight_path, dataset_path=dataset_path, output_path=output_path, device=device)





