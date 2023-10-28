import warnings

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

import webdataset as wds
from webdataset.handlers import warn_and_continue

from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP, DistributedDataParallel
from diffusers import AutoencoderKL, StableDiffusionKDiffusionPipeline

from torchtools.utils import Diffuzz, Diffuzz2
from torchtools.utils.diffusion2 import DDPMSampler

from ld_model import LDM
from utils import WebdatasetFilter

# PARAMETERS
updates = 300000  # 500000
warmup_updates = 10000
batch_size = 1536  # 1024 # 2048 # 4096
grad_accum_steps = 12
max_iters = updates * grad_accum_steps
print_every = 10 * grad_accum_steps
lr = 1e-4

dataset_path = "pipe:aws s3 cp s3://stability-west/laion-a-native-high-res/{part-0/{00000..18699}.tar,part-1/{00000..18699}.tar,part-2/{00000..18699}.tar,part-3/{00000..18699}.tar,part-4/{00000..18699}.tar} -"  # "pipe:aws s3 cp s3://laion-west/humans-7M-with-blip-caps+aesthetics+nsfw/00000{1..5499}.tar -"
dataset_path = "pipe:aws s3 cp s3://stability-west/laion-a-native-high-res/{part-0/{00000..18000}.tar,part-1/{00000..13500}.tar,part-2/{00000..13500}.tar,part-3/{00000..13500}.tar,part-4/{00000..14100}.tar} -"  # "pipe:aws s3 cp s3://laion-west/humans-7M-with-blip-caps+aesthetics+nsfw/00000{1..5499}.tar -"
clip_image_model_name = 'laion/CLIP-ViT-H-14-laion2B-s32B-b79K'
output_path = "../output/baseline/exp1/"
checkpoint_path = "../models/baseline/exp1.pt"
target = "e"

wandv_project = "LDBaseline"
wandv_entity = "mlrichter"
wandb_run_name = "LDTrain"
wandb_config = {
    "model_type": 'Latent Diffusion 0.87M',
    "target": f'{target}-target',
    "image_size": "512x512",
    "batch_size": batch_size,
    "warmup_updates": warmup_updates,
    "lr": lr,
    "description": "Latent Diffusion Model Similar to SD-1.4"
}

magic_norm = 0.18215
transforms = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Resize(512, interpolation=torchvision.transforms.InterpolationMode.BILINEAR, antialias=True),
    torchvision.transforms.CenterCrop(512),
    torchvision.transforms.Normalize([0.5], [0.5]),

])

def identity(x):
    return x

# --- PREPARE DATASET ---
# PREPARE DATASET
dataset = wds.WebDataset(
        dataset_path, resampled=True, handler=warn_and_continue
).select(
        WebdatasetFilter(min_size=512, max_pwatermark=0.5, aesthetic_threshold=5.0, unsafe_threshold=0.99)
).shuffle(690, handler=warn_and_continue).decode(
        "pilrgb", handler=warn_and_continue
).to_tuple(
        "jpg", "txt", handler=warn_and_continue
).map_tuple(
        transforms, identity, handler=warn_and_continue
)
real_batch_size = 256
dataloader = DataLoader(dataset, batch_size=real_batch_size, num_workers=8, pin_memory=True)
print("prepping dataloader")
#dataloader_iterator = dataloader
dataloader_iterator = iter(dataloader)

for i, x in enumerate(tqdm(dataset)):
    if i%5 == 0:
        print(i, x)