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
import json
class WebdatasetFilterCounter():
    def __init__(self, min_size=512, max_pwatermark=0.5, aesthetic_threshold=5.0, unsafe_threshold=0.99, text_conditions=None): # {'min_words': 2, 'forbidden_words': ["www.", ".com", "http", "-", "_", ":", ";", "(", ")", "/", "%", "|", "?", "download", "interior", "kitchen", "chair", "getty", "how", "what", "when", "why", "laminate", "furniture", "hair", "dress", "clothing"]}):
        self.min_size = min_size
        self.max_pwatermark = max_pwatermark
        self.aesthetic_threshold = aesthetic_threshold
        self.unsafe_threshold = unsafe_threshold
        self.text_conditions = text_conditions

        self.f_watermark = 0
        self.f_aesthetic = 0
        self.f_unsafe = 0
        self.f_size = 0
        self.total = 0

    def update_counters(self, filter_watermark, filter_aesthetics_a, filter_aesthetics_b, filter_unsafe, filter_size):
        self.f_watermark += filter_watermark
        self.f_aesthetic += (filter_aesthetics_a or filter_aesthetics_b)
        self.f_unsafe += filter_unsafe
        self.f_size += filter_size
        self.total += 1
        print("updating", self.total)

        if self.total%10 == 0:
            self.checkpoint(savepath="../stats/filter_stats_{}.json")

    def checkpoint(self, savepath: str):
        with open(savepath.format(self.total), "w") as fp:
            json.dump({
                "f_size": self.f_size,
                'f_watermark': self.f_watermark,
                'f_aesthetic': self.f_aesthetic,
                'f_unsafe': self.f_unsafe,
                'total': self.total,
            }, fp)

    def __call__(self, x_json):

            filter_size = x_json.get('original_width', 0.0) >= self.min_size and x_json.get('original_height', 0) >= self.min_size
            filter_watermark = x_json.get('pwatermark', 1.0) <= self.max_pwatermark
            filter_aesthetic_a = x_json.get('aesthetic', 0.0) >= self.aesthetic_threshold
            filter_aesthetic_b = x_json.get('AESTHETIC_SCORE', 0.0) >= self.aesthetic_threshold
            filter_unsafe = x_json.get('punsafe', 1.0) <= self.unsafe_threshold
            self.update_counters(
                filter_watermark=not filter_watermark,
                filter_aesthetics_a=not filter_aesthetic_a,
                filter_aesthetics_b=not filter_aesthetic_b,
                filter_unsafe=not filter_unsafe,
                filter_size=not filter_size
            )
            print("watermark", x_json.get('pwatermark', 1.0) , self.max_pwatermark)
            print("filter_aesthetic_a", x_json.get('aesthetic', 0.0), self.aesthetic_threshold)
            print("filter_aesthetic_b", x_json.get('AESTHETIC_SCORE', 0.0), self.aesthetic_threshold)
            print("punsafe", x_json.get('punsafe', 1.0), self.unsafe_threshold)
            print()



def identity(x):
    return x

# --- PREPARE DATASET ---
# PREPARE DATASET
dataset = wds.WebDataset(
        dataset_path, resampled=True, handler=warn_and_continue
).shuffle(690, handler=warn_and_continue).decode(
        "pilrgb", handler=warn_and_continue
).to_tuple(
        "jpg", "json", handler=warn_and_continue
).map_tuple(
        transforms, identity, handler=warn_and_continue
)
real_batch_size = 256
dataloader = DataLoader(dataset, batch_size=real_batch_size, num_workers=8, pin_memory=True)
print("prepping dataloader")
#dataloader_iterator = dataloader
dataloader_iterator = iter(dataloader)

counter = WebdatasetFilterCounter(min_size=512, max_pwatermark=0.5, aesthetic_threshold=5.0, unsafe_threshold=0.99)
for i, x in enumerate(tqdm(dataset)):
    to_be_counted = {"json": x[1]}
    counter(to_be_counted)
    if i%5 == 0:
        print(i, to_be_counted)