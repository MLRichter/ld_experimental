import json
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

    def update_counters(self, filter_watermark, filter_aesthetics_a, filter_aesthetics_b, filter_unsafe, filter_text):
        self.f_watermark += filter_watermark
        self.f_aesthetic += (filter_aesthetics_a or filter_aesthetics_b)
        self.f_unsafe += filter_unsafe
        self.f_size += filter_text
        self.total += 1

    def checkpoint(self, savepath: str):
        with open(savepath.format(self.total), "w") as fp:
            json.dump({
                'f_watermark': self.f_watermark,
                'f_aesthetic': self.f_aesthetic,
                'f_unsafe': self.f_unsafe,
                'f_size': self.f_size,
                'total': self.total,
            }, fp)

    def __call__(self, x):
        try:
            if 'json' in x:
                x_json = json.loads(x['json'])
                filter_size = (x_json.get('original_width', 0.0) or 0.0) >= self.min_size and x_json.get('original_height', 0) >= self.min_size
                filter_watermark = (x_json.get('pwatermark', 1.0) or 1.0) <= self.max_pwatermark
                filter_aesthetic_a = (x_json.get('aesthetic', 0.0) or 0.0) >= self.aesthetic_threshold
                filter_aesthetic_b = (x_json.get('AESTHETIC_SCORE', 0.0) or 0.0) >= self.aesthetic_threshold
                filter_unsafe = (x_json.get('punsafe', 1.0) or 1.0) <= self.unsafe_threshold
                if self.text_conditions is not None:
                    caption = x['txt'].decode("utf-8")
                    filter_min_words = len(caption.split(" ")) >= self.text_conditions['min_words']
                    filter_ord_128 = all([ord(c) < 128 for c in caption])
                    filter_forbidden_words = all([c not in caption.lower() for c in self.text_conditions['forbidden_words']])
                    filter_text = filter_min_words and filter_ord_128 and filter_forbidden_words
                else:
                    filter_text = True
                return filter_size and filter_watermark and (filter_aesthetic_a or filter_aesthetic_b) and filter_unsafe and filter_text
            else:
                return False
        except:
            return False


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
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
dataset = wds.WebDataset(
    dataset_path, resampled=False, handler=warn_and_continue
).select(
    WebdatasetFilter(min_size=512)
)
real_batch_size = 1000
dataloader = DataLoader(dataset, batch_size=real_batch_size, num_workers=8, pin_memory=True)

max_pwatermark=0.5
aesthetic_threshold=5.0
unsafe_threshold=0.99
min_size = 512

filter_counter = WebdatasetFilterCounter(
    min_size=min_size, max_pwatermark=max_pwatermark, aesthetic_threshold=aesthetic_threshold,
    unsafe_threshold=unsafe_threshold)

dataloader_iterator = iter(dataloader)
for i, x in enumerate(tqdm(dataloader, total=2600000000)):
    for xi in x:
        filter_counter(x)
    if i%100 == 0:
        filter_counter.checkpoint(savepath="../stats/filter_stats_{}.json")

