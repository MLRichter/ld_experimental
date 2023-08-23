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
from transformers import AutoTokenizer, CLIPTextModel, CLIPVisionModel
import time

import webdataset as wds
from webdataset.handlers import warn_and_continue

from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
from diffusers import AutoencoderKL, StableDiffusionKDiffusionPipeline

from torchtools.utils import Diffuzz, Diffuzz2
from torchtools.utils.diffusion2 import DDPMSampler

from ld_model import LDM
from utils import WebdatasetFilter

# PARAMETERS
updates = 300000  # 500000
warmup_updates = 10000
batch_size = 1536  # 1024 # 2048 # 4096
grad_accum_steps = 12 * 8
max_iters = updates * grad_accum_steps
print_every = 10 * grad_accum_steps
lr = 1e-4

dataset_path = "pipe:aws s3 cp s3://stability-west/laion-a-native-high-res/{part-0/{00000..18699}.tar,part-1/{00000..18699}.tar,part-2/{00000..18699}.tar,part-3/{00000..18699}.tar,part-4/{00000..18699}.tar} -"  # "pipe:aws s3 cp s3://laion-west/humans-7M-with-blip-caps+aesthetics+nsfw/00000{1..5499}.tar -"
dataset_path = "pipe:aws s3 cp s3://stability-west/laion-a-native-high-res/{part-0/{00000..18000}.tar,part-1/{00000..13500}.tar,part-2/{00000..13500}.tar,part-3/{00000..13500}.tar,part-4/{00000..14100}.tar} -"  # "pipe:aws s3 cp s3://laion-west/humans-7M-with-blip-caps+aesthetics+nsfw/00000{1..5499}.tar -"
clip_image_model_name = 'laion/CLIP-ViT-H-14-laion2B-s32B-b79K'
output_path = "../output/experimental/exp1/"
checkpoint_path = "../models/experimental/exp1.pt"
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

transforms = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Resize(512, interpolation=torchvision.transforms.InterpolationMode.BILINEAR, antialias=True),
    torchvision.transforms.CenterCrop(512)
])


def do_tokenize(tokenizer, prompt):
    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_input_ids = text_inputs.input_ids
    untruncated_ids = tokenizer(prompt, padding="longest", return_tensors="pt").input_ids
    if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(
            text_input_ids, untruncated_ids):
        removed_text = tokenizer.batch_decode(
            untruncated_ids[:, tokenizer.model_max_length - 1: -1]
        )
        warnings.warn("The following part of your input was truncated because CLIP can only handle sequences up to"
                      f" {tokenizer.model_max_length} tokens: {removed_text}"
                      )
    return text_inputs, text_input_ids


def prompts2embed(prompt, tokenizer, text_encoder, device):
    text_inputs, text_input_ids = do_tokenize(tokenizer, prompt)

    if hasattr(text_encoder.config, "use_attention_mask") and text_encoder.config.use_attention_mask:
        attention_mask = text_inputs.attention_mask.to(device)
    else:
        attention_mask = None

    prompt_embeds = text_encoder(
        text_input_ids.to(device),
        attention_mask=attention_mask,
    )
    prompt_embeds = prompt_embeds[0]
    prompt_embeds = prompt_embeds.to(dtype=text_encoder.dtype, device=device)
    return prompt_embeds.float()


def identity(x):
    return x


def ddp_setup(rank, world_size, n_node, node_id):  # <--- DDP
    rk = int(os.environ.get("SLURM_PROCID"))
    print(f"Rank {rk} setting device to {rank}")
    torch.cuda.set_device(rank)
    init_process_group(
        backend="nccl",
        rank=rk, world_size=world_size * n_node,
        #init_method="file:///fsx/mlrichter/ld_experimental/dist_file_experimental_exp1",
        init_method="file:///fsx/mlrichter/dist_file_experimental_exp1"
    )
    print(f"[GPU {rk}] READY")


def tokenizer_text_encoder_factory(device: str):
    model = StableDiffusionKDiffusionPipeline.from_pretrained(
        "CompVis/stable-diffusion-v1-4", revision="fp16", torch_dtype=torch.float16
    )

    model.set_scheduler("sample_dpmpp_2m")
    model.to(device)
    tokenizer = model.tokenizer
    text_encoder = model.text_encoder
    return tokenizer, text_encoder


def train(gpu_id, world_size, n_nodes):
    node_id = int(os.environ["SLURM_PROCID"]) // world_size
    is_main_node = int(os.environ.get("SLURM_PROCID")) == 0
    ddp_setup(gpu_id, world_size, n_nodes, node_id)  # <--- DDP
    device = torch.device(gpu_id)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # --- PREPARE DATASET ---
    # PREPARE DATASET
    dataset = wds.WebDataset(
        dataset_path, resampled=True, handler=warn_and_continue
    ).select(
        WebdatasetFilter(min_size=512, max_pwatermark=0.5, aesthetic_threshold=5.0, unsafe_threshold=0.99)
    ).shuffle(690, handler=warn_and_continue).decode(
        "pilrgb", handler=warn_and_continue
    ).to_tuple(
        "jpg", "combined_txt", handler=warn_and_continue
    ).map_tuple(
        transforms, identity, handler=warn_and_continue
    )
    real_batch_size = min(batch_size // (world_size * n_nodes * grad_accum_steps), 1)
    dataloader = DataLoader(dataset, batch_size=real_batch_size, num_workers=8, pin_memory=True)
    dataloader_iterator = iter(dataloader)
    pbar = tqdm(range(start_iter, max_iters + 1)) if is_main_node else range(start_iter, max_iters + 1)  # <--- DDP
    print("Switching Train Mode")
    start_iter = 1
    for it in pbar:
        images, captions = next(dataloader_iterator)
        images = images.to(device)


if __name__ == '__main__':
    print("Launching Script")
    world_size = torch.cuda.device_count()
    n_node = 1
    local_rank = int(os.environ.get("SLURM_LOCALID"))
    print("Detecting", torch.cuda.device_count(), "GPUs for each of the", n_node, "nodes")
    train(local_rank, world_size, n_node)
