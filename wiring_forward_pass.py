import warnings

import torch
import torchvision
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
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

from torch.distributed import init_process_group, destroy_process_group, new_group, get_world_size
from torch.nn.parallel import DistributedDataParallel as DDP
from diffusers import AutoencoderKL, StableDiffusionKDiffusionPipeline
import torch.multiprocessing as mp

from torchtools.utils import Diffuzz, Diffuzz2
from torchtools.utils.diffusion2 import DDPMSampler
from torchtools.transforms import SmartCrop

from ld_model import LDM
from modules_experimental_1 import StageX
from utils import WebdatasetFilterHumans

# PARAMETERS
updates = 300000 # 500000
warmup_updates = 10000
batch_size = 2 # 1024 # 2048 # 4096
grad_accum_steps = 1
max_iters = updates * grad_accum_steps
print_every = 500 * grad_accum_steps
lr = 1e-4

dataset_path = "pipe:aws s3 cp s3://laion-west/humans-7M-with-blip-caps+aesthetics+nsfw/00000{1..5499}.tar -"
clip_image_model_name = 'laion/CLIP-ViT-H-14-laion2B-s32B-b79K'
output_path = "./output/experimental/exp1/"
checkpoint_path = "./output/experimental/exp1.pt"
target = "v"

transforms = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Resize(64, interpolation=torchvision.transforms.InterpolationMode.BILINEAR, antialias=True),
    SmartCrop(64, randomize_p=0.3, randomize_q=0.2)
])
clip_preprocess = torchvision.transforms.Compose([
    torchvision.transforms.Resize(64, interpolation=torchvision.transforms.InterpolationMode.BICUBIC),
    torchvision.transforms.CenterCrop(64),
    torchvision.transforms.Normalize(
        mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711)
    )
])


def identity(x):
    return x


def get_scores(x_json):
    try:
        similarity = torch.tensor(x_json.get('similarity', 0.0))
        pwatermark = torch.tensor(x_json.get('pwatermark', 0.0))
        width, height = x_json.get('width', 1.0), x_json.get('height', 1.0)
        owidth, oheight = x_json.get('original_width', 1.0), x_json.get('original_height', 1.0)
        return similarity, pwatermark, torch.tensor(width / height), torch.tensor(min(owidth, oheight, width, height))
    except:
        return torch.tensor(0.0), torch.tensor(1.0), torch.tensor(1.0), torch.tensor(0.0)


def ddp_setup(rank, world_size, n_node, node_id):  # <--- DDP
    rk = int(os.environ.get("SLURM_PROCID"))
    print(f"Rank {rk} setting device to {rank} for world size {world_size * n_node}")
    torch.cuda.set_device(rank)
    init_process_group(
        backend="nccl",
        rank=rk, world_size=world_size * n_node,
        init_method="file:///fsx/home-pablo/src/wurstchen_experimental/dist_file_experimental_exp1",
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


def do_tokenize(tokenizer, prompt):
    text_inputs = tokenizer(
                    list(prompt),
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
        untruncated_ids[:, tokenizer.model_max_length - 1 : -1]
        )
        warnings.warn("The following part of your input was truncated because CLIP can only handle sequences up to"
        f" {tokenizer.model_max_length} tokens: {removed_text}"
                      )
    return text_inputs, text_input_ids


def prompts2embed(prompt, tokenizer, text_encoder, device):
    prompt = list(prompt)
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


def train(gpu_id, world_size, n_nodes):
    node_id = int(os.environ["SLURM_PROCID"]) // world_size
    is_main_node = int(os.environ.get("SLURM_PROCID")) == 0
    #ddp_setup(gpu_id, world_size, n_nodes, node_id) # <--- DDP
    device = torch.device(gpu_id)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    real_batch_size = batch_size//(world_size*n_nodes*grad_accum_steps)
    dataset = CIFAR10(root="./data/", train=True, transform=transforms, download=True)
    dataloader = DataLoader(dataset, batch_size=real_batch_size, num_workers=8, pin_memory=True)

    if is_main_node:
        print("REAL BATCH SIZE / DEVICE:", real_batch_size)
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device) if os.path.exists(checkpoint_path) else None
    except RuntimeError as e:
        if os.path.exists(f"{checkpoint_path}.bak"):
            os.remove(checkpoint_path)
            shutil.copyfile(f"{checkpoint_path}.bak", checkpoint_path)
            checkpoint = torch.load(checkpoint_path, map_location=device)
        else:
            raise e

    # - utils -
    # diffuzz = Diffuzz(device=device)
    if target == 'e':
        diffuzz = Diffuzz2(device=device, scaler=1, clamp_range=(0.0001, 0.9999))
        diffuzz_sampler = DDPMSampler(diffuzz, mode=target)
    elif target == 'v':
        diffuzz = Diffuzz2(device=device, scaler=1, clamp_range=(0, 1 - 1e-7))

    vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae").to(device)
    vae.eval().requires_grad_(False)

    clip_tokenizer, clip_model = tokenizer_text_encoder_factory(device=device)
    generator = LDM(c_hidden=[320, 640, 1280, 1280], nhead=[5, 10, 20, 20], blocks=[[2, 4, 14, 4], [5, 15, 5, 3]], level_config=['CTA', 'CTA', 'CTA', 'CTA']).to(device)
    if checkpoint is not None:
        generator.load_state_dict(checkpoint['state_dict'])
    #generator = DDP(generator, device_ids=[gpu_id], output_device=device) # <--- DDP
    if is_main_node: # <--- DDP
        print("Num trainable params:", sum(p.numel() for p in generator.parameters() if p.requires_grad))

    # SETUP OPTIMIZER, SCHEDULER & CRITERION
    optimizer = optim.SGD(generator.parameters(), lr=lr)  # , eps=1e-7, betas=(0.9, 0.95))
    scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=warmup_updates)
    if checkpoint is not None:
        try:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        except:
            print("Failed loading optimizer, skipping...")
        scheduler.last_epoch = checkpoint['scheduler_last_step']


    start_iter = 1
    grad_norm = torch.tensor(0, device=device)

    ema_loss = None
    if checkpoint is not None:
        ema_loss = checkpoint['metrics']['ema_loss']

    if checkpoint is not None:
        del checkpoint  # cleanup memory
        torch.cuda.empty_cache()

    # -------------- START TRAINING --------------
    dataloader_iterator = iter(dataloader)
    pbar = tqdm(range(start_iter, max_iters + 1)) if is_main_node else range(start_iter, max_iters + 1)  # <--- DDP
    generator.train()

    for it in pbar:
        images, captions = next(dataloader_iterator)
        scores = [1.0]*batch_size
        images = images.to(device)

        captions = ["this is a caption"]*batch_size
        sca = torch.from_numpy(np.asarray(scores))
        #_, _, _, sca = scores
        sca = (sca.to(device) / min(images.size(-1), images.size(-2))).clamp(max=1)

        if np.random.rand() < 0.05:
            sca = torch.zeros_like(sca)

        with torch.no_grad():
            embeds = prompts2embed(captions, clip_tokenizer, clip_model, device)
            rand_idx = np.random.rand(embeds.size(0)) <= 0.05
            if not rand_idx.any():
                embeds[rand_idx] = 0
                print("zeroed something")

            t = (1 - torch.rand(images.size(0), device=device)).add(0.001).clamp(0.001, 1.0)
            latents = vae.encode(images).latent_dist.mode()
            noised_latents, noise = diffuzz.diffuse(latents, t)
            target_v = diffuzz.get_v(latents, t, noise)
            print(noised_latents.size(), embeds.size())

        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            if target == 'e':
                pred_e = diffuzz.noise_from_v(noised_latents, pred_v, t)
                loss = nn.functional.mse_loss(pred_e, noise, reduction='none').mean(dim=[1, 2, 3])
                loss_adjusted = (loss * diffuzz.p2_weight(t)).mean() / grad_accum_steps
            elif target == 'v':
                pred_v = generator(noised_latents, t, embeds)#, sca=sca)
                loss = nn.functional.mse_loss(pred_v, target_v, reduction='none').mean(dim=[1, 2, 3])
                loss_adjusted = loss.mean() / grad_accum_steps
        if it % grad_accum_steps == 0 or it == max_iters:
            loss_adjusted.backward()
            grad_norm = nn.utils.clip_grad_norm_(generator.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
        else:
            with generator.no_sync():
                loss_adjusted.backward()

        if not np.isnan(loss.mean().item()):
            ema_loss = loss.mean().item() if ema_loss is None else ema_loss * 0.99 + loss.mean().item() * 0.01

        if is_main_node: # <--- DDP
            pbar.set_postfix({
                'bs': images.size(0),
                'loss': ema_loss,
                'raw_loss': loss.mean().item(),
                'grad_norm': grad_norm.item(),
                'lr': optimizer.param_groups[0]['lr'],
                'total_steps': scheduler.last_epoch,
            })

        if is_main_node and (scheduler.last_epoch == 1 or scheduler.last_epoch % print_every == 0 or it == max_iters): # <--- DDP
            if np.isnan(loss.mean().item()):
                tqdm.write(f"Skipping sampling & checkpoint because the loss is NaN")
            else:
                tqdm.write(f"ITER {it}/{max_iters} - loss {ema_loss}")
                generator.eval()
                images, captions = next(dataloader_iterator)
                captions = ["this is a caption"] * (len(images))
                while images.size(0) < 8:  # 8
                    #_images, _captions, _scores = next(dataloader_iterator)
                    _images, _captions = next(dataloader_iterator)
                    scores = [1.0] * len(images) * 2
                    #images = images.to(device)

                    _captions = ["this is a caption"] * (len(_images))
                    _sca = torch.from_numpy(np.asarray(scores))
                    images = torch.cat([images.cuda(), _images.cuda()], dim=0).cuda()
                    print(type(captions), type(_captions))
                    captions += _captions
#                    sca = torch.ones(images.size(0), device=device)
                    with torch.no_grad():
                        # clip stuff
                        embeds = prompts2embed(captions, clip_tokenizer, clip_model, device)
                        clip_image_embeddings_uncond = torch.zeros_like(embeds)
                        # ---

                        t = (1 - torch.rand(images.size(0), device=device)).add(0.001).clamp(0.001, 1.0)
                        latents = vae.encode(images).latent_dist.mode()
                        noised_latents, noise = diffuzz.diffuse(latents, t)
                        if target == 'e':
                            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                                pred_e = generator(noised_latents, t, embeds)
                            pred = \
                            diffuzz.undiffuse(noised_latents, t, torch.zeros_like(t), pred_e, sampler=diffuzz_sampler)[
                                0]
                        elif target == 'v':
                            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                                print(noised_latents.size(), embeds.size())
                                pred_v = generator(noised_latents, t, embeds)
                            pred = diffuzz.x0_from_v(noised_latents, pred_v, t)

                        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                            *_, (sampled, _, _) = diffuzz.sample(generator, {
                                'clip': embeds,
                            }, latents.shape, unconditional_inputs={
                                'clip': clip_image_embeddings_uncond,
                            }, cfg=7, sample_mode=target)
                        noised_images = vae.decode(noised_latents).sample.clamp(0, 1)
                        pred_images = vae.decode(pred).sample.clamp(0, 1)
                        sampled_images = vae.decode(sampled).sample.clamp(0, 1)
                    generator.train()

                    torchvision.utils.save_image(torch.cat([
                        torch.cat([i for i in images.cpu()], dim=-1),
                        torch.cat([i for i in noised_images.cpu()], dim=-1),
                        torch.cat([i for i in pred_images.cpu()], dim=-1),
                        torch.cat([i for i in sampled_images.cpu()], dim=-1)
                    ], dim=-2), f'{output_path}{it:06d}.jpg')

                    try:
                        os.remove(f"{checkpoint_path}.bak")
                    except OSError:
                        pass

                    try:
                        os.rename(checkpoint_path, f"{checkpoint_path}.bak")
                    except OSError:
                        pass

                    torch.save({
                        'state_dict': generator.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_last_step': scheduler.last_epoch,
                        'iter': it,
                        'metrics': {
                            'ema_loss': ema_loss,
                        },
                        'wandb_run_id': "lul",
                    }, checkpoint_path)

                    if scheduler.last_epoch % 20000 == 0:
                        chkpt_extension = f"_{scheduler.last_epoch // 1000}k.pt"
                        shutil.copyfile(checkpoint_path, checkpoint_path.replace(".pt", chkpt_extension))


if __name__ == '__main__':
    print("Launching Script")
    world_size = torch.cuda.device_count()
    n_node = 1
    local_rank = int(os.environ.get("SLURM_LOCALID"))
    print("Detecting", torch.cuda.device_count(), "GPUs for each of the", n_node, "nodes")
    train(local_rank, world_size, n_node)
