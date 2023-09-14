from typing import Any, List, Union

import PIL
import torch
import os

from diffusers import AutoencoderKL, StableDiffusionKDiffusionPipeline, StableDiffusionPipeline
from torchtools.utils import Diffuzz, Diffuzz2

from ld_model import LDM
from pathlib import Path

from train import prompts2embed, tokenizer_text_encoder_factory
from attrs import define
from typing import Protocol
import torch
from diffusers import AutoPipelineForText2Image
from diffusers.pipelines.wuerstchen import DEFAULT_STAGE_C_TIMESTEPS


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
    return denormalized_image


def get_processed_images(dst_path: Path) -> List[str]:
    filenames = os.listdir(dst_path)
    return filenames


class Inferencer(Protocol):

    def __call__(self, captions: List[str], device_lang: str, batch_size: int) -> torch.Tensor:
        ...


@define
class DiffusionModelInferencer(Inferencer):
    generator: LDM
    clip_tokenizer: Any
    clip_model: torch.nn.Module
    vae: torch.nn.Module
    diffuzz: Diffuzz2

    def __call__(self, captions: List[str], device_lang: str = "cpu", batch_size = 2) -> Union[torch.Tensor, List[PIL.Image.Image]]:
        with torch.no_grad():
            # clip stuff
            embeds = prompts2embed(captions, self.clip_tokenizer, self.clip_model, device_lang)
            embeds_uncond = prompts2embed([""] * len(captions), self.clip_tokenizer, self.clip_model, device_lang)
            # ---
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                *_, (sampled, _, _) = self.diffuzz.sample(self.generator, {
                    'clip': embeds,
                }, (batch_size, 4, 64, 64), unconditional_inputs={
                    'clip': embeds_uncond,
                }, cfg=7, sample_mode="e")

                magic_norm = 0.18215
                sampled_images = self.vae.decode(sampled / magic_norm).sample.clamp(0, 1)
            #sampled_images = denormalize_image(image=sampled_images, mean=0.5, std=0.5)
        return sampled_images


@define
class StableDiffusionInferencer(Inferencer):
    generator: StableDiffusionPipeline

    def __call__(self, captions: List[str], device_lang: str = "cpu", batch_size = 2):
        sampled = self.generator(captions)
        return sampled.images


@define
class WuerstchenInferencer(Inferencer):
    generator: StableDiffusionPipeline

    def __call__(self, captions: List[str], device_lang: str = "cpu", batch_size = 2):
        images = []
        for caption in captions:
            sampled = self.generator(
            caption,
            height=512,
            width=512,
            prior_timesteps=DEFAULT_STAGE_C_TIMESTEPS,
            prior_guidance_scale=4.0,
            num_images_per_prompt=1,
        ).images
            images += sampled
        return images


def ldm14(weight_path: Path = "./models/baseline/exp1.pt", device: str = "cpu") -> Inferencer:
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


def sd14(weight_path: Path = "CompVis/stable-diffusion-v1-4", device: str = "cpu") -> Inferencer:
    model = StableDiffusionKDiffusionPipeline.from_pretrained(
        weight_path, revision="fp16", torch_dtype=torch.float16
    )
    model.set_scheduler("sample_dpmpp_2m")
    model.to(device)
    return StableDiffusionInferencer(generator=model)


def wuerstchen(weight_path: Path = "warp-ai/wuerstchen", device: str = "cuda:0", compile: bool = False) -> Inferencer:
    pipeline = AutoPipelineForText2Image.from_pretrained(weight_path, torch_dtype=torch.float16).to(device)
    if compile:
        pipeline.prior_prior = torch.compile(pipeline.prior_prior, mode="reduce-overhead", fullgraph=True)
        pipeline.decoder = torch.compile(pipeline.decoder, mode="reduce-overhead", fullgraph=True)

    pipeline.set_progress_bar_config(leave=True)
    model = WuerstchenInferencer(pipeline)
    return model


if __name__ == '__main__':
    pipeline = AutoPipelineForText2Image.from_pretrained("warp-ai/wuerstchen", torch_dtype=torch.float16).to("cuda")
    caption = ["Anthropomorphic cat dressed as a firefighter", "Anthropomorphic cat doing a poledance"]
    images = wuerstchen()(caption)

    print(images)
    images[0].save("img1.jpg")
    images[1].save("img2.jpg")
    print(len(images))



