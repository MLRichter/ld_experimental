from typing import Any, List, Union

import PIL
import torch
import os

import torchvision
from diffusers import AutoencoderKL, StableDiffusionKDiffusionPipeline, StableDiffusionPipeline, \
    DPMSolverMultistepScheduler, DiffusionPipeline
from torchtools.utils import Diffuzz, Diffuzz2

from ld_model import LDM
from pathlib import Path

from train import prompts2embed, tokenizer_text_encoder_factory
from attrs import define
from typing import Protocol
import torch
from diffusers import AutoPipelineForText2Image
from diffusers.pipelines.wuerstchen import DEFAULT_STAGE_C_TIMESTEPS
from torchmetrics.multimodal import clip_score
from torchmetrics.multimodal.clip_score import CLIPScore



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

    def __call__(self, captions: List[str], device_lang: str = "cuda:0", batch_size = 2) -> Union[torch.Tensor, List[PIL.Image.Image]]:
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
                }, cfg=7, sample_mode="e", timesteps=60)

                magic_norm = 0.18215
                sampled_images = self.vae.decode(sampled / magic_norm).sample.clamp(0, 1)
            #sampled_images = denormalize_image(image=sampled_images, mean=0.5, std=0.5)
        return sampled_images


@define
class StableDiffusionInferencer(Inferencer):
    generator: StableDiffusionPipeline

    def __call__(self, captions: List[str], device_lang: str = "cpu", batch_size = 2, cfg: float = 7.5):
        sampled = self.generator(captions, guidance_scale=cfg if cfg != 1.0 else cfg + .00001)
        return sampled.images


@define
class WuerstchenInferencer(Inferencer):
    generator: StableDiffusionPipeline

    def __call__(self, captions: List[str], device_lang: str = "cpu", batch_size = 2, cfg: float = 4.0, spl: int = 60):
        images = []
        for caption in captions:
            sampled = self.generator(
            caption,
            height=1024,
            width=1024,
            #prior_timesteps=DEFAULT_STAGE_C_TIMESTEPS,
            prior_guidance_scale=cfg, prior_num_inference_steps=spl,
            num_images_per_prompt=1,
        ).images
            images += sampled
        return images


def ldm14(weight_path: Path = "../models/baseline/exp1.pt.bak", device: str = "cpu") -> Inferencer:
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

def ldm30k(weight_path: Path = "../models/baseline_highres/exp1.pt.bak", device: str = "cpu") -> Inferencer:
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


class BaseWuerstchenInferencer:

    def __init__(self):
        from diffusers import WuerstchenDecoderPipeline, WuerstchenPriorPipeline
        from diffusers.pipelines.wuerstchen import WuerstchenPrior
        self.default_stage_c_timesteps = DEFAULT_STAGE_C_TIMESTEPS
        device = "cuda"
        dtype = torch.float16
        num_images_per_prompt = 2

        prior = WuerstchenPrior.from_pretrained("warp-ai/wuerstchen-prior-model-base", torch_dtype=dtype).to(device)
        prior_pipeline = WuerstchenPriorPipeline.from_pretrained(
            "warp-ai/wuerstchen-prior", prior=prior, torch_dtype=dtype
        ).to(device)
        decoder_pipeline = WuerstchenDecoderPipeline.from_pretrained(
            "warp-ai/wuerstchen", torch_dtype=dtype
        ).to(device)

        negative_prompt = "bad anatomy, blurry, fuzzy, extra arms, extra fingers, poorly drawn hands, disfigured, tiling, deformed, mutated, drawing"
        self.prior = prior
        self.prior_pipeline = prior_pipeline
        self.decoder_pipeline = decoder_pipeline
        self.negative_prompt = negative_prompt
        self.num_clip_samples = 10
        if self.num_clip_samples > 1:
            self.clip_score = CLIPScore()

    def _get_one_sample(self, caption: str):
        prior_output = self.prior_pipeline(
            prompt=caption,
            height=1024,
            width=1024,
            timesteps=self.default_stage_c_timesteps,
            negative_prompt=self.negative_prompt,
            guidance_scale=8.0,
            num_images_per_prompt=1,
        )
        decoder_output = self.decoder_pipeline(
            image_embeddings=prior_output.image_embeddings,
            prompt=caption,
            negative_prompt=self.negative_prompt,
            num_images_per_prompt=1,
            guidance_scale=0.0,
            output_type="pt",
        ).images
        return decoder_output[0]

    def _get_best_clip_sample(self, caption: str):
        images = []
        scores = []
        for i in range(self.num_clip_samples):
            image = self._get_one_sample(caption)
            score = self.clip_score(image.cpu(), caption).detach().cpu().item()
            images.append(image)
            scores.append(score)
            self.clip_score.reset()
            print("Image", i, ":", score)
        index_of_max = scores.index(max(scores))
        selected_value = images[index_of_max]
        return [selected_value]

    def __call__(self, captions: List[str], device_lang: str = "cpu", batch_size = 2):

        images = []
        for caption in captions:
            """
            prior_output = self.prior_pipeline(
                prompt=caption,
                height=1024,
                width=1024,
                timesteps=self.default_stage_c_timesteps,
                negative_prompt=self.negative_prompt,
                guidance_scale=8.0,
                num_images_per_prompt=1,
            )
            decoder_output = self.decoder_pipeline(
                image_embeddings=prior_output.image_embeddings,
                prompt=caption,
                negative_prompt=self.negative_prompt,
                num_images_per_prompt=1,
                guidance_scale=0.0,
                output_type="pil",
            ).images
            """
            decoder_output = self._get_best_clip_sample(caption)
            images += decoder_output
        return images


def sdxl(weight_path: Path = "stabilityai/stable-diffusion-xl-base-1.0", device: str = "cpu") -> Inferencer:
    pipe = DiffusionPipeline.from_pretrained(weight_path, torch_dtype=torch.float16,
                                             use_safetensors=True, variant="fp16")
    pipe.to(device)
    return StableDiffusionInferencer(generator=pipe)

def sd21(weight_path: Path = "stabilityai/stable-diffusion-2-1", device: str = "cuda:0") -> Inferencer:
    pipe = StableDiffusionPipeline.from_pretrained(weight_path, torch_dtype=torch.float16)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(device)
    return StableDiffusionInferencer(generator=pipe)


def sd14(weight_path: Path = "CompVis/stable-diffusion-v1-4", device: str = "cuda:0") -> Inferencer:
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


class TextEncoderWarapper:

    def __init__(self, to_wrap):
        self.to_wrap = to_wrap

    def __call__(self, *args, **kwargs):
        x = self.to_wrap(*args, **kwargs)
        x.last_hidden_state = x.last_hidden_state * 0
        x.pooler_output = x.pooler_output * 0
        return x


def wuerstchen_no_text(weight_path: Path = "warp-ai/wuerstchen", device: str = "cuda:0", compile: bool = False) -> Inferencer:
    pipeline = AutoPipelineForText2Image.from_pretrained(weight_path, torch_dtype=torch.float16).to(device)
    pipeline.generator.decoder_pipe.text_encoder = TextEncoderWarapper(pipeline.generator.decoder_pipe.text_encoder)
    if compile:
        pipeline.prior_prior = torch.compile(pipeline.prior_prior, mode="reduce-overhead", fullgraph=True)
        pipeline.decoder = torch.compile(pipeline.decoder, mode="reduce-overhead", fullgraph=True)

    pipeline.set_progress_bar_config(leave=True)
    model = WuerstchenInferencer(pipeline)
    return model


def wuerstchen_base(weight_path: Path = "warp-ai/wuerstchen", device: str = "cuda:0", compile: bool = False) -> Inferencer:
    pipeline = AutoPipelineForText2Image.from_pretrained(weight_path, torch_dtype=torch.float16).to(device)
    if compile:
        pipeline.prior_prior = torch.compile(pipeline.prior_prior, mode="reduce-overhead", fullgraph=True)
        pipeline.decoder = torch.compile(pipeline.decoder, mode="reduce-overhead", fullgraph=True)

    pipeline.set_progress_bar_config(leave=True)
    model = WuerstchenInferencer(pipeline)
    return model

def wuerstchen_base(weight_path: Path = "warp-ai/wuerstchen", device: str = "cuda:0", compile: bool = False) -> Inferencer:
    model = BaseWuerstchenInferencer()
    return model



if __name__ == '__main__':
    #torch.set_default_device("cuda:0")
    #pipeline: StableDiffusionPipeline = ldm30k("./models/baseline/exp1_120k.pt", device="cuda:0")
    caption = [
        "Cute cat, big eyes pixar style.",
        "Cute cat, big eyes pixar style.",
               ]
    import numpy
    pipeline = wuerstchen()
    pipeline.generator.decoder_pipe.text_encoder = TextEncoderWarapper(pipeline.generator.decoder_pipe.text_encoder)
    images = pipeline(caption)

    #torchvision.utils.save_image(images.float().cpu(), "imgs.jpg")

    #print(images)
    images[0].save("img1.jpg")
    images[1].save("img2.jpg")
    #print(len(images))
    """
    text_enc = sum([numpy.prod(p.size()) for p in pipeline.generator.text_encoder.base_model.parameters()]) / 1000000
    text_enc2 = sum([numpy.prod(p.size()) for p in pipeline.generator.prior_text_encoder.base_model.parameters()]) / 1000000
    stage_c = sum([numpy.prod(p.size()) for p in pipeline.generator.prior_prior.parameters()]) / 1000000
    stage_b = sum([numpy.prod(p.size()) for p in pipeline.generator.decoder.parameters()]) / 1000000
    stage_a = sum([numpy.prod(p.size()) for p in pipeline.generator.vqgan.parameters()]) / 1000000
    print(text_enc)
    print(text_enc2)
    print(stage_c)
    print(stage_b)
    print(stage_a)
    total_params_wuerstchen = text_enc + stage_a + stage_b + stage_c
    print(total_params_wuerstchen) # 2.42B parameters

    print("\n\nLDM")

    pipeline: DiffusionModelInferencer = ldm14("./models/baseline/exp1.pt.bak")
    text_enc = sum([numpy.prod(p.size()) for p in pipeline.clip_model.parameters()]) / 1000000
    diffusion = sum([numpy.prod(p.size()) for p in pipeline.generator.parameters()]) / 1000000
    decoder = sum([numpy.prod(p.size()) for p in pipeline.vae.parameters()]) / 1000000
    total_params_wuerstchen = text_enc + diffusion + decoder
    print(text_enc)
    print(diffusion)
    print(decoder)
    print("total", total_params_wuerstchen)

    print("\n\nSD 1.4")
    pipeline: StableDiffusionPipeline = sd14()
    text_enc = sum([numpy.prod(p.size()) for p in pipeline.generator.text_encoder.base_model.parameters()]) / 1000000
    vae = sum([numpy.prod(p.size()) for p in pipeline.generator.vae.parameters()]) / 1000000
    unet = sum([numpy.prod(p.size()) for p in pipeline.generator.unet.parameters()]) / 1000000

    print(text_enc)
    print(vae)
    print(unet)
    total_params_wuerstchen = text_enc + unet + vae
    print("total", total_params_wuerstchen)

    print("\n\nSD 2.1")
    pipeline: StableDiffusionPipeline = sd21()
    text_enc = sum([numpy.prod(p.size()) for p in pipeline.generator.text_encoder.base_model.parameters()]) / 1000000
    vae = sum([numpy.prod(p.size()) for p in pipeline.generator.vae.parameters()]) / 1000000
    unet = sum([numpy.prod(p.size()) for p in pipeline.generator.unet.parameters()]) / 1000000

    print(text_enc)
    print(vae)
    print(unet)
    total_params_wuerstchen = text_enc + unet + vae
    print("total", total_params_wuerstchen)


    print("\n\nSD XL")
    pipeline: StableDiffusionPipeline = sdxl()
    text_enc = sum([numpy.prod(p.size()) for p in pipeline.generator.text_encoder.base_model.parameters()]) / 1000000
    vae = sum([numpy.prod(p.size()) for p in pipeline.generator.vae.parameters()]) / 1000000
    unet = sum([numpy.prod(p.size()) for p in pipeline.generator.unet.parameters()]) / 1000000

    print(text_enc)
    print(vae)
    print(unet)
    total_params_wuerstchen = text_enc + unet + vae
    print("total", total_params_wuerstchen)













    #images = pipeline(caption, cfg=0.0)

    #print(images)
    #images[0].save("img1.jpg")
    #images[1].save("img2.jpg")
    #print(len(images))
    """



