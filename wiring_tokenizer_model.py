from diffusers import StableDiffusionKDiffusionPipeline
import torch
from skimage.io import imshow, show
import numpy as np
import warnings

from ld_model import LDM
from modules_experimental_1 import StageX


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
        untruncated_ids[:, tokenizer.model_max_length - 1 : -1]
        )
        warnings.warn("The following part of your input was truncated because CLIP can only handle sequences up to"
        f" {tokenizer.model_max_length} tokens: {removed_text}"
                      )
    return text_inputs, text_input_ids


def prompts2embed(prompt, tokenizer, text_encoder):
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


if __name__ == '__main__':
    prompt = "A post for the movie mad mats fury road"
    device = "cuda:0"

    model = StableDiffusionKDiffusionPipeline.from_pretrained(
        "CompVis/stable-diffusion-v1-4", revision="fp16", torch_dtype=torch.float16
    )
    model.set_scheduler("sample_dpmpp_2m")
    model.to(device)

    tokenizer, text_encoder = tokenizer_text_encoder_factory(device)

    embeds = prompts2embed(prompt, tokenizer, text_encoder)
    print(embeds.size())

    #ldm_model = StageX(c_clip=768).cuda()
    ldm_model = LDM(c_hidden=[320, 640, 1280, 1280], nhead=[5, 10, 20, 20], blocks=[[2, 4, 14, 4], [5, 15, 5, 3]], level_config=['CTA', 'CTA', 'CTA', 'CTA']).cuda()

    img = torch.zeros((1, 4, 56, 56)).cuda()
    r = (1-torch.rand(img.size(0), device=device)).add(0.001).clamp(0.001, 1.0)
    #embeds = torch.zeros(img.size(0), 1, 1280).to(device)

    ldm_model(img, r=r, clip=embeds)

    #result = model(prompt)
    result = model(prompt_embeds=embeds)

    print(result)
    result.images[0].show()

    total_params = sum(p.numel() for p in ldm_model.parameters())
    print(f"Number of parameters: {total_params / 1000000}")