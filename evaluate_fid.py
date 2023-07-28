import os
import torch
import torchvision
from tqdm import tqdm
from transformers import AutoTokenizer, CLIPTextModel, CLIPVisionModel
from torchtools.utils import Diffuzz, Diffuzz2
from vqgan import VQModel
from modules_experimental_1 import StageX
from PIL import Image
import subprocess
from glob import glob
import math

image_size = 256
batch_size = 8
# sampling_mode = "e"
sampling_mode = "v"
# model_arch = "base"
model_arch = "asymmetric"
base_path = f"../../datasets/humans_10k_fid_{image_size}/"
save_path = f"../../datasets/generated_{sampling_mode}_{model_arch}_{image_size}/"
# model_path = "../../models/experimental/exp1b_100k.pt"
# model_path = "../../models/experimental/exp1c_100k.pt"
model_path = "../../models/experimental/exp1d_100k.pt"
clip_image_model_name = 'laion/CLIP-ViT-H-14-laion2B-s32B-b79K'

transforms = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Resize(image_size, interpolation=torchvision.transforms.InterpolationMode.BILINEAR, antialias=True),
    torchvision.transforms.CenterCrop(image_size),
])
clip_preprocess = torchvision.transforms.Compose([
    torchvision.transforms.Resize(224, interpolation=torchvision.transforms.InterpolationMode.BICUBIC),
    torchvision.transforms.CenterCrop(224),
    torchvision.transforms.Normalize(
        mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711)
    )
])

def evaluate(proc_id, gpu_id):
    device = torch.device(gpu_id)

    print(f"{proc_id} --- MODELS SETUP ---")
    if sampling_mode == 'e':
        diffuzz = Diffuzz2(device=device, scaler=1, clamp_range=(0.0001, 0.9999))
    else:    
        diffuzz = Diffuzz2(device=device, scaler=1, clamp_range=(0, 1-1e-7))

    vqmodel = VQModel().to(device)
    vqmodel.load_state_dict(torch.load(f"../../models/vqgan/vqgan_f4_v1_500k.pt", map_location=device)['state_dict'])
    vqmodel.eval().requires_grad_(False)

    clip_image_model = CLIPVisionModel.from_pretrained(clip_image_model_name).to(device).eval().requires_grad_(False)

    if model_arch == "base":
        generator = StageX().to(device)
    elif model_arch == "asymmetric":
        generator = StageX(blocks=[[2, 4], [36, 6]], c_hidden=[512, 1024], nhead=[-1, 16], level_config=['CT', 'CTA'], dropout=[0, 0.1]).to(device)
    generator.load_state_dict(torch.load(model_path, map_location=device)['state_dict'])
    generator.eval().requires_grad_(False)
    
    os.makedirs(save_path, exist_ok=True) 
    image_paths = glob(f"{base_path}*.jpg")

    chunk_size = int(math.ceil(len(image_paths) / torch.cuda.device_count()))
    image_paths = image_paths[proc_id*chunk_size:proc_id*chunk_size+chunk_size]
    print(f"{proc_id} --- DATASET IMAGE GENERATION ({proc_id*chunk_size} - {proc_id*chunk_size + chunk_size}) ---")

    with torch.no_grad():
        for i in tqdm(range(0, len(image_paths), batch_size)):
            image_names = [path.split("/")[-1] for path in image_paths[i:i+batch_size]]
            if all([os.path.exists(f"{save_path}{image_name}") for image_name in image_names]):
                continue
            images = torch.stack([transforms(Image.open(path).convert("RGB")).to(device) for path in image_paths[i:i+batch_size]], dim=0)

            clip_image_embeddings = clip_image_model(clip_preprocess(images)).pooler_output.unsqueeze(1)
            clip_image_embeddings_uncond = torch.zeros_like(clip_image_embeddings)
            sca = torch.ones(images.size(0), device=device)

            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                *_, (sampled, _, _) = diffuzz.sample(generator, {
                    'clip': clip_image_embeddings, "sca": sca,
                }, (images.size(0), 4, image_size//4, image_size//4), unconditional_inputs = {
                    'clip': clip_image_embeddings_uncond, "sca": sca,
                }, cfg=7, sample_mode=sampling_mode)

            sampled_images = torch.cat([vqmodel.decode(sampled[j:j+1]).clamp(0, 1) for j in range(sampled.size(0))], dim=0)

            for sampled_image, image_name in zip(sampled_images, image_names):
                image = torchvision.transforms.ToPILImage()(sampled_image)
                image.save(f"{save_path}{image_name}", format="JPEG", dpi=(300, 300), optimize=False, quality=100)

    if proc_id == 0:
        proc = subprocess.Popen(f"python3 -m pytorch_fid {base_path} {save_path} --device cuda" ,stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=True)
        fid_score = str(proc.communicate()[0]).split("FID:")[-1].replace("\\n", "").replace("'", "").strip()
        print("FID SCORE:", float(fid_score))

if __name__ == '__main__':
    local_rank = int(os.environ.get("SLURM_LOCALID"))
    proc_id = int(os.environ.get("SLURM_PROCID"))
    evaluate(proc_id, local_rank)