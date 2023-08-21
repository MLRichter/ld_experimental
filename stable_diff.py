from diffusers import AutoencoderKL, StableDiffusionKDiffusionPipeline, UNet2DConditionModel
from fvcore.nn import FlopCountAnalysis
from rfa_toolbox import visualize_architecture, create_graph_from_pytorch_model
from skimage.io import imshow, show

from skimage.data import astronaut

import torch

vqmodel = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae")
vqmodel.eval().requires_grad_(False)

data = torch.zeros((1, 3, 224, 224))
data = torch.from_numpy(astronaut()).float()
data: torch.Tensor = data.permute(dims=(2, 0, 1)).unsqueeze(dim=0) / 255.
print(data.size())

out2 = vqmodel.encode(data).latent_dist.mode()
out = vqmodel.decode(out2).sample

print(out.size())
print(out2.size())

tensor = data.cpu().squeeze().permute(dims=(1, 2, 0)).numpy()
imshow(tensor.astype(float))
show()

tensor = out.cpu().squeeze().permute(dims=(1, 2, 0)).numpy()
imshow(tensor)
show()