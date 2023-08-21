from vqgan import VQModel
import torch

device = "cuda:0"

vqmodel = VQModel().to(device)

data = torch.zeros((1, 3, 224, 224)).cuda()

out = vqmodel(data)[0]
out2 = vqmodel.encode(data)[0]

print(out.size())
print(out2.size())