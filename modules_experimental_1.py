import torch
from torch import nn
import numpy as np
import math

class Attention2D(nn.Module):
    def __init__(self, c, nhead, dropout=0.0):
        super().__init__()
        self.attn = torch.nn.MultiheadAttention(c, nhead, dropout=dropout, bias=True, batch_first=True)

    def forward(self, x, kv, self_attn=False):
        orig_shape = x.shape
        x = x.view(x.size(0), x.size(1), -1).permute(0, 2, 1) # Bx4xHxW -> Bx(HxW)x4
        if self_attn:
            kv = torch.cat([x, kv], dim=1)
        x = self.attn(x, kv, kv, need_weights=False)[0]
        x = x.permute(0, 2, 1).view(*orig_shape)
        return x

class LayerNorm2d(nn.LayerNorm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x):
        return super().forward(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

class GlobalResponseNorm(nn.Module): # from https://github.com/facebookresearch/ConvNeXt-V2/blob/3608f67cc1dae164790c5d0aead7bf2d73d9719b/models/utils.py#L105
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, dim))

    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=(1,2), keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x
    
# class SAMBlock(nn.Module):
#     def __init__(self, c, num_heads=4, expansion=2):
#         super().__init__()
#         assert c % num_heads == 0, "channels not divisible by num_heads"
#         self.num_heads = num_heads
#         self.channels = c // num_heads
#         self.projection_a = nn.Conv2d(c, c, kernel_size=1)
#         self.projection_b = nn.Conv2d(c, c, kernel_size=1)
#         self.convolutions = nn.ModuleList()
#         for i in range(num_heads):
#             kernel_size = 3+(2*i)
#             self.convolutions.append(
#                 nn.Conv2d(self.channels, self.channels, kernel_size=kernel_size, padding=kernel_size//2, groups=self.channels)
#             )
            
#         self.saa = nn.Sequential(
#             nn.Conv2d(c, c*expansion, kernel_size=1, groups=self.channels),
#             nn.Conv2d(c*expansion, c, kernel_size=1),
#         )
        
#     def forward(self, x):
#         residual = x
#         projected = self.projection_a(x)
#         orig_shape = x.shape
#         # MHMC
#         x = self.projection_b(x)
#         x = x.chunk(self.num_heads, dim=1)
#         x = [conv(xc) for xc, conv in zip(x, self.convolutions)]
#         x = torch.stack(x, dim=2).reshape(*orig_shape)
#         # SAA
#         x = self.saa(x)
#         return (x * projected) + residual

class ResBlock(nn.Module):
    def __init__(self, c, c_skip=0, kernel_size=3, dropout=0.0, num_heads=4, expansion=2):
        super().__init__()
        self.depthwise = nn.Conv2d(c, c, kernel_size=kernel_size, padding=kernel_size//2, groups=c)
#         self.depthwise = SAMBlock(c, num_heads, expansion)
        self.norm = LayerNorm2d(c, elementwise_affine=False, eps=1e-6)
        self.channelwise = nn.Sequential(
            nn.Linear(c+c_skip, c*4),
            nn.GELU(),
            GlobalResponseNorm(c*4),
            nn.Dropout(dropout),
            nn.Linear(c*4, c)
        )

    def forward(self, x, x_skip=None): 
        x_res = x
        x = self.norm(self.depthwise(x))
        if x_skip is not None:
            x = torch.cat([x, x_skip], dim=1)
        x = self.channelwise(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        return x + x_res
    
class AttnBlock(nn.Module):
    def __init__(self, c, c_cond, nhead, self_attn=True, dropout=0.0):
        super().__init__()
        self.self_attn = self_attn 
        self.norm = LayerNorm2d(c, elementwise_affine=False, eps=1e-6)
        self.attention = Attention2D(c, nhead, dropout)
        self.kv_mapper = nn.Sequential(
            nn.SiLU(),
            nn.Linear(c_cond, c)
        )

    def forward(self, x, kv):
        kv = self.kv_mapper(kv)
        x = x + self.attention(self.norm(x), kv, self_attn=self.self_attn)
        return x
    
class FeedForwardBlock(nn.Module):
    def __init__(self, c, dropout=0.0):
        super().__init__()
        self.norm = LayerNorm2d(c, elementwise_affine=False, eps=1e-6)
        self.channelwise = nn.Sequential(
            nn.Linear(c, c*4),
            nn.GELU(),
            GlobalResponseNorm(c*4),
            nn.Dropout(dropout),
            nn.Linear(c*4, c)
        )

    def forward(self, x):
        x = x + self.channelwise(self.norm(x).permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        return x
    
class TimestepBlock(nn.Module):
    def __init__(self, c, c_timestep, conds=['sca']):
        super().__init__()
        self.mapper = nn.Linear(c_timestep, c*2)
        self.conds = conds
        for cname in conds:
            setattr(self, f"mapper_{cname}", nn.Linear(c_timestep, c*2))
        
    def forward(self, x, t):
        t = t.chunk(len(self.conds)+1, dim=1)
        a, b = self.mapper(t[0])[:, :, None, None].chunk(2, dim=1)
        for i, c in enumerate(self.conds):
            ac, bc = getattr(self, f"mapper_{c}")(t[i+1])[:, :, None, None].chunk(2, dim=1)
            a, b = a+ac, b+bc
        return x * (1 + a) + b
    
class StageX(nn.Module):
    def __init__(self, c_in=4, c_out=4, c_r=64, patch_size=2, c_cond=1024, c_hidden=[512, 1024, 1024], nhead=[-1, 16, 16], blocks=[[4, 16, 4], [4, 16, 4]], level_config=['CT', 'CTA', 'CTA'], c_clip=1280, kernel_size=3, dropout=[0, 0.1, 0.1], self_attn=True, t_conds=['sca']):
        super().__init__()
        self.c_r = c_r
        self.t_conds = t_conds
        blocks[1] = list(reversed(blocks[1]))
        if not isinstance(dropout, list):
            dropout = [dropout] * len(c_hidden)
        if not isinstance(self_attn, list):
            self_attn = [self_attn] * len(c_hidden)
        
        # CONDITIONING
        self.clip_mapper = nn.Linear(c_clip, c_cond*4)
        self.clip_norm = nn.LayerNorm(c_cond, elementwise_affine=False, eps=1e-6)
            
        self.embedding = nn.Sequential(
            nn.PixelUnshuffle(patch_size),
            nn.Conv2d(c_in*(patch_size**2), c_hidden[0], kernel_size=1),
            LayerNorm2d(c_hidden[0], elementwise_affine=False, eps=1e-6)
        )
        
        def get_block(block_type, c_hidden, nhead, c_skip=0, dropout=0, self_attn=True):
            if block_type == 'C':
                return ResBlock(c_hidden, c_skip, kernel_size=kernel_size, dropout=dropout)
            elif block_type == 'A':
                return AttnBlock(c_hidden, c_cond, nhead, self_attn=self_attn, dropout=dropout)
            elif block_type == 'F':
                return FeedForwardBlock(c_hidden, dropout=dropout)
            elif block_type == 'T':
                return TimestepBlock(c_hidden, c_r, conds=t_conds)
            else:
                raise Exception(f'Block type {block_type} not supported')
        
        # BLOCKS
        # -- down blocks
        self.down_blocks = nn.ModuleList()
        for i in range(len(c_hidden)):
            down_block = nn.ModuleList()
            if i > 0:
                down_block.append(nn.Sequential(
                    LayerNorm2d(c_hidden[i-1], elementwise_affine=False, eps=1e-6),
                    nn.Conv2d(c_hidden[i-1], c_hidden[i], kernel_size=2, stride=2),
                ))
            for _ in range(blocks[0][i]):
                for block_type in level_config[i]:
                    down_block.append(
                        get_block(block_type, c_hidden[i], nhead[i], dropout=dropout[i], self_attn=self_attn[i])
                    )
            self.down_blocks.append(down_block)
          
        # -- up blocks
        self.up_blocks = nn.ModuleList()
        for i in reversed(range(len(c_hidden))):
            up_block = nn.ModuleList()
            for j in range(blocks[1][i]):
                for k, block_type in enumerate(level_config[i]):
                    c_skip = c_hidden[i] if i < len(c_hidden)-1 and j==k==0 else 0
                    up_block.append(
                        get_block(block_type, c_hidden[i], nhead[i], c_skip=c_skip, dropout=dropout[i], self_attn=self_attn[i])
                    )
            if i > 0:
                up_block.append(nn.Sequential(
                    LayerNorm2d(c_hidden[i], elementwise_affine=False, eps=1e-6),
                    nn.ConvTranspose2d(c_hidden[i], c_hidden[i-1], kernel_size=2, stride=2),
                ))
            self.up_blocks.append(up_block)
            
        # OUTPUT
        self.clf = nn.Sequential(
            LayerNorm2d(c_hidden[0], elementwise_affine=False, eps=1e-6),
            nn.Conv2d(c_hidden[0], c_out*(patch_size**2), kernel_size=1),
            nn.PixelShuffle(patch_size),
        )
        
        # --- WEIGHT INIT ---
        self.apply(self._init_weights) # General init
        nn.init.normal_(self.clip_mapper.weight, std=0.02) # conditionings
        torch.nn.init.xavier_uniform_(self.embedding[1].weight, 0.02) # inputs
        nn.init.constant_(self.clf[1].weight, 0) # outputs
        
        # blocks
        for level_block in self.down_blocks + self.up_blocks:
            for block in level_block:
                if isinstance(block, ResBlock) or isinstance(block, FeedForwardBlock):
                    block.channelwise[-1].weight.data *= np.sqrt(1 / sum(blocks[0]))
                elif isinstance(block, TimestepBlock):
                    nn.init.constant_(block.mapper.weight, 0)
      
    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
            
    def gen_r_embedding(self, r, max_positions=10000):
        r = r * max_positions
        half_dim = self.c_r // 2
        emb = math.log(max_positions) / (half_dim - 1)
        emb = torch.arange(half_dim, device=r.device).float().mul(-emb).exp()
        emb = r[:, None] * emb[None, :]
        emb = torch.cat([emb.sin(), emb.cos()], dim=1)
        if self.c_r % 2 == 1:  # zero pad
            emb = nn.functional.pad(emb, (0, 1), mode='constant')
        return emb
    
    def gen_c_embeddings(self, clip):
        clip = self.clip_norm(self.clip_mapper(clip).view(clip.size(0), 4, -1))
        return clip
    
    def _down_encode(self, x, r_embed, clip):
        level_outputs = []
        for i, down_block in enumerate(self.down_blocks):
            for block in down_block:
                if isinstance(block, ResBlock):
                    x = block(x)
                elif isinstance(block, AttnBlock):
                    x = block(x, clip)
                elif isinstance(block, TimestepBlock):
                    x = block(x, r_embed)
                else:
                    x = block(x)
            level_outputs.insert(0, x)
        return level_outputs
    
    def _up_decode(self, level_outputs, r_embed, clip):
        x = level_outputs[0]
        for i, up_block in enumerate(self.up_blocks):
            for j, block in enumerate(up_block):
                if isinstance(block, ResBlock):
                    skip = level_outputs[i] if j == 0 and i > 0 else None
                    if skip is not None and (x.size(-1) != skip.size(-1) or x.size(-2) != skip.size(-2)):
                        x = torch.nn.functional.interpolate(x.float(), skip.shape[-2:], mode='bilinear', align_corners=True)
                    x = block(x, skip)
                elif isinstance(block, AttnBlock):
                    x = block(x, clip)
                elif isinstance(block, TimestepBlock):
                    x = block(x, r_embed)
                else:
                    x = block(x)
        return x
        
    def forward(self, x, r, clip, **kwargs):
        # Process the conditioning embeddings
        r_embed = self.gen_r_embedding(r)
        for c in self.t_conds:
            t_cond = kwargs.get(c, torch.ones_like(r))
            r_embed = torch.cat([r_embed, self.gen_r_embedding(t_cond)], dim=1)
        clip = self.gen_c_embeddings(clip)
        
        # Model Blocks
        x = self.embedding(x)
        level_outputs = self._down_encode(x, r_embed, clip)
        x = self._up_decode(level_outputs, r_embed, clip)
        return self.clf(x)
    
    def update_weights_ema(self, src_model, beta=0.999):
        for self_params, src_params in zip(self.parameters(), src_model.parameters()):
            self_params.data = self_params.data * beta + src_params.data * (1-beta)
    
