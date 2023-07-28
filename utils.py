import numpy as np
import torch
from torch import nn
import json 
import copy

# DATA FILTERS
class WebdatasetFilter():
    def __init__(self, min_size=512, max_pwatermark=0.5, aesthetic_threshold=5.0, unsafe_threshold=0.99, text_conditions=None): # {'min_words': 2, 'forbidden_words': ["www.", ".com", "http", "-", "_", ":", ";", "(", ")", "/", "%", "|", "?", "download", "interior", "kitchen", "chair", "getty", "how", "what", "when", "why", "laminate", "furniture", "hair", "dress", "clothing"]}):
        self.min_size = min_size
        self.max_pwatermark = max_pwatermark
        self.aesthetic_threshold = aesthetic_threshold
        self.unsafe_threshold = unsafe_threshold
        self.text_conditions = text_conditions 

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

class WebdatasetFilterHumans():
    def __init__(self, min_size=512, max_pwatermark=0.5, unsafe_threshold=0.99, text_conditions=None): # {'min_words': 2, 'forbidden_words': ["www.", ".com", "http", "-", "_", ":", ";", "(", ")", "/", "%", "|", "?", "download", "interior", "kitchen", "chair", "getty", "how", "what", "when", "why", "laminate", "furniture", "hair", "dress", "clothing"]}):
        self.min_size = min_size
        self.max_pwatermark = max_pwatermark
        self.unsafe_threshold = unsafe_threshold
        self.text_conditions = text_conditions 

    def __call__(self, x):
        try:
            if 'json' in x:
                x_json = json.loads(x['json'])
                filter_size = (x_json.get('original_width', 0.0) or 0.0) >= self.min_size and x_json.get('original_height', 0) >= self.min_size
                filter_watermark = (x_json.get('p_watermark', 1.0) or 1.0) <= self.max_pwatermark
                filter_unsafe = (x_json.get('p_nsfw', 1.0) or 1.0) <= self.unsafe_threshold
                if self.text_conditions is not None:
                    caption = x['combined_txt'].decode("utf-8") 
                    filter_min_words = len(caption.split(" ")) >= self.text_conditions['min_words']
                    filter_ord_128 = all([ord(c) < 128 for c in caption])
                    filter_forbidden_words = all([c not in caption.lower() for c in self.text_conditions['forbidden_words']])
                    filter_text = filter_min_words and filter_ord_128 and filter_forbidden_words
                else:
                    filter_text = True
                return filter_size and filter_watermark and filter_unsafe and filter_text
            else:
                return False
        except:
            return False

class WebdatasetFilterSize():
    def __init__(self, min_size=256): 
        self.min_size = min_size

    def __call__(self, x):
        if 'json' in x:
            x_json = json.loads(x['json'])
            return x_json.get('original_width', 0.0) >= self.min_size and x_json.get('original_height', 0) >= self.min_size
        else:
            return False

def fit_state_dict(source_state_dict_orig, target_state_dict_orig, resize_input_output = True):
    source_state_dict = copy.deepcopy(source_state_dict_orig)
    target_state_dict = copy.deepcopy(target_state_dict_orig)
    target_keys = [k for k in source_state_dict.keys() if "down_blocks" in k]
    target_levels = [int(k.split(".")[1]) for k in source_state_dict.keys() if "down_blocks" in k]
    src_levels = [int(k.split(".")[1]) for k in target_state_dict.keys() if "down_blocks" in k]
    shift = max(src_levels) - max(target_levels)
    if shift > 0:
        for k in target_keys:
            splits = k.split(".")
            idx = int(splits[1]) + 1
            splits[1] = f"{idx}"
            new_k = ".".join(splits)
            del source_state_dict[k]
            source_state_dict[new_k] = source_state_dict_orig[k]
    
    resized_params = []
    for key in target_state_dict.keys():
        if key in source_state_dict:
            src_shape = source_state_dict[key].shape
            tgt_shape = target_state_dict[key].shape
            if tgt_shape == src_shape:
                target_state_dict[key] = source_state_dict[key]
            elif all([d1 >= d2 for d1, d2 in zip(tgt_shape, src_shape)]):
                    if not resize_input_output and ("embedding." in key or "clf." in key):
                        continue
                    resized_params.append(key)
                    target_state_dict[key] *= 1e-9
                    if len(tgt_shape) == 1:
                        target_state_dict[key][:src_shape[0]] = source_state_dict[key]
                    elif len(tgt_shape) == 2:
                        target_state_dict[key][:src_shape[0], :src_shape[1]] = source_state_dict[key]
                    elif len(tgt_shape) == 3:
                        target_state_dict[key][:src_shape[0], :src_shape[1], :src_shape[2]] = source_state_dict[key]
                    elif len(tgt_shape) == 4:
                        target_state_dict[key][:src_shape[0], :src_shape[1], :src_shape[2], :src_shape[3]] = source_state_dict[key]
                    elif len(tgt_shape) == 5:
                        target_state_dict[key][:src_shape[0], :src_shape[1], :src_shape[2], :src_shape[3], :src_shape[4]] = source_state_dict[key]
                    elif len(tgt_shape) == 6:
                        target_state_dict[key][:src_shape[0], :src_shape[1], :src_shape[2], :src_shape[3], :src_shape[4], :src_shape[5]] = source_state_dict[key]     
                    else:
                        print("!", key, src_shape, tgt_shape)
            else:
                print("!!", key, src_shape, tgt_shape)

    return target_state_dict, resized_params

class RandomMask(nn.Module):
    def __init__(self, mask_value=0.0, mask_modes=['courtain', 'center', 'border', 'patches'], max_patches=7, append_mask=False):
        super().__init__()
        self.mask_value = mask_value
        self.mask_modes = mask_modes
        self.max_patches = max_patches
        self.append_mask = append_mask
        
    def forward(self, x):
        mask = x.new_zeros(x.size(0), 1, *x.shape[2:])
        for i in range(x.size(0)):
            mode = np.random.choice(self.mask_modes)
            if mode == 'courtain':
                edge = np.random.choice(['left', 'right', 'top', 'bottom'])
                pct = np.random.uniform(0.1, 0.7)
                if edge == 'left':
                    mask[i, :, :int(pct*mask.size(-2))] = 1
                elif edge == 'right':
                    mask[i, :, int(pct*mask.size(-2)):] = 1
                elif edge == 'top':
                    mask[i, :, :, :int(pct*mask.size(-1))] = 1
                elif edge == 'bottom':
                    mask[i, :, :, int(pct*mask.size(-1)):] = 1
            elif mode == 'center' or mode == 'border':
                pad_w, pad_h = int(np.random.uniform(0.1, 0.5)*mask.size(-2)), int(np.random.uniform(0.1, 0.5)*mask.size(-1))
                mask[i, :, pad_w:-pad_w, pad_h:-pad_h] = 1
                if mode == 'border':
                    mask[i] = 1 - mask[i]
            elif mode == 'patches':
                num_patches = np.random.randint(1, self.max_patches+1)
                for _ in range(num_patches):
                    sx, sy = int(np.random.uniform(0.01, 0.9)*mask.size(-1)), int(np.random.uniform(0.01, 0.9)*mask.size(-1))
                    px, py = int(np.random.uniform(0, 0.99)*mask.size(-1)), int(np.random.uniform(0, 0.99)*mask.size(-1))
                    mask[i, :, py:py+sy, px:px+sx]
        x = x * (1-mask) + mask * self.mask_value
        if self.append_mask:
            x = torch.cat([x, mask], dim=1)
        return x