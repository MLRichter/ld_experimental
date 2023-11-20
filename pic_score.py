# import
import json
import os.path
from pathlib import Path
from typing import Tuple

import numpy as np
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import AutoProcessor, AutoModel
from PIL import Image
import torch
from joblib import Memory

mem = Memory(".cache", verbose=0)

# load model
device = "cuda"
processor_name_or_path = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
model_pretrained_name_or_path = "yuvalkirstain/PickScore_v1"

processor = AutoProcessor.from_pretrained(processor_name_or_path)
model = AutoModel.from_pretrained(model_pretrained_name_or_path).eval().to(device)


def calc_probs(prompt, images):
    # preprocess
    image_inputs = processor(
        images=images,
        padding=True,
        truncation=True,
        max_length=77,
        return_tensors="pt",
    ).to(device)

    text_inputs = processor(
        text=prompt,
        padding=True,
        truncation=True,
        max_length=77,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        # embed
        image_embs = model.get_image_features(**image_inputs)
        image_embs = image_embs / torch.norm(image_embs, dim=-1, keepdim=True)

        text_embs = model.get_text_features(**text_inputs)
        text_embs = text_embs / torch.norm(text_embs, dim=-1, keepdim=True)

        # score
        scores = model.logit_scale.exp() * (text_embs @ image_embs.T)[0]

        # get probabilities if you have multiple images to choose from
        probs = torch.softmax(scores, dim=-1)

    return probs.cpu().tolist()

@mem.cache
def get_score(image1, image2, caption):
    pil_images = [Image.open(image1),
                  Image.open(image2)]
    score = np.asarray(calc_probs(caption, pil_images))
    return score

def coco_caption_loader_pyarrow(data_path: str) -> Tuple[int, str]:
    import pandas as pd

    # Read the Parquet file into a pandas DataFrame
    df = pd.read_parquet(data_path, engine="pyarrow")
    content = df['caption'].values
    file_names = df['file_name'].values
    for i, (file_name, caption) in enumerate(zip(file_names, content)):
        yield file_name, caption


class DoubleDataset(Dataset):
    def __init__(self, dataset1: str, dataset2: str, parquet_file: str, transform=None):
        self.ds1 = []
        self.ds2 = []
        self.captions = []
        for file_name, caption in (coco_caption_loader_pyarrow(parquet_file)):
            image1 = os.path.join(dataset1, file_name)
            image2 = os.path.join(dataset2, file_name)
            if os.path.exists(image1) and os.path.exists(image2):
                self.ds1.append(image1)
                self.ds2.append(image2)
                self.captions.append(caption)
            else:
                print("nonexistant file:", file_name)
        print(len(self.ds1))
        print(len(self.ds2))
        print(len(self.captions))


    def get_class_label(self, image_name):
        # your method here
        return 0

    def __getitem__(self, index):
        return Image.open(self.ds1[index]), \
               Image.open(self.ds2[index]), \
               self.captions[index]

    def __len__(self):
        return len(self.ds1)


def main2(dataset1: str, dataset2: str, parquet_file: str = "../coco2017/coco_30k.parquet"):
    ds = DoubleDataset(dataset1=dataset1, dataset2=dataset2, parquet_file=parquet_file)
    ldr = DataLoader(ds, batch_size=1, num_workers=8)
    for img1, img2, caption in ldr:
        print(img1, img2, ldr)

def main(dataset1: str, dataset2: str, parquet_file: str = "../coco2017/coco_30k.parquet"):
    dataset1_name = Path(dataset1).name
    dataset2_name = Path(dataset2).name
    scores = np.zeros(2)
    num = len(list(coco_caption_loader_pyarrow(parquet_file)))
    pbar = tqdm(coco_caption_loader_pyarrow(parquet_file), total=num)
    total = 0
    picks = [0, 0]
    for i, (file_name, caption) in enumerate(pbar):
        image1 = os.path.join(dataset1, file_name) if os.path.exists(os.path.join(dataset1, file_name)) else Path(os.path.join(dataset1, file_name)).with_suffix(".png")
        image2 = os.path.join(dataset2, file_name) if os.path.exists(os.path.join(dataset2, file_name)) else Path(os.path.join(dataset2, file_name)).with_suffix(".png")
        if os.path.exists(image1)  and (os.path.exists(image2)):
            #pil_images = [Image.open(image1),
            #              Image.open(image2)]
            #score = np.asarray(calc_probs(caption, pil_images))
            score = get_score(image1, image2, caption)
            pick = np.argmax(score)
            picks[pick] += 1
            scores += score
            total += 1
            x = scores / total
            msg = {f'{dataset1_name} conf': x[0], f'{dataset2_name} conf': x[1],
                   f'{dataset1_name} pick ratio': picks[0] / total, f'{dataset2_name} pick ratio': picks[1] / total}
            pbar.set_postfix(msg)

        else:
            print("nonexistant file:", file_name, os.path.exists(image1), os.path.exists(image2))
    x = scores / total
    return {
        f'{dataset1_name} conf': x[0],
        f'{dataset2_name} conf': x[1],
        f'{dataset1_name} pick ratio': picks[0] / total,
        f'{dataset2_name} pick ratio': picks[1] / total
    }


if __name__ == '__main__':
    #parquet_file = "./results/partiprompts.parquet"
    #parquet_file = "../coco2017/long_context_val.parquet"
    parquet_file = "../coco2017/coco_30k.parquet"
    dataset1 = "output/wuerstchen_generated"
    datasets2 = [
        "output/wuerstchen_no_text_generated",
        "output/wuerstchen_no_prior_text_generated",
        #"output/ldm14_partiprompts_generated",
        #"output/df_gan_partiprompts_generated",
        #"output/GALIP_partiprompts_generated",
        #"output/sdxl_partiprompts_generated",
        #"output/sd14_partiprompts_generated",
        #"output/sd21_partiprompts_generated",
    ]
    final_result = {}
    for dataset2 in datasets2:
        result = main(dataset1, dataset2, parquet_file=parquet_file)
        final_result[dataset2] = result
        if (Path("results") / Path(Path(dataset1).with_suffix(".json").name)).exists():
            with (Path("results") / Path(Path(dataset1).with_suffix(".json").name)).open("r") as fp:
                previous_results = json.load(fp)
                previous_results.update(final_result)
        else:
            previous_results = final_result
        with (Path("results") / Path(Path(dataset1).with_suffix(".json").name)).open("w") as fp:
            json.dump(previous_results, fp)

#pil_images = [Image.open(r"C:\Users\matsl\Documents\ld_experimental\output\wuerstchen_generated\COCO_val2014_000000409181.jpg"), Image.open(r"C:\Users\matsl\Documents\ld_experimental\output\wuerstchen_base_generated\COCO_val2014_000000007961.jpg")]
#prompt = "A person holds a beverage and a chili dog."
#print(calc_probs(prompt, pil_images))