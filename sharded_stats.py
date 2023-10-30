import click
import torchvision
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import json

import webdataset as wds
from webdataset.handlers import warn_and_continue


magic_norm = 0.18215
transforms = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Resize(512, interpolation=torchvision.transforms.InterpolationMode.BILINEAR, antialias=True),
    torchvision.transforms.CenterCrop(512),
    torchvision.transforms.Normalize([0.5], [0.5]),

])


class WebdatasetFilterCounter():
    def __init__(self, min_size=512, max_pwatermark=0.5, aesthetic_threshold=5.0, unsafe_threshold=0.99, text_conditions=None): # {'min_words': 2, 'forbidden_words': ["www.", ".com", "http", "-", "_", ":", ";", "(", ")", "/", "%", "|", "?", "download", "interior", "kitchen", "chair", "getty", "how", "what", "when", "why", "laminate", "furniture", "hair", "dress", "clothing"]}):
        self.min_size = min_size
        self.max_pwatermark = max_pwatermark
        self.aesthetic_threshold = aesthetic_threshold
        self.unsafe_threshold = unsafe_threshold
        self.text_conditions = text_conditions

        self.f_watermark = 0
        self.f_aesthetic_a = 0
        self.f_aesthetic_b = 0
        self.f_unsafe = 0
        self.f_size = 0
        self.total = 0
        self.total_filtered = 0

    def update_counters(self, filter_watermark, filter_aesthetics_a, filter_aesthetics_b, filter_unsafe, filter_size):
        self.f_watermark += filter_watermark
        self.f_aesthetic_a += (filter_aesthetics_a)
        self.f_aesthetic_b += (filter_aesthetics_b)
        self.f_unsafe += filter_unsafe
        self.f_size += filter_size
        self.total_filtered += filter_watermark or (filter_aesthetics_a and filter_aesthetics_b) or filter_unsafe or filter_size
        self.total += 1

    def checkpoint(self, savepath: str, part: int, shard_range: str):
        """Save the stats to a json file, including part and shard info in filename."""
        filename = f"filter_stats_part{part}_shard{shard_range}.json"
        full_savepath = os.path.join(savepath, filename)
        with open(full_savepath, "w") as fp:
            json.dump({
                "f_size": self.f_size,
                'f_watermark': self.f_watermark,
                'f_aesthetic_a': self.f_aesthetic_a,
                'f_aesthetic_b': self.f_aesthetic_b,
                'f_unsafe': self.f_unsafe,
                'total': self.total,
                "total_filtered": self.total_filtered,
            }, fp)


    def __call__(self, x_json):

            filter_size = x_json.get('original_width', 0.0) >= self.min_size and x_json.get('original_height', 0) >= self.min_size
            filter_watermark = x_json.get('pwatermark', 1.0) <= self.max_pwatermark
            filter_aesthetic_a = x_json.get('aesthetic', 0.0) >= self.aesthetic_threshold
            filter_aesthetic_b = x_json.get('AESTHETIC_SCORE', 0.0) >= self.aesthetic_threshold
            filter_unsafe = x_json.get('punsafe', 1.0) <= self.unsafe_threshold
            self.update_counters(
                filter_watermark=not filter_watermark,
                filter_aesthetics_a=not filter_aesthetic_a,
                filter_aesthetics_b=not filter_aesthetic_b,
                filter_unsafe=not filter_unsafe,
                filter_size=not filter_size
            )
            #print("watermark", x_json.get('pwatermark', 1.0) , self.max_pwatermark)
            #print("filter_aesthetic_a", x_json.get('aesthetic', 0.0), self.aesthetic_threshold)
            #print("filter_aesthetic_b", x_json.get('AESTHETIC_SCORE', 0.0), self.aesthetic_threshold)
            #print("punsafe", x_json.get('punsafe', 1.0), self.unsafe_threshold)
            #print()


def identity(x):
    return x

@click.command()
@click.option('--part', default=0, type=int, help='Part of the dataset to process.')
@click.option('--shard_range', default="00000", type=str, help='Range of shards for the dataset part.')
def main(part, shard_range):
    dataset_path = f"pipe:aws s3 cp s3://stability-west/laion-a-native-high-res/part-{part}/{shard_range}.tar -"

    # --- PREPARE DATASET ---
    # PREPARE DATASET
    dataset = wds.WebDataset(
            dataset_path, resampled=False, handler=warn_and_continue
    ).decode(
            "pilrgb", handler=warn_and_continue
    ).to_tuple(
            "jpg", "json", handler=warn_and_continue
    ).map_tuple(
            transforms, identity, handler=warn_and_continue
    )
    real_batch_size = 256
    dataloader = DataLoader(dataset, batch_size=real_batch_size, num_workers=8, pin_memory=True)
    print("prepping dataloader")
    #dataloader_iterator = dataloader
    dataloader_iterator = iter(dataloader)

    counter = WebdatasetFilterCounter(min_size=512, max_pwatermark=0.5, aesthetic_threshold=5.0, unsafe_threshold=0.99)
    for i, x in enumerate(tqdm(dataset)):
        to_be_counted = x[1]
        counter(to_be_counted)
        if (counter.total % 10000) == 0:
            counter.checkpoint(savepath="../stats/sharded/", part=part, shard_range=shard_range)

    # Note: When calling checkpoint, include part and shard_range:

    counter.checkpoint(savepath="../stats/sharded/", part=part, shard_range=shard_range)


if __name__ == "__main__":
    main()