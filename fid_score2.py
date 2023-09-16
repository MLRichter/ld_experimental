import os.path

from PIL import Image
from torchmetrics.image.fid import FrechetInceptionDistance
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Compose, ToTensor, Resize, CenterCrop, Lambda
from tqdm import tqdm

def to_rgb(x):
    return x.repeat(3, 1, 1) if x.size(0) == 1 else x

transform = Compose([
    Resize(size=(512, 512)),
    ToTensor(),
    Lambda(to_rgb)
])


class MyDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = [os.path.join(image_paths, file) for file in os.listdir(image_paths)]
        self.transform = transform

    def get_class_label(self, image_name):
        # your method here
        return 0

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        x = Image.open(image_path)
        y = self.get_class_label(image_path.split('/')[-1])
        if self.transform is not None:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.image_paths)


def generate_dataset(ds_path: str):
    return DataLoader(MyDataset(ds_path, transform), batch_size=32, shuffle=False, num_workers=4)


def main(dataset_1: str, dataset_2: str):
    ds1 = generate_dataset(dataset_1)
    ds2 = generate_dataset(dataset_2)
    fid_score = FrechetInceptionDistance(normalize=True)

    for (img1, _), (img2, _) in zip(tqdm(ds1), ds2):
        fid_score.update(img1, real=True)
        fid_score.update(img2, real=False)
    print("FID Score:", fid_score.compute())





if __name__ == '__main__':
    main('../coco2017/coco_subset/', 'output/sd14_generated/')