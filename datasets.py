from collections import deque

from os import listdir
from os.path import isfile, join, splitext

from PIL import Image
from PIL import ImageFile

import numpy as np

import torch
from torch.utils.data import Dataset
from torchvision import transforms as T
from torch import Tensor

from typing import Optional, Callable

ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None

IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')


class MyDataset(Dataset):

    def __init__(self, root_dir: str, transform: Optional[Callable] = None, img_limit: Optional[int] = None):
        super(MyDataset, self).__init__()
        self.root_dir = root_dir
        if transform is None:
            self.transform = T.Compose([T.ToTensor()])
        else:
            self.transform = transform

        # parse all images from root_dir
        self.img_names = deque()
        for f in listdir(root_dir):
            if len(self.img_names) == img_limit:
                break
            if isfile(join(root_dir, f)) and splitext(f)[-1] in IMG_EXTENSIONS:
                self.img_names.append(f)

    def __getitem__(self, index: int) -> Tensor:
        name = self.img_names[index]
        x = self.transform(self.load_image(join(self.root_dir, name)))
        return x

    def get_no_transform(self, index: int) -> Tensor:
        name = self.img_names[index]
        return T.ToTensor()(self.load_image(join(self.root_dir, name)))

    def load_image(self, path) -> Image:
        img = Image.open(path)
        shape = np.array(img).shape
        if shape[-1] != 3 or len(shape) != 3:
            return img.convert('RGB')
        return img

    def name(self) -> str:
        return "My Dataset"

    def __len__(self) -> int:
        return len(self.img_names)


class CachedDataset(MyDataset):

    def __init__(self, root_dir: str, transform, img_limit: Optional[int], max_cache_size: int,
                 use_cache: bool = False):
        super(CachedDataset, self).__init__(root_dir, transform, img_limit)

        self.use_cache = use_cache
        self.max_cache_size = self.cache_available = max_cache_size
        self.cache = []

    def __getitem__(self, index) -> Tensor:
        name = self.img_names[index]
        if not self.use_cache:
            img = self.load_image(join(self.root_dir, name))
            if self.cache_available > 0:
                self.cache.append(img)
            return self.transform(self.cache[-1])
        else:
            if index > len(self.img_names) - 1:
                return self.transform(self.load_image(join(self.root_dir, name)))
            return self.transform(self.cache[index])

    def set_use_cache(self, use_cache):
        if use_cache:
            self.cache = torch.stack(tuple(self.cache))
        else:
            self.cache = []
        self.use_cache = use_cache

    def name(self) -> str:
        return "Cached Dataset"


if __name__ == '__main__':
    ds = MyDataset(root_dir='data/wikiart', transform=None, img_limit=100)
    print(ds[1])

    transform = T.Compose([T.Resize(256), T.ToTensor()])

    cache_ds = CachedDataset(root_dir='data/wikiart', transform=transform, img_limit=500, max_cache_size=1)
    print(cache_ds[0])
    print(cache_ds[42])
