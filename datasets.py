from collections import deque

from os import listdir
from os.path import isfile, join, splitext

from PIL import Image
import numpy as np

import torch
from torch.utils.data import Dataset
from torchvision import transforms as tf
from torch import Tensor

from typing import Optional, Callable

IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')


class MyDataset(Dataset):

    def __init__(self, root_dir: str, transform: Optional[Callable] = None, img_limit: Optional[int] = None):
        super(MyDataset, self).__init__()
        self.root_dir = root_dir
        if transform is None:
            self.transform = tf.Compose([tf.ToTensor()])
        else:
            self.transform = transform

        # parse all images from root_dir
        self.img_names = deque()
        for f in listdir(root_dir):
            if len(self.img_names) == img_limit:
                break
            if isfile(join(root_dir, f)) and splitext(f)[-1] in IMG_EXTENSIONS:
                # x = tf.ToTensor()(self.load_image(join(root_dir,f)))
                # if x.shape[0] == 4:
                #     print(join(root_dir, f))
                # if len(self.img_names) % 100 == 0:
                #     print(len(self.img_names))
                self.img_names.append(f)

    def __getitem__(self, index: int) -> Tensor:
        name = self.img_names[index]
        x = self.transform(self.load_image(join(self.root_dir, name)))
        return x

    def load_image(self, path) -> Image:
        img = Image.open(path)
        if len(np.array(img).shape) == 2:
            img = img.convert('RGB')
        return img

    def name(self) -> str:
        return "My Datset"

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
                # self.cache.append(img)
                self.cache.append(self.transform(img))
            # return self.transform(img)
            return self.cache[-1]
        else:
            if index > len(self.img_names) - 1:
                return self.transform(self.load_image(join(self.root_dir, name)))
            # return self.transform(self.cache[index])
            return self.cache[index]

    def set_use_cache(self, use_cache):
        if use_cache:
            self.cache = torch.stack(tuple(self.cache))
        else:
            self.cache = []
        self.use_cache = use_cache

    def name(self) -> str:
        return "Cached Datset"


if __name__ == '__main__':
    ds = MyDataset(root_dir='data/wikiart', transform=None, img_limit=100)
    print(ds[1])

    transform = tf.Compose([
        tf.Resize(256),  # rescale
        tf.ToTensor(),  # convert to [0, 1] range
    ])

    cache_ds = CachedDataset(root_dir='data/wikiart', transform=transform, img_limit=500, max_cache_size=1)
    print(cache_ds[0])
    print(cache_ds[42])
