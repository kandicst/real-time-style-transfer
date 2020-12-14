from os import listdir
from os.path import isfile, join, splitext

from PIL import Image
import numpy as np
from torch.utils.data import Dataset

from typing import Optional, Callable

IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')


class MyDataset(Dataset):

    def __init__(self, root_dir: str, transforms: Optional[Callable], img_limit: Optional[int]):
        super(MyDataset, self).__init__()
        self.root_dir = root_dir
        self.transforms = transforms

        # parse all images from root_dir
        self.img_names = []
        for f in listdir(root_dir):
            if isfile(join(root_dir, f)) and splitext(f)[-1] in IMG_EXTENSIONS:
                self.img_names.append(f)
            if len(self.img_names) == img_limit:
                break

    def __getitem__(self, index: int):
        name = self.img_names[index]
        if self.transforms:
            return self.transforms(self.load_image(join(self.root_dir, name)))

        return self.load_image(join(self.root_dir, name))

    def load_image(self, path):
        img = Image.open(path)
        if len(np.array(img).shape) == 2:
            img = img.convert('RGB')
        return img

    def __len__(self):
        return len(self.img_names)


class CacheDataset(MyDataset):

    def __init__(self, root_dir: str, transforms, img_limit: Optional[int], max_cache_size: int):
        super(CacheDataset, self).__init__(root_dir, transforms, img_limit)

        self.max_cache_size = self.cache_available = max_cache_size
        self.cache = dict()

    def __getitem__(self, index):
        name = self.img_names[index]

        if name in self.cache:
            return self.transforms(self.cache[name]) if self.transforms else self.cache[name]

        # add to cache
        if self.cache_available > 0:
            self.cache[name] = self.load_image(join(self.root_dir, name))
            self.cache_available -= 1
            return self.cache[name]

        # cache is full
        return super(CacheDataset, self).__getitem__(index)


if __name__ == '__main__':
    ds = MyDataset(root_dir='data/wikiart', transforms=None, img_limit=10000)

    from torchvision import transforms

    transform = transforms.Compose([
        transforms.Resize(256),  # rescale
        transforms.ToTensor(),  # convert to [0, 1] range
    ])

    cache_ds = CacheDataset(root_dir='data/wikiart', transforms=transform, img_limit=500, max_cache_size=1)
    print(cache_ds[0])
    print(cache_ds[42])
