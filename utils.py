import yaml
import numpy as np

from PIL import Image

Image.MAX_IMAGE_PIXELS = None

from os import listdir, remove
from os.path import isfile, join, splitext

from datasets import IMG_EXTENSIONS


def load_yaml(path):
    with open(path, 'r') as stream:
        return yaml.load(stream)


def load_image(path) -> Image:
    img = Image.open(path)
    shape = np.array(img).shape
    if shape[-1] != 3 or len(shape) != 3:
        return img.convert('RGB')
    return img


def clean_img_folder(root_dir):
    for idx, f in enumerate(listdir(root_dir)):
        if isfile(join(root_dir, f)) and splitext(f)[-1] in IMG_EXTENSIONS:
            shape = np.array(load_image(join(root_dir, f))).shape
            if shape[-1] != 3 or len(shape) != 3:
                file_path = join(root_dir, f)
                try:
                    remove(file_path)
                    print(f'{file_path} removed from dataset')
                except OSError:
                    print(f'Failed to remove {file_path}')
            if idx and idx % 500 == 0:
                print(idx)
