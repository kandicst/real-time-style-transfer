import yaml
import numpy as np
from PIL import Image


def load_yaml(path):
    with open(path, 'r') as stream:
        return yaml.load(stream)


def load_image(path) -> Image:
    img = Image.open(path)
    if len(np.array(img).shape) == 2:
        img = img.convert('RGB')
    return img
