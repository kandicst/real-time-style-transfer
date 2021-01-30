import yaml
import numpy as np
import math
from PIL import Image
from os import listdir, remove
from os.path import isfile, join, splitext

from matplotlib.axes import Axes

from datasets import IMG_EXTENSIONS
import matplotlib.pyplot as plt
from torch import Tensor
from typing import Callable, Tuple

Image.MAX_IMAGE_PIXELS = None


def load_yaml(path):
    with open(path, 'r') as stream:
        return yaml.load(stream)


def load_image(path) -> Image:
    img = Image.open(path)
    shape = np.array(img).shape
    if shape[-1] != 3 or len(shape) != 3:
        return img.convert('RGB')
    return img


def show_content_style_tradeoff_grid(content_img: Tensor, style_img: Tensor, fn: Callable, inverse_transform: Callable,
                                     alpha_min: float = 0.,
                                     alpha_max: float = 1.,
                                     alpha_step: float = 0.25):
    num_cols = math.ceil((alpha_max - alpha_min) / alpha_step) + 3
    f, axarr = plt.subplots(1, num_cols, figsize=(50, 100))

    # set first and last column as images
    set_axis_text(axarr[0], 'Content Image')
    content_img_pil: Image = inverse_transform(content_img.cpu())
    axarr[0].imshow(content_img_pil)

    set_axis_text(axarr[num_cols - 1], 'Style Image')
    style_img_pil: Image = inverse_transform(style_img.cpu())
    style_img_pil = style_img_pil.resize(content_img_pil.size)
    axarr[num_cols - 1].imshow(style_img_pil)

    for i in range(1, num_cols - 1):
        set_axis_text(axarr[i], f'\u03B1 = {alpha_min}')
        axarr[i].imshow(fn(content_img, style_img, alpha_min))
        alpha_min += alpha_step

    plt.tight_layout()
    plt.show()
    plt.figure()


def set_axis_text(ax: Axes, text: str, size: int = 36):
    ax.axis('off')
    ax.text(0.5, -0.1, text, size=size, ha="center", transform=ax.transAxes)


def show_style_and_content_img(content_img: Tensor, style_img: Tensor, inverse_transform: Callable, figsize=(7, 14)):
    f, axarr = plt.subplots(1, 2, figsize=figsize)
    axarr[0].axis('off')
    axarr[1].axis('off')
    axarr[0].set_title('Content Image')
    axarr[1].set_title('Style Image')
    cont_image_pil = inverse_transform(content_img.cpu())
    axarr[0].imshow(cont_image_pil)
    axarr[1].imshow(inverse_transform(style_img.cpu()).resize(cont_image_pil.size))

    plt.tight_layout()


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


if __name__ == '__main__':
    load_image('')