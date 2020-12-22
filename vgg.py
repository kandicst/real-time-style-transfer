from torch import nn
import torch


def get_vgg():
    vgg = nn.Sequential(
        nn.Conv2d(3, 3, (1, 1)),
        nn.Conv2d(3, 64, (3, 3), padding=1, padding_mode='reflect'),
        nn.ReLU(inplace=True),  # relu1-1
        nn.Conv2d(64, 64, (3, 3), padding=1, padding_mode='reflect'),
        nn.ReLU(inplace=True),  # relu1-2
        nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),

        nn.Conv2d(64, 128, (3, 3), padding=1, padding_mode='reflect'),
        nn.ReLU(inplace=True),  # relu2-1
        nn.Conv2d(128, 128, (3, 3), padding=1, padding_mode='reflect'),
        nn.ReLU(inplace=True),  # relu2-2
        nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),

        nn.Conv2d(128, 256, (3, 3), padding=1, padding_mode='reflect'),
        nn.ReLU(inplace=True),  # relu3-1
        nn.Conv2d(256, 256, (3, 3), padding=1, padding_mode='reflect'),
        nn.ReLU(inplace=True),  # relu3-2
        nn.Conv2d(256, 256, (3, 3), padding=1, padding_mode='reflect'),
        nn.ReLU(inplace=True),  # relu3-3
        nn.Conv2d(256, 256, (3, 3), padding=1, padding_mode='reflect'),
        nn.ReLU(inplace=True),  # relu3-4
        nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),

        nn.Conv2d(256, 512, (3, 3), padding=1, padding_mode='reflect'),
        nn.ReLU(inplace=True),  # relu4-1
        nn.Conv2d(512, 512, (3, 3), padding=1, padding_mode='reflect'),
        nn.ReLU(),  # relu4-2
        nn.Conv2d(512, 512, (3, 3), padding=1, padding_mode='reflect'),
        nn.ReLU(),  # relu4-3
        nn.Conv2d(512, 512, (3, 3), padding=1, padding_mode='reflect'),
        nn.ReLU(),  # relu4-4
        nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),

        nn.Conv2d(512, 512, (3, 3), padding=1, padding_mode='reflect'),
        nn.ReLU(),  # relu5-1
        nn.Conv2d(512, 512, (3, 3), padding=1, padding_mode='reflect'),
        nn.ReLU(),  # relu5-2
        nn.Conv2d(512, 512, (3, 3), padding=1, padding_mode='reflect'),
        nn.ReLU(),  # relu5-3
        nn.Conv2d(512, 512, (3, 3), padding=1, padding_mode='reflect'),
        nn.ReLU()  # relu5-4
    )
    vgg.load_state_dict(torch.load("saved_models/encoderVGG_weights.pt"))
    return vgg
