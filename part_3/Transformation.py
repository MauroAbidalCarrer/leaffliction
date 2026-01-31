import os
import sys
import argparse

import torch
import torchvision
import plotly.express as px
from torch import Tensor, nn
import torch.nn.functional as F


def main():
    if len(sys.argv) != 2:
        print("Provide path to image.")
        exit(1)
    img = torchvision.io.decode_image(sys.argv[1])
    img = (img.permute(1, 2, 0) / 255).unsqueeze(0)
    mask = green_adaptive_threshold(img, 0.15, torch.float)
    colored_mask = mask.repeat(1, 1, 1, 3)
    imgs = torch.concat((
        img,
        gaussian_blur(img),
        colored_mask,
        horizontal_diff(img),
        vertical_diff(img),
        diff_intensity(img),
        grayscale_intensity(img).repeat(1, 1, 1, 3),
    ))
    fig = px.imshow(imgs, facet_col=0)
    titles = [
        "original",
        "blured",
        "green adaptive threshold",
        "horizontal diff",
        "vertical diff",
        "diff mean",
        "grayscale",
    ]
    fig.for_each_annotation(
        lambda a: a.update(text=titles[int(a.text.split("=")[-1])])
    )
    fig.show()

gaussian_blur = torchvision.transforms.GaussianBlur(5, 3)

def diff_intensity(x: Tensor) -> Tensor:
    h_diff = horizontal_diff(x)
    v_diff = vertical_diff(x)
    return (h_diff + v_diff) / 2

def grayscale_intensity(x: Tensor) -> Tensor:
    return x.sum(dim=-1, keepdim=True) / 3

def green_adaptive_threshold(
        x: Tensor, 
        quantile_threshold: float,
        dtype: torch.dtype=torch.bool,
    ) -> Tensor:
    threshold = torch.quantile(x[..., 1], quantile_threshold)
    return (x[..., 1] > threshold).to(dtype=dtype).unsqueeze(-1)

def horizontal_diff(x: Tensor) -> Tensor:
    h_diff = x[:, :, 1:] - x[:, :, :-1]
    return F.pad(h_diff, (0, 0, 1, 0))

def vertical_diff(x: Tensor) -> Tensor:
    v_diff = x[:, 1:] - x[:, :-1]
    return F.pad(
        v_diff,
        (
            0, 0,
            0, 0,
            1, 0,
        )
    )
    
if __name__ == "__main__":
    main()