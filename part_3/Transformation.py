import os
import argparse

import torch
import torchvision
from torch import Tensor
import plotly.express as px
import matplotlib.pyplot as plt
import torch.nn.functional as F


PTH_NOT_FILE_ERR = "--{pth} must be a file when --dst is not provided"
gaussian_blur = torchvision.transforms.GaussianBlur(5, 3)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True)
    parser.add_argument("--dst", default=None)
    args = parser.parse_args()

    if args.dst is None:
        assert os.path.isfile(args.src), PTH_NOT_FILE_ERR.format(pth="src")
        plt_transforms_of_single_img(args.src)
    else:
        assert os.path.isdir(args.src), PTH_NOT_FILE_ERR.format(pth="dst")
        transform_dataset(args.src, args.dst)


def plt_transforms_of_single_img(img_path: str) -> None:
    img = torchvision.io.decode_image(img_path)
    img = (img.permute(1, 2, 0) / 255).unsqueeze(0)  # (1, H, W, C)
    imgs = apply_transforms(img).squeeze(0)        # (T, H, W, C)

    titles = [
        "original",
        "blurred",
        "green adaptive threshold",
        "horizontal diff",
        "vertical diff",
        "diff mean",
        "grayscale",
    ]

    T = imgs.shape[0]
    fig, axes = plt.subplots(1, T, figsize=(3 * T, 3))

    # Handle the case T == 1
    if T == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        ax.imshow(imgs[i])
        ax.set_title(titles[i])
        ax.axis("off")

    plt.tight_layout()
    plt.show()


def transform_dataset(src_dir: str, dst: str) -> None:
    img_paths = [
        os.path.join(src_dir, f)
        for f in sorted(os.listdir(src_dir))
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]
    assert len(img_paths) > 0, "No images found in src directory."

    transform_names = [
        "original",
        "blurred",
        "green_adaptive_threshold",
        "horizontal_diff",
        "vertical_diff",
        "diff_mean",
        "grayscale",
    ]

    # Create output directories
    for name in transform_names:
        os.makedirs(os.path.join(dst, name), exist_ok=True)

    imgs = []
    for p in img_paths:
        img = torchvision.io.decode_image(p)
        img = img.permute(1, 2, 0) / 255  # (H, W, C)
        imgs.append(img)

    batch = torch.stack(imgs, dim=0)      # (B, H, W, C)

    transformed = apply_transforms(batch)  # (B, T, H, W, C)

    for t, name in enumerate(transform_names):
        imgs_t = transformed[:, t]        # (B, H, W, C)

        # Convert to uint8 CHW for PNG writing
        imgs_t = (imgs_t.clamp(0, 1) * 255).to(torch.uint8)
        imgs_t = imgs_t.permute(0, 3, 1, 2)  # (B, C, H, W)

        for i, src_path in enumerate(img_paths):
            base = os.path.splitext(os.path.basename(src_path))[0]
            out_path = os.path.join(dst, name, f"{base}.png")
            torchvision.io.write_png(imgs_t[i], out_path)


def plt_transformed_dataset(
        pt_path: str,
        num_samples: int = 4,
) -> None:
    data = torch.load(pt_path)  # (B, T, H, W, C)
    assert data.ndim == 5, "Expected tensor of shape (B, T, H, W, C)"

    B, T, H, W, C = data.shape
    num_samples = min(num_samples, B)

    # Select first N samples
    imgs = data[:num_samples]          # (N, T, H, W, C)
    imgs = imgs.reshape(-1, H, W, C)    # (N*T, H, W, C)

    fig = px.imshow(imgs, facet_col=0, facet_col_wrap=7)
    fig.show()


def apply_transforms(img: Tensor) -> Tensor:
    mask = green_adaptive_threshold(img, 0.15, torch.float)
    colored_mask = mask.repeat(1, 1, 1, 3)
    transforms = [
        img,
        gaussian_blur(img),
        colored_mask,
        horizontal_diff(img),
        vertical_diff(img),
        diff_intensity(img),
        grayscale_intensity(img).repeat(1, 1, 1, 3),
    ]
    return torch.stack(transforms, dim=1)


def diff_intensity(x: Tensor) -> Tensor:
    h_diff = horizontal_diff(x)
    v_diff = vertical_diff(x)
    return (h_diff + v_diff) / 2


def grayscale_intensity(x: Tensor) -> Tensor:
    return x.sum(dim=-1, keepdim=True) / 3


def green_adaptive_threshold(
        x: Tensor,
        quantile_threshold: float,
        dtype: torch.dtype = torch.bool,
) -> Tensor:
    x_green = x[..., 1]
    threshold = torch.quantile(
        x_green.flatten(1),
        quantile_threshold,
        dim=1,
    )
    return (
        (x_green > threshold.reshape(-1, 1, 1))
        .to(dtype=dtype)
        .unsqueeze(-1)
    )


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
    try:
        main()
    except Exception as e:
        print("Error: ", e)