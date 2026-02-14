import os
import zipfile
import shutil
import urllib.request

import torch
from torch import Tensor
from torch.utils.data import (
    Subset,
    DataLoader,
    TensorDataset,
)

import leaffliction.constants as consts
from leaffliction.utils import load_image_as_tensor
from leaffliction.constants import LABEL2ID, DEVICE, TRAINING, DATA, PATHS


def downloadZip():
    print("Downloading Dataset Zip")
    urllib.request.urlretrieve(consts.PATHS["dataset_url"], "dataset.zip")
    print("Download Completed")
    unzipData()


def unzipData():
    try:
        with zipfile.ZipFile("dataset.zip", "r") as z:
            if z.testzip() is not None:
                print("Zip file is Corrupted")
            else:
                z.extractall(PATHS["dataset_dir"])
                print("Unzip Successful")
    except zipfile.BadZipFile:
        print("Error: The file is not a zip file or is corrupted.")
    except FileNotFoundError:
        print("Error: The zip file was not found.")
    except PermissionError:
        print("Error: You don't have permission to write to that folder.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


def get_dataset_for_training() -> dict[str, Tensor]:
    imgs_lst: list[Tensor] = []
    labels_lst: list[int] = []
    for img_class in os.listdir(PATHS["training_dataset_dir"]):
        class_idx = LABEL2ID[img_class]
        dataset_dir = PATHS["training_dataset_dir"]
        for img in os.listdir(os.path.join(dataset_dir, img_class)):
            img_pth = os.path.join(dataset_dir, img_class, img)
            img = load_image_as_tensor(img_pth)
            imgs_lst.append(img)
            labels_lst.append(class_idx)

    raw_imgs = torch.stack(imgs_lst, dim=0)  # dataset_size, C, H, W
    labels = torch.IntTensor(labels_lst)

    return raw_imgs, labels


def mk_data_loaders(
    x: Tensor,
    y: Tensor,
    val_fraction=DATA["val_fraction"],
    batch_size=TRAINING["batch_size"],
    seed: int = consts.SEED,
) -> dict[str, Tensor]:
    torch.manual_seed(seed)

    train_idx = []
    val_idx = []

    for cls in torch.unique(y):
        cls_idx = torch.where(y == cls)[0]
        cls_idx = cls_idx[torch.randperm(len(cls_idx))]

        n_val = int(len(cls_idx) * val_fraction)
        val_idx.append(cls_idx[:n_val])
        train_idx.append(cls_idx[n_val:])

    train_idx = torch.cat(train_idx)
    val_idx = torch.cat(val_idx)
    full_dataset = TensorDataset(x, y)
    train_dataset = Subset(full_dataset, train_idx)
    val_dataset = Subset(full_dataset, val_idx)
    return (
        DataLoader(train_dataset, batch_size, shuffle=True),
        DataLoader(val_dataset, batch_size),
    )


def preprocess_batch(x: Tensor, y: Tensor) -> tuple[Tensor, Tensor]:
    return (
        x.to(device=DEVICE, dtype=torch.bfloat16),
        y.to(device=DEVICE, dtype=torch.long),
    )


def ensure_dataset_present():
    """Check if dataset exists, download if not present."""
    if not os.path.exists(PATHS["training_dataset_dir"]):
        print("Dataset not found. Downloading...")
        downloadZip()
    else:
        print("Dataset already present.")


if __name__ == "__main__":
    downloadZip()
