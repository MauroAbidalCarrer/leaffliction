import os
import urllib.request
import zipfile
import shutil

import torch
from torch.utils.data import (
    Subset,
    DataLoader,
    TensorDataset,
)
from torch import Tensor

import leaffliction.constants as consts
from leaffliction.constants import LABEL2ID, DEVICE, TRAINING, DATA, PATHS
from leaffliction.utils import load_image_as_tensor


def downloadZip():
    if os.path.exists("dataset.zip"):
        os.remove("dataset.zip")
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
                z.extractall()
                print("Unzip Successful")
                replaceData()
    except zipfile.BadZipFile:
        print("Error: The file is not a zip file or is corrupted.")
    except FileNotFoundError:
        print("Error: The zip file was not found.")
    except PermissionError:
        print("Error: You don't have permission to write to that folder.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


def replaceData():
    if os.path.exists(PATHS["dataset_dir"]):
        shutil.rmtree(PATHS["dataset_dir"])
        print("Old data removed")
    os.rename("images", PATHS["dataset_dir"])
    os.remove("dataset.zip")

def get_dataset_for_training() -> dict[str, Tensor]:
    imgs_lst: list[Tensor] = []
    labels_lst: list[int] = []
    for img_class in os.listdir(PATHS["training_dataset_dir"]):
        class_idx = LABEL2ID[img_class]
        for img in os.listdir(os.path.join(PATHS["training_dataset_dir"], img_class)):
            img_pth = os.path.join(PATHS["training_dataset_dir"], img_class, img)
            img = load_image_as_tensor(img_pth)
            imgs_lst.append(img)
            labels_lst.append(class_idx)

    raw_imgs = torch.stack(imgs_lst, dim=0) # dataset_size, C, H, W
    labels = torch.IntTensor(labels_lst)
    
    return raw_imgs, labels

def mk_data_loaders(
    x: Tensor,
    y: Tensor,
    val_fraction=DATA["val_fraction"],
    batch_size=TRAINING["batch_size"],
    seed: int=DATA["seed"]
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