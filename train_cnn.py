import os
from typing import Any
from itertools import pairwise, repeat

import torch
import torchvision
import pandas as pd
import plotly.express as px
from torch import nn, Tensor

LABEL2ID = {
    "Apple_Black_rot": 0,
    "Apple_rust": 1,
    "Apple_healthy": 2,
    "Apple_scab": 3,
    "Grape_Black_rot": 4,
    "Grape_Esca": 5,
    "Grape_healthy": 6,
    "Grape_spot": 7,
}
DEVICE = torch.device("cuda")
BATCH_SIZE = 32


class CNN(nn.Module):
    def __init__(
        self,
        kernels_per_layer: list[int],
        mlp_width: int,
        mlp_depth: int,
        n_classes: int,
    ):
        super().__init__()
        conv_layers = []
        channels = [3] + kernels_per_layer
        for in_channels, out_channels in pairwise(channels):
            conv_layer = nn.Conv2d(
                in_channels,
                out_channels,
                5,
                padding=2,
            )
            conv_layers.append(conv_layer)
        self.conv_layers = nn.ModuleList(conv_layers)
        self.batch_norms = nn.ModuleList([nn.LazyBatchNorm2d() for _ in range(len(conv_layers))])
        
        self.linear_layers = []
        for width in repeat(mlp_width, mlp_depth - 1):
            self.linear_layers.append(nn.LazyLinear(width))
        self.linear_layers.append(nn.LazyLinear(n_classes))
        self.linear_layers = nn.ModuleList(self.linear_layers)
    
    def forward(self, x: Tensor) -> Tensor:
        for layer_idx, (b_norm, conv) in enumerate(zip(self.batch_norms, self.conv_layers)):
            x = b_norm(x)
            x = conv(x)
            x = nn.functional.relu(x)
            x = nn.functional.max_pool2d(x, 2)

        x = x.flatten(1)
        for layer_idx, linear_layer in enumerate(self.linear_layers):
            x = linear_layer(x)
            if layer_idx != len(self.linear_layers) - 1:
                x = nn.functional.relu(x)
        return x


def training_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    criterion,
    x: Tensor,
    y: Tensor,
) -> dict[str, float]:
    x = x.to(dtype=torch.float32, device=DEVICE)
    y = y.to(dtype=torch.long, device=DEVICE)
    optimizer.zero_grad()
    model_output = model(x)
    # loss = nn.functional.cross_entropy(model_output, y)
    loss = criterion(model_output, y)
    loss.backward()
    loss_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    with torch.no_grad():
        accuracy = (model_output.argmax(dim=-1) == y).float().mean()
    return {
        "loss": loss.item(),
        "accuracy": accuracy.item(),
        "loss_norm": loss_norm.item(),
    }

def train_model_for_single_epoch(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    data_loader: torch.utils.data.DataLoader,
    criterion,
) -> list[dict]:
    model = model.train()
    step_dicts = []
    for x, y in data_loader:
        step_dict = training_step(model, optimizer, criterion, x, y)
        step_dicts.append(step_dict)
    return step_dicts

def train_model(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    data_loader: torch.utils.data.DataLoader,
    criterion,
    n_epochs: int,
) -> pd.DataFrame:
    training_data = []
    for epoch in range(n_epochs):
        epoch_data = train_model_for_single_epoch(\
            model,
            optimizer,
            data_loader,
            criterion,
        )
        training_data.extend(epoch_data)
    return pd.DataFrame.from_records(training_data)


if __name__ == "__main__":
    print("Making dataset and loader")
    imgs_lst: list[Tensor] = []
    labels_lst: list[int] = []
    for img_class in os.listdir("dataset"):
        class_idx = LABEL2ID[img_class]
        for img in os.listdir(os.path.join("dataset", img_class)):
            img_pth = os.path.join("dataset", img_class, img)
            img = torchvision.io.decode_image(img_pth)
            imgs_lst.append(img)
            labels_lst.append(class_idx)

    raw_imgs = torch.stack(imgs_lst, dim=0) # dataset_size, C, H, W
    labels = torch.IntTensor(labels_lst)
    preprocessed_imgs = (
        raw_imgs #dataset_size, C, H, W uint8
        .to(dtype=torch.bfloat16)  #dataset_size, H, W, C float 32
    )
    preprocessed_imgs = (preprocessed_imgs - preprocessed_imgs.mean(dim=3, keepdim=True)) / preprocessed_imgs.std(dim=3, keepdim=True)
    dataset = torch.utils.data.TensorDataset(preprocessed_imgs, labels)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        BATCH_SIZE,
        shuffle=True,
    )

    print("Making model")
    model = (
        CNN(
            kernels_per_layer=[32, 64, 128, 256],
            mlp_width=128,
            mlp_depth=3,
            n_classes=len(LABEL2ID)
        )
        .to(device=DEVICE)
    )
    print("making optim and loss")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()
    print("Training model")
    step_dicts = train_model(model, optimizer, data_loader, criterion, 2)
    print("nan loss value count:", step_dicts["loss"].isna().value_counts())
    print("best step", step_dicts.aggregate({"loss": "min", "accuracy": "max"}))


