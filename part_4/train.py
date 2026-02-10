import os
import zipfile
import torch
from torch import nn
from leaffliction.constants import (
    LABEL2ID,
    DEVICE,
    PATHS,
    MODEL,
    TRAINING,
)
from leaffliction.dataset import get_raw_dataset, mk_data_loaders, ensure_dataset_present
from leaffliction.models import CNN
from leaffliction.training import Trainer

if __name__ == "__main__":
    ensure_dataset_present()
    
    print("Making dataset and loader")
    raw_x, raw_y = get_raw_dataset()
    train_dl, val_dl = mk_data_loaders(raw_x, raw_y)

    print("Making model")
    model = (
        CNN(
            kernels_per_layer=MODEL["kernels_per_layer"],
            mlp_width=MODEL["mlp_width"],
            mlp_depth=MODEL["mlp_depth"],
            n_classes=MODEL["n_classes"]
        )
        .to(device=DEVICE)
    )
    
    print("Making optim and loss")
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=TRAINING["learning_rate"]
    )
    
    criterion = nn.CrossEntropyLoss()
    
    print("Training model")
    run = (
        Trainer(model, optimizer)
        .train_model(train_dl, val_dl, criterion, TRAINING["n_epochs"])
    )
    
    print(f"Saving model to {PATHS['model']}")
    torch.save(model.state_dict(), PATHS["model"])
    
    print(f"Zipping model to {PATHS['zip']}")
    with zipfile.ZipFile(PATHS["zip"], 'w', zipfile.ZIP_DEFLATED) as zipf:
        zipf.write(PATHS["model"], os.path.basename(PATHS["model"]))
    
    print("Model saved and zipped successfully!")
