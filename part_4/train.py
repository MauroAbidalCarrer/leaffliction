import os
import zipfile
import torch
from torch import nn
from leaffliction.constants import (
    LABEL2ID,
    DEVICE,
    N_EPOCHS,
    MODEL_PATH,
    ZIP_PATH,
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
            kernels_per_layer=[32, 64, 128, 256],
            mlp_width=128,
            mlp_depth=3,
            n_classes=len(LABEL2ID)
        )
        .to(device=DEVICE)
    )
    
    print("Making optim and loss")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    criterion = nn.CrossEntropyLoss()
    
    print("Training model")
    run = (
        Trainer(model, optimizer)
        .train_model(train_dl, val_dl, criterion, N_EPOCHS)
    )
    
    print(f"Saving model to {MODEL_PATH}")
    torch.save(model.state_dict(), MODEL_PATH)
    
    print(f"Zipping model to {ZIP_PATH}")
    with zipfile.ZipFile(ZIP_PATH, 'w', zipfile.ZIP_DEFLATED) as zipf:
        zipf.write(MODEL_PATH, os.path.basename(MODEL_PATH))
    
    print("Model saved and zipped successfully!")
