import os
import zipfile

import torch
from torch import nn
from leaffliction.constants import (
    DEVICE,
    PATHS,
    TRAINING,
    DFLT_MODEL_KWARGS,
    DFLT_OPTIMIZER_KWARGS,
)

from leaffliction.dataset import get_dataset_for_training, \
                                mk_data_loaders, \
                                ensure_dataset_present
from Part_2.Augmentation import Balance, Augmentation
from leaffliction.models import CNN
from leaffliction.training import Trainer
from leaffliction import constants

if __name__ == "__main__":
    try:
        torch.manual_seed(constants.SEED)
        augmentation: Augmentation = Augmentation()
        balance = Balance(augmentation)

        ensure_dataset_present()

        balance.balance_dataset(
            PATHS["dataset_dir"]
        )

        print("Making dataset and loader")
        X, y = get_dataset_for_training()
        train_dl, val_dl = mk_data_loaders(X, y)

        print(f'Making model for {DEVICE} device')
        model = CNN(**DFLT_MODEL_KWARGS).to(device=DEVICE)

        print("Making optim and loss")
        optimizer = torch.optim.Adam(
            model.parameters(),
            **DFLT_OPTIMIZER_KWARGS)

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
    except Exception as e:
        print(f'Error: {e}')
