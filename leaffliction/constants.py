import torch

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

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PATHS = {
    "dataset_url": "https://cdn.intra.42.fr/document/document/42144/leaves.zip",
    "dataset_dir": "dataset",
    "model": "model.pt",
    "zip": "model.zip",
}

MODEL = {
    "kernels_per_layer": [32, 64, 128, 256],
    "mlp_width": 128,
    "mlp_depth": 3,
    "n_classes": len(LABEL2ID),
}

TRAINING = {
    "batch_size": 32,
    "n_epochs": 3,
    "learning_rate": 1e-4,
    "optimizer": "Adam",
    "gradient_clip_norm": 1.0,
}

DATA = {
    "val_fraction": 0.2,
    "seed": 42,
}