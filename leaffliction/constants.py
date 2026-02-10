import torch

DATASET_URL = "https://cdn.intra.42.fr/document/document/42144/leaves.zip"

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

BATCH_SIZE = 32

N_EPOCHS = 3

MODEL_PATH = "model.pt"

ZIP_PATH = "model.zip"