import wandb
import torch
from typing import (
    Callable,
    Tuple,
    Dict,
)
from torch import nn, Tensor
from torch.utils.data import (
    DataLoader,
)
from tqdm import tqdm
from leaffliction.dataset import preprocess_batch
from leaffliction.constants import BATCH_SIZE, DEVICE

criterion_t = Callable[[Tensor, Tensor], Tuple[Tensor, Dict[str, Tensor]]]

class Trainer:
    def __init__(self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
    ):
        self.model = model
        self.optimizer = optimizer   
    
    def train_model(
        self,
        train_dl: torch.utils.data.DataLoader,
        val_dl: torch.utils.data.DataLoader,
        criterion,
        n_epochs: int,
    ) -> wandb.Run:
        self.epoch = 0
        self.step = 0
        wandb_run = wandb.init(project="leaffliction")
        self.eval_model(val_dl)
        for _ in range(n_epochs):
            print('epochs', n_epochs)
            self.train_model_for_single_epoch(
                train_dl,
                criterion,
            )
            self.eval_model(val_dl)
        return wandb_run

    @torch.no_grad
    def eval_model(self, data_loader: DataLoader):
        self.model = self.model.eval()
        accuracy = 0
        for x, y in data_loader:
            x, y = preprocess_batch(x, y)
            with torch.autocast(DEVICE.type, torch.bfloat16):
                model_output = self.model(x)
                accuracy += (
                    (model_output.argmax(dim=-1) == y)
                    .float()
                    .mean()
                    .item()
                    / len(data_loader)
                )
        self.wandb_log_with_prefix({"accuracy": accuracy}, "validation")
        return accuracy

    def train_model_for_single_epoch(
        self,
        data_loader: torch.utils.data.DataLoader,
        criterion,
    ):
        self.model = self.model.train()
        for x, y in tqdm(data_loader):
            step_dict = self.train_model_for_single_step(criterion, x, y)
            self.wandb_log_with_prefix(step_dict, "training")
        self.epoch += 1

    def train_model_for_single_step(
        self,
        criterion: criterion_t,
        x: Tensor,
        y: Tensor,
    ) -> dict[str, float]:
        x, y = preprocess_batch(x, y)
        self.optimizer.zero_grad()
        with torch.autocast(DEVICE.type, torch.bfloat16):
            model_output = self.model(x)
            loss = criterion(model_output, y)
            loss.backward()
        loss_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        with torch.no_grad():
            accuracy = (model_output.argmax(dim=-1) == y).float().mean()
        self.step += 1
        return {
            "loss": loss.item(),
            "accuracy": accuracy.item(),
            "loss_norm": loss_norm.item(),
        }

    def wandb_log_with_prefix(self, data: dict, prefix: str):
        data = {prefix + "/" + k: v for k, v in data.items()}
        data["epoch"] = self.epoch
        data["step"] = self.step
        data["training_samples_seen"] = self.step * BATCH_SIZE
        wandb.log(data, commit=True)