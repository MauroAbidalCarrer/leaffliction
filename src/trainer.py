from datetime import datetime
from IPython.display import display
from dataclasses import dataclass, field

import torch
from plotly.express import line
from torch.optim import Optimizer
from pandas import DataFrame as DF
from torch.utils.data import DataLoader as DL
from plotly.graph_objects import FigureWidget
from  torch.optim.lr_scheduler import LRScheduler
from torch.nn import utils, Module, CrossEntropyLoss


@dataclass
class Trainer:
    model:Module
    optimizer:Optimizer
    step: int = field(default=0, init=False)
    epoch: int = field(default=0, init=False)
    lr_scheduler: LRScheduler = field(default=None)
    grad_clip: float = field(default=0.0)
    fig: FigureWidget = field(default=None, init=False)
    loss:CrossEntropyLoss = field(default_factory=CrossEntropyLoss)
    training_metrics:list[dict] = field(default_factory=list, init=False)

    def optimize_nn(self, epochs, train_dl:DL, test_dl:DL, *, catch_key_int=True, plt_kwargs: dict=None) -> DF:
        """Optimizes the neural network and returns a dataframe of the training metrics."""
        if catch_key_int:
            try:
                self._training_loop(epochs, train_dl, test_dl, plt_kwargs)
            except KeyboardInterrupt:
                print("Caught KeyboardInterrupt exception, returning training metrics.")
        else:
            self._training_loop(epochs, train_dl, test_dl, plt_kwargs)
        return DF.from_records(self.training_metrics)

    def _training_loop(self, epochs, train_dl:DL, test_dl:DL, plt_kwargs=None):
        model_device = next(self.model.parameters()).device
        if model_device.type != "cuda":
            print("Warning: Model is not on a cuda device.")
        # Use self.epoch instead of for epoch in range(epochs).
        # This avoids resetting new metrics DF lines to the same epoch value in case this method gets recalled.
        if self.epoch == 0:
            self.record_and_display_metrics(train_dl, test_dl, plt_kwargs)
        for _ in range(epochs):
            total_loss = 0
            total_accuracy = 0
            # Count nb samples instead of accessing len(data_loader.dataset)
            # in case the data lauder augments the number of samples
            nb_batches = 0
            nb_samples = 0
            for batch_x, batch_y in train_dl:
                nb_batches += 1
                nb_samples += len(batch_x)
                self.model.train()
                self.optimizer.zero_grad()
                batch_y_pred = self.model(batch_x)
                loss_value: torch.Tensor = self.loss(batch_y_pred, batch_y)
                loss_value.retain_grad()
                loss_value.backward()
                print(loss_value)
                print(loss_value.grad)
                print(loss_value.grad.shape)
                print("==========")
                if self.grad_clip:
                    utils.clip_grad_value_(self.model.parameters(), self.grad_clip)
                total_loss += loss_value.item()
                total_accuracy += (torch.max(batch_y_pred, 1)[1] == batch_y).sum().item()
                self.optimizer.step()
                if self.lr_scheduler:
                    self.lr_scheduler.step()
                self.step += 1
            self.epoch += 1
            train_metrics = {
                "train_loss": total_loss / len(train_dl),
                "train_accuracy": total_accuracy / nb_samples
            }
            self.record_and_display_metrics(train_metrics, test_dl, plt_kwargs)

    def record_and_display_metrics(self, train_dl: DL, test_dl: DL, plt_kwargs: dict):
        self.training_metrics.append(self.record_metrics(train_dl, test_dl))
        if plt_kwargs is not None:
            if self.fig is None:
                self.create_figure_widget(plt_kwargs)
            self.update_figure(plt_kwargs)

    def record_metrics(self, train_dl: DL|dict, test_dl: DL) -> dict[str, any]:
        # This is ugly but it does the trick
        if isinstance(train_dl, DL):
            train_metrics = self.metrics_of_dataset(train_dl, "train")
        else: 
            train_metrics = train_dl
        return {
            "epoch": self.epoch,
            "step": self.step,
            "date": datetime.now(),
            **self.metrics_of_dataset(test_dl, "test"),
            **train_metrics,
            **self.optimizer.state_dict()["param_groups"][-1],
        }

    def metrics_of_dataset(self, data_loader: DL, dl_prefix: str) -> dict:
        self.model.eval()
        model_device = next(self.model.parameters()).device
        total_loss = 0
        total_accuracy = 0
        # Count nb samples instead of accessing len(data_loader.dataset)
        # in case the data lauder augments the number of samples
        nb_batches = 0
        nb_samples = 0
        for batch_x, batch_y in data_loader:
            nb_batches += 1
            with torch.no_grad():
                batch_y_pred = self.model(batch_x)
                total_loss += self.loss(batch_y_pred, batch_y).item()
                total_accuracy += (torch.max(batch_y_pred, 1)[1] == batch_y).sum().item()
            nb_samples += len(batch_x)
        return {
            # Divide loss by nb batches
            dl_prefix + "_loss": total_loss / len(data_loader),
            # Divide accuracy by nb elements
            dl_prefix + "_accuracy": total_accuracy / nb_samples,
        }

    def create_figure_widget(self, plt_kwargs: dict) -> FigureWidget:
        df = (
            DF.from_records(self.training_metrics)
            .melt(plt_kwargs["x"], plt_kwargs["y"])
        )
        self.fig = (
            line(
                data_frame=df,
                y="value",
                facet_row="variable",
                color="variable",
                markers=True,
                **{k: v for k, v in plt_kwargs.items() if k != "y"},
            )
            .update_yaxes(matches=None)
        )
        self.fig = FigureWidget(self.fig)
        display(self.fig)

    def update_figure(self, plt_kwargs: dict):
        df = DF.from_records(self.training_metrics)
        with self.fig.batch_update():
            for i, plt_y in enumerate(plt_kwargs["y"]):
                self.fig.data[i].x = df[plt_kwargs["x"]]
                self.fig.data[i].y = df[plt_y]
