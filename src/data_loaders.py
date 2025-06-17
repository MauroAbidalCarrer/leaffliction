from collections.abc import Sequence, Callable

from torch import Tensor, cat
from torch import device as dev
from torch.utils.data import Dataset, DataLoader


class MyDataLoader(DataLoader):

    def __init__(
        self,
        dataset: Dataset,
        batch_size: int,
        device: dev,
        attacker: Callable[[Tensor, Tensor], Tensor]|None=None,
        **kwargs,
        ):
        super().__init__(dataset, batch_size, **kwargs)
        self.device = device
        self.attacker = attacker

    def __iter__(self):
        for batch_x, batch_y in super().__iter__():
            batch_x, batch_y = self._to_device(batch_x), self._to_device(batch_y)
            if self.attacker:
                attacks = self.attacker.generate_attacks(batch_x, batch_y)
                batch_x = cat((batch_x, attacks))
                batch_y = batch_y.tile((2,))
            yield batch_x, batch_y

    def _to_device(self, tensor: Tensor) -> Tensor:
        if not isinstance(tensor, Tensor):
            raise ValueError(f"Batch of the wrapped dataset does not yield tensor, got {type(tensor)}")
        return tensor.to(self.device)