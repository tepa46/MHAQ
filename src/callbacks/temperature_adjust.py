import logging

import torch

from lightning.pytorch.callbacks.callback import Callback
from lightning.pytorch import Trainer, LightningModule


logger = logging.getLogger("lightning.pytorch")


class TemperatureScale(Callback):
    """
    Updates the criterion temperature `t` on every training batch.

    After `warmup` batches, `t` grows by `pl_module.lr * scale_t` per step,
    and the resulting value is written into `pl_module.wrapped_criterion.t`.

    LR scheduling is handled by `src.training.lr_schedulers.QuantizationLR`
    and `ConvergenceTrigger`.
    """

    def __init__(self, scale_t: float = 2, warmup: int = 50) -> None:
        super().__init__()
        self.scale_t = scale_t
        self.warmup = warmup

    def on_train_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self.lr = pl_module.lr
        self.total_batch = 0
        self.t = 0
        return super().on_train_start(trainer, pl_module)

    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs,
        batch,
        batch_idx: int,
    ) -> None:
        self.total_batch += 1

        pl_module.log("temperature", self.t, prog_bar=True, sync_dist=True)

        if self.total_batch > self.warmup:
            self.t = self.t + self.lr * self.scale_t

        pl_module.wrapped_criterion.t = torch.tensor(self.t)

        return super().on_train_batch_end(
            trainer, pl_module, outputs, batch, batch_idx
        )
