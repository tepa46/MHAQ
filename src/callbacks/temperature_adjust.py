import logging
import math
import torch

from lightning.pytorch.callbacks.callback import Callback
from lightning.pytorch import Trainer, LightningModule


from src.quantization.gdnsq.utils import model_stats

logger = logging.getLogger("lightning.pytorch")

class TemperatureScale(Callback):
    def __init__(
        self,
        scale_anneal=0.9985,
        scale_lr=1.0,
        scale_t=2,
        warmup=50,
        lr_schedule="exponential",
        min_lr_ratio=0.01,
    ) -> None:
        self.scale_anneal = scale_anneal
        self.scale_lr = scale_lr
        self.warmup = warmup
        self.lr_schedule = lr_schedule.lower()
        if self.lr_schedule not in {"exponential", "cosine"}:
            raise ValueError(
                f"Unknown lr_schedule: {lr_schedule}. Expected 'exponential' or 'cosine'."
            )
        if min_lr_ratio < 0:
            raise ValueError("min_lr_ratio must be non-negative.")
        self.min_lr_ratio = min_lr_ratio
        self.converged = False
        self.scale_t = scale_t
        self.anneal_start_step = None
        self.anneal_start_lr = None
        super().__init__()


    def on_fit_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        return super().on_fit_start(trainer, pl_module)

    def on_train_start(self, trainer, pl_module):
        self.lr = pl_module.lr
        self.total_batch = 0
        self.t = 0
        self.lr_t = 1.0
        self.anneal_start_step = None
        self.anneal_start_lr = None
        self.change_lr(pl_module, trainer, 0)

        return super().on_train_start(trainer, pl_module)

    def on_train_batch_end(self, trainer: Trainer, pl_module: LightningModule, outputs, batch, batch_idx: int) -> None:

        self.total_batch += 1
                
        pl_module.log("temperature", self.t, prog_bar=True, sync_dist=True)

        self.t = (self.t + self.lr * self.scale_t) if self.total_batch > self.warmup else self.t

        loss = pl_module.wrapped_criterion

        loss.t = torch.tensor(self.t)

        new_lr = self._get_lr(trainer)
        self.change_lr(pl_module, trainer, new_lr)
            
        return super().on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx)


    def on_train_epoch_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        return super().on_train_epoch_start(trainer, pl_module)


    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if self.lr_schedule == "cosine":
            current_converged = model_stats.is_converged(pl_module)
            if current_converged and self.anneal_start_step is None:
                self.anneal_start_step = self.total_batch
                self.anneal_start_lr = pl_module.lr
            self.converged = self.converged or current_converged
        else:
            self.converged = model_stats.is_converged(pl_module)
        
        return super().on_train_epoch_end(trainer, pl_module)


    def _get_lr(self, trainer: Trainer) -> float:
        if self.warmup > 0 and self.total_batch <= self.warmup:
            return self.lr * self.total_batch / self.warmup

        if self.lr_schedule == "cosine":
            if self.converged:
                return self._get_cosine_lr(trainer)
            self.lr_t = self.lr_t * self.scale_lr
            return self.lr * self.lr_t

        scale_lr = self.scale_lr if not self.converged else self.scale_anneal
        self.lr_t = self.lr_t * scale_lr
        return self.lr * self.lr_t

    def _get_cosine_lr(self, trainer: Trainer) -> float:
        anneal_start_step = self.anneal_start_step
        if anneal_start_step is None:
            anneal_start_step = self.warmup

        anneal_start_lr = self.anneal_start_lr
        if anneal_start_lr is None:
            anneal_start_lr = self.lr * self.lr_t

        min_lr = self.lr * self.min_lr_ratio
        total_steps = self._get_total_steps(trainer)
        anneal_steps = max(total_steps - anneal_start_step, 1)
        progress = (self.total_batch - anneal_start_step) / anneal_steps
        progress = min(max(progress, 0.0), 1.0)
        cosine_factor = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr + (anneal_start_lr - min_lr) * cosine_factor

    def _get_total_steps(self, trainer: Trainer) -> float:
        estimated_steps = getattr(trainer, "estimated_stepping_batches", None)
        if (
            isinstance(estimated_steps, (int, float))
            and math.isfinite(estimated_steps)
            and estimated_steps > 0
        ):
            return estimated_steps

        max_epochs = trainer.max_epochs if trainer.max_epochs is not None else 1
        num_batches = getattr(trainer, "num_training_batches", 0)
        if not isinstance(num_batches, int) or num_batches <= 0:
            return max(self.total_batch, 1)
        return max_epochs * num_batches

    def change_lr(self, pl_module: LightningModule, trainer: Trainer, new_lr: float) -> None:
        optimizer = trainer.optimizers[0]
        for param_group in trainer.optimizers[0].param_groups:
            param_group['lr'] = new_lr

        pl_module.lr = new_lr
        trainer.optimizers[0] = optimizer
