from lightning.pytorch import Trainer, LightningModule
from lightning.pytorch.callbacks.callback import Callback

from src.quantization.gdnsq.utils import model_stats


class ConvergenceTrigger(Callback):
    """
    Latches `signal_converged` on the active LR scheduler the first time
    `model_stats.is_converged(pl_module)` returns True.

    Designed to work with `QuantizationLR` (or any scheduler exposing a
    `signal_converged(step)` method). If no compatible scheduler is registered,
    the callback is a no-op.
    """

    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        sched_configs = getattr(trainer, "lr_scheduler_configs", None)
        if not sched_configs:
            return
        scheduler = sched_configs[0].scheduler
        if not hasattr(scheduler, "signal_converged"):
            return
        if model_stats.is_converged(pl_module):
            scheduler.signal_converged(trainer.global_step)
