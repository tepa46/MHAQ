import inspect
import math

from torch.optim.lr_scheduler import LRScheduler


def build_scheduler(scheduler_cls, optimizer, scheduler_params: dict, total_steps: int):
    """
    Instantiate a LR scheduler, injecting `total_steps` only if the class accepts it.

    YAML-provided params always win. This lets the codebase host both schedulers
    that need `total_steps` (e.g. `QuantizationLR`) and stock torch schedulers
    that do not (e.g. `ExponentialLR`).
    """
    params = dict(scheduler_params or {})
    sig = inspect.signature(scheduler_cls.__init__)
    accepts_total_steps = (
        "total_steps" in sig.parameters
        or any(p.kind is inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
    )
    if accepts_total_steps and "total_steps" not in params:
        params["total_steps"] = int(total_steps)
    return scheduler_cls(optimizer, **params)


class QuantizationLR(LRScheduler):
    """
    LR scheduler for GDNSQ training.

    Phases:
        1. Warmup: linear from 0 to base_lr over `warmup` steps.
        2. Pre-convergence: lr = base_lr * scale_lr ** (step - warmup).
           With default scale_lr=1.0 this is a no-op (LR stays at base_lr).
        3. Post-convergence (sticky): starts when an external observer calls
           signal_converged(step). Once set, never resets.
           - lr_schedule="exponential": lr = base_lr * scale_anneal ** (step - conv_step)
           - lr_schedule="cosine": cosine anneal from base_lr down to
             base_lr * min_lr_ratio over (total_steps - conv_step) steps.

    `total_steps` is only used for cosine. Pass total stepping batches.
    """

    def __init__(
        self,
        optimizer,
        total_steps: int,
        warmup: int = 0,
        scale_lr: float = 1.0,
        scale_anneal: float = 0.9985,
        lr_schedule: str = "exponential",
        min_lr_ratio: float = 0.01,
        last_epoch: int = -1,
    ) -> None:
        lr_schedule = lr_schedule.lower()
        if lr_schedule not in {"exponential", "cosine"}:
            raise ValueError(
                f"Unknown lr_schedule: {lr_schedule}. Expected 'exponential' or 'cosine'."
            )
        if min_lr_ratio < 0:
            raise ValueError("min_lr_ratio must be non-negative.")
        if warmup < 0:
            raise ValueError("warmup must be non-negative.")

        self.total_steps = int(total_steps)
        self.warmup = int(warmup)
        self.scale_lr = float(scale_lr)
        self.scale_anneal = float(scale_anneal)
        self.lr_schedule = lr_schedule
        self.min_lr_ratio = float(min_lr_ratio)
        self.converged = False
        self.conv_step: int | None = None

        super().__init__(optimizer, last_epoch=last_epoch)

    def signal_converged(self, step: int) -> None:
        """Latch the convergence event. Idempotent: only the first call takes effect."""
        if not self.converged:
            self.converged = True
            self.conv_step = int(step)

    def get_lr(self):
        step = self.last_epoch
        return [self._compute(base_lr, step) for base_lr in self.base_lrs]

    def _compute(self, base_lr: float, step: int) -> float:
        if self.warmup > 0 and step < self.warmup:
            return base_lr * step / self.warmup

        if not self.converged:
            return base_lr * (self.scale_lr ** max(step - self.warmup, 0))

        rel = step - self.conv_step
        if self.lr_schedule == "exponential":
            return base_lr * (self.scale_anneal ** max(rel, 0))

        # cosine
        horizon = max(self.total_steps - self.conv_step, 1)
        progress = min(max(rel / horizon, 0.0), 1.0)
        min_lr = base_lr * self.min_lr_ratio
        return min_lr + (base_lr - min_lr) * 0.5 * (1.0 + math.cos(math.pi * progress))
