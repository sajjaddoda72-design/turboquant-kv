"""
OpenEvolve: Universal Adaptive Training Plugin — EVOLVABLE SEED
================================================================
This file is the *initial seed* that OpenEvolve will evolve.
The decision logic is isolated between # EVOLVE-BLOCK-START / END
markers — OpenEvolve rewrites ONLY that block.  Everything else
(observer, actuator, samplers, dataloader, config dataclasses) is
treated as fixed scaffolding and must NOT be modified.
"""

from __future__ import annotations

import math
import time
import logging
import warnings
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Sampler

__all__ = [
    "ControllerConfig",
    "ControllerState",
    "ControllerAction",
    "UniversalAdaptiveController",
    "HardnessAwareSampler",
    "AdaptiveBatchSampler",
    "build_dataloader",
]

log = logging.getLogger("openevolve")
if not log.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[%(name)s] %(levelname)s: %(message)s"))
    log.addHandler(handler)
log.setLevel(logging.INFO)


# ============================================================================
#  Configuration, State, Action
# ============================================================================

@dataclass
class ControllerConfig:
    memory_soft_limit: float = 0.85
    memory_hard_limit: float = 0.95
    oom_recovery_factor: float = 0.5
    plateau_window: int = 50
    plateau_velocity_eps: float = 1e-4
    plateau_variance_eps: float = 1e-5
    loss_spike_factor: float = 0.10
    min_batch_size: int = 1
    max_batch_size: int = 1024
    batch_grow_factor: float = 1.10
    batch_shrink_factor: float = 0.90
    min_pruning_ratio: float = 0.00
    max_pruning_ratio: float = 0.50
    pruning_grow_step: float = 0.05
    lr_decay_factor: float = 0.90
    lr_recovery_factor: float = 1.02
    lr_min: float = 1e-7
    lr_max: float = 1.0
    step_time_alpha: float = 0.10
    loss_ema_short_alpha: float = 0.20
    loss_ema_long_alpha: float = 0.01
    grad_norm_alpha: float = 0.10
    cooldown_steps: int = 5
    warmup_steps: int = 10
    grad_clip_norm: float = 1.0
    checkpoint_recompute_steps: int = 1
    log_every: int = 25
    verbose: bool = False


@dataclass
class ControllerState:
    loss_current: float = 0.0
    loss_ema_short: float = 0.0
    loss_ema_long: float = 0.0
    loss_velocity: float = 0.0
    loss_variance: float = 0.0
    loss_trend: int = 0
    memory_used_bytes: int = 0
    memory_total_bytes: int = 1
    memory_pressure: float = 0.0
    oom_detected: bool = False
    oom_recovery_count: int = 0
    step_time_ema: float = 0.0
    throughput_samples_per_sec: float = 0.0
    current_batch_size: int = 32
    current_amp_enabled: bool = True
    current_checkpointing_enabled: bool = False
    current_pruning_ratio: float = 0.0
    current_lr: float = 1e-3
    grad_norm_ema: float = 0.0
    step: int = 0


@dataclass
class ControllerAction:
    target_batch_size: int
    target_amp_enabled: bool
    target_checkpointing_enabled: bool
    target_pruning_ratio: float
    target_lr: float
    skip_step: bool = False
    recompute_loss: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "batch_size": self.target_batch_size,
            "amp": self.target_amp_enabled,
            "checkpointing": self.target_checkpointing_enabled,
            "pruning_ratio": self.target_pruning_ratio,
            "lr": self.target_lr,
            "skip_step": self.skip_step,
            **self.metadata,
        }


# ============================================================================
#  Hardness-Aware Sampler
# ============================================================================

class HardnessAwareSampler(Sampler[int]):
    def __init__(
        self,
        dataset: Dataset,
        pruning_ratio: float = 0.0,
        hardness_ema_alpha: float = 0.1,
        seed: int = 0,
    ) -> None:
        self.dataset = dataset
        self.n = len(dataset)
        self.pruning_ratio = float(pruning_ratio)
        self.alpha = hardness_ema_alpha
        self.rng = torch.Generator().manual_seed(seed)
        self.hardness = torch.zeros(self.n)
        self.seen = torch.zeros(self.n, dtype=torch.bool)
        self._cold_start_filled = False
        self._pending_losses: Dict[int, float] = {}

    def set_pruning_ratio(self, ratio: float) -> None:
        self.pruning_ratio = max(0.0, min(1.0, float(ratio)))

    def report_losses(self, indices: Iterable[int], losses: Iterable[float]) -> None:
        for idx, loss in zip(indices, losses):
            idx_i = int(idx)
            l = float(loss)
            if not torch.isfinite(torch.tensor(l)):
                continue
            if not self.seen[idx_i]:
                self.hardness[idx_i] = l
                self.seen[idx_i] = True
            else:
                self.hardness[idx_i] = (
                    (1.0 - self.alpha) * self.hardness[idx_i]
                    + self.alpha * l
                )

    def __iter__(self) -> Iterable[int]:
        if not self.seen.any():
            perm = torch.randperm(self.n, generator=self.rng).tolist()
        else:
            med = float(self.hardness[self.seen].median()) if self.seen.any() else 1.0
            hardness_full = torch.where(
                self.seen, self.hardness, torch.full_like(self.hardness, med)
            )
            k = 1.0 + 4.0 * self.pruning_ratio
            weights = (hardness_full + 1e-6) ** k
            weights = weights / weights.sum()
            if self.pruning_ratio > 0:
                kth = int((1.0 - self.pruning_ratio) * self.n)
                if kth > 0 and kth < self.n:
                    threshold = torch.kthvalue(weights, kth).values
                    mask = weights >= threshold
                    weights = weights * mask.float()
                    weights = weights / weights.sum().clamp_min(1e-12)
            idxs = torch.multinomial(
                weights, num_samples=self.n, replacement=True, generator=self.rng
            )
            perm = idxs.tolist()
        return iter(perm)

    def __len__(self) -> int:
        return self.n


# ============================================================================
#  Adaptive Batch Sampler
# ============================================================================

class AdaptiveBatchSampler(Sampler[List[int]]):
    def __init__(self, base_sampler: Sampler[int], batch_size: int) -> None:
        self.base_sampler = base_sampler
        self.batch_size = int(batch_size)

    def set_batch_size(self, batch_size: int) -> None:
        self.batch_size = max(1, int(batch_size))

    def __iter__(self) -> Iterable[List[int]]:
        batch: List[int] = []
        for idx in self.base_sampler:
            batch.append(idx)
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        if batch:
            yield batch

    def __len__(self) -> int:
        return (len(self.base_sampler) + self.batch_size - 1) // self.batch_size


def build_dataloader(
    dataset: Dataset,
    initial_batch_size: int = 32,
    pruning_ratio: float = 0.0,
    num_workers: int = 0,
    pin_memory: bool = True,
    seed: int = 0,
) -> Tuple[DataLoader, HardnessAwareSampler, AdaptiveBatchSampler]:
    base_sampler = HardnessAwareSampler(dataset, pruning_ratio=pruning_ratio, seed=seed)
    batch_sampler = AdaptiveBatchSampler(base_sampler, initial_batch_size)
    loader = DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return loader, base_sampler, batch_sampler


# ============================================================================
#  Universal Adaptive Controller
# ============================================================================

class UniversalAdaptiveController:
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        train_loader: DataLoader,
        device: torch.device,
        config: Optional[ControllerConfig] = None,
        amp_dtype: torch.dtype = torch.float16,
        hardness_sampler: Optional[HardnessAwareSampler] = None,
        batch_sampler: Optional[AdaptiveBatchSampler] = None,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.loader = train_loader
        self.device = device
        self.cfg = config or ControllerConfig()
        self.amp_dtype = amp_dtype
        self.hardness_sampler = hardness_sampler
        self.batch_sampler = batch_sampler
        if self.batch_sampler is None and isinstance(train_loader.batch_sampler, AdaptiveBatchSampler):
            self.batch_sampler = train_loader.batch_sampler
        if self.hardness_sampler is None and self.batch_sampler is not None:
            if isinstance(self.batch_sampler.base_sampler, HardnessAwareSampler):
                self.hardness_sampler = self.batch_sampler.base_sampler
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.cfg_check_amp())
        self._amp_enabled = True
        self._checkpoint_modules: List[nn.Module] = []
        self._loss_hist: deque[float] = deque(maxlen=self.cfg.plateau_window)
        self._loss_ema_short: float = float("nan")
        self._loss_ema_long: float = float("nan")
        self._loss_velocity: float = 0.0
        self._loss_variance: float = 0.0
        self._step_time_ema: float = 0.0
        self._grad_norm_ema: float = 0.0
        self._step: int = 0
        self._steps_since_config: int = self.cfg.cooldown_steps
        self._steps_since_amp_toggle: int = 10_000
        self._oom_pending: bool = False
        self._oom_recovery_count: int = 0
        self._lr_current: float = optimizer.param_groups[0]["lr"]
        self._current_batch_indices: List[int] = []
        self.history: List[Dict[str, Any]] = []

    def cfg_check_amp(self) -> bool:
        return self.device.type == "cuda" and torch.cuda.is_available()

    def observe(
        self,
        loss: torch.Tensor,
        batch_indices: Optional[Iterable[int]] = None,
        per_sample_losses: Optional[Iterable[float]] = None,
        step_time_s: Optional[float] = None,
        oom: bool = False,
    ) -> ControllerState:
        loss_val = float(loss.detach().item())
        if not math.isfinite(loss_val):
            warnings.warn(f"Non-finite loss detected: {loss_val}; treating as OOM.")
            oom = True
            loss_val = float("nan")

        if math.isnan(self._loss_ema_short):
            self._loss_ema_short = loss_val
            self._loss_ema_long = loss_val
            self._loss_velocity = 0.0
        else:
            prev_short = self._loss_ema_short
            self._loss_ema_short = (
                (1 - self.cfg.loss_ema_short_alpha) * self._loss_ema_short
                + self.cfg.loss_ema_short_alpha * loss_val
            )
            self._loss_ema_long = (
                (1 - self.cfg.loss_ema_long_alpha) * self._loss_ema_long
                + self.cfg.loss_ema_long_alpha * loss_val
            )
            self._loss_velocity = self._loss_ema_short - prev_short
        self._loss_hist.append(loss_val)
        if len(self._loss_hist) > 5:
            arr = torch.tensor(list(self._loss_hist), dtype=torch.float64)
            self._loss_variance = float(arr.var(unbiased=False).item())
        else:
            self._loss_variance = 0.0

        if self._loss_ema_short < self._loss_ema_long * (1 - 1e-3):
            trend = -1
        elif self._loss_ema_short > self._loss_ema_long * (1 + 1e-3):
            trend = +1
        else:
            trend = 0

        if step_time_s is not None and step_time_s > 0:
            if self._step_time_ema == 0.0:
                self._step_time_ema = step_time_s
            else:
                a = self.cfg.step_time_alpha
                self._step_time_ema = (1 - a) * self._step_time_ema + a * step_time_s

        if self.device.type == "cuda" and torch.cuda.is_available():
            used = torch.cuda.max_memory_allocated(self.device)
            total = torch.cuda.get_device_properties(self.device).total_memory
        else:
            used, total = 0, 1
        pressure = used / max(1, total)

        gn = float(self._grad_norm_ema)
        if self._step_time_ema > 0:
            tput = (self._current_effective_batch_size() / self._step_time_ema)
        else:
            tput = 0.0

        if oom:
            self._oom_pending = True
            self._oom_recovery_count += 1
        else:
            self._oom_pending = False

        if (
            self.hardness_sampler is not None
            and batch_indices is not None
            and per_sample_losses is not None
        ):
            try:
                self.hardness_sampler.report_losses(batch_indices, per_sample_losses)
            except Exception as exc:
                log.debug("hardness feedback failed: %s", exc)

        state = ControllerState(
            loss_current=loss_val,
            loss_ema_short=self._loss_ema_short,
            loss_ema_long=self._loss_ema_long,
            loss_velocity=self._loss_velocity,
            loss_variance=self._loss_variance,
            loss_trend=trend,
            memory_used_bytes=int(used),
            memory_total_bytes=int(total),
            memory_pressure=float(pressure),
            oom_detected=self._oom_pending,
            oom_recovery_count=self._oom_recovery_count,
            step_time_ema=self._step_time_ema,
            throughput_samples_per_sec=float(tput),
            current_batch_size=self._current_effective_batch_size(),
            current_amp_enabled=self._amp_enabled,
            current_checkpointing_enabled=bool(self._checkpoint_modules),
            current_pruning_ratio=(self.hardness_sampler.pruning_ratio
                                   if self.hardness_sampler else 0.0),
            current_lr=self._lr_current,
            grad_norm_ema=gn,
            step=self._step,
        )
        self._step += 1
        self._steps_since_config += 1
        self._steps_since_amp_toggle += 1

        return state

    def decide(self, state: ControllerState) -> ControllerAction:
        action = ControllerAction(
            target_batch_size=state.current_batch_size,
            target_amp_enabled=state.current_amp_enabled,
            target_checkpointing_enabled=state.current_checkpointing_enabled,
            target_pruning_ratio=state.current_pruning_ratio,
            target_lr=state.current_lr,
        )

        # ------------------------------------------------------------------
        #  EVOLVE-BLOCK-START
        # ------------------------------------------------------------------
        cfg = self.cfg
        c = cfg
        m_soft = c.memory_soft_limit
        m_hard = c.memory_hard_limit

        if state.oom_detected or state.memory_pressure >= m_hard:
            action.target_batch_size = max(
                c.min_batch_size,
                int(state.current_batch_size * c.oom_recovery_factor),
            )
            action.target_checkpointing_enabled = True
            action.target_amp_enabled = True
            action.target_pruning_ratio = min(
                c.max_pruning_ratio,
                state.current_pruning_ratio + c.pruning_grow_step * 2.0,
            )
            action.skip_step = state.oom_recovery_count <= 1
            action.recompute_loss = True
            action.metadata["reason"] = "oom_hard"
            return action

        if state.memory_pressure >= m_soft:
            action.target_batch_size = max(
                c.min_batch_size,
                int(state.current_batch_size * c.batch_shrink_factor),
            )
            action.target_checkpointing_enabled = True
            if not state.current_amp_enabled:
                action.target_amp_enabled = True
            action.metadata["reason"] = "mem_soft"
            return action

        if (
            state.loss_ema_long > 0
            and state.loss_current
            > state.loss_ema_long * (1.0 + c.loss_spike_factor)
            and state.loss_trend == 1
            and state.step > c.warmup_steps
        ):
            action.target_lr = max(c.lr_min, state.current_lr * c.lr_decay_factor)
            action.metadata["reason"] = "loss_spike"
            return action

        is_plateau = (
            state.step > c.warmup_steps
            and abs(state.loss_velocity) < c.plateau_velocity_eps
            and state.loss_variance < c.plateau_variance_eps
            and state.loss_trend == 0
        )
        if is_plateau and state.current_pruning_ratio < c.max_pruning_ratio:
            action.target_pruning_ratio = min(
                c.max_pruning_ratio, state.current_pruning_ratio + c.pruning_grow_step
            )
            action.target_batch_size = max(
                c.min_batch_size,
                int(state.current_batch_size * c.batch_shrink_factor),
            )
            action.metadata["reason"] = "plateau"
            return action

        if (
            state.memory_pressure < m_soft * 0.75
            and state.loss_trend <= 0
            and state.step > c.warmup_steps
        ):
            action.target_batch_size = min(
                c.max_batch_size,
                max(c.min_batch_size, int(state.current_batch_size * c.batch_grow_factor)),
            )
            action.metadata["reason"] = "throughput"
            return action

        action.metadata["reason"] = "steady"
        return action
        # ------------------------------------------------------------------
        #  EVOLVE-BLOCK-END
        # ------------------------------------------------------------------

    def act(self, action: ControllerAction) -> None:
        if self._steps_since_config < self.cfg.cooldown_steps and not action.skip_step:
            return

        new_bs = int(action.target_batch_size)
        new_bs = max(self.cfg.min_batch_size, min(self.cfg.max_batch_size, new_bs))
        if self.batch_sampler is not None and new_bs != self.batch_sampler.batch_size:
            self.batch_sampler.set_batch_size(new_bs)

        if (
            action.target_amp_enabled != self._amp_enabled
            and self._steps_since_amp_toggle >= 5
        ):
            self._amp_enabled = bool(action.target_amp_enabled)
            try:
                self.scaler = torch.amp.GradScaler("cuda", enabled=self._amp_enabled)
            except Exception:
                self.scaler = torch.amp.GradScaler(enabled=self._amp_enabled)
            self._steps_since_amp_toggle = 0
        else:
            self._amp_enabled = bool(action.target_amp_enabled)

        self._set_checkpointing(bool(action.target_checkpointing_enabled))

        if self.hardness_sampler is not None:
            self.hardness_sampler.set_pruning_ratio(
                max(self.cfg.min_pruning_ratio,
                    min(self.cfg.max_pruning_ratio, float(action.target_pruning_ratio)))
            )

        new_lr = float(action.target_lr)
        new_lr = max(self.cfg.lr_min, min(self.cfg.lr_max, new_lr))
        if abs(new_lr - self._lr_current) > 1e-12:
            for pg in self.optimizer.param_groups:
                pg["lr"] = new_lr
            self._lr_current = new_lr

        self._steps_since_config = 0
        self.history.append({"step": self._step, **action.as_dict()})

    def backward(self, loss: torch.Tensor) -> None:
        if self._amp_enabled:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

    def optimizer_step(self) -> None:
        params = [p for g in self.optimizer.param_groups for p in g["params"] if p.grad is not None]
        if params:
            gn = torch.norm(
                torch.stack([p.grad.detach().float().norm(2) for p in params]), 2
            ).item()
            a = self.cfg.grad_norm_alpha
            self._grad_norm_ema = (1 - a) * self._grad_norm_ema + a * gn
            if self.cfg.grad_clip_norm and self.cfg.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(params, self.cfg.grad_clip_norm)
        if self._amp_enabled:
            self.scaler.unscale_(self.optimizer)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)

    def amp_autocast(self) -> torch.amp.autocast:
        if self._amp_enabled and self.device.type == "cuda":
            return torch.amp.autocast("cuda", dtype=self.amp_dtype)
        class _NullCtx:
            def __enter__(self): return self
            def __exit__(self, *a): return False
        return _NullCtx()

    def register_for_checkpointing(self, *modules: nn.Module) -> None:
        for m in modules:
            if m not in self._checkpoint_modules:
                self._checkpoint_modules.append(m)

    def _set_checkpointing(self, enabled: bool) -> None:
        for m in self._checkpoint_modules:
            try:
                m.gradient_checkpointing = bool(enabled)
            except AttributeError:
                self._wrap_module_forward(m, bool(enabled))

    @staticmethod
    def _wrap_module_forward(m: nn.Module, enabled: bool) -> None:
        if getattr(m, "_oe_wrapped", False):
            return
        orig_forward = m.forward

        def wrapped_forward(*args, **kwargs):
            from torch.utils.checkpoint import checkpoint
            return checkpoint(orig_forward, *args, use_reentrant=False, **kwargs)

        if enabled:
            m.forward = wrapped_forward
        m._oe_wrapped = True

    def _current_effective_batch_size(self) -> int:
        if self.batch_sampler is not None:
            return int(self.batch_sampler.batch_size)
        return int(self.loader.batch_size) if self.loader.batch_size else 1

    @torch.no_grad()
    def compute_per_sample_loss(
        self,
        model_forward: Callable[[torch.Tensor], torch.Tensor],
        x: torch.Tensor,
        y: torch.Tensor,
        loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    ) -> torch.Tensor:
        logits = model_forward(x)
        per = loss_fn(logits, y)
        if per.ndim == 0:
            per = per.expand(x.shape[0])
        return per.detach()

    def step(
        self,
        loss: torch.Tensor,
        per_sample_losses: Optional[torch.Tensor] = None,
        batch_indices: Optional[List[int]] = None,
        step_time_s: Optional[float] = None,
        oom: bool = False,
    ) -> Tuple[ControllerState, ControllerAction]:
        state = self.observe(
            loss,
            batch_indices=batch_indices,
            per_sample_losses=(per_sample_losses.tolist() if per_sample_losses is not None else None),
            step_time_s=step_time_s,
            oom=oom,
        )
        action = self.decide(state)
        self.act(action)
        return state, action
