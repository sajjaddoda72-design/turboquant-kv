"""
Evaluator for Universal Adaptive Controller — REAL metrics (v2).

Four metrics extracted from a single deterministic 200-step run:

  1. loss_score     (50%) — convergence quality: initial → final loss
  2. stability_score (20%) — smoothness of the final training phase
  3. speed_score    (15%) — step throughput (samples / second)
  4. vram_score     (15%) — peak VRAM efficiency on GPU

Why 4 metrics?
  - loss_score alone can reach 0 by bad luck on short runs.
  - stability_score rewards controllers that *stabilise* training,
    not just ones that accidentally drop one loss value.
  - speed_score rewards smart batch-size growth when memory allows.
  - vram_score catches controllers that leak GPU memory.

Additive formula ensures evolution always has a non-zero learning signal:
  combined = 0.50 × loss + 0.20 × stability + 0.15 × speed + 0.15 × vram
"""

from __future__ import annotations

import importlib.util
import json
import math
import os
import sys
import time
import traceback
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
#  OpenEvolve integration (optional — falls back gracefully)
# ---------------------------------------------------------------------------

try:
    from openevolve.evaluation_result import EvaluationResult
    _HAS_OPENEVOLVE = True
except ImportError:
    _HAS_OPENEVOLVE = False

    class EvaluationResult:  # type: ignore[no-redef]
        """Minimal shim when openevolve is not installed."""

        def __init__(self, metrics: Dict[str, Any], artifacts: Dict[str, Any] | None = None):
            self.metrics = metrics
            self.artifacts = artifacts or {}

        @classmethod
        def from_dict(cls, d: Dict[str, Any]) -> "EvaluationResult":
            return cls(metrics=d)


# ---------------------------------------------------------------------------
#  Internal helpers
# ---------------------------------------------------------------------------

def _import_controller(path: str) -> Any:
    """Load a candidate program file as a Python module."""
    spec = importlib.util.spec_from_file_location("candidate", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load spec from: {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


class _FixedDataset(torch.utils.data.Dataset):
    """
    Deterministic, fixed-label synthetic dataset.

    Every call to __getitem__(i) returns the *same* (x, label) pair for
    index i, so the model can learn a consistent mapping and the evaluation
    score is reproducible across runs.
    """

    def __init__(
        self,
        n: int = 512,
        input_dim: int = 64,
        n_classes: int = 10,
        seed: int = 42,
    ) -> None:
        gen = torch.Generator().manual_seed(seed)
        self.x = torch.randn(n, input_dim, generator=gen)
        self.y = torch.randint(0, n_classes, (n,), generator=gen)

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, i: int):
        return self.x[i], self.y[i]


class _TinyModel(nn.Module):
    """
    3-layer MLP (64 → 256 → 128 → 10).

    ~50 K parameters — fast to train, large enough to overfit the
    512-sample fixed dataset in ~200 steps.
    """

    def __init__(
        self, input_dim: int = 64, hidden: int = 256, n_classes: int = 10
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
#  Core simulation
# ---------------------------------------------------------------------------

def _run_real_simulation(
    controller_module: Any, steps: int = 200
) -> Dict[str, Any]:
    """
    Run a short, fully deterministic training loop and collect telemetry.

    Four signals are returned so *evaluate()* can compute four independent
    scores:
      losses       — per-step cross-entropy (finite values only)
      step_times   — wall-clock time for each step (seconds)
      batch_sizes  — effective batch size reported by the controller
      peak_vram_gb — peak GPU memory allocated (0.0 on CPU)
    """
    # Reproducible initialisation
    torch.manual_seed(42)
    np.random.seed(42)

    cfg = controller_module.ControllerConfig(
        cooldown_steps=2,
        warmup_steps=10,
        log_every=99999,   # silence per-step logging during evaluation
        verbose=False,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _TinyModel().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    ds = _FixedDataset()
    loader, hard_sampler, batch_sampler = controller_module.build_dataloader(
        ds,
        initial_batch_size=32,
        pruning_ratio=0.0,
        num_workers=0,
        pin_memory=False,
        seed=0,
    )

    controller = controller_module.UniversalAdaptiveController(
        model=model,
        optimizer=optimizer,
        train_loader=loader,
        device=device,
        config=cfg,
        amp_dtype=torch.float16 if device.type == "cuda" else torch.float32,
        hardness_sampler=hard_sampler,
        batch_sampler=batch_sampler,
    )

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize()

    losses: List[float] = []
    step_times: List[float] = []
    batch_sizes: List[int] = []
    data_iter = iter(loader)

    for _ in range(steps):
        # Refill iterator at epoch boundary
        try:
            batch_x, batch_y = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            batch_x, batch_y = next(data_iter)

        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        batch_indices = list(range(batch_x.size(0)))

        t0 = time.perf_counter()

        with controller.amp_autocast():
            logits = model(batch_x)
            loss = nn.functional.cross_entropy(logits, batch_y)

        controller.backward(loss)
        controller.optimizer_step()

        step_dt = time.perf_counter() - t0

        per_loss = torch.full((batch_x.size(0),), loss.item())
        _state, _action = controller.step(
            loss,
            per_sample_losses=per_loss,
            batch_indices=batch_indices,
            step_time_s=step_dt,
        )

        loss_val = loss.item()
        if math.isfinite(loss_val):
            losses.append(loss_val)
        step_times.append(step_dt)
        batch_sizes.append(controller._current_effective_batch_size())

        if device.type == "cuda":
            torch.cuda.synchronize()

    if not losses:
        raise RuntimeError("All training steps produced non-finite loss.")

    peak_vram_gb = (
        torch.cuda.max_memory_allocated(device) / 1024 ** 3
        if device.type == "cuda"
        else 0.0
    )

    return {
        "losses": losses,
        "step_times": step_times,
        "batch_sizes": batch_sizes,
        "peak_vram_gb": peak_vram_gb,
        "device": device.type,
    }


# ---------------------------------------------------------------------------
#  Public API — called by OpenEvolve
# ---------------------------------------------------------------------------

def evaluate(program_path: str) -> Any:
    """
    Evaluate a candidate controller; return a dict or EvaluationResult.

    The *combined_score* key drives OpenEvolve's selection pressure.
    """
    try:
        eval_dir = os.path.dirname(os.path.abspath(program_path))
        if eval_dir not in sys.path:
            sys.path.insert(0, eval_dir)

        mod = _import_controller(program_path)
        tel = _run_real_simulation(mod, steps=200)
        losses: List[float] = tel["losses"]
        n = len(losses)

        # ── Metric 1: loss_score ────────────────────────────────────────
        #   Compare average of first 10 steps vs average of last 20 steps.
        #   A ≥33 % drop → score ≈ 1.0; proportional below that.
        warmup_n = max(1, min(10, n // 5))
        tail_n = max(1, min(20, n // 4))
        initial_loss = float(np.mean(losses[:warmup_n]))
        final_loss = float(np.mean(losses[-tail_n:]))
        loss_drop = max(0.0, (initial_loss - final_loss) / max(initial_loss, 1e-6))
        diverge_penalty = 1.0 if final_loss < initial_loss * 1.5 else 0.3
        loss_score = float(min(1.0, loss_drop * 3.0) * diverge_penalty)

        # ── Metric 2: stability_score ───────────────────────────────────
        #   Coefficient of variation (σ / μ) of the final 50 loss values.
        #   Low CV → smooth final phase → high score.
        #   A good controller settles the loss; a bad one keeps it oscillating.
        window = losses[-min(50, n):]
        cv = float(np.std(window) / (np.mean(window) + 1e-6))
        stability_score = float(1.0 / (1.0 + cv * 5.0))

        # ── Metric 3: speed_score ───────────────────────────────────────
        #   Rewards controllers that grow batch size over time (more
        #   samples per second) without hurting convergence.
        avg_step_s = float(np.mean(tel["step_times"]))
        speed_score = float(1.0 / (1.0 + avg_step_s * 20.0))

        # ── Metric 4: vram_score ────────────────────────────────────────
        #   On CPU the GPU memory is 0 → perfect score.
        #   On T4 (16 GB), target is well under 8 GB for a tiny model.
        vram_gb = float(tel["peak_vram_gb"])
        vram_score = 1.0 if vram_gb == 0.0 else float(max(0.0, 1.0 - vram_gb / 8.0))

        # ── Combined (additive — never collapses to zero) ───────────────
        combined = float(
            0.50 * loss_score
            + 0.20 * stability_score
            + 0.15 * speed_score
            + 0.15 * vram_score
        )

        metrics: Dict[str, Any] = {
            "combined_score": combined,
            "loss_score": loss_score,
            "stability_score": stability_score,
            "speed_score": speed_score,
            "vram_score": vram_score,
            # Diagnostic fields (visible in OpenEvolve logs)
            "initial_loss": initial_loss,
            "final_loss": final_loss,
            "loss_drop_pct": float(loss_drop * 100.0),
            "avg_step_ms": float(avg_step_s * 1000.0),
            "peak_vram_gb": vram_gb,
        }

        if _HAS_OPENEVOLVE:
            bs = tel["batch_sizes"]
            artifacts = {
                "device": tel["device"],
                "training_steps": n,
                "convergence": (
                    f"loss {initial_loss:.4f} → {final_loss:.4f} "
                    f"(↓{loss_drop * 100:.1f}%)"
                ),
                "stability": f"CV(last 50 steps) = {cv:.4f}",
                "batch_evolution": (
                    f"start={bs[0]}, end={bs[-1]}, "
                    f"min={min(bs)}, max={max(bs)}"
                ),
            }
            return EvaluationResult(metrics=metrics, artifacts=artifacts)

        return metrics

    except Exception as exc:  # noqa: BLE001
        tb = traceback.format_exc()
        error_metrics: Dict[str, Any] = {
            "combined_score": 0.0,
            "loss_score": 0.0,
            "stability_score": 0.0,
            "speed_score": 0.0,
            "vram_score": 0.0,
            "error": str(exc)[:300],
        }
        if _HAS_OPENEVOLVE:
            return EvaluationResult(
                metrics=error_metrics,
                artifacts={
                    "traceback": tb[:1000],
                    "error_type": type(exc).__name__,
                },
            )
        error_metrics["traceback"] = tb[:500]
        return error_metrics


# ---------------------------------------------------------------------------
#  Standalone runner — quick sanity check without OpenEvolve
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    program = sys.argv[1] if len(sys.argv) > 1 else "initial_program.py"
    result = evaluate(program)
    payload = result.metrics if hasattr(result, "metrics") else result
    print(json.dumps(payload, indent=2))
