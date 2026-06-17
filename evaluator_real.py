"""
Evaluator for Universal Adaptive Controller — REAL metrics.

Measures 3 real-world dimensions on GPU (T4) or CPU:
  1) loss_score  (60%) — convergence quality
  2) speed_score (20%) — throughput (steps/sec)
  3) vram_score  (20%) — peak VRAM efficiency

Design goals:
  - Deterministic: seeded dataset → same controller → same score.
  - Non-collapsing: additive scoring (never zeros out from a single bad metric).
  - Realistic: fixed-label dataset so the model can actually learn.
"""

from __future__ import annotations

import importlib.util
import math
import os
import sys
import time
import traceback
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn as nn

# Support both standalone use and use from within OpenEvolve.
try:
    from openevolve.evaluation_result import EvaluationResult
    _HAS_OPENEVOLVE = True
except ImportError:
    _HAS_OPENEVOLVE = False

    class EvaluationResult:  # type: ignore[no-redef]
        """Minimal fallback when openevolve is not installed."""
        def __init__(self, metrics: Dict[str, Any], artifacts: Dict[str, Any] | None = None):
            self.metrics = metrics
            self.artifacts = artifacts or {}

        @classmethod
        def from_dict(cls, d: Dict[str, Any]) -> "EvaluationResult":
            return cls(metrics=d)


# ---------------------------------------------------------------------------
#  Internal helpers
# ---------------------------------------------------------------------------

def _import_controller(path: str):
    """Load a candidate program file as a Python module."""
    spec = importlib.util.spec_from_file_location("candidate", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load spec from: {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


class _FixedDataset(torch.utils.data.Dataset):
    """
    Deterministic synthetic dataset.

    Each index always returns the SAME (x, label) pair so that:
    - The model can actually learn (no contradictory gradients).
    - Repeated evaluations of the same controller give the same score.
    """

    def __init__(self, n: int = 512, input_dim: int = 64, n_classes: int = 10, seed: int = 42):
        gen = torch.Generator().manual_seed(seed)
        self.x = torch.randn(n, input_dim, generator=gen)
        self.y = torch.randint(0, n_classes, (n,), generator=gen)

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, i: int):
        return self.x[i], self.y[i]


class _TinyModel(nn.Module):
    """Small 3-layer MLP — fast to train, large enough to overfit the fixed dataset."""

    def __init__(self, input_dim: int = 64, hidden: int = 256, n_classes: int = 10):
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


def _run_real_simulation(controller_module: Any, steps: int = 150) -> Dict[str, Any]:
    """
    Run a real (short) training loop and return telemetry.

    The controller_module is the candidate EVOLVE-BLOCK under evaluation.
    """
    # --- setup ---
    torch.manual_seed(42)
    np.random.seed(42)

    cfg = controller_module.ControllerConfig(
        cooldown_steps=2,
        warmup_steps=10,
        log_every=1000,  # suppress per-step logging during evaluation
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
        pin_memory=False,  # pin_memory=True causes issues when device is CPU
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

    # --- training loop ---
    losses: List[float] = []
    step_times: List[float] = []
    data_iter = iter(loader)

    for _ in range(steps):
        # Refill iterator when the epoch ends.
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
        state, _action = controller.step(
            loss,
            per_sample_losses=per_loss,
            batch_indices=batch_indices,
            step_time_s=step_dt,
        )

        loss_val = loss.item()
        if math.isfinite(loss_val):
            losses.append(loss_val)
        step_times.append(step_dt)

        if device.type == "cuda":
            torch.cuda.synchronize()

    if not losses:
        raise RuntimeError("All training steps produced non-finite loss.")

    peak_vram_gb = (
        torch.cuda.max_memory_allocated(device) / 1024 ** 3
        if device.type == "cuda"
        else 0.0
    )

    # Robust initial/final loss: average first 10 and last 20 finite steps.
    warmup = min(10, len(losses) // 4)
    tail = min(20, len(losses) // 4)
    initial_loss = float(np.mean(losses[:warmup]))
    final_loss = float(np.mean(losses[-tail:]))

    return {
        "losses": losses,
        "step_times": step_times,
        "peak_vram_gb": peak_vram_gb,
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "device": device.type,
    }


# ---------------------------------------------------------------------------
#  Public evaluate() — called by OpenEvolve
# ---------------------------------------------------------------------------

def evaluate(program_path: str) -> Dict[str, Any]:
    """
    Evaluate a candidate controller and return metrics.

    Returns a dict (or EvaluationResult) with at minimum:
      combined_score  ∈ [0, 1] — higher is better
    """
    try:
        eval_dir = os.path.dirname(os.path.abspath(program_path))
        if eval_dir not in sys.path:
            sys.path.insert(0, eval_dir)

        mod = _import_controller(program_path)
        tel = _run_real_simulation(mod, steps=150)

        # --- loss score (0-1) ---
        init_loss = tel["initial_loss"]
        final_loss = tel["final_loss"]
        loss_drop = max(0.0, (init_loss - final_loss) / max(init_loss, 1e-6))
        # Penalise if training diverged (final > 1.5× initial).
        stability_penalty = 1.0 if final_loss < init_loss * 1.5 else 0.3
        # Scale: a 33 % drop → score ≈ 1.0; smaller drops score proportionally.
        loss_score = float(min(1.0, loss_drop * 3.0) * stability_penalty)

        # --- speed score (0-1) ---
        avg_step_s = float(np.mean(tel["step_times"]))
        # At 20 steps/sec the denominator = 2 → score = 0.5; at 50 steps/sec = 1.
        speed_score = float(1.0 / (1.0 + avg_step_s * 20.0))

        # --- VRAM score (0-1) ---
        vram_gb = float(tel["peak_vram_gb"])
        # On CPU vram_gb == 0 → perfect score.
        vram_score = 1.0 if vram_gb == 0.0 else float(max(0.0, 1.0 - vram_gb / 8.0))

        # --- combined (additive — never collapses to zero from one bad metric) ---
        combined = float(0.6 * loss_score + 0.2 * speed_score + 0.2 * vram_score)

        metrics = {
            "combined_score": combined,
            "loss_score": loss_score,
            "speed_score": speed_score,
            "vram_score": vram_score,
            "initial_loss": init_loss,
            "final_loss": final_loss,
            "loss_drop_pct": float(loss_drop * 100.0),
            "avg_step_ms": float(avg_step_s * 1000.0),
            "peak_vram_gb": vram_gb,
        }

        if _HAS_OPENEVOLVE:
            artifacts = {
                "device": tel["device"],
                "training_steps": len(tel["losses"]),
                "convergence_info": (
                    f"loss {init_loss:.4f} → {final_loss:.4f} "
                    f"(drop {loss_drop * 100:.1f}%)"
                ),
            }
            return EvaluationResult(metrics=metrics, artifacts=artifacts)

        return metrics

    except Exception as exc:  # noqa: BLE001
        tb = traceback.format_exc()
        error_metrics = {
            "combined_score": 0.0,
            "loss_score": 0.0,
            "speed_score": 0.0,
            "vram_score": 0.0,
            "error": str(exc)[:300],
        }
        if _HAS_OPENEVOLVE:
            return EvaluationResult(
                metrics=error_metrics,
                artifacts={"traceback": tb[:1000], "error_type": type(exc).__name__},
            )
        error_metrics["traceback"] = tb[:500]
        return error_metrics


# ---------------------------------------------------------------------------
#  Standalone runner — for quick local testing
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    program = sys.argv[1] if len(sys.argv) > 1 else "initial_program.py"
    result = evaluate(program)
    if hasattr(result, "metrics"):
        print(result.metrics)
    else:
        print(result)
