%%writefile evaluator_real.py
"""
Evaluator for Universal Adaptive Controller - REAL metrics

Measures 3 real-world dimensions on T4:
  1) loss_score (50%) - preserves accuracy
  2) speed_score (25%) - step time
  3) vram_score (25%) - peak memory

Designed for user goal: quality first, then speed and RAM.
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

def _import_controller(path: str):
    spec = importlib.util.spec_from_file_location("candidate", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod

def _run_real_simulation(controller_module, steps: int = 40):
    """Short live training run on available CPU/GPU"""
    cfg = controller_module.ControllerConfig(
        cooldown_steps=2, warmup_steps=5, log_every=100, verbose=False
    )

    class TinyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(nn.Linear(128, 256), nn.ReLU(), nn.Linear(256, 10))
        def forward(self, x): return self.net(x)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TinyModel().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

    class DS(torch.utils.data.Dataset):
        def __len__(self): return 512
        def __getitem__(self, i):
            return torch.randn(128), torch.randint(0,10,(1,)).squeeze()

    ds = DS()
    loader, hard_sampler, batch_sampler = controller_module.build_dataloader(
        ds, initial_batch_size=32, pruning_ratio=0.0, num_workers=0, pin_memory=False, seed=0
    )

    controller = controller_module.UniversalAdaptiveController(
        model=model, optimizer=opt, train_loader=loader, device=device,
        config=cfg, amp_dtype=torch.float16 if device.type=='cuda' else torch.float32,
        hardness_sampler=hard_sampler, batch_sampler=batch_sampler,
    )

    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    losses, step_times = [], []

    for step in range(steps):
        x = torch.randn(controller._current_effective_batch_size(), 128, device=device)
        y = torch.randint(0,10,(x.size(0),), device=device)

        t0 = time.time()
        with controller.amp_autocast():
            logits = model(x)
            loss = nn.functional.cross_entropy(logits, y)

        controller.backward(loss)
        controller.optimizer_step()

        per_loss = torch.full((x.size(0),), loss.item())
        state, action = controller.step(loss, per_sample_losses=per_loss,
                                       batch_indices=list(range(x.size(0))),
                                       step_time_s=time.time()-t0)
        losses.append(loss.item())
        step_times.append(time.time()-t0)

        if device.type == 'cuda': torch.cuda.synchronize()

    peak_vram = torch.cuda.max_memory_allocated()/1024**3 if device.type=='cuda' else 0.0

    return {
        "losses": losses,
        "step_times": step_times,
        "peak_vram_gb": peak_vram,
        "final_loss": losses[-1],
        "initial_loss": losses[0],
    }

def evaluate(program_path: str) -> Dict[str, Any]:
    try:
        sys.path.insert(0, os.path.dirname(program_path))
        mod = _import_controller(program_path)
        tel = _run_real_simulation(mod, steps=40)

        loss_drop = max(0, (tel["initial_loss"] - tel["final_loss"]) / max(tel["initial_loss"],1e-6))
        loss_stable = 1.0 if tel["final_loss"] < tel["initial_loss"]*1.5 else 0.2
        loss_score = min(1.0, loss_drop*2) * loss_stable

        avg_step = np.mean(tel["step_times"])
        speed_score = 1.0 / (1.0 + avg_step*20)

        vram = tel["peak_vram_gb"]
        vram_score = max(0, 1.0 - vram/8.0)

        combined = (loss_score**0.5) * (speed_score**0.25) * (vram_score**0.25)

        return {
            "combined_score": float(combined),
            "loss_score": float(loss_score),
            "speed_score": float(speed_score),
            "vram_score": float(vram_score),
            "final_loss": float(tel["final_loss"]),
            "avg_step_ms": float(avg_step*1000),
            "peak_vram_gb": float(vram),
            "loss_drop_pct": float(loss_drop*100),
        }
    except Exception as e:
        return {"combined_score": 0.0, "error": str(e)[:200], "traceback": traceback.format_exc()[:300]}

if __name__ == "__main__":
    import sys
    p = sys.argv[1] if len(sys.argv)>1 else "initial_program.py"
    print(evaluate(p))