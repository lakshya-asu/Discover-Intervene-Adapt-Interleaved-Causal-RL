from __future__ import annotations

from typing import Any, Dict, Optional
import os
import time

try:
    from torch.utils.tensorboard import SummaryWriter  # type: ignore
except Exception:
    SummaryWriter = None


class TBLogger:
    def __init__(self, logdir: str):
        os.makedirs(logdir, exist_ok=True)
        self.logdir = logdir
        self.writer = SummaryWriter(logdir) if SummaryWriter else None
        self.step = 0

    def add_scalar(self, tag: str, value: float, step: Optional[int] = None):
        self.step = int(self.step + 1 if step is None else step)
        if self.writer:
            self.writer.add_scalar(tag, value, self.step)
        else:
            print(f"[TBLogger:{self.step}] {tag}={value:.6f}")

    def flush(self):
        if self.writer:
            self.writer.flush()

    def close(self):
        if self.writer:
            self.writer.close()
