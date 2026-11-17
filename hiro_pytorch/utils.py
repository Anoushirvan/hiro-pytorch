"""Small helpers: seeding and CSV logging."""
from __future__ import annotations

import csv
import os
import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class CSVLogger:
    """Appends dict rows to a CSV file; the header is taken from the first row."""

    def __init__(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.path = path
        self._fields = None

    def log(self, row: dict) -> None:
        new = self._fields is None
        if new:
            self._fields = list(row.keys())
        with open(self.path, "w" if new else "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=self._fields)
            if new:
                w.writeheader()
            w.writerow(row)
