"""Saving and loading: genome checkpoints (.npz) and training metrics (CSV).

A checkpoint always embeds the full Config as JSON. A genome is meaningless
without the physics and sensor layout it was evolved for — change the ray
angles and the same weights drive a different (usually broken) car — so the
two travel together, and loaders can rebuild the exact world the champion
was trained in.
"""

from __future__ import annotations

import csv
import os
from typing import Any

import numpy as np

from .config import Config


def save_genome(path: str, genome: np.ndarray, config: Config,
                meta: dict[str, Any] | None = None) -> None:
    """Write one genome + its config + optional scalar metadata to .npz."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    payload: dict[str, Any] = {
        "genome": genome.astype(np.float32),
        "config_json": np.array(config.to_json()),
    }
    for key, value in (meta or {}).items():
        payload[f"meta_{key}"] = np.array(value)
    # Write-then-rename so an interrupt (Ctrl-C mid-save) can never leave a
    # truncated checkpoint: os.replace is atomic, so the file at `path` is
    # always either the complete old champion or the complete new one.
    tmp = path + ".tmp.npz"
    np.savez(tmp, **payload)
    os.replace(tmp, path)


def load_genome(path: str) -> tuple[np.ndarray, Config, dict[str, Any]]:
    """Read back (genome, config, meta) from a checkpoint written above."""
    with np.load(path) as data:
        genome = data["genome"].astype(np.float32)
        config = Config.from_json(str(data["config_json"]))
        meta = {key[5:]: data[key].item() for key in data.files
                if key.startswith("meta_")}
    return genome, config, meta


class MetricsLogger:
    """Append-only CSV, one row per generation. Safe to plot mid-training.

    Columns missing from a row (e.g. validation columns on non-validation
    generations) are left empty rather than faked — sparse but honest.
    """

    COLUMNS = [
        "gen", "difficulty", "track_seeds", "sigma",
        "best_fit", "mean_fit", "median_fit", "p90_fit",
        "best_laps", "alive_rate_end", "crash_rate",
        "steps_simulated", "wall_s",
        "val_mean", "val_min", "val_gap",
        # realized per-track difficulties (";"-joined) — the generator may
        # back off below what the curriculum requested
        "realized_ds",
        # where on the track cars crashed: 10-bin histogram over lap
        # fraction (";"-joined counts) — the cheap failure-mode diagnostic
        "crash_hist",
    ]

    def __init__(self, path: str):
        self.path = path
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        if not os.path.exists(path):
            with open(path, "w", newline="") as f:
                csv.writer(f).writerow(self.COLUMNS)

    def log(self, **row: Any) -> None:
        unknown = set(row) - set(self.COLUMNS)
        if unknown:
            raise ValueError(f"unknown metric columns: {sorted(unknown)}")
        with open(self.path, "a", newline="") as f:
            csv.writer(f).writerow([row.get(col, "") for col in self.COLUMNS])
