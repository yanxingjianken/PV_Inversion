"""sh_ppvi.clim — 30-day running-mean climatology helper.

Builds a 30-day running mean centred on a chosen event date from daily ERA5
files. The mean is used as the BALNC reference state; the event-day anomaly
(q' = q_event - q_clim) drives the inversion.
"""
from __future__ import annotations
from pathlib import Path
from datetime import datetime, timedelta
from typing import List
import numpy as np
import xarray as xr

from .io import load_global_era5

__all__ = ["list_clim_files", "load_clim_mean"]


def list_clim_files(data_dir: Path, event_date: str, half_window: int = 15,
                    pattern: str = "era5_{date}_00Z.nc") -> List[Path]:
    """List of daily ERA5 files within ±half_window days of event_date."""
    ev = datetime.strptime(event_date, "%Y-%m-%d")
    files = []
    for off in range(-half_window, half_window + 1):
        d = ev + timedelta(days=off)
        p = data_dir / pattern.format(date=d.strftime("%Y-%m-%d"))
        if p.exists():
            files.append(p)
    return files


def load_clim_mean(files: List[Path]) -> xr.Dataset:
    """Open the listed files and return the time-mean dataset.

    Each file is a single-time ERA5 snapshot; we concatenate along a synthetic
    "time" axis and take the mean.
    """
    if not files:
        raise FileNotFoundError("No climatology files found.")
    # Use load_global_era5 so each file gets lat-reversed, level-reordered.
    dsets = [load_global_era5(f) for f in files]
    stacked = xr.concat(dsets, dim="member")
    return stacked.mean(dim="member", keep_attrs=True)
