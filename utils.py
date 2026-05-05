"""
utils.py — Shared utilities for the CUDA avalanche benchmark suite.
"""

import os
import csv
import time
import numpy as np
from contextlib import contextmanager

import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family":      "serif",
    "font.serif":       ["DejaVu Serif", "Computer Modern Roman", "Times New Roman"],
    "font.size":        11,
    "axes.titlesize":   12,
    "axes.labelsize":   11,
    "legend.fontsize":  10,
})

# ---------------------------------------------------------------------------
# Numerical helpers
# ---------------------------------------------------------------------------

def nearest_time_index(t_array, target_t):
    """Return the index of t_array closest to target_t."""
    return int(np.argmin(np.abs(t_array - target_t)))


def binned_profile(x, h_vals, x_min, x_max, nbins):
    """Bin scattered (x, h) data into uniform bins.

    Parameters
    ----------
    x, h_vals       : 1-D arrays of equal length
    x_min, x_max    : float — bin range
    nbins           : int

    Returns
    -------
    mids  : (nbins,) bin mid-points
    hb    : (nbins,) mean h per bin; NaN for empty bins
    """
    edges = np.linspace(x_min, x_max, nbins + 1)
    mids  = 0.5 * (edges[:-1] + edges[1:])
    hb    = np.full(nbins, np.nan, dtype=np.float64)
    idx   = np.searchsorted(edges, x, side="right") - 1
    valid = (idx >= 0) & (idx < nbins)
    idx   = idx[valid]
    hv    = h_vals[valid]
    s     = np.zeros(nbins, dtype=np.float64)
    c     = np.zeros(nbins, dtype=np.int64)
    np.add.at(s, idx, hv)
    np.add.at(c, idx, 1)
    mask       = c > 0
    hb[mask]   = s[mask] / c[mask]
    return mids, hb


def fill_nans_linear(x, y):
    """Linearly interpolate NaN gaps in y (in-place copy)."""
    y    = y.copy()
    mask = np.isfinite(y)
    if mask.sum() < 2:
        return y
    y[~mask] = np.interp(x[~mask], x[mask], y[mask])
    return y


def calculate_rms_error(x_num, h_num, x_ref, h_ref):
    """Interpolate numerical h onto the reference x grid and return RMSE."""
    h_interp = np.interp(x_ref, x_num, h_num)
    return np.sqrt(np.mean((h_interp - h_ref) ** 2))

# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------

@contextmanager
def timer(label=""):
    """Context manager that measures wall time.

    Usage::

        with timer("my step") as t:
            do_work()
        print(t['elapsed'])
    """
    start  = time.perf_counter()
    result = {"elapsed": 0.0}
    try:
        yield result
    finally:
        result["elapsed"] = time.perf_counter() - start
        if label:
            print(f"[TIMER] {label}: {result['elapsed']:.3f}s")

# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def ensure_results_dir(subdir=""):
    """Create results/<subdir>/ next to this file if it does not exist.

    Returns the absolute path.
    """
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", subdir)
    os.makedirs(path, exist_ok=True)
    return path


def save_benchmark_csv(filepath_or_rows, headers_or_filepath=None, rows=None):
    """Write benchmark results to a CSV file.

    Accepts two calling conventions:

    1. New style (used by data_layout, shared_mem_impact, precision_study,
       block_size_sweep): ``save_benchmark_csv(rows, filepath)``
       where *rows* is a list of dicts.  Column headers are derived from the
       keys of the first dict.

    2. Legacy style: ``save_benchmark_csv(filepath, headers, rows)``
       where *rows* is a list of sequences.
    """
    if isinstance(filepath_or_rows, list):
        # New style: (rows_of_dicts, filepath)
        dict_rows = filepath_or_rows
        filepath  = headers_or_filepath
        if not dict_rows:
            open(filepath, 'w').close()
            print(f"[SAVED] {filepath}")
            return
        headers  = list(dict_rows[0].keys())
        raw_rows = [[r.get(h, "") for h in headers] for r in dict_rows]
    else:
        # Legacy style: (filepath, headers, rows_of_sequences)
        filepath  = filepath_or_rows
        headers   = headers_or_filepath
        raw_rows  = rows

    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(raw_rows)
    print(f"[SAVED] {filepath}")
