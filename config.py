"""
config.py — Shared physics constants and helpers for the CUDA avalanche solver.
Savage-Hutter model for 2-D granular avalanche flow on an inclined plane.
"""

import math
import numpy as np

# ---------------------------------------------------------------------------
# Physics constants
# ---------------------------------------------------------------------------
zeta  = math.radians(37.4)   # slope angle (rad)
delta = math.radians(27.0)   # basal friction angle (rad)
phi   = math.radians(37.3)   # internal friction angle (rad)

g  = 9.81
gx = g * math.sin(zeta)      # down-slope gravity component
gy = 0.0
gz = g * math.cos(zeta)      # slope-normal gravity component
mu = math.tan(delta)         # basal friction coefficient

DRY = 1e-4                   # dry-bed depth threshold (m)

# ---------------------------------------------------------------------------
# Earth pressure coefficients (Savage-Hutter active/passive)
# ---------------------------------------------------------------------------
_inner  = 1.0 - (1.0 + math.tan(delta)**2) * math.cos(phi)**2
k_act   = 2.0 * (1.0 - math.sqrt(max(_inner, 0.0))) / math.cos(phi)**2 - 1.0
k_pass  = 2.0 * (1.0 + math.sqrt(max(_inner, 0.0))) / math.cos(phi)**2 - 1.0

# ---------------------------------------------------------------------------
# Domain defaults
# ---------------------------------------------------------------------------
Lx          = 4.0   # metres
Ly          = 1.3   # metres
DEFAULT_CFL = 0.3

# ---------------------------------------------------------------------------
# Precision helper
# ---------------------------------------------------------------------------

def get_precision(precision='float32'):
    """Return (numpy_dtype, cupy_dtype, c_type_string) for the requested precision.

    Parameters
    ----------
    precision : str
        'float32' or 'float64'

    Returns
    -------
    tuple
        (np_dtype, cp_dtype, c_str)  e.g. (np.float32, cp.float32, 'float')
    """
    if precision not in ('float32', 'float64'):
        raise ValueError(f"precision must be 'float32' or 'float64', got {precision!r}")

    np_dtype = np.float32 if precision == 'float32' else np.float64
    c_str    = 'float'    if precision == 'float32' else 'double'

    try:
        import cupy as cp
        cp_dtype = cp.float32 if precision == 'float32' else cp.float64
    except ImportError:
        cp_dtype = None

    return np_dtype, cp_dtype, c_str

# ---------------------------------------------------------------------------
# Initial condition
# ---------------------------------------------------------------------------

def init_heap(nx, ny, Lx=4.0, Ly=1.3, dtype=np.float64):
    """Create the initial hemispherical pile for the Savage-Hutter problem.

    Parameters
    ----------
    nx, ny : int
        Grid resolution.
    Lx, Ly : float
        Domain size in metres.
    dtype : numpy dtype
        Floating-point type for the output arrays.

    Returns
    -------
    h  : ndarray (ny, nx) — initial depth field
    hu : ndarray (ny, nx) — initial x-momentum (zeros)
    hv : ndarray (ny, nx) — initial y-momentum (zeros)
    x  : ndarray (nx,)   — x coordinates
    y  : ndarray (ny,)   — y coordinates

    Notes
    -----
    Pile is a circle centred at (Xc=0.4, Yc=0.65) with radius R=0.06 m;
    depth H0 = R/1.5 = 0.04 m inside, 0 outside.
    """
    x  = np.linspace(0, Lx, nx)
    y  = np.linspace(0, Ly, ny)
    X, Y = np.meshgrid(x, y)
    R2 = (X - 0.4)**2 + (Y - 0.65)**2
    h  = np.where(R2 <= 0.06**2, 0.06 / 1.5, 0.0).astype(dtype)
    hu = np.zeros((ny, nx), dtype=dtype)
    hv = np.zeros((ny, nx), dtype=dtype)
    return h, hu, hv, x, y
