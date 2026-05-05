"""
titan2d_compare.py — Centerline validation and mass conservation plots.
Run from d:\\cuda\\:  python -m validation.titan2d_compare
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt

from utils import nearest_time_index, binned_profile, fill_nans_linear, ensure_results_dir
from solver import run_gpu_shared
from config import Lx, Ly, DEFAULT_CFL

# ── Config ────────────────────────────────────────────────────────────────────
NX, NY       = 800, 260
SNAP_TIMES   = [0.0, 0.5, 1.0, 1.5, 2.0]
CFL          = 0.3
Y_TARGET     = 0.65
X_MIN, X_MAX = 0.0, 1.5
NBINS        = 400
COLORS       = ['black', 'blue', 'green', 'orange', 'red']
MASS_INTERVAL = 50


# ── Load TITAN2D ──────────────────────────────────────────────────────────────
def load_titan():
    npz = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "titan_height_cells_0to2s.npz")
    data = np.load(npz)
    return data['t'], data['xc'], data['yc'], data['H']


# ── Run solver with mass tracking ─────────────────────────────────────────────
def run_with_mass():
    """Run gpu_shared and collect mass every MASS_INTERVAL steps."""
    import math, time
    import cupy as cp
    from config import gx, gy, gz, mu, k_act, k_pass, DRY, init_heap, get_precision
    from solver.kernels import compile_kernels

    precision = 'float32'
    np_dtype, cp_dtype, ctype = get_precision(precision)

    kernels   = compile_kernels(ctype)
    k_grid_fn = kernels['compute_k_grid']
    wave_fn   = kernels['compute_wave_speeds']
    flux_fn   = kernels['compute_fluxes_shared']
    update_fn = kernels['update_state']

    nx, ny = NX, NY
    bx, by = 16, 16
    grid   = (math.ceil(nx / bx), math.ceil(ny / by))
    block  = (bx, by)

    h0, hu0, hv0, x, y = init_heap(nx, ny, Lx, Ly, np_dtype)
    dx = np.float64(Lx / (nx - 1))
    dy = np.float64(Ly / (ny - 1))

    h   = cp.asarray(h0,  dtype=cp_dtype)
    hu  = cp.asarray(hu0, dtype=cp_dtype)
    hv  = cp.asarray(hv0, dtype=cp_dtype)
    Fx_h  = cp.zeros((ny, nx+1), dtype=cp_dtype)
    Fx_hu = cp.zeros((ny, nx+1), dtype=cp_dtype)
    Fx_hv = cp.zeros((ny, nx+1), dtype=cp_dtype)
    Gy_h  = cp.zeros((ny+1, nx), dtype=cp_dtype)
    Gy_hu = cp.zeros((ny+1, nx), dtype=cp_dtype)
    Gy_hv = cp.zeros((ny+1, nx), dtype=cp_dtype)
    k_grid_d    = cp.full((ny, nx), k_act, dtype=cp_dtype)
    wave_speeds = cp.zeros((ny, nx), dtype=cp_dtype)

    _nx, _ny = np.int32(nx), np.int32(ny)
    _gz      = np_dtype(gz)
    _gx, _gy = np_dtype(gx), np_dtype(gy)
    _mu      = np_dtype(mu)
    _k_act   = np_dtype(k_act)
    _k_pass  = np_dtype(k_pass)
    _DRY     = np_dtype(DRY)
    _dx, _dy = np_dtype(dx), np_dtype(dy)

    targets  = sorted(set(SNAP_TIMES))
    history  = {}
    mass_t   = []
    mass_m   = []
    t_idx    = 0
    t        = 0.0
    steps    = 0

    # initial mass
    initial_mass = float(cp.sum(h)) * float(dx) * float(dy)

    cp.cuda.Stream.null.synchronize()

    while t_idx < len(targets):
        if t >= targets[t_idx] - 1e-10:
            history[targets[t_idx]] = cp.asnumpy(h)
            t_idx += 1
            continue

        k_grid_fn(grid, block, (h, hu, hv, k_grid_d, _nx, _ny, _dx, _dy, _k_act, _k_pass, _DRY))
        wave_fn(grid, block, (h, hu, hv, k_grid_d, wave_speeds, _nx, _ny, _gz, _DRY))
        max_speed = float(cp.max(wave_speeds))
        dt_val    = CFL * min(float(dx), float(dy)) / (max_speed + 1e-8)
        if t_idx < len(targets):
            dt_val = min(dt_val, targets[t_idx] - t)
        _dt = np_dtype(dt_val)

        flux_fn(grid, block, (h, hu, hv, k_grid_d,
                              Fx_h, Fx_hu, Fx_hv, Gy_h, Gy_hu, Gy_hv,
                              _nx, _ny, _gz, _DRY))
        update_fn(grid, block, (h, hu, hv,
                                Fx_h, Fx_hu, Fx_hv, Gy_h, Gy_hu, Gy_hv,
                                _nx, _ny, _dt, _dx, _dy, _DRY,
                                _gx, _gy, _gz, _mu))
        t     += dt_val
        steps += 1

        if steps % MASS_INTERVAL == 0:
            mass_t.append(t)
            mass_m.append(float(cp.sum(h)) * float(dx) * float(dy))

    return history, x, y, mass_t, mass_m, initial_mass


# ── Centerline helpers ────────────────────────────────────────────────────────
def get_cuda_centerline(h_field, x, y):
    jc = int(np.argmin(np.abs(y - Y_TARGET)))
    mask = (x >= X_MIN) & (x <= X_MAX)
    return x[mask], h_field[jc, mask]


def get_titan_centerline(t_titan, xc, yc, H, target_t):
    tidx = nearest_time_index(t_titan, target_t)
    sel  = (np.abs(yc - Y_TARGET) <= 0.002) & (xc >= X_MIN) & (xc <= X_MAX)
    mids, hb = binned_profile(xc[sel], H[tidx, sel], X_MIN, X_MAX, NBINS)
    hb = fill_nans_linear(mids, hb)
    return mids, hb


# ── Plots ─────────────────────────────────────────────────────────────────────
def plot_centerlines(history, x, y, t_titan, xc, yc, H, out_dir):
    fig, ax = plt.subplots(figsize=(8, 4))
    for snap, color in zip(SNAP_TIMES, COLORS):
        if snap not in history:
            continue
        xn, hn = get_cuda_centerline(history[snap], x, y)
        xt, ht = get_titan_centerline(t_titan, xc, yc, H, snap)
        ax.plot(xt, ht, '--', color=color, linewidth=2.5, label=f"TITAN2D (t={snap}s)")
        ax.plot(xn, hn, '-',  color=color, linewidth=2.0, alpha=0.8, label=f"CUDA (t={snap}s)")

    ax.set_xlim(X_MIN, X_MAX)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Downslope Distance x (m)")
    ax.set_ylabel("Depth h (m)")
    ax.set_title("Centerline Validation: CUDA vs TITAN2D")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = os.path.join(out_dir, "centerline_validation.pdf")
    fig.savefig(out)
    print(f"[SAVED] {out}")
    plt.close(fig)


def plot_mass(mass_t, mass_m, initial_mass, out_dir):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(mass_t, mass_m, 'b-', linewidth=1.5)
    ax.axhline(initial_mass, color='k', linestyle='--', linewidth=1, label='Initial mass')
    ax.set_ylim(0, initial_mass * 1.5)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Total mass (m³)")
    ax.set_title("Mass Conservation")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = os.path.join(out_dir, "mass_conservation.pdf")
    fig.savefig(out)
    print(f"[SAVED] {out}")
    plt.close(fig)


# ── Public entry point (called by GPU runner scripts) ─────────────────────────
def run_titan2d_compare():
    """Run the TITAN2D centerline + mass-conservation validation and save plots."""
    print(f"Running gpu_shared {NX}x{NY} for t={SNAP_TIMES} ...")
    history, x, y, mass_t, mass_m, initial_mass = run_with_mass()

    print("Loading TITAN2D data...")
    t_titan, xc, yc, H = load_titan()

    out_dir = ensure_results_dir("validation")
    plot_centerlines(history, x, y, t_titan, xc, yc, H, out_dir)

    if mass_t:
        plot_mass(mass_t, mass_m, initial_mass, out_dir)

    drift = abs(mass_m[-1] - initial_mass) / initial_mass * 100 if mass_m else 0
    print(f"Mass drift: {drift:.3f}%")
    print("TITAN2D comparison done.")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"Running gpu_shared {NX}x{NY} for t={SNAP_TIMES} ...")
    history, x, y, mass_t, mass_m, initial_mass = run_with_mass()

    print("Loading TITAN2D data...")
    t_titan, xc, yc, H = load_titan()

    out_dir = ensure_results_dir("validation")
    plot_centerlines(history, x, y, t_titan, xc, yc, H, out_dir)

    if mass_t:
        plot_mass(mass_t, mass_m, initial_mass, out_dir)

    drift = abs(mass_m[-1] - initial_mass) / initial_mass * 100 if mass_m else 0
    print(f"Mass drift: {drift:.3f}%")
    print("Done.")
