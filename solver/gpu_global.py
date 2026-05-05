"""
gpu_global.py — CuPy GPU solver, SoA layout, global-memory flux kernel.
"""

import math
import time
import numpy as np

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from config import gx, gy, gz, mu, k_act, k_pass, DRY, Lx, Ly, DEFAULT_CFL, init_heap, get_precision
from .kernels import compile_kernels


def run_gpu_global(nx, ny, target_times, CFL=DEFAULT_CFL,
                   precision='float32', block_size=(16, 16)):
    """Run GPU solver with global memory flux kernel (SoA layout).

    Returns:
        history: dict {time: h_field_numpy_array}
        x, y: 1-D coordinate arrays (numpy)
        wall_time: float seconds (excludes kernel compilation)
        step_count: int
        perf_metrics: dict with 'times', 'mcups', 'effective_bw_gbs'
    """
    import cupy as cp

    np_dtype, cp_dtype, ctype = get_precision(precision)
    itemsize = np.dtype(np_dtype).itemsize

    # Compile kernels (excluded from timing)
    kernels = compile_kernels(ctype)
    k_grid_fn   = kernels['compute_k_grid']
    wave_fn     = kernels['compute_wave_speeds']
    flux_fn     = kernels['compute_fluxes_global']
    update_fn   = kernels['update_state']

    # Grid / block
    bx, by = block_size
    grid = (math.ceil(nx / bx), math.ceil(ny / by))
    block = (bx, by)

    h0, hu0, hv0, x, y = init_heap(nx, ny, Lx, Ly, np_dtype)
    dx = np.float64(Lx / (nx - 1))
    dy = np.float64(Ly / (ny - 1))

    # Device arrays (SoA)
    h   = cp.asarray(h0,  dtype=cp_dtype)
    hu  = cp.asarray(hu0, dtype=cp_dtype)
    hv  = cp.asarray(hv0, dtype=cp_dtype)
    Fx_h  = cp.zeros((ny, nx+1), dtype=cp_dtype)
    Fx_hu = cp.zeros((ny, nx+1), dtype=cp_dtype)
    Fx_hv = cp.zeros((ny, nx+1), dtype=cp_dtype)
    Gy_h  = cp.zeros((ny+1, nx), dtype=cp_dtype)
    Gy_hu = cp.zeros((ny+1, nx), dtype=cp_dtype)
    Gy_hv = cp.zeros((ny+1, nx), dtype=cp_dtype)
    k_grid_d     = cp.full((ny, nx), k_act, dtype=cp_dtype)
    wave_speeds  = cp.zeros((ny, nx), dtype=cp_dtype)

    # Scalar args (must match CUDA types)
    _nx, _ny     = np.int32(nx), np.int32(ny)
    _gz          = np_dtype(gz)
    _gx, _gy     = np_dtype(gx), np_dtype(gy)
    _mu          = np_dtype(mu)
    _k_act       = np_dtype(k_act)
    _k_pass      = np_dtype(k_pass)
    _DRY         = np_dtype(DRY)
    _dx, _dy     = np_dtype(dx), np_dtype(dy)

    targets  = sorted(set(target_times))
    history  = {}
    t_idx    = 0
    t        = 0.0
    steps    = 0
    perf     = {'times': [], 'mcups': [], 'effective_bw_gbs': []}
    _step_timer = time.perf_counter()

    cp.cuda.Stream.null.synchronize()
    t0 = time.perf_counter()

    while t_idx < len(targets):
        if t >= targets[t_idx] - 1e-10:
            history[targets[t_idx]] = cp.asnumpy(h)
            t_idx += 1
            continue

        # k_grid  (called every step — critical correctness fix)
        k_grid_fn(grid, block,
                  (h, hu, hv, k_grid_d, _nx, _ny, _dx, _dy,
                   _k_act, _k_pass, _DRY))

        # wave speeds → dt
        wave_fn(grid, block,
                (h, hu, hv, k_grid_d, wave_speeds, _nx, _ny, _gz, _DRY))
        max_speed = float(cp.max(wave_speeds))
        dt_val    = CFL * min(float(dx), float(dy)) / (max_speed + 1e-8)
        if t_idx < len(targets):
            dt_val = min(dt_val, targets[t_idx] - t)
        _dt = np_dtype(dt_val)

        # fluxes
        flux_fn(grid, block,
                (h, hu, hv, k_grid_d,
                 Fx_h, Fx_hu, Fx_hv, Gy_h, Gy_hu, Gy_hv,
                 _nx, _ny, _gz, _DRY))

        # update
        update_fn(grid, block,
                  (h, hu, hv,
                   Fx_h, Fx_hu, Fx_hv, Gy_h, Gy_hu, Gy_hv,
                   _nx, _ny, _dt, _dx, _dy, _DRY,
                   _gx, _gy, _gz, _mu))

        t     += dt_val
        steps += 1

        # Performance every 50 steps
        if steps % 50 == 0:
            cp.cuda.Stream.null.synchronize()
            elapsed = time.perf_counter() - _step_timer
            mcups   = (nx * ny * 50) / elapsed / 1e6
            bytes_per_cell = 9 * itemsize
            bw_gbs  = (bytes_per_cell * nx * ny * 50) / elapsed / 1e9
            perf['times'].append(t)
            perf['mcups'].append(mcups)
            perf['effective_bw_gbs'].append(bw_gbs)
            _step_timer = time.perf_counter()

    cp.cuda.Stream.null.synchronize()
    wall_time = time.perf_counter() - t0
    return history, x, y, wall_time, steps, perf
