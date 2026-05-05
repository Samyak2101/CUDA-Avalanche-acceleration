"""
cpu_numba.py — Numba JIT CPU solver for the 2D Savage-Hutter avalanche model.
Rusanov (Local Lax-Friedrichs) finite volume, forward Euler time integration.
"""

import time
import numpy as np
from numba import njit, prange

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from config import gx, gy, gz, mu, k_act, k_pass, DRY, Lx, Ly, DEFAULT_CFL, init_heap


# ── Numba kernels ─────────────────────────────────────────────────────────────

@njit(parallel=True)
def compute_k_grid(U, k_grid, nx, ny, dx, dy, k_act, k_pass, DRY):
    for j in prange(1, ny - 1):
        for i in range(1, nx - 1):
            h  = U[0, j, i]
            if h <= DRY:
                k_grid[j, i] = k_act
                continue
            hL = U[0, j, i-1]; hR = U[0, j, i+1]
            hD = U[0, j-1, i]; hU = U[0, j+1, i]
            uL = U[1, j, i-1] / hL if hL > DRY else 0.0
            uR = U[1, j, i+1] / hR if hR > DRY else 0.0
            vD = U[2, j-1, i] / hD if hD > DRY else 0.0
            vU = U[2, j+1, i] / hU if hU > DRY else 0.0
            div = (uR - uL) / (2.0 * dx) + (vU - vD) / (2.0 * dy)
            k_grid[j, i] = k_pass if div < 0.0 else k_act
    # transmissive BC on edges
    k_grid[0,  :] = k_grid[1,    :]
    k_grid[-1, :] = k_grid[-2,   :]
    k_grid[:,  0] = k_grid[:, 1   ]
    k_grid[:, -1] = k_grid[:, -2  ]


@njit
def compute_dt(U, k_grid, nx, ny, gz, dx, dy, CFL, DRY):
    max_speed = 1e-8
    for j in range(ny):
        for i in range(nx):
            h = U[0, j, i]
            if h <= DRY:
                continue
            u = U[1, j, i] / h
            v = U[2, j, i] / h
            c = (k_grid[j, i] * gz * h) ** 0.5
            s = max(abs(u) + c, abs(v) + c)
            if s > max_speed:
                max_speed = s
    return CFL * min(dx, dy) / max_speed


@njit(parallel=True)
def compute_fluxes_and_sources(U, k_grid, Fx, Gy, S, nx, ny, gz, gx, gy, mu, DRY):
    # X-fluxes  Fx shape (3, ny, nx+1)
    for j in prange(ny):
        for i in range(nx - 1):
            hL  = U[0, j, i];   hR  = U[0, j, i+1]
            huL = U[1, j, i];   huR = U[1, j, i+1]
            hvL = U[2, j, i];   hvR = U[2, j, i+1]
            uL  = huL/hL if hL > DRY else 0.0
            uR  = huR/hR if hR > DRY else 0.0
            vL  = hvL/hL if hL > DRY else 0.0
            vR  = hvR/hR if hR > DRY else 0.0
            kL  = k_grid[j, i]; kR = k_grid[j, i+1]
            cL  = (kL * gz * max(hL, 0.0)) ** 0.5
            cR  = (kR * gz * max(hR, 0.0)) ** 0.5
            a   = max(abs(uL)+cL, abs(uR)+cR)
            Fx[0, j, i+1] = 0.5*(hL*uL + hR*uR) - 0.5*a*(hR - hL)
            Fx[1, j, i+1] = 0.5*(hL*uL*uL + 0.5*kL*gz*hL*hL
                                + hR*uR*uR + 0.5*kR*gz*hR*hR) - 0.5*a*(hR*uR - hL*uL)
            Fx[2, j, i+1] = 0.5*(hL*vL*uL + hR*vR*uR) - 0.5*a*(hR*vR - hL*vL)

    # Y-fluxes  Gy shape (3, ny+1, nx)
    for j in prange(ny - 1):
        for i in range(nx):
            hD  = U[0, j,   i];  hU  = U[0, j+1, i]
            huD = U[1, j,   i];  huU = U[1, j+1, i]
            hvD = U[2, j,   i];  hvU = U[2, j+1, i]
            uD  = huD/hD if hD > DRY else 0.0
            uU  = huU/hU if hU > DRY else 0.0
            vD  = hvD/hD if hD > DRY else 0.0
            vU  = hvU/hU if hU > DRY else 0.0
            kD  = k_grid[j, i]; kU = k_grid[j+1, i]
            cD  = (kD * gz * max(hD, 0.0)) ** 0.5
            cU  = (kU * gz * max(hU, 0.0)) ** 0.5
            a   = max(abs(vD)+cD, abs(vU)+cU)
            Gy[0, j+1, i] = 0.5*(hD*vD + hU*vU) - 0.5*a*(hU - hD)
            Gy[1, j+1, i] = 0.5*(hD*uD*vD + hU*uU*vU) - 0.5*a*(hU*uU - hD*uD)
            Gy[2, j+1, i] = 0.5*(hD*vD*vD + 0.5*kD*gz*hD*hD
                                + hU*vU*vU + 0.5*kU*gz*hU*hU) - 0.5*a*(hU*vU - hD*vD)

    # Source terms
    for j in prange(ny):
        for i in range(nx):
            h = U[0, j, i]
            if h > DRY:
                u = U[1, j, i] / h
                v = U[2, j, i] / h
                vmag = (u*u + v*v + 1e-6) ** 0.5
                S[1, j, i] = gx*h - h*gz*mu*u/vmag
                S[2, j, i] = gy*h - h*gz*mu*v/vmag
            else:
                S[1, j, i] = 0.0
                S[2, j, i] = 0.0
            S[0, j, i] = 0.0


@njit(parallel=True)
def update_state(U, Fx, Gy, S, dt, dx, dy, nx, ny, DRY):
    for j in prange(1, ny - 1):
        for i in range(1, nx - 1):
            for f in range(3):
                dFlux = (Fx[f, j, i+1] - Fx[f, j, i]) / dx \
                      + (Gy[f, j+1, i] - Gy[f, j, i]) / dy
                U[f, j, i] -= dt * (dFlux - S[f, j, i])
            # positivity
            if U[0, j, i] < 0.0:
                U[0, j, i] = 0.0
            if U[0, j, i] < DRY:
                U[1, j, i] = 0.0
                U[2, j, i] = 0.0
    # transmissive BCs
    for f in range(3):
        U[f, 0,  :] = U[f, 1,    :]
        U[f, -1, :] = U[f, -2,   :]
        U[f, :,  0] = U[f, :, 1   ]
        U[f, :, -1] = U[f, :, -2  ]


# ── Public interface ──────────────────────────────────────────────────────────

def run_cpu_numba(nx, ny, target_times, CFL=DEFAULT_CFL):
    """Run Numba JIT CPU solver.

    Args:
        nx, ny: grid dimensions
        target_times: list of simulation times to snapshot (must include 0.0 or
                      first snapshot is taken at t=0 automatically)
        CFL: Courant number

    Returns:
        history: dict {time: h_field_2d_array}
        x: 1-D x-coordinates (numpy)
        y: 1-D y-coordinates (numpy)
        wall_time: total wall-clock seconds (excluding JIT warm-up)
        step_count: total timesteps taken
    """
    dtype = np.float64
    h0, hu0, hv0, x, y = init_heap(nx, ny, Lx, Ly, dtype)
    dx = Lx / (nx - 1)
    dy = Ly / (ny - 1)

    # Warm-up JIT with tiny arrays
    _U  = np.zeros((3, 4, 4), dtype)
    _U[0] = 0.01
    _kg = np.full((4, 4), k_act, dtype)
    _Fx = np.zeros((3, 4, 5), dtype)
    _Gy = np.zeros((3, 5, 4), dtype)
    _S  = np.zeros((3, 4, 4), dtype)
    compute_k_grid(_U, _kg, 4, 4, dx, dy, k_act, k_pass, DRY)
    compute_dt(_U, _kg, 4, 4, gz, dx, dy, CFL, DRY)
    compute_fluxes_and_sources(_U, _kg, _Fx, _Gy, _S, 4, 4, gz, gx, gy, mu, DRY)
    update_state(_U, _Fx, _Gy, _S, 1e-4, dx, dy, 4, 4, DRY)

    # Initialise state
    U      = np.zeros((3, ny, nx), dtype)
    U[0]   = h0
    U[1]   = hu0
    U[2]   = hv0
    k_grid = np.full((ny, nx), k_act, dtype)
    Fx     = np.zeros((3, ny, nx + 1), dtype)
    Gy     = np.zeros((3, ny + 1, nx), dtype)
    S      = np.zeros((3, ny, nx), dtype)

    targets  = sorted(set(target_times))
    history  = {}
    t_idx    = 0
    t        = 0.0
    steps    = 0

    t0 = time.perf_counter()

    while t_idx < len(targets):
        # save snapshot
        if t >= targets[t_idx] - 1e-10:
            history[targets[t_idx]] = U[0].copy()
            t_idx += 1
            continue

        compute_k_grid(U, k_grid, nx, ny, dx, dy, k_act, k_pass, DRY)
        dt = compute_dt(U, k_grid, nx, ny, gz, dx, dy, CFL, DRY)

        # clamp to not overshoot next target
        if t_idx < len(targets):
            dt = min(dt, targets[t_idx] - t)

        compute_fluxes_and_sources(U, k_grid, Fx, Gy, S, nx, ny, gz, gx, gy, mu, DRY)
        update_state(U, Fx, Gy, S, dt, dx, dy, nx, ny, DRY)
        t     += dt
        steps += 1

    wall_time = time.perf_counter() - t0
    return history, x, y, wall_time, steps
