"""
giid.py — Grid-independent convergence (self-convergence without reference data).
Uses a different domain (Lx=Ly=15m) and simpler physics parameters.
Run from d:\\cuda\\:  python -m validation.giid
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from math import log

from utils import ensure_results_dir
from solver import run_cpu_numba

# ── Local physics overrides (do NOT modify config.py) ─────────────────────────
import math as _math
_zeta = _math.radians(32)
_delta = _math.radians(22)
_phi  = _math.radians(29)

LX_GIID = 15.0
LY_GIID = 7.5   # Y_centre = 7.5 so heap centred correctly
T_END   = 1.0

RESOLUTIONS = [51, 101, 201, 401, 801]


# ── GIID-specific solver ─────────────────────────────────────────────────────
def run_giid(N):
    """Run simplified CPU solver with GIID physics on an (N x N) grid."""
    import time
    from numba import njit, prange

    # Local parameters
    gz_   = 9.81 * _math.cos(_zeta)
    gx_   = 9.81 * _math.sin(_zeta)
    gy_   = 0.0
    mu_   = _math.tan(_delta)
    k_    = (2.0 * _math.cos(_zeta) * (1 - _math.cos(_phi) / _math.cos(_delta))
             / (1 + _math.sin(_phi) * _math.sin(_delta)) - 1)  # simplified k_act
    DRY_  = 1e-4
    CFL_  = 0.3

    dx = LX_GIID / (N - 1)
    dy = LX_GIID / (N - 1)   # square domain

    x = np.linspace(0, LX_GIID, N, dtype=np.float64)
    y = np.linspace(0, LX_GIID, N, dtype=np.float64)
    X, Y = np.meshgrid(x, y)

    R2   = (X - 3.0)**2 + (Y - 7.5)**2
    h0   = np.maximum(1.0 - R2, 0.0).astype(np.float64)
    hu0  = np.zeros_like(h0)
    hv0  = np.zeros_like(h0)

    U = np.zeros((3, N, N), np.float64)
    U[0] = h0; U[1] = hu0; U[2] = hv0
    k_grid = np.full((N, N), max(k_, 0.5), np.float64)
    Fx = np.zeros((3, N, N+1), np.float64)
    Gy = np.zeros((3, N+1, N), np.float64)
    S  = np.zeros((3, N, N),   np.float64)

    # ── inline Rusanov (to avoid reusing config-dependent Numba kernels) ──
    def step(U, k_grid, Fx, Gy, S, dt):
        ny_, nx_ = N, N
        # x-fluxes
        for j in range(ny_):
            for i in range(nx_-1):
                hL, hR = U[0,j,i], U[0,j,i+1]
                uL = U[1,j,i]/hL if hL > DRY_ else 0.0
                uR = U[1,j,i+1]/hR if hR > DRY_ else 0.0
                vL = U[2,j,i]/hL if hL > DRY_ else 0.0
                vR = U[2,j,i+1]/hR if hR > DRY_ else 0.0
                k  = k_grid[j,i]
                cL = (k * gz_ * max(hL, 0.0))**0.5
                cR = (k * gz_ * max(hR, 0.0))**0.5
                a  = max(abs(uL)+cL, abs(uR)+cR)
                Fx[0,j,i+1] = 0.5*(hL*uL+hR*uR) - 0.5*a*(hR-hL)
                Fx[1,j,i+1] = 0.5*(hL*uL**2+0.5*k*gz_*hL**2
                                  +hR*uR**2+0.5*k*gz_*hR**2) - 0.5*a*(hR*uR-hL*uL)
                Fx[2,j,i+1] = 0.5*(hL*vL*uL+hR*vR*uR) - 0.5*a*(hR*vR-hL*vL)
        # y-fluxes
        for j in range(ny_-1):
            for i in range(nx_):
                hD, hU = U[0,j,i], U[0,j+1,i]
                uD = U[1,j,i]/hD if hD > DRY_ else 0.0
                uU = U[1,j+1,i]/hU if hU > DRY_ else 0.0
                vD = U[2,j,i]/hD if hD > DRY_ else 0.0
                vU = U[2,j+1,i]/hU if hU > DRY_ else 0.0
                k  = k_grid[j,i]
                cD = (k * gz_ * max(hD, 0.0))**0.5
                cU = (k * gz_ * max(hU, 0.0))**0.5
                a  = max(abs(vD)+cD, abs(vU)+cU)
                Gy[0,j+1,i] = 0.5*(hD*vD+hU*vU) - 0.5*a*(hU-hD)
                Gy[1,j+1,i] = 0.5*(hD*uD*vD+hU*uU*vU) - 0.5*a*(hU*uU-hD*uD)
                Gy[2,j+1,i] = 0.5*(hD*vD**2+0.5*k*gz_*hD**2
                                  +hU*vU**2+0.5*k*gz_*hU**2) - 0.5*a*(hU*vU-hD*vD)
        # source
        for j in range(ny_):
            for i in range(nx_):
                h = U[0,j,i]
                if h > DRY_:
                    u = U[1,j,i]/h; v = U[2,j,i]/h
                    vm = (u*u+v*v+1e-6)**0.5
                    S[1,j,i] = gx_*h - h*gz_*mu_*u/vm
                    S[2,j,i] = gy_*h - h*gz_*mu_*v/vm
                else:
                    S[1,j,i] = S[2,j,i] = 0.0
                S[0,j,i] = 0.0
        # update
        for j in range(1, ny_-1):
            for i in range(1, nx_-1):
                for f in range(3):
                    df = (Fx[f,j,i+1]-Fx[f,j,i])/dx + (Gy[f,j+1,i]-Gy[f,j,i])/dy
                    U[f,j,i] -= dt*(df - S[f,j,i])
                if U[0,j,i] < 0: U[0,j,i] = 0.0
                if U[0,j,i] < DRY_: U[1,j,i] = U[2,j,i] = 0.0
        # BCs
        for f in range(3):
            U[f,0,:]  = U[f,1,:]
            U[f,-1,:] = U[f,-2,:]
            U[f,:,0]  = U[f,:,1]
            U[f,:,-1] = U[f,:,-2]

    # time-march
    t = 0.0
    while t < T_END - 1e-10:
        # wave speed
        max_s = 1e-8
        for j in range(N):
            for i in range(N):
                h = U[0,j,i]
                if h > DRY_:
                    u = U[1,j,i]/h; v = U[2,j,i]/h
                    c = (k_grid[j,i]*gz_*h)**0.5
                    s = max(abs(u)+c, abs(v)+c)
                    if s > max_s: max_s = s
        dt = CFL_ * min(dx, dy) / max_s
        dt = min(dt, T_END - t)
        step(U, k_grid, Fx, Gy, S, dt)
        t += dt

    return U[0].copy(), dx


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=== GIID Self-Convergence Study ===")
    print(f"Resolutions: {RESOLUTIONS}  (T_end={T_END}s)")

    h_fields = []
    dxs      = []
    for N in RESOLUTIONS:
        print(f"  N={N} ...", flush=True)
        h, dx = run_giid(N)
        h_fields.append(h)
        dxs.append(dx)
        print(f"    dx={dx:.5f}  max_h={h.max():.4f}")

    # L1 errors between successive grids (subsample fine -> coarse)
    errors = []
    print("\n  L1 errors (successive grid pairs):")
    for i in range(len(RESOLUTIONS) - 1):
        h_coarse = h_fields[i]
        h_fine   = h_fields[i+1]
        # subsample fine (2N-1) every 2 -> N
        h_fine_sub = h_fine[::2, ::2]
        Nc = RESOLUTIONS[i]
        # trim to same shape
        s = min(h_coarse.shape[0], h_fine_sub.shape[0])
        err = np.mean(np.abs(h_fine_sub[:s, :s] - h_coarse[:s, :s]))
        errors.append(err)
        print(f"    N={RESOLUTIONS[i]}->{RESOLUTIONS[i+1]}  L1={err:.6f}")

    print("\n  Orders of accuracy:")
    for i in range(len(errors) - 1):
        if errors[i] > 0 and errors[i+1] > 0:
            p = log(errors[i] / errors[i+1]) / log(2.0)
            print(f"    N={RESOLUTIONS[i+1]}->{RESOLUTIONS[i+2]}:  p = {p:.3f}")

    # Plot
    out_dir = ensure_results_dir("validation")
    fig, ax = plt.subplots(figsize=(6, 4))
    dx_arr  = np.array(dxs[:-1])
    err_arr = np.array(errors)
    ax.loglog(dx_arr, err_arr, 'o-', label='L1 error')
    ref = err_arr[0] * (dx_arr / dx_arr[0])
    ax.loglog(dx_arr, ref, 'k--', label='O(Δx)')
    ax.set_xlabel('Δx (m)')
    ax.set_ylabel('L1 error (m)')
    ax.set_title('GIID Self-Convergence (no reference)')
    ax.legend()
    ax.grid(True, which='both', alpha=0.3)
    fig.tight_layout()
    out = os.path.join(out_dir, "giid_convergence.pdf")
    fig.savefig(out)
    print(f"\n[SAVED] {out}")
    plt.close(fig)
    print("Done.")
