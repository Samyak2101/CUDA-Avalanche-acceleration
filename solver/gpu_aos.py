"""
gpu_aos.py — GPU solver with Array-of-Structures (AoS) data layout.
Purpose: demonstrate non-coalesced access penalty vs. SoA (gpu_global.py).
U shape: (ny, nx, 3)  where U[:,:,0]=h, U[:,:,1]=hu, U[:,:,2]=hv.
"""

import math
import time
import numpy as np

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from config import gx, gy, gz, mu, k_act, k_pass, DRY, Lx, Ly, DEFAULT_CFL, init_heap, get_precision


_AOS_KERNEL_TEMPLATE = """
extern "C" {{
typedef {ctype} real;

__global__ void compute_k_grid_aos_kernel(
    const real* U, real* k_grid,
    int nx, int ny, real dx, real dy,
    real k_act, real k_pass, real DRY)
{{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i <= 0 || i >= nx-1 || j <= 0 || j >= ny-1) return;
    int idx = j*nx + i;
    real h = U[idx*3 + 0];
    if (h <= DRY) {{ k_grid[idx] = k_act; return; }}
    real hL = U[(j*nx + i-1)*3];  real hR = U[(j*nx + i+1)*3];
    real hD = U[((j-1)*nx + i)*3]; real hU = U[((j+1)*nx + i)*3];
    real uL = (hL > DRY) ? U[(j*nx + i-1)*3+1]/hL : (real)0;
    real uR = (hR > DRY) ? U[(j*nx + i+1)*3+1]/hR : (real)0;
    real vD = (hD > DRY) ? U[((j-1)*nx+i)*3+2]/hD : (real)0;
    real vU = (hU > DRY) ? U[((j+1)*nx+i)*3+2]/hU : (real)0;
    real div = (uR-uL)/((real)2*dx) + (vU-vD)/((real)2*dy);
    k_grid[idx] = (div < (real)0) ? k_pass : k_act;
}}

__global__ void compute_wave_speeds_aos_kernel(
    const real* U, const real* k_grid, real* wave_speeds,
    int nx, int ny, real gz, real DRY)
{{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= nx || j >= ny) return;
    int idx = j*nx + i;
    real h = U[idx*3];
    if (h <= DRY) {{ wave_speeds[idx] = (real)0; return; }}
    real u = U[idx*3+1]/h, v = U[idx*3+2]/h;
    real c = sqrt(k_grid[idx] * gz * h);
    wave_speeds[idx] = fmax(fabs(u)+c, fabs(v)+c);
}}

__global__ void compute_fluxes_aos_kernel(
    const real* U, const real* k_grid,
    real* Fx_h, real* Fx_hu, real* Fx_hv,
    real* Gy_h, real* Gy_hu, real* Gy_hv,
    int nx, int ny, real gz, real DRY)
{{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < nx-1 && j < ny) {{
        int idL = (j*nx + i)*3,  idR = (j*nx + i+1)*3;
        real hL = U[idL], hR = U[idR];
        real uL = (hL>DRY)?U[idL+1]/hL:(real)0, uR = (hR>DRY)?U[idR+1]/hR:(real)0;
        real vL = (hL>DRY)?U[idL+2]/hL:(real)0, vR = (hR>DRY)?U[idR+2]/hR:(real)0;
        real kL = k_grid[j*nx+i], kR = k_grid[j*nx+i+1];
        real cL = sqrt(kL*gz*fmax(hL,(real)0)), cR = sqrt(kR*gz*fmax(hR,(real)0));
        real a  = fmax(fabs(uL)+cL, fabs(uR)+cR);
        int fx  = j*(nx+1) + (i+1);
        Fx_h [fx] = (real)0.5*(hL*uL+hR*uR) - (real)0.5*a*(hR-hL);
        Fx_hu[fx] = (real)0.5*(hL*uL*uL+(real)0.5*kL*gz*hL*hL+hR*uR*uR+(real)0.5*kR*gz*hR*hR) - (real)0.5*a*(hR*uR-hL*uL);
        Fx_hv[fx] = (real)0.5*(hL*vL*uL+hR*vR*uR) - (real)0.5*a*(hR*vR-hL*vL);
    }}

    if (i < nx && j < ny-1) {{
        int idD = (j*nx+i)*3,     idU = ((j+1)*nx+i)*3;
        real hD = U[idD], hU = U[idU];
        real uD = (hD>DRY)?U[idD+1]/hD:(real)0, uU = (hU>DRY)?U[idU+1]/hU:(real)0;
        real vD = (hD>DRY)?U[idD+2]/hD:(real)0, vU = (hU>DRY)?U[idU+2]/hU:(real)0;
        real kD = k_grid[j*nx+i], kU = k_grid[(j+1)*nx+i];
        real cD = sqrt(kD*gz*fmax(hD,(real)0)), cU = sqrt(kU*gz*fmax(hU,(real)0));
        real a  = fmax(fabs(vD)+cD, fabs(vU)+cU);
        int gy_idx = (j+1)*nx + i;
        Gy_h [gy_idx] = (real)0.5*(hD*vD+hU*vU) - (real)0.5*a*(hU-hD);
        Gy_hu[gy_idx] = (real)0.5*(hD*uD*vD+hU*uU*vU) - (real)0.5*a*(hU*uU-hD*uD);
        Gy_hv[gy_idx] = (real)0.5*(hD*vD*vD+(real)0.5*kD*gz*hD*hD+hU*vU*vU+(real)0.5*kU*gz*hU*hU) - (real)0.5*a*(hU*vU-hD*vD);
    }}
}}

__global__ void update_state_aos_kernel(
    real* U,
    const real* Fx_h,  const real* Fx_hu, const real* Fx_hv,
    const real* Gy_h,  const real* Gy_hu, const real* Gy_hv,
    int nx, int ny, real dt, real dx, real dy, real DRY,
    real gx, real gy, real gz, real mu)
{{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i<=0||i>=nx-1||j<=0||j>=ny-1) return;
    int idx = j*nx+i;
    real h  = U[idx*3];
    real dFh  = (Fx_h [j*(nx+1)+i+1]-Fx_h [j*(nx+1)+i])/dx + (Gy_h [(j+1)*nx+i]-Gy_h [j*nx+i])/dy;
    real dFhu = (Fx_hu[j*(nx+1)+i+1]-Fx_hu[j*(nx+1)+i])/dx + (Gy_hu[(j+1)*nx+i]-Gy_hu[j*nx+i])/dy;
    real dFhv = (Fx_hv[j*(nx+1)+i+1]-Fx_hv[j*(nx+1)+i])/dx + (Gy_hv[(j+1)*nx+i]-Gy_hv[j*nx+i])/dy;
    real h_new = fmax(h - dt*dFh, (real)0);
    real S_hu = (real)0, S_hv = (real)0;
    if (h > DRY) {{
        real u = U[idx*3+1]/h, v = U[idx*3+2]/h;
        real vm = sqrt(u*u + v*v + (real)1e-6);
        S_hu = gx*h - h*gz*mu*u/vm;
        S_hv = gy*h - h*gz*mu*v/vm;
    }}
    U[idx*3]   = h_new;
    U[idx*3+1] = (h_new>DRY) ? U[idx*3+1] - dt*(dFhu-S_hu) : (real)0;
    U[idx*3+2] = (h_new>DRY) ? U[idx*3+2] - dt*(dFhv-S_hv) : (real)0;
}}
}} // extern "C"
"""


def _compile_aos(ctype):
    import cupy as cp
    code   = _AOS_KERNEL_TEMPLATE.format(ctype=ctype)
    module = cp.RawModule(code=code, options=('--std=c++14',))
    return {
        'compute_k_grid':    module.get_function('compute_k_grid_aos_kernel'),
        'compute_wave_speeds': module.get_function('compute_wave_speeds_aos_kernel'),
        'compute_fluxes':    module.get_function('compute_fluxes_aos_kernel'),
        'update_state':      module.get_function('update_state_aos_kernel'),
    }


def run_gpu_aos(nx, ny, target_times, CFL=DEFAULT_CFL,
                precision='float32', block_size=(16, 16)):
    """Same interface as run_gpu_global but uses AoS data layout."""
    import cupy as cp

    np_dtype, cp_dtype, ctype = get_precision(precision)
    itemsize = np.dtype(np_dtype).itemsize

    kernels   = _compile_aos(ctype)
    k_grid_fn = kernels['compute_k_grid']
    wave_fn   = kernels['compute_wave_speeds']
    flux_fn   = kernels['compute_fluxes']
    update_fn = kernels['update_state']

    bx, by = block_size
    grid   = (math.ceil(nx / bx), math.ceil(ny / by))
    block  = (bx, by)

    h0, hu0, hv0, x, y = init_heap(nx, ny, Lx, Ly, np_dtype)
    dx = np.float64(Lx / (nx - 1))
    dy = np.float64(Ly / (ny - 1))

    # AoS: interleave (h, hu, hv) per cell
    U_np = np.stack([h0, hu0, hv0], axis=-1)   # (ny, nx, 3)
    U    = cp.asarray(U_np, dtype=cp_dtype)

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

    targets = sorted(set(target_times))
    history = {}
    t_idx   = 0
    t       = 0.0
    steps   = 0
    perf    = {'times': [], 'mcups': [], 'effective_bw_gbs': []}
    _step_timer = time.perf_counter()

    cp.cuda.Stream.null.synchronize()
    t0 = time.perf_counter()

    while t_idx < len(targets):
        if t >= targets[t_idx] - 1e-10:
            U_host = cp.asnumpy(U)
            history[targets[t_idx]] = U_host[:, :, 0]
            t_idx += 1
            continue

        k_grid_fn(grid, block,
                  (U, k_grid_d, _nx, _ny, _dx, _dy, _k_act, _k_pass, _DRY))
        wave_fn(grid, block,
                (U, k_grid_d, wave_speeds, _nx, _ny, _gz, _DRY))
        max_speed = float(cp.max(wave_speeds))
        dt_val    = CFL * min(float(dx), float(dy)) / (max_speed + 1e-8)
        if t_idx < len(targets):
            dt_val = min(dt_val, targets[t_idx] - t)
        _dt = np_dtype(dt_val)

        flux_fn(grid, block,
                (U, k_grid_d,
                 Fx_h, Fx_hu, Fx_hv, Gy_h, Gy_hu, Gy_hv,
                 _nx, _ny, _gz, _DRY))
        update_fn(grid, block,
                  (U,
                   Fx_h, Fx_hu, Fx_hv, Gy_h, Gy_hu, Gy_hv,
                   _nx, _ny, _dt, _dx, _dy, _DRY,
                   _gx, _gy, _gz, _mu))

        t     += dt_val
        steps += 1

        if steps % 50 == 0:
            cp.cuda.Stream.null.synchronize()
            elapsed = time.perf_counter() - _step_timer
            mcups   = (nx * ny * 50) / elapsed / 1e6
            bw_gbs  = (9 * itemsize * nx * ny * 50) / elapsed / 1e9
            perf['times'].append(t)
            perf['mcups'].append(mcups)
            perf['effective_bw_gbs'].append(bw_gbs)
            _step_timer = time.perf_counter()

    cp.cuda.Stream.null.synchronize()
    wall_time = time.perf_counter() - t0
    return history, x, y, wall_time, steps, perf
