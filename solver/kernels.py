"""
kernels.py — CUDA C++ kernel source templates for the Savage-Hutter solver.
"""


def get_kernel_code(ctype='float'):
    """Returns complete CUDA C++ source string with typedef real = ctype."""
    return f"""
typedef {ctype} real;

extern "C" {{

// -- Kernel 1: compute_k_grid -----------------------------------------------
__global__ void compute_k_grid_kernel(
    const real* h, const real* hu, const real* hv,
    real* k_grid,
    int nx, int ny, real dx, real dy,
    real k_act, real k_pass, real DRY)
{{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i <= 0 || i >= nx-1 || j <= 0 || j >= ny-1) return;
    int idx = j * nx + i;
    if (h[idx] <= DRY) {{
        k_grid[idx] = k_act;
        return;
    }}
    // central-difference divergence with dry-bed guards
    real hL = h[j*nx + i-1], hR = h[j*nx + i+1];
    real hD = h[(j-1)*nx + i], hU = h[(j+1)*nx + i];
    real uL = (hL > DRY) ? hu[j*nx + i-1] / hL : (real)0;
    real uR = (hR > DRY) ? hu[j*nx + i+1] / hR : (real)0;
    real vD = (hD > DRY) ? hv[(j-1)*nx + i] / hD : (real)0;
    real vU = (hU > DRY) ? hv[(j+1)*nx + i] / hU : (real)0;
    real div = (uR - uL) / ((real)2 * dx) + (vU - vD) / ((real)2 * dy);
    k_grid[idx] = (div < (real)0) ? k_pass : k_act;
}}

// -- Kernel 2: compute_wave_speeds ------------------------------------------
__global__ void compute_wave_speeds_kernel(
    const real* h, const real* hu, const real* hv,
    const real* k_grid, real* wave_speeds,
    int nx, int ny, real gz, real DRY)
{{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= nx || j >= ny) return;
    int idx = j * nx + i;
    if (h[idx] <= DRY) {{ wave_speeds[idx] = (real)0; return; }}
    real u = hu[idx] / h[idx];
    real v = hv[idx] / h[idx];
    real c = sqrt(k_grid[idx] * gz * h[idx]);
    real sx = fabs(u) + c;
    real sy = fabs(v) + c;
    wave_speeds[idx] = (sx > sy) ? sx : sy;
}}

// -- Kernel 3: compute_fluxes_global (SoA, global memory) -------------------
__global__ void compute_fluxes_global_kernel(
    const real* h, const real* hu, const real* hv, const real* k_grid,
    real* Fx_h, real* Fx_hu, real* Fx_hv,
    real* Gy_h, real* Gy_hu, real* Gy_hv,
    int nx, int ny, real gz, real DRY)
{{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    // X-flux at interface (j, i+1) between cell i and i+1
    if (i < nx-1 && j < ny) {{
        int idL = j*nx + i,  idR = j*nx + i+1;
        real hL = h[idL], hR = h[idR];
        real uL = (hL > DRY) ? hu[idL]/hL : (real)0;
        real uR = (hR > DRY) ? hu[idR]/hR : (real)0;
        real vL = (hL > DRY) ? hv[idL]/hL : (real)0;
        real vR = (hR > DRY) ? hv[idR]/hR : (real)0;
        real kL = k_grid[idL], kR = k_grid[idR];
        real cL = sqrt(kL * gz * fmax(hL, (real)0));
        real cR = sqrt(kR * gz * fmax(hR, (real)0));
        real a  = fmax(fabs(uL)+cL, fabs(uR)+cR);
        real FL0 = hL*uL,  FR0 = hR*uR;
        real FL1 = hL*uL*uL + (real)0.5*kL*gz*hL*hL;
        real FR1 = hR*uR*uR + (real)0.5*kR*gz*hR*hR;
        real FL2 = hL*vL*uL, FR2 = hR*vR*uR;
        int fx = j*(nx+1) + (i+1);
        Fx_h [fx] = (real)0.5*(FL0+FR0) - (real)0.5*a*(hR -hL );
        Fx_hu[fx] = (real)0.5*(FL1+FR1) - (real)0.5*a*(hR*uR-hL*uL);
        Fx_hv[fx] = (real)0.5*(FL2+FR2) - (real)0.5*a*(hR*vR-hL*vL);
    }}

    // Y-flux at interface (j+1, i) between cell j and j+1
    if (i < nx && j < ny-1) {{
        int idD = j*nx + i,  idU = (j+1)*nx + i;
        real hD = h[idD], hU = h[idU];
        real uD = (hD > DRY) ? hu[idD]/hD : (real)0;
        real uU = (hU > DRY) ? hu[idU]/hU : (real)0;
        real vD = (hD > DRY) ? hv[idD]/hD : (real)0;
        real vU = (hU > DRY) ? hv[idU]/hU : (real)0;
        real kD = k_grid[idD], kU = k_grid[idU];
        real cD = sqrt(kD * gz * fmax(hD, (real)0));
        real cU = sqrt(kU * gz * fmax(hU, (real)0));
        real a  = fmax(fabs(vD)+cD, fabs(vU)+cU);
        real GL0 = hD*vD,  GR0 = hU*vU;
        real GL1 = hD*uD*vD, GR1 = hU*uU*vU;
        real GL2 = hD*vD*vD + (real)0.5*kD*gz*hD*hD;
        real GR2 = hU*vU*vU + (real)0.5*kU*gz*hU*hU;
        int gy_idx = (j+1)*nx + i;
        Gy_h [gy_idx] = (real)0.5*(GL0+GR0) - (real)0.5*a*(hU   -hD   );
        Gy_hu[gy_idx] = (real)0.5*(GL1+GR1) - (real)0.5*a*(hU*uU-hD*uD);
        Gy_hv[gy_idx] = (real)0.5*(GL2+GR2) - (real)0.5*a*(hU*vU-hD*vD);
    }}
}}

// -- Kernel 4: compute_fluxes_shared (tiled 16x16) --------------------------
__global__ void compute_fluxes_shared_kernel(
    const real* h, const real* hu, const real* hv, const real* k_grid,
    real* Fx_h, real* Fx_hu, real* Fx_hv,
    real* Gy_h, real* Gy_hu, real* Gy_hv,
    int nx, int ny, real gz, real DRY)
{{
    __shared__ real s_h [17][17];
    __shared__ real s_hu[17][17];
    __shared__ real s_hv[17][17];
    __shared__ real s_k [17][17];

    int i  = blockIdx.x * blockDim.x + threadIdx.x;
    int j  = blockIdx.y * blockDim.y + threadIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;

    // helper lambda-style: clamp and load
    auto load = [&](int gi, int gj, int si, int sj) {{
        if (gi >= 0 && gi < nx && gj >= 0 && gj < ny) {{
            int idx = gj*nx + gi;
            s_h [sj][si] = h [idx];
            s_hu[sj][si] = hu[idx];
            s_hv[sj][si] = hv[idx];
            s_k [sj][si] = k_grid[idx];
        }} else {{
            s_h [sj][si] = (real)0;
            s_hu[sj][si] = (real)0;
            s_hv[sj][si] = (real)0;
            s_k [sj][si] = (real)1;
        }}
    }};

    load(i, j, tx, ty);
    if (tx == blockDim.x - 1) load(i+1, j,   tx+1, ty  );
    if (ty == blockDim.y - 1) load(i,   j+1,  tx,   ty+1);
    __syncthreads();

    // X-flux
    if (i < nx-1 && j < ny) {{
        real hL = s_h [ty][tx], hR = s_h [ty][tx+1];
        real uL = (hL > DRY) ? s_hu[ty][tx]/hL : (real)0;
        real uR = (hR > DRY) ? s_hu[ty][tx+1]/hR : (real)0;
        real vL = (hL > DRY) ? s_hv[ty][tx]/hL : (real)0;
        real vR = (hR > DRY) ? s_hv[ty][tx+1]/hR : (real)0;
        real kL = s_k[ty][tx], kR = s_k[ty][tx+1];
        real cL = sqrt(kL * gz * fmax(hL, (real)0));
        real cR = sqrt(kR * gz * fmax(hR, (real)0));
        real a  = fmax(fabs(uL)+cL, fabs(uR)+cR);
        int fx = j*(nx+1) + (i+1);
        Fx_h [fx] = (real)0.5*(hL*uL+hR*uR) - (real)0.5*a*(hR-hL);
        Fx_hu[fx] = (real)0.5*(hL*uL*uL+(real)0.5*kL*gz*hL*hL + hR*uR*uR+(real)0.5*kR*gz*hR*hR) - (real)0.5*a*(hR*uR-hL*uL);
        Fx_hv[fx] = (real)0.5*(hL*vL*uL+hR*vR*uR) - (real)0.5*a*(hR*vR-hL*vL);
    }}

    // Y-flux
    if (i < nx && j < ny-1) {{
        real hD = s_h [ty][tx], hU = s_h [ty+1][tx];
        real uD = (hD > DRY) ? s_hu[ty][tx]/hD : (real)0;
        real uU = (hU > DRY) ? s_hu[ty+1][tx]/hU : (real)0;
        real vD = (hD > DRY) ? s_hv[ty][tx]/hD : (real)0;
        real vU = (hU > DRY) ? s_hv[ty+1][tx]/hU : (real)0;
        real kD = s_k[ty][tx], kU = s_k[ty+1][tx];
        real cD = sqrt(kD * gz * fmax(hD, (real)0));
        real cU = sqrt(kU * gz * fmax(hU, (real)0));
        real a  = fmax(fabs(vD)+cD, fabs(vU)+cU);
        int gy_idx = (j+1)*nx + i;
        Gy_h [gy_idx] = (real)0.5*(hD*vD+hU*vU) - (real)0.5*a*(hU-hD);
        Gy_hu[gy_idx] = (real)0.5*(hD*uD*vD+hU*uU*vU) - (real)0.5*a*(hU*uU-hD*uD);
        Gy_hv[gy_idx] = (real)0.5*(hD*vD*vD+(real)0.5*kD*gz*hD*hD + hU*vU*vU+(real)0.5*kU*gz*hU*hU) - (real)0.5*a*(hU*vU-hD*vD);
    }}
}}

// -- Kernel 5: update_state -------------------------------------------------
__global__ void update_state_kernel(
    real* h, real* hu, real* hv,
    const real* Fx_h,  const real* Fx_hu, const real* Fx_hv,
    const real* Gy_h,  const real* Gy_hu, const real* Gy_hv,
    int nx, int ny,
    real dt, real dx, real dy, real DRY,
    real gx, real gy, real gz, real mu)
{{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i <= 0 || i >= nx-1 || j <= 0 || j >= ny-1) return;
    int idx = j*nx + i;

    real dFh  = (Fx_h [j*(nx+1)+i+1] - Fx_h [j*(nx+1)+i]) / dx
              + (Gy_h [(j+1)*nx+i]    - Gy_h [j*nx+i]   ) / dy;
    real dFhu = (Fx_hu[j*(nx+1)+i+1] - Fx_hu[j*(nx+1)+i]) / dx
              + (Gy_hu[(j+1)*nx+i]    - Gy_hu[j*nx+i]   ) / dy;
    real dFhv = (Fx_hv[j*(nx+1)+i+1] - Fx_hv[j*(nx+1)+i]) / dx
              + (Gy_hv[(j+1)*nx+i]    - Gy_hv[j*nx+i]   ) / dy;

    real h_new = h[idx] - dt * dFh;
    h_new = fmax(h_new, (real)0);

    real S_hu = (real)0, S_hv = (real)0;
    if (h[idx] > DRY) {{
        real u = hu[idx] / h[idx];
        real v = hv[idx] / h[idx];
        real vmag = sqrt(u*u + v*v + (real)1e-6);
        S_hu = gx * h[idx] - h[idx] * gz * mu * u / vmag;
        S_hv = gy * h[idx] - h[idx] * gz * mu * v / vmag;
    }}

    h [idx] = h_new;
    hu[idx] = h_new > DRY ? hu[idx] - dt*(dFhu - S_hu) : (real)0;
    hv[idx] = h_new > DRY ? hv[idx] - dt*(dFhv - S_hv) : (real)0;
}}

}} // extern "C"
"""


def compile_kernels(ctype='float'):
    """Compiles and returns dict of CuPy RawKernel functions."""
    import cupy as cp
    code = get_kernel_code(ctype)
    module = cp.RawModule(code=code, options=('--std=c++14',))
    return {
        'compute_k_grid':      module.get_function('compute_k_grid_kernel'),
        'compute_wave_speeds': module.get_function('compute_wave_speeds_kernel'),
        'compute_fluxes_global': module.get_function('compute_fluxes_global_kernel'),
        'compute_fluxes_shared': module.get_function('compute_fluxes_shared_kernel'),
        'update_state':        module.get_function('update_state_kernel'),
    }
