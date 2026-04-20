import numpy as np
import cupy as cp
import time
import math
import matplotlib.pyplot as plt

# ==========================================
# 1. Configuration & Matplotlib Setup
# ==========================================
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman", "Times New Roman"],
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 10
})

# Single precision for maximum throughput on consumer Turing architectures
DTYPE_NP = np.float32
DTYPE_CP = cp.float32
CTYPE = 'float'

zeta  = np.radians(37.4)
delta = np.radians(27.0)
phi   = np.radians(37.3)
g     = 9.81
gx = g * np.sin(zeta)
gy = 0.0
gz = g * np.cos(zeta)
mu = np.tan(delta)

inner = 1.0 - (1.0 + np.tan(delta)**2) * np.cos(phi)**2
k_act  = 2.0 * (1.0 - np.sqrt(max(inner, 0.0))) / np.cos(phi)**2 - 1.0
k_pass = 2.0 * (1.0 + np.sqrt(max(inner, 0.0))) / np.cos(phi)**2 - 1.0
DRY = 1e-4

# ==========================================
# 2. CUDA C++ Kernel Definitions
# ==========================================
kernel_code = f"""
typedef {CTYPE} real;

extern "C" __global__
void compute_k_grid_kernel(const real* h, const real* hu, const real* hv, 
                           real* k_grid, int nx, int ny, real dx, real dy, 
                           real k_act, real k_pass, real DRY) {{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (i <= 0 || i >= nx-1 || j <= 0 || j >= ny-1) return;
    int idx = j * nx + i;
    
    if (h[idx] > DRY) {{
        real u_R = (h[j*nx + i+1] > DRY) ? hu[j*nx + i+1] / h[j*nx + i+1] : 0.0;
        real u_L = (h[j*nx + i-1] > DRY) ? hu[j*nx + i-1] / h[j*nx + i-1] : 0.0;
        real v_T = (h[(j+1)*nx + i] > DRY) ? hv[(j+1)*nx + i] / h[(j+1)*nx + i] : 0.0;
        real v_B = (h[(j-1)*nx + i] > DRY) ? hv[(j-1)*nx + i] / h[(j-1)*nx + i] : 0.0;
        
        real div_u = (u_R - u_L) / (2.0 * dx) + (v_T - v_B) / (2.0 * dy);
        k_grid[idx] = (div_u < 0.0) ? k_pass : k_act;
    }} else {{
        k_grid[idx] = k_act;
    }}
}}

extern "C" __global__
void compute_fluxes_kernel(const real* h, const real* hu, const real* hv, const real* k_grid,
                           real* Fx_h, real* Fx_hu, real* Fx_hv,
                           real* Gy_h, real* Gy_hu, real* Gy_hv,
                           int nx, int ny, real gz, real DRY) {{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    // X-direction fluxes (Fx)
    if (i < nx - 1 && j < ny) {{
        int idxL = j * nx + i;
        int idxR = j * nx + i + 1;
        int fidx = j * (nx + 1) + i + 1;

        real hL = h[idxL], huL = hu[idxL], hvL = hv[idxL], kL = k_grid[idxL];
        real hR = h[idxR], huR = hu[idxR], hvR = hv[idxR], kR = k_grid[idxR];

        real uL = (hL > DRY) ? huL / hL : 0.0;
        real vL = (hL > DRY) ? hvL / hL : 0.0;
        real uR = (hR > DRY) ? huR / hR : 0.0;
        real vR = (hR > DRY) ? hvR / hR : 0.0;

        real f0L = huL, f1L = huL * uL + 0.5 * kL * gz * hL * hL, f2L = hvL * uL;
        real f0R = huR, f1R = huR * uR + 0.5 * kR * gz * hR * hR, f2R = hvR * uR;

        real cL = (hL > DRY) ? sqrt(kL * gz * hL) : 0.0;
        real cR = (hR > DRY) ? sqrt(kR * gz * hR) : 0.0;
        real a = max(abs(uL) + cL, abs(uR) + cR);

        Fx_h[fidx]  = 0.5 * (f0L + f0R) - 0.5 * a * (hR - hL);
        Fx_hu[fidx] = 0.5 * (f1L + f1R) - 0.5 * a * (huR - huL);
        Fx_hv[fidx] = 0.5 * (f2L + f2R) - 0.5 * a * (hvR - hvL);
    }}

    // Y-direction fluxes (Gy)
    if (i < nx && j < ny - 1) {{
        int idxB = j * nx + i;
        int idxT = (j + 1) * nx + i;
        int gidx = (j + 1) * nx + i;

        real hB = h[idxB], huB = hu[idxB], hvB = hv[idxB], kB = k_grid[idxB];
        real hT = h[idxT], huT = hu[idxT], hvT = hv[idxT], kT = k_grid[idxT];

        real uB = (hB > DRY) ? huB / hB : 0.0;
        real vB = (hB > DRY) ? hvB / hB : 0.0;
        real uT = (hT > DRY) ? huT / hT : 0.0;
        real vT = (hT > DRY) ? hvT / hT : 0.0;

        real g0B = hvB, g1B = huB * vB, g2B = hvB * vB + 0.5 * kB * gz * hB * hB;
        real g0T = hvT, g1T = huT * vT, g2T = hvT * vT + 0.5 * kT * gz * hT * hT;

        real cB = (hB > DRY) ? sqrt(kB * gz * hB) : 0.0;
        real cT = (hT > DRY) ? sqrt(kT * gz * hT) : 0.0;
        real a = max(abs(vB) + cB, abs(vT) + cT);

        Gy_h[gidx]  = 0.5 * (g0B + g0T) - 0.5 * a * (hT - hB);
        Gy_hu[gidx] = 0.5 * (g1B + g1T) - 0.5 * a * (huT - huB);
        Gy_hv[gidx] = 0.5 * (g2B + g2T) - 0.5 * a * (hvT - hvB);
    }}
}}

extern "C" __global__
void update_state_kernel(real* h, real* hu, real* hv, 
                         const real* Fx_h, const real* Fx_hu, const real* Fx_hv,
                         const real* Gy_h, const real* Gy_hu, const real* Gy_hv,
                         int nx, int ny, real dt, real dx, real dy, real DRY,
                         real gx, real gy, real gz, real mu) {{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (i <= 0 || i >= nx-1 || j <= 0 || j >= ny-1) return;
    int idx = j * nx + i;
    
    real S_hu = 0.0, S_hv = 0.0;
    real h_local = h[idx];
    if (h_local > DRY) {{
        real u = hu[idx] / h_local;
        real v = hv[idx] / h_local;
        real vmag = sqrt(u*u + v*v + 1e-6);
        S_hu = gx * h_local - (h_local * gz * mu * u / vmag);
        S_hv = gy * h_local - (h_local * gz * mu * v / vmag);
    }}
    
    h[idx]  -= (dt/dx)*(Fx_h[j*(nx+1) + i+1] - Fx_h[j*(nx+1) + i]) + (dt/dy)*(Gy_h[(j+1)*nx + i] - Gy_h[j*nx + i]);
    hu[idx] -= (dt/dx)*(Fx_hu[j*(nx+1) + i+1] - Fx_hu[j*(nx+1) + i]) + (dt/dy)*(Gy_hu[(j+1)*nx + i] - Gy_hu[j*nx + i]) - dt*S_hu;
    hv[idx] -= (dt/dx)*(Fx_hv[j*(nx+1) + i+1] - Fx_hv[j*(nx+1) + i]) + (dt/dy)*(Gy_hv[(j+1)*nx + i] - Gy_hv[j*nx + i]) - dt*S_hv;
    
   // 1. Fix numerical undershoot (preserve mass)
    if (h[idx] < 0.0) {{
        h[idx] = 0.0;
    }}
    
    // 2. Freeze momentum for microscopic layers (prevent NaNs)
    if (h[idx] < DRY) {{
        hu[idx] = 0.0;
        hv[idx] = 0.0;
    }}
}}
"""

module = cp.RawModule(code=kernel_code)
compute_k_grid_kernel = module.get_function('compute_k_grid_kernel')
compute_fluxes_kernel = module.get_function('compute_fluxes_kernel')
update_state_kernel   = module.get_function('update_state_kernel')

dt_reduction = cp.ReductionKernel(
    in_params='T h, T hu, T hv, T k, T gz, T DRY',
    out_params='T max_spd',
    map_expr='(h > DRY) ? max(abs(hu/h) + sqrt(k * gz * h), abs(hv/h) + sqrt(k * gz * h)) : (T)0.0',
    reduce_expr='max(a, b)',
    post_map_expr='max_spd = a',
    identity='0.0',
    name='max_wave_speed'
)

# ==========================================
# 3. Main GPU Execution Loop
# ==========================================
def run_rusanov_cuda(nx, ny, target_times, CFL=0.3):
    Lx, Ly = 4.0, 1.3
    dx, dy = Lx / nx, Ly / ny
    
    h  = cp.zeros((ny, nx), dtype=DTYPE_CP)
    hu = cp.zeros((ny, nx), dtype=DTYPE_CP)
    hv = cp.zeros((ny, nx), dtype=DTYPE_CP)
    
    Fx_h, Fx_hu, Fx_hv = [cp.zeros((ny, nx+1), dtype=DTYPE_CP) for _ in range(3)]
    Gy_h, Gy_hu, Gy_hv = [cp.zeros((ny+1, nx), dtype=DTYPE_CP) for _ in range(3)]
    k_grid = cp.full((ny, nx), k_act, dtype=DTYPE_CP)
    
    x = np.linspace(0, Lx, nx)
    y = np.linspace(0, Ly, ny)
    X, Y = np.meshgrid(x, y)
    R2 = (X - 0.4)**2 + (Y - 0.65)**2
    h_init = np.where(R2 <= 0.06**2, 0.06 / 1.5, 0.0).astype(DTYPE_NP)
    h[:] = cp.asarray(h_init)

    threads_per_block = (16, 16)
    blocks_per_grid = (math.ceil(nx / 16), math.ceil(ny / 16))

    history = {}
    mass_history = []
    
    # Analytics arrays
    analytics_t = []
    analytics_mcups = []
    analytics_bw = []
    
    t = 0.0
    steps = 0
    save_id = 0
    bytes_per_cell = 9 * h.itemsize # Conservative estimate for SoA
    
    print(f"Executing SoA CuPy Rusanov ({nx}x{ny}) in {CTYPE} precision...")
    
    # Timing variables for analytics
    step_start = time.perf_counter()

    while save_id < len(target_times):
        if t >= target_times[save_id] or math.isclose(t, target_times[save_id], abs_tol=1e-4):
            history[target_times[save_id]] = cp.asnumpy(h)
            save_id += 1
            if save_id >= len(target_times): break

        compute_k_grid_kernel(blocks_per_grid, threads_per_block, 
                              (h, hu, hv, k_grid, nx, ny, DTYPE_NP(dx), DTYPE_NP(dy), 
                               DTYPE_NP(k_act), DTYPE_NP(k_pass), DTYPE_NP(DRY)))

        max_speed = dt_reduction(h, hu, hv, k_grid, DTYPE_NP(gz), DTYPE_NP(DRY))
        dt = CFL * min(dx, dy) / (max_speed + 1e-8)
        dt = float(dt.get()) 
        
        if save_id < len(target_times) and t + dt > target_times[save_id]:
            dt = target_times[save_id] - t

        compute_fluxes_kernel(blocks_per_grid, threads_per_block,
                              (h, hu, hv, k_grid, Fx_h, Fx_hu, Fx_hv, Gy_h, Gy_hu, Gy_hv,
                               nx, ny, DTYPE_NP(gz), DTYPE_NP(DRY)))
            
        update_state_kernel(blocks_per_grid, threads_per_block,
                            (h, hu, hv, Fx_h, Fx_hu, Fx_hv, Gy_h, Gy_hu, Gy_hv,
                             nx, ny, DTYPE_NP(dt), DTYPE_NP(dx), DTYPE_NP(dy), DTYPE_NP(DRY),
                             DTYPE_NP(gx), DTYPE_NP(gy), DTYPE_NP(gz), DTYPE_NP(mu)))
        
        # Tracking metrics every 50 steps
        if steps % 50 == 0:
            cp.cuda.Stream.null.synchronize()
            step_end = time.perf_counter()
            elapsed = step_end - step_start
            
            if elapsed > 0 and steps > 0:
                mcups = (nx * ny * 50) / elapsed / 1e6
                bw_gbs = (bytes_per_cell * nx * ny * 50) / elapsed / 1e9
                analytics_t.append(t)
                analytics_mcups.append(mcups)
                analytics_bw.append(bw_gbs)
                
            step_start = time.perf_counter()
            mass = float(cp.sum(h).get()) * dx * dy
            mass_history.append((t, mass))
            
        t += dt
        steps += 1

    print(f"Simulation Finished. Total Steps: {steps}")
    return history, x, y, mass_history, analytics_t, analytics_mcups, analytics_bw

# ==========================================
# 4. Utilities & Plotting
# ==========================================
def nearest_time_index(t_array, target_t): 
    return int(np.argmin(np.abs(t_array - target_t)))

def binned_profile(x, h_vals, x_min, x_max, nbins):
    edges = np.linspace(x_min, x_max, nbins + 1)
    mids  = 0.5 * (edges[:-1] + edges[1:])
    hb    = np.full(nbins, np.nan, dtype=np.float64)
    idx   = np.searchsorted(edges, x, side="right") - 1
    valid = (idx >= 0) & (idx < nbins)
    idx   = idx[valid]
    hv    = h_vals[valid]
    s = np.zeros(nbins, dtype=np.float64)
    c = np.zeros(nbins, dtype=np.int64)
    np.add.at(s, idx, hv)
    np.add.at(c, idx, 1)
    mask = c > 0
    hb[mask] = s[mask] / c[mask]
    return mids, hb

def fill_nans_linear(x, y):
    y = y.copy()
    mask = np.isfinite(y)
    if mask.sum() < 2: return y
    y[~mask] = np.interp(x[~mask], x[mask], y[mask])
    return y

if __name__ == "__main__":
    TIMES_TO_PLOT = [0.0, 0.5, 1.0, 1.5, 2.0]
    Y_TARGET = 0.65
    
    # Execute the CUDA solver
    history_rus, x_rus, y_rus, mass_hist, an_t, an_mcups, an_bw = run_rusanov_cuda(nx=800, ny=260, target_times=TIMES_TO_PLOT, CFL=0.3)
    y_idx_rus = np.argmin(np.abs(y_rus - Y_TARGET))

    print("Loading TITAN2D data...")
    try:
        data = np.load("titan_height_cells_0to2s.npz")
        t_titan, xc_titan, yc_titan, H_titan = data['t'], data['xc'], data['yc'], data['H']
        band_mask = np.abs(yc_titan - Y_TARGET) <= 0.002
        xmask = (xc_titan >= 0.0) & (xc_titan <= 4.0)
        sel = band_mask & xmask
        x_line_titan = xc_titan[sel]
        has_titan = True
    except FileNotFoundError:
        print("WARNING: 'titan_height_cells_0to2s.npz' not found. Skipping TITAN2D overlays.")
        has_titan = False

    # ------------------------------------------
    # PLOT 1: Centerline Validation
    # ------------------------------------------
    plt.figure(figsize=(12, 6))
    colors = ['black', 'blue', 'green', 'orange', 'red']

    for i, target_t in enumerate(TIMES_TO_PLOT):
        c = colors[i]
        if has_titan:
            it = nearest_time_index(t_titan, target_t)
            h_line = H_titan[it][sel].astype(np.float64)
            x_mid, h_prof = binned_profile(x_line_titan, h_line, 0.0, 4.0, nbins=800)
            h_prof = fill_nans_linear(x_mid, h_prof)
            h_prof = np.nan_to_num(h_prof, nan=0.0)
            plt.plot(x_mid, h_prof, color=c, linestyle='--', linewidth=2.5, label=f"TITAN2D (t={target_t}s)")
        
        h_rus = history_rus[target_t][y_idx_rus, :]
        h_rus = np.where(h_rus > DRY, h_rus, 0.0)
        plt.plot(x_rus, h_rus, color=c, linestyle='-', linewidth=2.0, alpha=0.8, label=f"CUDA (t={target_t}s)")

    plt.xlim(0.0, 1.5) 
    plt.ylim(bottom=0.0)
    plt.xlabel("Downslope Distance x (m)", fontsize=12)
    plt.ylabel("Depth h (m)", fontsize=12)
    plt.grid(True, alpha=0.4, linestyle=':')
    plt.tight_layout()
    plt.savefig('centerline_val_cuda.pdf', format='pdf', bbox_inches='tight')
    plt.show()

    # ------------------------------------------
    # PLOT 2: Mass Conservation
    # ------------------------------------------
    t_vals = [m[0] for m in mass_hist]
    mass_vals = [m[1] for m in mass_hist]
    plt.figure(figsize=(8, 4))
    plt.plot(t_vals, mass_vals, 'k-', linewidth=2)
    plt.ylim(0, mass_vals[0] * 1.5 if mass_vals else 1) 
    plt.xlabel("Time (s)", fontsize=12)
    plt.ylabel("Total Volume (m³)", fontsize=12)
    plt.grid(True, alpha=0.4, linestyle=':')
    plt.tight_layout()
    plt.savefig('mass_conservation_cuda.pdf', format='pdf', bbox_inches='tight')
    plt.show()

    # ------------------------------------------
    # PLOT 3: HPC Analytics
    # ------------------------------------------
    fig, ax1 = plt.subplots(figsize=(10, 4))
    
    color = 'tab:red'
    ax1.set_xlabel('Simulation Time (s)')
    ax1.set_ylabel('Performance (MCUPS)', color=color)
    ax1.plot(an_t, an_mcups, color=color, linewidth=2, label="MCUPS")
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, alpha=0.4, linestyle=':')

    ax2 = ax1.twinx()  
    color = 'tab:blue'
    ax2.set_ylabel('Effective Bandwidth (GB/s)', color=color)  
    ax2.plot(an_t, an_bw, color=color, linewidth=2, linestyle='--', label="Bandwidth")
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout() 
    plt.title("HPC Solver Analytics")
    plt.savefig('hpc_analytics_cuda.pdf', format='pdf', bbox_inches='tight')
    plt.show()