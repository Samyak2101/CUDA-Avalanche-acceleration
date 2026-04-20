import numpy as np
import matplotlib.pyplot as plt
import time
from numba import njit, prange
# Force Matplotlib to use LaTeX-style Serif fonts
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman", "Times New Roman"],
    "font.size": 11, # Adjust to match your FMFP template base size
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 10
})


zeta  = np.radians(37.4)  
delta = np.radians(27.0)  
phi   = np.radians(37.3)  
g     = 9.81

gx = g * np.sin(zeta)
gy = 0.0
gz = g * np.cos(zeta)
mu = np.tan(delta)

# The Dynamic Earth Pressure Coefficients
inner = 1.0 - (1.0 + np.tan(delta)**2) * np.cos(phi)**2
k_act  = 2.0 * (1.0 - np.sqrt(max(inner, 0.0))) / np.cos(phi)**2 - 1.0
k_pass = 2.0 * (1.0 + np.sqrt(max(inner, 0.0))) / np.cos(phi)**2 - 1.0 # NEW: Passive State
DRY = 1e-4


@njit(parallel=True)
def compute_k_grid(U, k_grid, nx, ny, dx, dy, k_act, k_pass, DRY):
    """Calculates 2D velocity divergence to assign Active or Passive pressure states."""
    for j in prange(1, ny-1):
        for i in range(1, nx-1):
            h = U[0, j, i]
            if h > DRY:
                # Get velocities of neighboring cells (with dry-bed safeguards)
                u_R = U[1, j, i+1] / U[0, j, i+1] if U[0, j, i+1] > DRY else 0.0
                u_L = U[1, j, i-1] / U[0, j, i-1] if U[0, j, i-1] > DRY else 0.0
                v_T = U[2, j+1, i] / U[0, j+1, i] if U[0, j+1, i] > DRY else 0.0
                v_B = U[2, j-1, i] / U[0, j-1, i] if U[0, j-1, i] > DRY else 0.0
                
                # Central difference divergence: div(u) = du/dx + dv/dy
                div_u = (u_R - u_L) / (2.0 * dx) + (v_T - v_B) / (2.0 * dy)
                
                # If compressing (divergence < 0), use k_pass. Otherwise, k_act.
                if div_u < 0.0:
                    k_grid[j, i] = k_pass
                else:
                    k_grid[j, i] = k_act
            else:
                k_grid[j, i] = k_act
                
    # Transmissive boundaries for k_grid
    for j in prange(ny):
        k_grid[j, 0] = k_grid[j, 1]
        k_grid[j, nx-1] = k_grid[j, nx-2]
    for i in prange(nx):
        k_grid[0, i] = k_grid[1, i]
        k_grid[ny-1, i] = k_grid[ny-2, i]

@njit
def compute_dt(U, k_grid, nx, ny, gz, dx, dy, CFL, DRY):
    """Calculates the CFL time step using local wave speeds."""
    max_speed_x = 1e-8
    max_speed_y = 1e-8
    
    for j in range(ny):
        for i in range(nx):
            h = U[0, j, i]
            if h > DRY:
                u = abs(U[1, j, i] / h)
                v = abs(U[2, j, i] / h)
                c = np.sqrt(k_grid[j, i] * gz * h) # Wave speed now depends on local k
                if u + c > max_speed_x: max_speed_x = u + c
                if v + c > max_speed_y: max_speed_y = v + c
                
    return CFL * min(dx / max_speed_x, dy / max_speed_y)

@njit(parallel=True)
def compute_fluxes_and_sources(U, k_grid, Fx, Gy, S, nx, ny, gz, gx, gy, mu, DRY):
    """Calculates all Rusanov interface fluxes using the dynamic k_grid."""
    # 1. Calculate Fx (X-direction Rusanov Fluxes)
    for j in prange(ny):
        for i in range(nx - 1):
            hL, huL, hvL = U[0, j, i], U[1, j, i], U[2, j, i]
            hR, huR, hvR = U[0, j, i+1], U[1, j, i+1], U[2, j, i+1]
            kL, kR = k_grid[j, i], k_grid[j, i+1]
            
            uL = huL / hL if hL > DRY else 0.0
            vL = hvL / hL if hL > DRY else 0.0
            uR = huR / hR if hR > DRY else 0.0
            vR = hvR / hR if hR > DRY else 0.0
            
            f0L, f1L, f2L = huL, huL*uL + 0.5*kL*gz*hL**2, hvL*uL
            f0R, f1R, f2R = huR, huR*uR + 0.5*kR*gz*hR**2, hvR*uR
            
            cL = np.sqrt(kL * gz * hL) if hL > DRY else 0.0
            cR = np.sqrt(kR * gz * hR) if hR > DRY else 0.0
            a = max(abs(uL) + cL, abs(uR) + cR)
            
            Fx[0, j, i+1] = 0.5*(f0L + f0R) - 0.5*a*(hR - hL)
            Fx[1, j, i+1] = 0.5*(f1L + f1R) - 0.5*a*(huR - huL)
            Fx[2, j, i+1] = 0.5*(f2L + f2R) - 0.5*a*(hvR - hvL)

    # 2. Calculate Gy (Y-direction Rusanov Fluxes)
    for j in prange(ny - 1):
        for i in range(nx):
            hL, huL, hvL = U[0, j, i], U[1, j, i], U[2, j, i]
            hR, huR, hvR = U[0, j+1, i], U[1, j+1, i], U[2, j+1, i]
            kL, kR = k_grid[j, i], k_grid[j+1, i]
            
            uL = huL / hL if hL > DRY else 0.0
            vL = hvL / hL if hL > DRY else 0.0
            uR = huR / hR if hR > DRY else 0.0
            vR = hvR / hR if hR > DRY else 0.0
            
            g0L, g1L, g2L = hvL, huL*vL, hvL*vL + 0.5*kL*gz*hL**2
            g0R, g1R, g2R = hvR, huR*vR, hvR*vR + 0.5*kR*gz*hR**2
            
            cL = np.sqrt(kL * gz * hL) if hL > DRY else 0.0
            cR = np.sqrt(kR * gz * hR) if hR > DRY else 0.0
            a = max(abs(vL) + cL, abs(vR) + cR)
            
            Gy[0, j+1, i] = 0.5*(g0L + g0R) - 0.5*a*(hR - hL)
            Gy[1, j+1, i] = 0.5*(g1L + g1R) - 0.5*a*(huR - huL)
            Gy[2, j+1, i] = 0.5*(g2L + g2R) - 0.5*a*(hvR - hvL)

    # 3. Calculate Source Terms
    for j in prange(ny):
        for i in range(nx):
            h = U[0, j, i]
            if h > DRY:
                u, v = U[1, j, i] / h, U[2, j, i] / h
                vmag = np.sqrt(u*u + v*v + 1e-6)
                S[0, j, i] = 0.0
                S[1, j, i] = gx*h - (h * gz * mu * u / vmag)
                S[2, j, i] = gy*h - (h * gz * mu * v / vmag)
            else:
                S[0, j, i] = 0.0
                S[1, j, i] = 0.0
                S[2, j, i] = 0.0

@njit(parallel=True)
def update_state(U, Fx, Gy, S, dt, dx, dy, nx, ny, DRY):
    for j in prange(1, ny-1):
        for i in range(1, nx-1):
            for c in range(3):
                U[c, j, i] = U[c, j, i] - (dt/dx)*(Fx[c, j, i+1] - Fx[c, j, i]) \
                                        - (dt/dy)*(Gy[c, j+1, i] - Gy[c, j, i]) \
                                        + dt * S[c, j, i]
            
            if U[0, j, i] < 0.0: U[0, j, i] = 0.0
            if U[0, j, i] < DRY:
                U[1, j, i] = 0.0
                U[2, j, i] = 0.0

    for j in prange(ny):
        for c in range(3):
            U[c, j, 0], U[c, j, nx-1] = U[c, j, 1], U[c, j, nx-2]
    for i in prange(nx):
        for c in range(3):
            U[c, 0, i], U[c, ny-1, i] = U[c, 1, i], U[c, ny-2, i]

# ==========================================
# 3. Main Simulation Loop
# ==========================================
def run_rusanov(nx, ny, target_times, CFL=0.5): # Lowered CFL slightly for k_pass stability
    Lx, Ly = 4.0, 1.3
    dx, dy = Lx / nx, Ly / ny
    x = np.linspace(0, Lx, nx)
    y = np.linspace(0, Ly, ny)
    X, Y = np.meshgrid(x, y)

    U = np.zeros((3, ny, nx))
    Fx = np.zeros((3, ny, nx+1))
    Gy = np.zeros((3, ny+1, nx))
    S = np.zeros((3, ny, nx))
    k_grid = np.full((ny, nx), k_act) # NEW: 2D array to hold local k states
    
    Xc, Yc, R = 0.4, 0.65, 0.06
    H0 = R / 1.5
    R2 = (X - Xc)**2 + (Y - Yc)**2
    U[0] = np.where(R2 <= R**2, H0, 0.0)

    t = 0.0
    history = {}
    mass_history = []
    save_id = 0  

    print(f"Running Dynamic-k Rusanov ({nx}x{ny}) on a {Lx}m domain...")
    start_time = time.time()

    while save_id < len(target_times):
        if t >= target_times[save_id] or np.isclose(t, target_times[save_id], atol=1e-4):
            history[target_times[save_id]] = U[0].copy()
            save_id += 1
            if save_id >= len(target_times): break

        current_mass = np.sum(U[0]) * dx * dy
        mass_history.append((t, current_mass))

        # NEW: Compute dynamic k_grid before calculating time steps or fluxes
        compute_k_grid(U, k_grid, nx, ny, dx, dy, k_act, k_pass, DRY)

        dt = compute_dt(U, k_grid, nx, ny, gz, dx, dy, CFL, DRY)
        if save_id < len(target_times) and t + dt > target_times[save_id]:
            dt = target_times[save_id] - t

        compute_fluxes_and_sources(U, k_grid, Fx, Gy, S, nx, ny, gz, gx, gy, mu, DRY)
        update_state(U, Fx, Gy, S, dt, dx, dy, nx, ny, DRY)
        t += dt

    print(f"-> Simulation complete in {time.time() - start_time:.2f}s")
    return history, x, y, mass_history

# ==========================================
# 4. TITAN2D Data Loaders
# ==========================================
def nearest_time_index(t_array, target_t): return int(np.argmin(np.abs(t_array - target_t)))

def binned_profile(x, h, x_min, x_max, nbins):
    edges = np.linspace(x_min, x_max, nbins + 1)
    mids  = 0.5 * (edges[:-1] + edges[1:])
    hb    = np.full(nbins, np.nan, dtype=np.float64)
    idx   = np.searchsorted(edges, x, side="right") - 1
    valid = (idx >= 0) & (idx < nbins)
    idx   = idx[valid]
    hv    = h[valid]
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

# ==========================================
# 5. Main Execution & Plots
# ==========================================
if __name__ == "__main__":
    TIMES_TO_PLOT = [0.0, 0.5, 1.0, 1.5, 2.0]
    Y_TARGET = 0.65
    
    # Execute Dynamic-k solver. Lowered CFL to 0.5 because k_pass increases wave speeds!
    history_rus, x_rus, y_rus, mass_history = run_rusanov(nx=800, ny=260, target_times=TIMES_TO_PLOT, CFL=0.3)
    y_idx_rus = np.argmin(np.abs(y_rus - Y_TARGET))

    print("Loading TITAN2D data...")
    try:
        data = np.load("titan_height_cells_0to2s.npz")
        t_titan, xc_titan, yc_titan, H_titan = data['t'], data['xc'], data['yc'], data['H']
    except FileNotFoundError:
        print("ERROR: 'titan_height_cells_0to2s.npz' not found.")
        exit()

    band_mask = np.abs(yc_titan - Y_TARGET) <= 0.002
    xmask = (xc_titan >= 0.0) & (xc_titan <= 4.0)
    sel = band_mask & xmask
    x_line_titan = xc_titan[sel]

    # ------------------------------------------
    # PLOT 1: Centerline Validation
    # ------------------------------------------
    plt.figure(figsize=(12, 6))
    colors = ['black', 'blue', 'green', 'orange', 'red']

    for i, target_t in enumerate(TIMES_TO_PLOT):
        c = colors[i]
        it = nearest_time_index(t_titan, target_t)
        h_line = H_titan[it][sel].astype(np.float64)
        x_mid, h_prof = binned_profile(x_line_titan, h_line, 0.0, 4.0, nbins=800)
        h_prof = fill_nans_linear(x_mid, h_prof)
        h_prof = np.nan_to_num(h_prof, nan=0.0)
        plt.plot(x_mid, h_prof, color=c, linestyle='--', linewidth=2.5, label=f"TITAN2D (t={target_t}s)")
        
        h_rus = history_rus[target_t][y_idx_rus, :]
        h_rus = np.where(h_rus > DRY, h_rus, 0.0)
        plt.plot(x_rus, h_rus, color=c, linestyle='-', linewidth=2.0, alpha=0.8, label=f"Dynamic-k Rusanov (t={target_t}s)")

    plt.xlim(0.0, 1.5) 
    plt.ylim(bottom=0.0, top=1.2 * np.max(H_titan))
    plt.xlabel("Downslope Distance x (m)", fontsize=12)
    plt.ylabel("Depth h (m)", fontsize=12)
    #plt.title("Centerline Profile Validation: Dynamic-k Rusanov vs TITAN2D", fontsize=14, fontweight='bold')
    #plt.legend(ncol=2, fontsize=10)
    plt.grid(True, alpha=0.4, linestyle=':')
    plt.tight_layout()
    
    # SAVE PLOT 1 BEFORE SHOWING
    plt.savefig('centerline_val.pdf', format='pdf', bbox_inches='tight')
    plt.show()

    # ------------------------------------------
    # PLOT 2: Mass Conservation
    # ------------------------------------------
    t_vals = [m[0] for m in mass_history]
    mass_vals = [m[1] for m in mass_history]
    plt.figure(figsize=(8, 4))
    plt.plot(t_vals, mass_vals, 'k-', linewidth=2)
    plt.ylim(0, mass_vals[0] * 1.5) 
    plt.xlabel("Time (s)", fontsize=12)
    plt.ylabel("Total Volume ($m^3$)", fontsize=12)
    #plt.title("Domain Mass Conservation (Dynamic-k Rusanov)", fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.4, linestyle=':')
    plt.tight_layout()
    
    # SAVE PLOT 2 BEFORE SHOWING
    plt.savefig('mass_conservation.pdf', format='pdf', bbox_inches='tight')
    plt.show()