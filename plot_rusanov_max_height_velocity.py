import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, AutoMinorLocator
from scipy.interpolate import RegularGridInterpolator
from numba import njit, prange
import time

# ==========================================
# 1. Configuration & Physical Parameters
# ==========================================
T_MAX = 0.5          # Max time to analyze (script focuses on t <= 0.5s)
H_THRESH_VEL = 1e-3  # 1mm threshold for a cell to be considered "wet"
DRY = 1e-4
X_MIN, X_MAX = 0.0, 4.0
Y_MIN, Y_MAX = 0.0, 1.3

zeta  = np.radians(37.4)  
delta = np.radians(27.0)  
phi   = np.radians(37.3)  
g     = 9.81
gx, gy, gz = g * np.sin(zeta), 0.0, g * np.cos(zeta)
mu = np.tan(delta)

inner = 1.0 - (1.0 + np.tan(delta)**2) * np.cos(phi)**2
k_act  = 2.0 * (1.0 - np.sqrt(max(inner, 0.0))) / np.cos(phi)**2 - 1.0
k_pass = 2.0 * (1.0 + np.sqrt(max(inner, 0.0))) / np.cos(phi)**2 - 1.0

# ==========================================
# 2. Physics Kernels (Dynamic-k Rusanov)
# ==========================================
@njit(parallel=True)
def compute_k_grid(U, k_grid, nx, ny, dx, dy, k_act, k_pass, DRY):
    for j in prange(1, ny-1):
        for i in range(1, nx-1):
            h = U[0, j, i]
            if h > DRY:
                u_R = U[1, j, i+1] / U[0, j, i+1] if U[0, j, i+1] > DRY else 0.0
                u_L = U[1, j, i-1] / U[0, j, i-1] if U[0, j, i-1] > DRY else 0.0
                v_T = U[2, j+1, i] / U[0, j+1, i] if U[0, j+1, i] > DRY else 0.0
                v_B = U[2, j-1, i] / U[0, j-1, i] if U[0, j-1, i] > DRY else 0.0
                div_u = (u_R - u_L) / (2.0 * dx) + (v_T - v_B) / (2.0 * dy)
                k_grid[j, i] = k_pass if div_u < 0.0 else k_act
            else:
                k_grid[j, i] = k_act
    for j in prange(ny):
        k_grid[j, 0], k_grid[j, nx-1] = k_grid[j, 1], k_grid[j, nx-2]
    for i in prange(nx):
        k_grid[0, i], k_grid[ny-1, i] = k_grid[1, i], k_grid[ny-2, i]

@njit
def compute_dt(U, k_grid, nx, ny, gz, dx, dy, CFL, DRY):
    max_speed_x, max_speed_y = 1e-8, 1e-8
    for j in range(ny):
        for i in range(nx):
            h = U[0, j, i]
            if h > DRY:
                u, v = abs(U[1, j, i] / h), abs(U[2, j, i] / h)
                c = np.sqrt(k_grid[j, i] * gz * h)
                if u + c > max_speed_x: max_speed_x = u + c
                if v + c > max_speed_y: max_speed_y = v + c
    return CFL * min(dx / max_speed_x, dy / max_speed_y)

@njit(parallel=True)
def compute_fluxes_and_sources(U, k_grid, Fx, Gy, S, nx, ny, gz, gx, gy, mu, DRY):
    for j in prange(ny):
        for i in range(nx - 1):
            hL, huL, hvL = U[0, j, i], U[1, j, i], U[2, j, i]
            hR, huR, hvR = U[0, j, i+1], U[1, j, i+1], U[2, j, i+1]
            kL, kR = k_grid[j, i], k_grid[j, i+1]
            uL, vL = (huL/hL, hvL/hL) if hL > DRY else (0.0, 0.0)
            uR, vR = (huR/hR, hvR/hR) if hR > DRY else (0.0, 0.0)
            f0L, f1L, f2L = huL, huL*uL + 0.5*kL*gz*hL**2, hvL*uL
            f0R, f1R, f2R = huR, huR*uR + 0.5*kR*gz*hR**2, hvR*uR
            cL, cR = np.sqrt(kL * gz * hL) if hL > DRY else 0.0, np.sqrt(kR * gz * hR) if hR > DRY else 0.0
            a = max(abs(uL) + cL, abs(uR) + cR)
            Fx[0, j, i+1] = 0.5*(f0L + f0R) - 0.5*a*(hR - hL)
            Fx[1, j, i+1] = 0.5*(f1L + f1R) - 0.5*a*(huR - huL)
            Fx[2, j, i+1] = 0.5*(f2L + f2R) - 0.5*a*(hvR - hvL)

    for j in prange(ny - 1):
        for i in range(nx):
            hL, huL, hvL = U[0, j, i], U[1, j, i], U[2, j, i]
            hR, huR, hvR = U[0, j+1, i], U[1, j+1, i], U[2, j+1, i]
            kL, kR = k_grid[j, i], k_grid[j+1, i]
            uL, vL = (huL/hL, hvL/hL) if hL > DRY else (0.0, 0.0)
            uR, vR = (huR/hR, hvR/hR) if hR > DRY else (0.0, 0.0)
            g0L, g1L, g2L = hvL, huL*vL, hvL*vL + 0.5*kL*gz*hL**2
            g0R, g1R, g2R = hvR, huR*vR, hvR*vR + 0.5*kR*gz*hR**2
            cL, cR = np.sqrt(kL * gz * hL) if hL > DRY else 0.0, np.sqrt(kR * gz * hR) if hR > DRY else 0.0
            a = max(abs(vL) + cL, abs(vR) + cR)
            Gy[0, j+1, i] = 0.5*(g0L + g0R) - 0.5*a*(hR - hL)
            Gy[1, j+1, i] = 0.5*(g1L + g1R) - 0.5*a*(huR - huL)
            Gy[2, j+1, i] = 0.5*(g2L + g2R) - 0.5*a*(hvR - hvL)

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
                S[0, j, i], S[1, j, i], S[2, j, i] = 0.0, 0.0, 0.0

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
                U[1, j, i], U[2, j, i] = 0.0, 0.0
    for j in prange(ny):
        for c in range(3):
            U[c, j, 0], U[c, j, nx-1] = U[c, j, 1], U[c, j, nx-2]
    for i in prange(nx):
        for c in range(3):
            U[c, 0, i], U[c, ny-1, i] = U[c, 1, i], U[c, ny-2, i]

# ==========================================
# 3. Solver Execution
# ==========================================
def generate_prediction_history(nx, ny, target_times, CFL=0.5):
    """
    Runs the Rusanov solver and stores the full State (U) at specific timestamps.
    Unlike previous scripts, we need hu and hv to calculate velocity!
    """
    dx, dy = X_MAX / nx, Y_MAX / ny
    x = np.linspace(X_MIN, X_MAX, nx)
    y = np.linspace(Y_MIN, Y_MAX, ny)
    X, Y = np.meshgrid(x, y)

    U = np.zeros((3, ny, nx))
    Fx, Gy, S = np.zeros((3, ny, nx+1)), np.zeros((3, ny+1, nx)), np.zeros((3, ny, nx))
    k_grid = np.full((ny, nx), k_act)
    
    Xc, Yc, R = 0.4, 0.65, 0.06
    H0 = R / 1.5
    U[0] = np.where((X - Xc)**2 + (Y - Yc)**2 <= R**2, H0, 0.0)

    t = 0.0
    history = {}
    save_id = 0
    sorted_times = np.sort(np.unique(target_times))
    
    print(f"Running Dynamic-k Rusanov ({nx}x{ny}) to generate predictions...")
    start_time = time.time()

    if save_id < len(sorted_times) and (t >= sorted_times[save_id] or np.isclose(t, sorted_times[save_id], atol=1e-5)):
        history[sorted_times[save_id]] = U.copy() # Saving full state U
        save_id += 1

    while save_id < len(sorted_times):
        compute_k_grid(U, k_grid, nx, ny, dx, dy, k_act, k_pass, DRY)
        dt = compute_dt(U, k_grid, nx, ny, gz, dx, dy, CFL, DRY)
        
        if t + dt > sorted_times[save_id]:
            dt = sorted_times[save_id] - t

        compute_fluxes_and_sources(U, k_grid, Fx, Gy, S, nx, ny, gz, gx, gy, mu, DRY)
        update_state(U, Fx, Gy, S, dt, dx, dy, nx, ny, DRY)
        t += dt

        if t >= sorted_times[save_id] or np.isclose(t, sorted_times[save_id], atol=1e-5):
            history[sorted_times[save_id]] = U.copy() # Saving full state U
            save_id += 1

    print(f"-> Simulation complete in {time.time() - start_time:.2f}s")
    return history, x, y

# ==========================================
# 4. Metric Plotting Functions
# ==========================================
def plot_hmax_velocity_combined(t_titan, xc, yc, H_titan_all, history, x_rus, y_rus):
    print("\n[Figure] h_max + V_aw combined panel (Rusanov vs Titan2D)")
    t_out, hmax_titan_out, hmax_rus_out = [], [], []
    v_aw_rus_out = []

    for it, t_val in enumerate(t_titan):
        if t_val > T_MAX:
            continue
            
        # 1. Compute H_max for Titan2D
        h_titan = np.maximum(H_titan_all[it, :], 0.0)
        wet_titan = h_titan > H_THRESH_VEL
        if not wet_titan.any():
            continue
        hmax_titan = float(h_titan[wet_titan].max())

        # 2. Extract Rusanov state and interpolate onto Titan2D's coordinates
        U_rus = history[t_val]
        h_rus_grid = U_rus[0]
        u_rus_grid = np.where(h_rus_grid > DRY, U_rus[1] / h_rus_grid, 0.0)
        v_rus_grid = np.where(h_rus_grid > DRY, U_rus[2] / h_rus_grid, 0.0)
        
        interp_func_h = RegularGridInterpolator((y_rus, x_rus), h_rus_grid, bounds_error=False, fill_value=0.0)
        interp_func_u = RegularGridInterpolator((y_rus, x_rus), u_rus_grid, bounds_error=False, fill_value=0.0)
        interp_func_v = RegularGridInterpolator((y_rus, x_rus), v_rus_grid, bounds_error=False, fill_value=0.0)
        
        h_pred = interp_func_h((yc, xc))
        u_pred = interp_func_u((yc, xc))
        v_pred = interp_func_v((yc, xc))

        # 3. Compute H_max for Rusanov
        wet_pred = h_pred > H_THRESH_VEL
        hmax_rus = float(h_pred[wet_pred].max()) if wet_pred.any() else 0.0

        # 4. Compute Mass-Weighted Velocity (V_aw) for Rusanov
        spd = np.sqrt(u_pred**2 + v_pred**2)
        denom = np.sum(h_pred)
        v_aw_rus = float(np.sum(h_pred * spd) / denom) if denom > 0 else 0.0

        t_out.append(float(t_val))
        hmax_titan_out.append(hmax_titan)
        hmax_rus_out.append(hmax_rus)
        v_aw_rus_out.append(v_aw_rus)

    t_out = np.array(t_out)
    hmax_titan_out = np.array(hmax_titan_out)
    hmax_rus_out = np.array(hmax_rus_out)
    v_aw_rus_out = np.array(v_aw_rus_out)

    # Note: Because we do not have TITAN_VEL_CSV, we will just plot the Rusanov Velocity curve on the right.
    # If you later obtain the TITAN2D velocity CSV, you can overlay it just like the height!

    fig, (ax_h, ax_v) = plt.subplots(1, 2, figsize=(14, 5))

    # --- Plot h_max ---
    ax_h.plot(t_out, hmax_titan_out * 1e3, color="#1f77b4", linewidth=2.0, linestyle="--", label="TITAN2D")
    ax_h.plot(t_out, hmax_rus_out * 1e3, color="#d62728", linewidth=2.0, linestyle="-", label="Dynamic-k Rusanov")
    
    ax_h.set_xlabel(r"Time $t$ (s)", fontsize=11)
    ax_h.set_ylabel(r"$h_{\mathrm{max}}$ (mm)", fontsize=11)
    ax_h.set_xlim(0.0, float(t_out.max()))
    ax_h.set_ylim(bottom=0.0)
    ax_h.grid(True, linestyle=":", alpha=0.6)
    #ax_h.legend(loc="upper right")
    #ax_h.set_title("Maximum Flow Thickness over Time", fontweight="bold")

    # --- Plot Mass-Weighted Velocity ---
    ax_v.plot(t_out, v_aw_rus_out, color="#d62728", linewidth=2.0, linestyle="-", label="Dynamic-k Rusanov")
    
    ax_v.set_xlabel(r"Time $t$ (s)", fontsize=11)
    ax_v.set_ylabel(r"Mass-Weighted Velocity $\bar{V}$ (m/s)", fontsize=11)
    ax_v.set_xlim(0.0, float(t_out.max()))
    ax_v.set_ylim(bottom=0.0)
    ax_v.grid(True, linestyle=":", alpha=0.6)
    #ax_v.legend(loc="lower right")
    #ax_v.set_title("Mass-Weighted Avalanche Velocity", fontweight="bold")

    plt.tight_layout()
    plt.savefig("hmax_velocity_combined.pdf", format="pdf", bbox_inches="tight", pad_inches=0)
    plt.show()


# ==========================================
# 5. Main Execution
# ==========================================
if __name__ == "__main__":
    print("Loading TITAN2D reference data...")
    try:
        data = np.load("titan_height_cells_0to2s.npz")
        t_titan_full = data['t']
        xc_titan = data['xc']
        yc_titan = data['yc']
        H_titan_all_full = data['H']
    except FileNotFoundError:
        print("ERROR: 'titan_height_cells_0to2s.npz' not found.")
        exit()

    # Filter data up to T_MAX
    mask_time = t_titan_full <= T_MAX
    t_titan = t_titan_full[mask_time]
    H_titan_all = H_titan_all_full[mask_time, :]

    # Run our solver over a 400x130 grid.
    # Note: We must return the FULL state history (U) because we need momentum (hu, hv) to calculate velocity.
    history, x_rus, y_rus = generate_prediction_history(nx=800, ny=260, target_times=t_titan, CFL=0.5)

    # Generate Publication Plots
    plot_hmax_velocity_combined(t_titan, xc_titan, yc_titan, H_titan_all, history, x_rus, y_rus)