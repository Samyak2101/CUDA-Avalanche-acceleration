import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, AutoMinorLocator
from scipy.interpolate import RegularGridInterpolator
from numba import njit, prange
import time

# ==========================================
# 1. Configuration & Physical Parameters
# ==========================================
T_MAX = 0.55          # Max time to analyze
H_THRESH_VEL = 1e-3   # 1mm threshold for a cell to be considered "wet"
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
        history[sorted_times[save_id]] = U.copy()
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
            history[sorted_times[save_id]] = U.copy()
            save_id += 1

    print(f"-> Simulation complete in {time.time() - start_time:.2f}s")
    return history, X, Y

# ==========================================
# 4. Metric Plotting Functions
# ==========================================
def plot_vel_vs_dist_comparison(history, X, Y):
    print("\n[Figure] Velocity vs Distance – Rusanov Centroid Trajectory")

    # Arrays to track centroid position and velocity over time
    t_out = []
    dist_out = []
    v_aw_out = []
    
    xcen0, ycen0 = None, None

    # Calculate centroid parameters at each saved time step
    for t_val in sorted(history.keys()):
        U_rus = history[t_val]
        h = U_rus[0]
        u = np.where(h > DRY, U_rus[1] / h, 0.0)
        v = np.where(h > DRY, U_rus[2] / h, 0.0)
        
        mask = h > H_THRESH_VEL
        if not mask.any():
            continue

        h_wet = h[mask]
        denom = float(h_wet.sum())
        
        # Calculate Center of Mass (Centroid)
        xcen = float(np.sum(X[mask] * h_wet) / denom)
        ycen = float(np.sum(Y[mask] * h_wet) / denom)
        
        # Calculate Mass-Weighted Velocity
        spd = np.sqrt(u[mask]**2 + v[mask]**2)
        v_aw = float(np.sum(h_wet * spd) / denom)

        if xcen0 is None:
            xcen0, ycen0 = xcen, ycen
            print(f"  Initial Rusanov centroid:  Xcen0 = {xcen0:.4f} m,  Ycen0 = {ycen0:.4f} m")

        # Distance formula from original position
        dist = float(np.sqrt((xcen - xcen0)**2 + (ycen - ycen0)**2))
        
        t_out.append(t_val)
        dist_out.append(dist)
        v_aw_out.append(v_aw)
        
        print(f"  t = {t_val:.4f} s  dist = {dist:.4f} m  V_aw = {v_aw:.5f} m/s")

    # Plotting
    fig, ax = plt.subplots(figsize=(8, 5))

    # Note: We plot only the Rusanov solver here. 
    # If a TITAN2D centroid/distance CSV becomes available, it can be overlaid just like the PINN script.
    ax.plot(dist_out, v_aw_out, color="#d62728", linewidth=2.0, linestyle="-", label="Dynamic-k Rusanov")

    # Add timestamp annotations along the curve
    label_times = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.55]
    t_arr = np.array(t_out)
    
    for lt in label_times:
        if lt > T_MAX:
            continue
        idx = int(np.argmin(np.abs(t_arr - lt)))
        xp = dist_out[idx]
        yp = v_aw_out[idx]
        tp = t_arr[idx]
        
        ax.plot(xp, yp, marker="s", markersize=6, color="#d62728", markerfacecolor="white", markeredgewidth=1.5, zorder=5)
        ax.annotate(f"$t$={tp:.2g}s",
                    xy=(xp, yp), xytext=(4, -16),
                    textcoords="offset points",
                    fontsize=9, color="#d62728",
                    va="top", ha="left", zorder=6,
                    arrowprops=dict(arrowstyle="-", color="#d62728", lw=1.0, shrinkA=0, shrinkB=2))

    ax.set_xlabel(r"Distance from initial centroid $d$ (m)", fontsize=11)
    ax.set_ylabel(r"Mass-Weighted Velocity $\bar{V}$ (m/s)", fontsize=11)
    ax.set_xlim(left=0.0)
    ax.set_ylim(bottom=0.0)
    ax.grid(True, linestyle=":", alpha=0.6)
    #ax.legend(loc="upper left")
    #ax.set_title("Centroid Velocity vs. Distance Traveled", fontweight="bold")

    plt.tight_layout()
    plt.savefig("vel_vs_dist_comparison.pdf", format="pdf", bbox_inches="tight", pad_inches=0)
    plt.show()

# ==========================================
# 5. Main Execution
# ==========================================
if __name__ == "__main__":
    print("Loading TITAN2D timestamps...")
    try:
        data = np.load("titan_height_cells_0to2s.npz")
        t_titan_full = data['t']
    except FileNotFoundError:
        print("ERROR: 'titan_height_cells_0to2s.npz' not found.")
        exit()

    # Filter data up to T_MAX
    mask_time = t_titan_full <= T_MAX
    t_titan = t_titan_full[mask_time]

    # Run our solver over a 400x130 grid.
    history, X, Y = generate_prediction_history(nx=400, ny=130, target_times=t_titan, CFL=0.5)

    # Generate Publication Plots
    plot_vel_vs_dist_comparison(history, X, Y)