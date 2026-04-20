import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
from scipy.interpolate import RegularGridInterpolator
from scipy.spatial import cKDTree
from numba import njit, prange
import time

# ==========================================
# 1. Configuration & Physical Parameters
# ==========================================
T_MAX = 2.0          # Max time to analyze
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
        history[sorted_times[save_id]] = U[0].copy()
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
            history[sorted_times[save_id]] = U[0].copy()
            save_id += 1

    print(f"-> Simulation complete in {time.time() - start_time:.2f}s")
    return history, x, y

# ==========================================
# 4. Metric Plotting Functions
# ==========================================
def plot_iou_rmse_combined(t_titan, xc, yc, H_titan_all, history, x_rus, y_rus):
    print("\n[Figure] IoU + RMSE combined panel (Rusanov vs Titan2D)")
    t_out, iou_out, rmse_out, sq_errors_all = [], [], [], []

    for it, t_val in enumerate(t_titan):
        h_titan = np.maximum(H_titan_all[it, :], 0.0)
        
        # Pull correct history and interpolate onto Titan2D's coordinates
        h_rus_2d = history[t_val]
        interp_func = RegularGridInterpolator((y_rus, x_rus), h_rus_2d, bounds_error=False, fill_value=0.0)
        h_pred = interp_func((yc, xc))

        wet_titan = h_titan > H_THRESH_VEL
        if not wet_titan.any():
            continue

        wet_pred = h_pred > H_THRESH_VEL
        intersection = int(np.sum(wet_titan & wet_pred))
        union = int(np.sum(wet_titan | wet_pred))
        iou = intersection / union if union > 0 else 1.0

        err = h_pred[wet_titan] - h_titan[wet_titan]
        rmse = float(np.sqrt(np.mean(err ** 2)))
        sq_errors_all.extend(err ** 2)

        t_out.append(float(t_val))
        iou_out.append(iou)
        rmse_out.append(rmse)

    t_out, iou_out, rmse_out = np.array(t_out), np.array(iou_out), np.array(rmse_out)
    mean_iou = float(np.mean(iou_out)) * 100.0
    global_rmse = float(np.sqrt(np.mean(sq_errors_all)))
    
    print(f"  Global mean IoU = {mean_iou:.2f}%   Global RMSE = {global_rmse:.6f} m")

    fig, (ax_iou, ax_rmse) = plt.subplots(1, 2, figsize=(14, 5))

    # --- Plot IoU ---
    ax_iou.plot(t_out, iou_out * 100.0, color="#2ca02c", linewidth=2.0, linestyle="-")
    ax_iou.axhline(mean_iou, color="0.45", linewidth=1.2, linestyle="--", label=f"Mean = {mean_iou:.1f}%")
    ax_iou.set_xlabel(r"Time $t$ (s)", fontsize=11)
    ax_iou.set_ylabel(r"IoU (%)", fontsize=11)
    ax_iou.set_xlim(0.0, float(t_out.max()))
    ax_iou.set_ylim(0.0, 105.0)
    ax_iou.grid(True, linestyle=":", alpha=0.6)
    #ax_iou.legend(loc="lower right")
    #ax_iou.set_title("Wet-Cell Intersection over Union (IoU)", fontweight="bold")

    # --- Plot RMSE ---
    ax_rmse.plot(t_out, rmse_out, color="#1f77b4", linewidth=2.0, linestyle="-")
    ax_rmse.axhline(global_rmse, color="0.45", linewidth=1.2, linestyle="--", label=f"Global = {global_rmse:.4f} m")
    ax_rmse.set_xlabel(r"Time $t$ (s)", fontsize=11)
    ax_rmse.set_ylabel(r"RMSE of $h$ (m)", fontsize=11)
    ax_rmse.set_xlim(0.0, float(t_out.max()))
    ax_rmse.set_ylim(bottom=0.0)
    ax_rmse.grid(True, linestyle=":", alpha=0.6)
    #ax_rmse.legend(loc="upper right")
    #ax_rmse.set_title("Root Mean Square Error (RMSE) on Wet Cells", fontweight="bold")

    plt.tight_layout()
    plt.savefig("iou_rmse.pdf", format="pdf", bbox_inches="tight", pad_inches=0)
    plt.show()


def plot_wetted_area_comparison(t_titan, xc, yc, H_titan_all, history, x_rus, y_rus):
    print("\n[Figure] Cumulative wetted footprint (Rusanov vs Titan2D)")
    res = 300
    xs = np.linspace(X_MIN, X_MAX, res, dtype=np.float32)
    ys = np.linspace(Y_MIN, Y_MAX, res, dtype=np.float32)
    Xg, Yg = np.meshgrid(xs, ys, indexing="ij")
    x_flat, y_flat = Xg.ravel(), Yg.ravel()
    cell_area_grid = float(xs[1] - xs[0]) * float(ys[1] - ys[0])

    # --- Titan2D cumulative mask ---
    ever_wet_titan_cells = np.any(H_titan_all > H_THRESH_VEL, axis=0)
    tree = cKDTree(np.column_stack([xc, yc]))
    _, nn = tree.query(np.column_stack([x_flat, y_flat]), k=1)
    ever_wet_titan = ever_wet_titan_cells[nn].reshape(res, res)

    # --- Rusanov cumulative mask ---
    ever_wet_rus = np.zeros(res * res, dtype=bool)
    for t_val in history.keys():
        h_rus_2d = history[t_val]
        interp_func = RegularGridInterpolator((y_rus, x_rus), h_rus_2d, bounds_error=False, fill_value=0.0)
        h_pred = interp_func((y_flat, x_flat))
        ever_wet_rus |= (h_pred > H_THRESH_VEL)
    ever_wet_rus = ever_wet_rus.reshape(res, res)

    print(f"  Titan2D wetted area = {float(ever_wet_titan.sum()) * cell_area_grid:.5f} m²")
    print(f"  Rusanov wetted area = {float(ever_wet_rus.sum()) * cell_area_grid:.5f} m²")

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.set_facecolor("#f7f7f7")

    # Plot masks
    Z_t = np.where(ever_wet_titan, 1.0, np.nan)
    ax.pcolormesh(xs, ys, Z_t.T, cmap=mcolors.ListedColormap(["#d62728"]), vmin=0.5, vmax=1.5, alpha=0.45, shading="auto")
    
    Z_p = np.where(ever_wet_rus, 1.0, np.nan)
    ax.pcolormesh(xs, ys, Z_p.T, cmap=mcolors.ListedColormap(["#1f77b4"]), vmin=0.5, vmax=1.5, alpha=0.45, shading="auto")

    # Outline initial pile and bounds
    circle = plt.Circle((0.4, 0.65), 0.06, fill=False, edgecolor="black", linewidth=1.2, zorder=6)
    ax.add_patch(circle)
    bnd_kw = dict(color="0.20", linewidth=1.5, zorder=7)
    ax.plot([X_MIN, X_MIN, X_MAX, X_MAX, X_MIN], [Y_MIN, Y_MAX, Y_MAX, Y_MIN, Y_MIN], **bnd_kw)

    ax.set_aspect("equal")
    ax.set_xlim(X_MIN - 0.02, X_MAX + 0.02)
    ax.set_ylim(Y_MIN - 0.02, Y_MAX + 0.02)
    ax.set_xlabel(r"Downslope position $x$ (m)")
    ax.set_ylabel(r"Cross-slope position $y$ (m)")

    legend_elements = [
        Patch(facecolor="#1f77b4", alpha=0.55, edgecolor="none", label="Dynamic-k Rusanov"),
        Patch(facecolor="#d62728", alpha=0.55, edgecolor="none", label="TITAN2D"),
        Patch(facecolor="#6a3d9a", alpha=0.65, edgecolor="none", label="Overlap"),
    ]
    #ax.legend(handles=legend_elements, loc="upper right", frameon=True, fontsize=9)
    #plt.title("Cumulative Wetted Footprint Comparison", fontweight="bold")
    plt.tight_layout()
    plt.savefig("wetted_area.pdf", format="pdf", bbox_inches="tight", pad_inches=0)
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

    # Run our solver over a high-res grid (400x130 recommended for speed vs accuracy)
    # Using the exact timestamps from TITAN2D
    history, x_rus, y_rus = generate_prediction_history(nx=800, ny=260, target_times=t_titan, CFL=0.5)

    # Generate Publication Plots
    plot_iou_rmse_combined(t_titan, xc_titan, yc_titan, H_titan_all, history, x_rus, y_rus)
    plot_wetted_area_comparison(t_titan, xc_titan, yc_titan, H_titan_all, history, x_rus, y_rus)