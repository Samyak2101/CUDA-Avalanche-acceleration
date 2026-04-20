import numpy as np
import matplotlib.pyplot as plt
from numba import njit, prange

# ==========================================
# 1. Physical Parameters
# ==========================================
zeta  = np.radians(37.4)  
delta = np.radians(27.0)  
phi   = np.radians(37.3)  
g     = 9.81
gx, gy, gz = g * np.sin(zeta), 0.0, g * np.cos(zeta)
mu = np.tan(delta)

inner = 1.0 - (1.0 + np.tan(delta)**2) * np.cos(phi)**2
k_act  = 2.0 * (1.0 - np.sqrt(max(inner, 0.0))) / np.cos(phi)**2 - 1.0
k_pass = 2.0 * (1.0 + np.sqrt(max(inner, 0.0))) / np.cos(phi)**2 - 1.0
DRY = 1e-4

# ==========================================
# 2. Physics Kernels (Abridged for brevity, same as titan2d_val.py)
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
# 3. Solver: Extract Single Point at Target Time
# ==========================================
def extract_point_value(nx, ny, CFL, probe_x, probe_y, target_time):
    Lx, Ly = 4.0, 1.3
    dx, dy = Lx / nx, Ly / ny
    x = np.linspace(0, Lx, nx)
    y = np.linspace(0, Ly, ny)
    X, Y = np.meshgrid(x, y)

    U = np.zeros((3, ny, nx))
    Fx, Gy, S = np.zeros((3, ny, nx+1)), np.zeros((3, ny+1, nx)), np.zeros((3, ny, nx))
    k_grid = np.full((ny, nx), k_act)
    
    Xc, Yc, R = 0.4, 0.65, 0.06
    U[0] = np.where((X - Xc)**2 + (Y - Yc)**2 <= R**2, R / 1.5, 0.0)

    idx_x = np.argmin(np.abs(x - probe_x))
    idx_y = np.argmin(np.abs(y - probe_y))

    t = 0.0
    while t < target_time:
        compute_k_grid(U, k_grid, nx, ny, dx, dy, k_act, k_pass, DRY)
        dt = compute_dt(U, k_grid, nx, ny, gz, dx, dy, CFL, DRY)
        if t + dt > target_time:
            dt = target_time - t

        compute_fluxes_and_sources(U, k_grid, Fx, Gy, S, nx, ny, gz, gx, gy, mu, DRY)
        update_state(U, Fx, Gy, S, dt, dx, dy, nx, ny, DRY)
        t += dt
        
    return U[0, idx_y, idx_x]

# ==========================================
# 4. Execution & Diminishing Returns Plots
# ==========================================
if __name__ == "__main__":
    # The coordinate and time requested by your professor
    PROBE_X = 1.0
    PROBE_Y = 0.65
    TARGET_TIME = 1.0
    
    # ---------------------------------------------------------
    # TEST 1: Grid Size Convergence (Find optimal nx)
    # ---------------------------------------------------------
    print("Running Grid Convergence Study...")
    nx_values = [100, 200, 400, 800, 1200, 1600] 
    h_spatial_results = []
    
    for nx in nx_values:
        ny = int(nx * (1.3 / 4.0)) # Keep grid cells square
        print(f" -> Solving for {nx}x{ny}...")
        h_val = extract_point_value(nx, ny, CFL=0.5, probe_x=PROBE_X, probe_y=PROBE_Y, target_time=TARGET_TIME)
        h_spatial_results.append(h_val)

    # ---------------------------------------------------------
    # TEST 2: Time Step Convergence (Find optimal CFL)
    # ---------------------------------------------------------
    print("\nRunning Time Step Convergence Study...")
    # Varying CFL while keeping grid constant at a high resolution (400x130)
    cfl_values = [0.8, 0.4, 0.2, 0.1, 0.05]
    h_temporal_results = []
    
    for cfl in cfl_values:
        print(f" -> Solving for CFL = {cfl}...")
        h_val = extract_point_value(nx=400, ny=130, CFL=cfl, probe_x=PROBE_X, probe_y=PROBE_Y, target_time=TARGET_TIME)
        h_temporal_results.append(h_val)

    # ---------------------------------------------------------
    # Plotting the Asymptotes (Separate PDFs)
    # ---------------------------------------------------------
    
    # PLOT 1: Diminishing Returns for Grid Size
    fig1 = plt.figure(figsize=(7, 5))
    ax1 = fig1.add_subplot(111)
    
    ax1.plot(nx_values, h_spatial_results, 'bo-', linewidth=2, markersize=8)
    #ax1.set_title(f"Grid Convergence at x={PROBE_X}m, t={TARGET_TIME}s\n(Constant CFL=0.5)", fontweight='bold')
    ax1.set_xlabel("Number of Grid Cells in X ($N_x$)")
    ax1.set_ylabel("Measured Avalanche Height $h$ (m)")
    ax1.grid(True, linestyle=':', alpha=0.6)
    
    # Add percentage change annotations to prove diminishing returns
    for i in range(1, len(nx_values)):
        change = abs((h_spatial_results[i] - h_spatial_results[i-1]) / h_spatial_results[i-1]) * 100
        ax1.annotate(f"Change: {change:.2f}%", 
                     (nx_values[i], h_spatial_results[i]), 
                     textcoords="offset points", xytext=(0,10), ha='center', fontsize=9)

    plt.tight_layout()
    fig1.savefig("grid_convergence.pdf", bbox_inches='tight', pad_inches=0.1)
    print("Saved 'grid_convergence.pdf'")
    plt.show()

    # PLOT 2: Diminishing Returns for Time Step (CFL)
    fig2 = plt.figure(figsize=(7, 5))
    ax2 = fig2.add_subplot(111)
    
    ax2.plot(cfl_values, h_temporal_results, 'gs-', linewidth=2, markersize=8)
    ax2.invert_xaxis() 
    #ax2.set_title(f"Time Step Convergence at x={PROBE_X}m, t={TARGET_TIME}s\n(Constant Grid=400x130)", fontweight='bold')
    ax2.set_xlabel(r"Courant Number (Decreasing $\Delta t$ $\rightarrow$)")
    ax2.set_ylabel("Measured Avalanche Height $h$ (m)")
    ax2.grid(True, linestyle=':', alpha=0.6)

    plt.tight_layout()
    fig2.savefig("time_convergence.pdf", bbox_inches='tight', pad_inches=0.1)
    print("Saved 'time_convergence.pdf'")
    plt.show()