import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import time

# ==========================================
# 1. Global Physical Parameters
# ==========================================
zeta  = np.radians(32.0)
delta = np.radians(22.0)
phi   = np.radians(29.0)
g     = 9.81

gx = g * np.sin(zeta)
gy = 0.0
gz = g * np.cos(zeta)

mu = np.tan(delta)

inner = 1.0 - (1.0 + np.tan(delta)**2) * np.cos(phi)**2
k_act = 2.0 * (1.0 - np.sqrt(max(inner,0.0))) / np.cos(phi)**2 - 1.0

DRY = 1e-4

# ==========================================
# 2. Flux and Source Functions
# ==========================================
def primitive(U):
    h  = U[0]
    hu = U[1]
    hv = U[2]

    h_safe = np.maximum(h, DRY)
    u = np.where(h > DRY, hu / h_safe, 0.0)
    v = np.where(h > DRY, hv / h_safe, 0.0)

    return h, u, v

def flux_x(U):
    h, u, v = primitive(U)
    hu = U[1]
    hv = U[2]

    F = np.zeros_like(U)
    F[0] = hu
    F[1] = hu*u + 0.5 * k_act * gz * h**2
    F[2] = hv*u
    return F

def flux_y(U):
    h, u, v = primitive(U)
    hu = U[1]
    hv = U[2]

    G = np.zeros_like(U)
    G[0] = hv
    G[1] = hu*v
    G[2] = hv*v + 0.5 * k_act * gz * h**2
    return G

def source(U):
    h, u, v = primitive(U)
    S = np.zeros_like(U)

    vmag = np.sqrt(u*u + v*v + 1e-6)

    fric_x = h * gz * mu * u / vmag
    fric_y = h * gz * mu * v / vmag

    S[1] = gx*h - fric_x
    S[2] = gy*h - fric_y

    mask = h > DRY
    S[1] *= mask
    S[2] *= mask

    return S

def rusanov_x(UL, UR):
    FL = flux_x(UL)
    FR = flux_x(UR)

    hL, uL, vL = primitive(UL)
    hR, uR, vR = primitive(UR)

    cL = np.sqrt(k_act * gz * hL)
    cR = np.sqrt(k_act * gz * hR)

    a = np.maximum(np.abs(uL)+cL, np.abs(uR)+cR)

    return 0.5*(FL + FR) - 0.5*a*(UR - UL)

def rusanov_y(UL, UR):
    GL = flux_y(UL)
    GR = flux_y(UR)

    hL, uL, vL = primitive(UL)
    hR, uR, vR = primitive(UR)

    cL = np.sqrt(k_act * gz * hL)
    cR = np.sqrt(k_act * gz * hR)

    a = np.maximum(np.abs(vL)+cL, np.abs(vR)+cR)

    return 0.5*(GL + GR) - 0.5*a*(UR - UL)


# ==========================================
# 3. Main Simulation Function
# ==========================================
def run_simulation(nx, ny, Lx=15.0, Ly=15.0, T_end=1.5, CFL=0.4):
    """
    Runs the 2D Savage-Hutter avalanche simulation.
    
    Returns:
        history (list): List of tuples (time, depth_field)
        X, Y (ndarrays): 2D meshgrid arrays
        exec_time (float): Total execution time in seconds
    """
    # Start timer
    start_time = time.time()
    
    # Setup Grid
    dx = Lx / nx
    dy = Ly / ny

    x = np.linspace(0, Lx, nx)
    y = np.linspace(0, Ly, ny)
    X, Y = np.meshgrid(x, y)

    # Initialize State Vector
    U = np.zeros((3, ny, nx))
    R2 = (X - 3.0)**2 + (Y - 7.5)**2
    U[0] = np.maximum(1.0 - R2, 0.0)

    t = 0.0
    history = []
    save_times = np.linspace(0, T_end, 6)
    save_id = 1  

    history.append((0.0, U[0].copy()))

    while t < T_end:
        h, u, v = primitive(U)
        c = np.sqrt(k_act * gz * h)
        
        # 2D CFL Condition
        max_speed_x = np.max(np.abs(u) + c)
        max_speed_y = np.max(np.abs(v) + c)
        
        max_speed_x = max(max_speed_x, 1e-8)
        max_speed_y = max(max_speed_y, 1e-8)

        dt = CFL * min(dx / max_speed_x, dy / max_speed_y)
        
        if t + dt > T_end:
            dt = T_end - t

        Fx = np.zeros((3, ny, nx+1))
        Gy = np.zeros((3, ny+1, nx))

        Fx[:,:,1:-1] = rusanov_x(U[:,:, :-1], U[:,:,1:])
        Gy[:,1:-1,:] = rusanov_y(U[:,:-1,:], U[:,1:,:])

        U_new = U.copy()

        U_new -= dt/dx * (Fx[:,:,1:] - Fx[:,:,:-1])
        U_new -= dt/dy * (Gy[:,1:,:] - Gy[:,:-1,:])

        U_new += dt * source(U)

        # Positivity fix
        U_new[0] = np.maximum(U_new[0], 0.0)
        dry = U_new[0] < DRY
        U_new[1][dry] = 0.0
        U_new[2][dry] = 0.0

        # Transmissive BC
        U_new[:,:,0]  = U_new[:,:,1]
        U_new[:,:,-1] = U_new[:,:,-2]
        U_new[:,0,:]  = U_new[:,1,:]
        U_new[:,-1,:] = U_new[:,-2,:]

        U = U_new
        t += dt

        if save_id < len(save_times) and t >= save_times[save_id]:
            history.append((t, U[0].copy()))
            save_id += 1
            
    # End timer
    exec_time = time.time() - start_time
    
    return history, X, Y, exec_time


# ==========================================
# 4. Run Multiple Configurations & Plot
# ==========================================
if __name__ == "__main__":
    # Define the resolutions you want to test
    # (nx, ny) pairs
    resolutions_to_test = [
        (50, 50),
        (100, 100),
        (200, 200),
    ]
    
    results = {}
    
    for nx, ny in resolutions_to_test:
        print(f"Running simulation for {nx}x{ny} grid (dx={15.0/nx:.3f})...")
        history, X, Y, exec_time = run_simulation(nx=nx, ny=ny)
        print(f"-> Completed in {exec_time:.2f} seconds.\n")
        
        # Store results if you want to plot a specific one later
        results[(nx, ny)] = (history, X, Y)
        
    # Example: Plot the results of the highest resolution run
    best_res = resolutions_to_test[-1]
    history, X, Y = results[best_res]
    
    print(f"Plotting results for {best_res[0]}x{best_res[1]} resolution...")
    
    fig = plt.figure(figsize=(16,10))
    for i, (ts, hplot) in enumerate(history):
        ax = fig.add_subplot(2, 3, i+1, projection='3d')
        hplot = np.where(hplot > DRY, hplot, np.nan)

        ax.plot_surface(X, Y, hplot,
                        cmap=cm.viridis,
                        linewidth=0,
                        antialiased=True)

        ax.set_title(f"t = {ts:.2f} s")
        ax.set_xlim(0, 15.0)
        ax.set_ylim(0, 15.0)
        ax.set_zlim(0, 1)
        ax.view_init(35, -120)

    plt.tight_layout()
    plt.show()