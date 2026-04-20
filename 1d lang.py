import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d

# ==========================================
# 1. Physical Parameters (Huber Exp 106b)
# ==========================================
zeta  = np.radians(32.0)
delta = np.radians(22.0)
phi   = np.radians(29.0)
epsilon = 0.3218

inner = 1.0 - (1.0 + np.tan(delta)**2) * np.cos(phi)**2
k_act = 2.0 * (1.0 - np.sqrt(max(inner, 0.0))) / np.cos(phi)**2 - 1.0

sin_z = np.sin(zeta)
cos_z = np.cos(zeta)
tan_d = np.tan(delta)
beta  = epsilon * k_act * cos_z  
a     = sin_z - tan_d * cos_z

# ==========================================
# 2. Analytical Solution Setup
# ==========================================
T_end = 5.0
K = 1.0
K_ode = 2.0 * beta * K

def g_ode_rhs(t, y):
    g, p = y
    g = max(g, 1e-12)
    return [p, K_ode / g**2]

t_ode = np.linspace(0, T_end, 2000)
sol = solve_ivp(g_ode_rhs, (0, T_end), [1.0, 0.0], t_eval=t_ode, method='RK45')
g_interp = interp1d(sol.t, sol.y[0], kind='cubic', fill_value='extrapolate')

def h_analytical(x, t):
    g_t = float(g_interp(t))
    x_c = 0.5 * a * t**2
    eta = (x - x_c) / g_t
    h = (K / g_t) * (1.0 - eta**2)
    return np.maximum(h, 0.0)

def u_analytical(x, t):
    g_t = float(g_interp(t))
    x_c = 0.5 * a * t**2
    eta = (x - x_c) / g_t
    u0 = a * t
    factor = np.sqrt(2.0 * K_ode * max(g_t - 1.0, 0.0) / g_t)
    u = factor * eta + u0
    return np.where(np.abs(eta) <= 1.0, u, np.nan)

# ==========================================
# 3. Lagrangian Setup
# ==========================================
dt = 0.002
nt = int(T_end / dt) + 1
mu = 0.02
N_cells = 200

# Arrays for boundaries (j) and centers (i)
x_j = np.linspace(-1.0, 1.0, N_cells + 1)
x_i = np.zeros(N_cells)
h_i = np.zeros(N_cells)
m_i = np.zeros(N_cells)
u_j = np.zeros(N_cells + 1)

# Initialize cells using loops
for i in range(N_cells):
    x_i[i] = 0.5 * (x_j[i] + x_j[i+1])
    h_i[i] = max(K * (1.0 - x_i[i]**2), 0.0)
    dx = x_j[i+1] - x_j[i]
    m_i[i] = h_i[i] * dx

save_times = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
history = []

# ==========================================
# 4. Main Integration Loop (Explicit For-Loops)
# ==========================================
for n in range(nt):
    t_curr = n * dt
    
    if any(abs(t_curr - st) < dt/2 for st in save_times):
        history.append((t_curr, x_j.copy(), h_i.copy(), u_j.copy()))

    # -------------------------------------------------
    # Step 1: Advect boundaries
    # -------------------------------------------------
    for j in range(len(x_j)):
        x_j[j] = x_j[j] + u_j[j] * dt

    # -------------------------------------------------
    # Step 2: Update cell depths
    # -------------------------------------------------
    dx_curr = np.zeros(N_cells)
    for i in range(N_cells):
        dx_curr[i] = max(x_j[i+1] - x_j[i], 1e-10)
        h_i[i] = m_i[i] / dx_curr[i]

    for i in range(N_cells):
        x_i[i] = 0.5 * (x_j[i] + x_j[i+1])

    # -------------------------------------------------
    # Step 3: Pressure gradients
    # -------------------------------------------------
    dpdx = np.zeros_like(x_j)

    # interior nodes
    for j in range(1, N_cells):
        dpdx[j] = (h_i[j] - h_i[j-1]) / (x_i[j] - x_i[j-1])

    # boundaries
    dpdx[0]  =  h_i[0]   / (x_i[0]   - x_j[0])
    dpdx[-1] = -h_i[-1]  / (x_j[-1]  - x_i[-1])

    # -------------------------------------------------
    # Step 4: Source + artificial viscosity
    # -------------------------------------------------
    src = np.zeros_like(u_j)
    for j in range(len(u_j)):
        #src[j] = sin_z - np.sign(u_j[j] + 1e-12) * cos_z * tan_d
        src[j] = sin_z - 1 * cos_z * tan_d

    d2udx2 = np.zeros_like(u_j)
    for j in range(1, N_cells):
        dx_bound = 0.5 * (x_j[j+1] - x_j[j-1])
        term_r = (u_j[j+1] - u_j[j])   / (x_j[j+1] - x_j[j])
        term_l = (u_j[j]   - u_j[j-1]) / (x_j[j]   - x_j[j-1])
        d2udx2[j] = (term_r - term_l) / dx_bound

    # -------------------------------------------------
    # Step 5: Update velocities
    # -------------------------------------------------
    for j in range(len(u_j)):
        u_j[j] = u_j[j] + dt * (src[j] - beta * dpdx[j] + mu * d2udx2[j])

# ==========================================
# 5. Visualization
# ==========================================
fig, (ax_h, ax_u) = plt.subplots(1, 2, figsize=(15, 6))
x_dense = np.linspace(-2.0, 10.0, 1000)
colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(history)))

for idx, (t, x_saved, h_saved, u_saved) in enumerate(history):
    color = colors[idx]
    lbl_num = f'Num. t={t:.1f}' if idx == len(history)-1 else ""
    lbl_ana = f'Ana. t={t:.1f}' if idx == len(history)-1 else ""
    
    # Plot Depth (h)
    x_centers = np.zeros(N_cells)
    for i in range(N_cells):
        x_centers[i] = 0.5 * (x_saved[i] + x_saved[i+1])
    ax_h.plot(x_centers, h_saved, color=color, linewidth=2, label=lbl_num)
    
    h_ana = h_analytical(x_dense, t)
    mask_h = h_ana > 1e-4
    if np.any(mask_h):
        ax_h.plot(x_dense[mask_h], h_ana[mask_h], color=color, linewidth=1.5, linestyle='--', label=lbl_ana)
        
    # Plot Velocity (u)
    ax_u.plot(x_saved, u_saved, color=color, linewidth=2, label=lbl_num)
    
    u_ana = u_analytical(x_dense, t)
    if np.any(mask_h):
        ax_u.plot(x_dense[mask_h], u_ana[mask_h], color=color, linewidth=1.5, linestyle='--', label=lbl_ana)

#ax_h.set_title('Dimensionless Depth ($h$) vs Distance')
ax_h.set_xlabel('Dimensionless Distance ($x$)')
ax_h.set_ylabel('Depth ($h$)')
ax_h.grid(True, alpha=0.3)

#ax_u.set_title('Dimensionless Velocity ($u$) vs Distance')
ax_u.set_xlabel('Dimensionless Distance ($x$)')
ax_u.set_ylabel('Velocity ($u$)')
ax_u.grid(True, alpha=0.3)

# fig.suptitle(f'Savage-Hutter Parabolic Cap: Lagrangian vs Analytical\n'
#              rf'$\zeta={np.degrees(zeta):.0f}^\circ$, $\delta={np.degrees(delta):.0f}^\circ$, $\beta={beta:.3f}$', 
#              fontsize=14, y=1.02)

from matplotlib.lines import Line2D
custom_lines = [Line2D([0], [0], color='k', lw=2),
                Line2D([0], [0], color='k', lw=1.5, linestyle='--')]

ax_h.legend(custom_lines, ['Lagrangian Numerical', 'Analytical'], loc='upper right')
ax_u.legend(custom_lines, ['Lagrangian Numerical', 'Analytical'], loc='upper left')

plt.tight_layout()
plt.savefig("coupled.pdf", format="pdf", bbox_inches="tight", pad_inches=0)
plt.show()