import numpy as np
import matplotlib.pyplot as plt

# --- constants ---
g_const = 9.81
L_left = -2
L_right = 20
nx = 321
dx = (L_right - L_left) / (nx - 1)
dt = 0.00014
theta = np.pi / 6
nt = 1920
mu = 0.1
a = g_const * np.sin(theta) - mu * g_const * np.cos(theta)
beta = g_const * np.cos(theta)

# --- functions ---
def g(t):
    return 0.11613 * t**2 + 0.74497 * t + 1.0

def u_analytical(x, t):
    g_t = g(t)
    return np.sqrt((2/g_t)*(g_t - 1)) * (x - 0.5*a*t**2)/g_t + a*t

# --- grids ---
x = np.linspace(L_left, L_right, nx)
z = np.linspace(0, 3, 200)  # time array
X, Z = np.meshgrid(x, z)
U = u_analytical(X, Z)

# --- Heatmap plot ---
fig, ax = plt.subplots(figsize=(10, 5))

# Set smooth contour levels and color map
levels = np.linspace(np.min(U), np.max(U), 200)
c = ax.contourf(X, Z, U, levels=levels, cmap='turbo', extend='both')

# Add contour lines for better visual depth
contours = ax.contour(X, Z, U, levels=15, colors='k', linewidths=0.4, alpha=0.5)
ax.clabel(contours, inline=True, fontsize=8, fmt="%.2f")

# Labels and formatting
ax.set_title("Analytical Velocity Field  u(x, t)", fontsize=14, pad=12)
ax.set_xlabel("x (Position)", fontsize=12)
ax.set_ylabel("t (Time)", fontsize=12)
ax.tick_params(axis='both', labelsize=10)
ax.grid(alpha=0.3)

# Colorbar
cbar = fig.colorbar(c, ax=ax, pad=0.02)
cbar.set_label("Velocity  u(x, t)", fontsize=12)
cbar.ax.tick_params(labelsize=10)

plt.tight_layout()
plt.show()
