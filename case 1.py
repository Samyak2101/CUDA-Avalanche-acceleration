import numpy as np 
import matplotlib.pyplot as plt
import time

# Constants
g_const=9.81
L=10 #length of the slope
H = 1.0   # choose reference height scale (can be 1 since your solution is normalized)
nx=341 #number of spatial points
dx=L/(nx-1) #space step
dt=0.005 #time step
theta=np.pi/6 #angle of the slope
nt=200 #number of time steps
mu=0.1 #friction coefficient
a = np.sin(theta) - mu * np.cos(theta)   #acceleration
epsilon = H / L
k = 1.0   # assume active stress for now
beta = epsilon * k * np.cos(theta)
T = np.sqrt(L / g_const)
U = np.sqrt(g_const * L)

def g(t):
    return 0.11613 * t**2 + 0.74497 * t + 1.0

#analytical solution
def h_analytical(x, t):
    g_t = g(t)
    term = (x - 0.5*a*t**2) / g_t
    result = (1/g_t) * (1 - term**2)
    #return result
    return np.maximum(result, 0)  # Prevent negative heights

def u_analytical(x,t):
    g_t=g(t)
    return np.sqrt((2/g_t)*(g_t-1))*(x-0.5*a*t**2)/g_t+a*t

h=np.zeros(int(nx)) #height array
u=np.zeros(int(nx)) #velocity array

x_dim = np.linspace(0, L, nx)
x=x_dim/L #dimesionless
h = h_analytical(x, 0)
print (h)

#storage for plotting at different times
h_history=[h.copy()]
u_history=[u.copy()]
times=[0]

#record start time
start_time=time.time()

dt_star = dt / T
dx_star = dx / L

#time loop
for n in range(nt):
    t_dim=n*dt
    t=t_dim/T
    u_temp=u.copy()
    h_temp=h.copy()
    for i in range(1,nx-1):
        h[i]= h_temp[i]-dt_star/dx_star*(h_temp[i]*u_analytical(x[i],t)-h_temp[i-1]*u_analytical(x[i-1],t))
        
    # Boundary conditions
    h[0] = 0   # inflow Dirichlet
    h[-1]=0                 # outflow Neumann
    #h[-1] = h_analytical(L, t)

    if n%50==0:
        h_history.append(h.copy())
        u_history.append(u.copy())
        times.append(t)

#record end time
end_time=time.time()
print(f"Simulation completed in {end_time-start_time:.2f} seconds.")
#plotting
for i,t in enumerate(times):    
    plt.figure(1)
    plt.clf()
    plt.plot(x,h_history[i],'b-',label='Numerical')
    plt.plot(x,h_analytical(x,t),'r--',label='Analytical')
    plt.xlabel('Distance (m)')
    plt.ylabel('Height (m)')
    plt.title(f'Height Profile at t = {t:.4f} s', fontsize=14)
    plt.legend()
    plt.grid()
    plt.pause(0.1)
plt.show()
