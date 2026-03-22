import numpy as np 
import matplotlib.pyplot as plt
import time

# Constants
g_const=9.81
L_left = -2
L_right=30 #right boundary diatance from zero
nx= 641 #number of spatial points
dx=(L_right - L_left) /(nx-1) #space step
dt=0.0007 #time step
theta=np.pi/6 #angle of the slope
nt=3840 #number of time steps
mu=0.1 #friction coefficient
a=g_const*np.sin(theta)-mu*g_const*np.cos(theta) #acceleration
beta=g_const*np.cos(theta)


def g(t):
    return 0.11613 * t**2 + 0.74497 * t + 1.0

#analytical solution
def h_analytical(x, t):
    g_t = g(t)
    term = (x - a*t**2*0.5) / g_t
    result = (1/g_t) * (1 - term**2)
    #return result
    return np.maximum(result, 0)  # Prevent negative heights

def u_analytical(x,t):
    g_t=g(t)
    return np.sqrt((2/g_t)*(g_t-1))*(x-0.5*a*t**2)/g_t+a*t

h=np.zeros(int(nx)) #height array
u=np.zeros(int(nx)) #velocity array
x = np.linspace(L_left, L_right, nx)
z = np.linspace(0, 5, nx)  # Time array for analytical solution
u = u_analytical(x, 0)
print (u)

#storage for plotting at different times
h_history=[h.copy()]
u_history=[u.copy()]
times=[0]

#record start time
start_time=time.time()

#time loop
for n in range(nt):
    t=n*dt
    u_temp=u.copy()
    h_temp=h.copy()
    
    #boundary conditions
    u[0]=u_analytical(x[0], t)
    #u[0]=u[1]
    u[-1]=u[-2]


    for i in range(1,nx-1):
        if u_temp[i]>=0:
            u[i] = u_temp[i] - u_temp[i] * dt / dx * (u_temp[i] - u_temp[i-1]) + a * dt - beta * dt / dx  * (h_analytical(x[i], t) - h_analytical(x[i-1], t))
            
        else:
            u[i] = u_temp[i] - u_temp[i] * dt / dx * (u_temp[i+1] - u_temp[i]) + a * dt - beta * dt / dx * (h_analytical(x[i+1], t) - h_analytical(x[i], t))      

    
    if n%400==0:
        h_history.append(h.copy())
        u_history.append(u.copy())
        times.append(t)

#plotting
for i, t in enumerate(times):
    plt.figure(1)
    plt.clf()
    plt.plot(x, u_history[i], 'b-', label='Numerical')
    plt.plot(x, u_analytical(x, t), 'r--', label='Analytical')
    plt.xlabel('Distance along slope (m)')
    plt.ylabel('Velocity (m/s)')
    plt.title(f'Velocity Profile at t = {t:.4f} s', fontsize=14)
    plt.ylim(-2, 20)
    plt.xlim(L_left, L_right)
    plt.legend()
    plt.grid()
    plt.pause(0.1)
plt.show()
