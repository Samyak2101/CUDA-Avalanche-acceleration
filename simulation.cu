#include <cstdio>
#include <cmath>
#include <chrono>

#ifndef M_PI
#define M_PI 3.14159265358979323846 
#endif

// ======================================================
// CUDA KERNEL
// ======================================================
extern "C" __global__
void solve_step(double* h_new, const double* h_old, const double* x, 
                double t, double dt, double dx, double a, int nx) {
    
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= nx) return;

    double g_t = 0.11613 * t * t + 0.74497 * t + 1.0;

    // -------------------------------
    // Boundary Conditions
    // -------------------------------
    if (i == 0) {
        double term = (x[0] - 0.5 * a * t * t) / g_t;
        double res = (1.0 / g_t) * (1.0 - term * term);
        h_new[i] = (res > 0.0) ? res : 0.0;
        return;
    } 
    else if (i == nx - 1) {
        h_new[i] = h_old[i - 1];
        return;
    }

    // -------------------------------
    // Interior update
    // -------------------------------
    double x_c = x[i];
    double x_l = x[i - 1];

    // Analytical velocity
    double term_sqrt = sqrt((2.0/g_t) * (g_t - 1.0));

    double term_c = (x_c - 0.5 * a * t * t) / g_t;
    double term_l = (x_l - 0.5 * a * t * t) / g_t;

    double u_c = term_sqrt * term_c + a * t;
    double u_l = term_sqrt * term_l + a * t;

    // Clamp velocities
    u_c = (u_c > 0.0 ? u_c : 0.0);
    u_l = (u_l > 0.0 ? u_l : 0.0);

    // Fluxes, dry bed masking
    const double TOL = 1e-9;

    double flux_c = (h_old[i]   < TOL ? 0.0 : h_old[i]   * u_c);
    double flux_l = (h_old[i-1] < TOL ? 0.0 : h_old[i-1] * u_l);

    // Update
    double h_updated = h_old[i] - (dt / dx) * (flux_c - flux_l);

    // Positivity
    h_new[i] = (h_updated > 0.0 ? h_updated : 0.0);
}


// ======================================================
// MAIN PROGRAM
// ======================================================
int main() {
    // -------------------------------
    // Configuration (same as Python)
    // -------------------------------
    const double g_const = 9.81;
    const double L_left = -2.0;
    const double L_right = 10.0;
    const int nx = 100000;
    const int nt = 300000;
    const double dt = 1.0e-5;
    const double theta = M_PI / 6.0;
    const double mu = 0.1;

    double dx = (L_right - L_left) / (nx - 1);
    double a = g_const * sin(theta) - mu * g_const * cos(theta);

    printf("--- Simulation Setup ---\n");
    printf("Nodes: %d\n", nx);
    printf("Steps: %d\n", nt);
    printf("dx: %.6f\n", dx);
    printf("dt: %.6f\n", dt);
    printf("a : %.6f\n", a);

    // -------------------------------
    // Allocate host arrays
    // -------------------------------
    double* x_cpu = new double[nx];
    double* h_init = new double[nx];

    // Fill x grid
    for (int i = 0; i < nx; i++)
        x_cpu[i] = L_left + i * dx;

    // Initial condition at t=0
    double g_t0 = 0.11613 * 0.0 + 0.74497 * 0.0 + 1.0;
    for (int i = 0; i < nx; i++) {
        double term = (x_cpu[i]) / g_t0;
        double res = (1.0 / g_t0) * (1.0 - term * term);
        h_init[i] = (res > 0.0 ? res : 0.0);
    }

    // -------------------------------
    // Allocate device memory
    // -------------------------------
    double *d_x, *d_h_old, *d_h_new;
    cudaMalloc(&d_x,      nx * sizeof(double));
    cudaMalloc(&d_h_old,  nx * sizeof(double));
    cudaMalloc(&d_h_new,  nx * sizeof(double));

    cudaMemcpy(d_x,     x_cpu,  nx * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_h_old, h_init, nx * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_h_new, h_init, nx * sizeof(double), cudaMemcpyHostToDevice);

    // Kernel config
    int threads = 256;
    int blocks = (nx + threads - 1) / threads;

    printf("Running simulation...\n");

    // -------------------------------
    // Timing start
    // -------------------------------
    auto t_start = std::chrono::high_resolution_clock::now();

    // -------------------------------
    // Time stepping
    // -------------------------------
    for (int n = 1; n <= nt; n++) {
        double t = n * dt;

        solve_step<<<blocks, threads>>>(d_h_new, d_h_old, d_x, t, dt, dx, a, nx);

        cudaDeviceSynchronize();

        // Pointer swap
        double* tmp = d_h_old;
        d_h_old = d_h_new;
        d_h_new = tmp;
    }

    // -------------------------------
    // Timing end
    // -------------------------------
    auto t_end = std::chrono::high_resolution_clock::now();
    double sim_time = std::chrono::duration<double>(t_end - t_start).count();

    printf("Simulation finished.\n");
    printf("Total GPU simulation time: %.6f seconds\n", sim_time);

    // Cleanup
    cudaFree(d_x);
    cudaFree(d_h_old);
    cudaFree(d_h_new);
    delete[] x_cpu;
    delete[] h_init;

    return 0;
}
