<div align="center">

# 🏔️ CUDA Avalanche Solver

### GPU-Accelerated 2D Savage-Hutter Granular Flow Simulation

[![CUDA](https://img.shields.io/badge/CUDA-CuPy%20RawKernels-76B900?style=for-the-badge&logo=nvidia&logoColor=white)](https://cupy.dev/)
[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![License](https://img.shields.io/badge/License-Academic-orange?style=for-the-badge)]()
[![Platform](https://img.shields.io/badge/Platform-Windows%20%7C%20Linux-blue?style=for-the-badge)]()

*A high-performance finite volume solver for simulating granular avalanches on inclined surfaces, leveraging NVIDIA GPU parallelism through hand-written CUDA C++ kernels compiled JIT via CuPy.*

---

</div>

## 📖 Overview

This project implements a **2D Savage-Hutter shallow-water model** for granular avalanche flow simulation on an inclined plane with Coulomb basal friction. The numerical scheme employs the **Rusanov (Local Lax-Friedrichs) finite volume method** with forward Euler time integration.

The solver is designed as an HPC benchmarking suite that comprehensively evaluates GPU acceleration strategies across multiple NVIDIA architectures — from consumer-grade Turing GPUs to data-center Volta cards.

### Key Highlights

- 🚀 **5 hand-written CUDA C++ kernels** compiled JIT via CuPy `RawKernels`
- 📐 **Structure-of-Arrays (SoA)** memory layout for coalesced GPU access
- 🧊 **Shared memory tiling** for optimized flux computation
- 🔬 **Validated against TITAN2D** — an established geophysical flow simulator
- 📊 **Full benchmarking suite** with roofline analysis, scaling studies, and Nsight Compute integration
- 🖥️ **Multi-GPU profiled** across 4 architectures (GTX 1650, A5000, RTX 5070, V100)

---

## 🧮 Physics & Numerical Method

The solver models a hemispherical granular pile released on a 37.4° inclined plane, governed by:

| Parameter | Symbol | Value |
|---|---|---|
| Slope angle | ζ | 37.4° |
| Basal friction angle | δ | 27.0° |
| Internal friction angle | φ | 37.3° |
| Gravity | g | 9.81 m/s² |
| Dry-bed threshold | DRY | 10⁻⁴ m |
| Domain | L×W | 4.0 m × 1.3 m |

The conserved variables are **depth** (*h*), **x-momentum** (*hu*), and **y-momentum** (*hv*). Earth pressure coefficients (*k_act*, *k_pass*) switch dynamically between active and passive states based on the local velocity divergence — a hallmark of the Savage-Hutter model.

---

## 🏗️ Architecture

```
Python Control Plane (NumPy, CuPy, Matplotlib)
    │
    ├── config.py          ← Physics constants & initial conditions
    ├── gpu_profiles.py    ← Hardware specs for 4 GPU architectures
    ├── utils.py           ← Timers, I/O, plotting utilities
    │
    ├── solver/
    │   ├── kernels.py     ← 5 CUDA C++ kernel templates (JIT compiled)
    │   ├── gpu_global.py  ← Primary solver: SoA + global memory
    │   ├── gpu_shared.py  ← Advanced solver: SoA + shared memory tiling
    │   ├── gpu_aos.py     ← Comparison solver: AoS layout (non-coalesced)
    │   └── cpu_numba.py   ← CPU baseline: Numba JIT parallel
    │
    ├── validation/
    │   ├── titan2d_compare.py  ← Centerline profiles vs TITAN2D
    │   ├── convergence.py      ← Spatial/temporal convergence
    │   └── giid.py             ← Grid-independence verification
    │
    ├── benchmarks/
    │   ├── strong_scaling.py   ← Fixed-size speedup analysis
    │   ├── weak_scaling.py     ← Scaled-size efficiency (MCUPS)
    │   ├── roofline.py         ← Roofline model analysis
    │   ├── block_size_sweep.py ← Thread block optimization
    │   ├── data_layout.py      ← SoA vs AoS comparison
    │   ├── precision_study.py  ← FP32 vs FP64 trade-offs
    │   ├── shared_mem_impact.py← Shared vs global memory
    │   └── nsight_profile.py   ← Nsight Compute automation
    │
    └── run_*.py               ← Per-GPU & cross-GPU runner scripts
```

### CUDA Kernel Pipeline

Each time step executes **5 sequential kernels**:

```
┌─────────────────┐    ┌─────────────────────┐    ┌─────────────────────┐
│ 1. compute_k    │───▶│ 2. wave_speeds      │───▶│ 3. compute_fluxes   │
│    (active/     │    │    (CFL adaptive dt) │    │    (Rusanov solver)  │
│     passive)    │    │                      │    │    [global | shared] │
└─────────────────┘    └─────────────────────┘    └──────────┬──────────┘
                                                              │
                                                              ▼
                                                   ┌─────────────────────┐
                                                   │ 4. update_state     │
                                                   │    (Euler + source  │
                                                   │     + positivity)   │
                                                   └─────────────────────┘
```

Precision is controlled at compile time via C++ `typedef` injection — the same kernel source handles both `float` and `double` arithmetic without code duplication.

---

## 🖥️ Supported Hardware

| GPU | Architecture | VRAM | Memory BW | FP32 | FP64 Ratio |
|---|---|---|---|---|---|
| **GTX 1650** | Turing | 4 GB | 128 GB/s | 2.98 TFLOPS | 1:64 |
| **RTX A5000** | Ampere | 24 GB | 768 GB/s | 27.8 TFLOPS | 1:64 |
| **RTX 5070** | Blackwell | 12 GB | 672 GB/s | 31.0 TFLOPS | 1:64 |
| **Tesla V100** | Volta | 16 GB | 900 GB/s | 14.0 TFLOPS | 1:2 |

Grid resolutions are automatically scaled per GPU based on available VRAM. The V100 is the only card with full-rate FP64, making it the preferred target for double-precision workloads.

---

## ⚡ Quick Start

### Prerequisites

- **Python** ≥ 3.8
- **NVIDIA GPU** with CUDA toolkit installed
- **CuPy** (matching your CUDA version)
- **Numba** (for CPU baseline)
- **NumPy**, **Matplotlib**

### Installation

```bash
# Clone the repository
git clone https://github.com/<username>/cuda-avalanche-solver.git
cd cuda-avalanche-solver

# Install dependencies
pip install cupy-cuda12x numpy numba matplotlib
```

> [!NOTE]
> Replace `cupy-cuda12x` with the package matching your CUDA toolkit version (e.g., `cupy-cuda11x` for CUDA 11).

### Running the Solver

```bash
# Run the full benchmark suite on your detected GPU
python run_gtx1650.py     # For GTX 1650
python run_a5000.py       # For RTX A5000
python run_rtx5070.py     # For RTX 5070
python run_v100.py        # For Tesla V100

# Run TITAN2D validation
python -m validation.titan2d_compare

# Generate cross-GPU comparison plots (requires results from ≥2 GPUs)
python run_cross_gpu.py
```

### Running Individual Benchmarks

```python
from solver import run_gpu_global

# Run at 800×260 grid, snapshot at t = 0, 1, 2 seconds
history, x, y, wall_time, steps, perf = run_gpu_global(
    nx=800, ny=260,
    target_times=[0.0, 1.0, 2.0],
    precision='float32',
    block_size=(16, 16),
)

print(f"Completed in {wall_time:.2f}s over {steps} timesteps")
```

---

## 📊 Benchmark Suite

The benchmarking suite provides eight comprehensive analyses:

| Benchmark | Description | Output |
|---|---|---|
| **Strong Scaling** | GPU speedup over Numba CPU at fixed grid sizes | `strong_scaling.csv/pdf` |
| **Weak Scaling** | Throughput (MCUPS) as grid size increases | `weak_scaling.csv/pdf` |
| **Roofline Model** | Arithmetic intensity vs hardware ceilings | `roofline.csv/pdf` |
| **Block Size Sweep** | SM occupancy optimization across block configs | `block_size_sweep.csv/pdf` |
| **Data Layout** | SoA vs AoS coalesced memory access comparison | `data_layout.csv/pdf` |
| **Precision Study** | FP32 vs FP64 performance and accuracy trade-offs | `precision_study.csv/pdf` |
| **Shared Memory** | Global vs shared memory tiling impact | `shared_mem_impact.csv/pdf` |
| **Nsight Compute** | Per-kernel L1/L2/DRAM throughput profiling | `ncu_*.csv/ncu-rep` |

All results are saved to `results/<gpu_key>/` as both CSV data and publication-quality PDF plots.

---

## ✅ Validation

Physical accuracy is verified through three independent methods:

1. **TITAN2D Comparison** — Centerline height profiles at *t* = 0.0, 0.5, 1.0, 1.5, 2.0 s are compared against the established TITAN2D geophysical flow simulator using reference data from `titan_height_cells_0to2s.npz`.

2. **Spatial Convergence** — Grid refinement studies confirm the expected first-order convergence rate of the Rusanov scheme.

3. **Grid Independence (GIID)** — Self-convergence under successive mesh doublings verifies that solutions stabilize at higher resolutions.

4. **Mass Conservation** — Total fluid volume is tracked every 50 timesteps to confirm negligible numerical mass drift.

---

## 🔑 Key Design Decisions

| Decision | Rationale |
|---|---|
| **Python + CuPy RawKernels** | Maximum GPU performance with zero C++/CMake boilerplate; Python handles orchestration, visualization, and data wrangling |
| **SoA over AoS** | Coalesced global memory access is critical — the solver is memory-bandwidth bound (confirmed by roofline analysis) |
| **FP32 default** | Consumer GPUs penalize FP64 at 1:64 ratio; FP32 provides sufficient accuracy for this simulation |
| **Dynamic `typedef` injection** | Single CUDA source handles both precisions via f-string templating |
| **Shared memory tiling** | 17×17 `__shared__` buffers with halo padding eliminate redundant global memory fetches for neighbor data |

---

## 📁 Project Structure

```
d:\cuda\
├── config.py               # Physics constants, initial conditions, precision helper
├── gpu_profiles.py          # Hardware specs for 4 GPU architectures
├── utils.py                 # Timers, binning, I/O, matplotlib config
│
├── solver/                  # Numerical solver implementations
│   ├── __init__.py          # Exports: run_cpu_numba, run_gpu_global, run_gpu_shared, run_gpu_aos
│   ├── kernels.py           # 5 CUDA C++ kernel templates (JIT via CuPy RawModule)
│   ├── gpu_global.py        # Primary GPU solver (SoA + global memory)
│   ├── gpu_shared.py        # GPU solver with shared memory tiling (16×16 blocks)
│   ├── gpu_aos.py           # GPU solver with AoS layout (for comparison)
│   └── cpu_numba.py         # CPU baseline solver (Numba JIT, parallel)
│
├── validation/              # Physical accuracy verification
│   ├── titan2d_compare.py   # Centerline profiles vs TITAN2D reference
│   ├── convergence.py       # Spatial/temporal convergence analysis
│   └── giid.py              # Grid-independence study
│
├── benchmarks/              # HPC performance analysis suite
│   ├── __init__.py          # run_all_benchmarks() orchestrator
│   ├── strong_scaling.py    # CPU vs GPU speedup at fixed sizes
│   ├── weak_scaling.py      # Throughput scaling (MCUPS)
│   ├── roofline.py          # Roofline model construction
│   ├── block_size_sweep.py  # Thread block size optimization
│   ├── data_layout.py       # SoA vs AoS comparison
│   ├── precision_study.py   # FP32 vs FP64 analysis
│   ├── shared_mem_impact.py # Shared vs global memory comparison
│   └── nsight_profile.py    # Nsight Compute CLI automation
│
├── run_gtx1650.py           # Runner: GTX 1650 (4 GB VRAM)
├── run_a5000.py             # Runner: RTX A5000 (24 GB VRAM)
├── run_rtx5070.py           # Runner: RTX 5070 (12 GB VRAM)
├── run_v100.py              # Runner: Tesla V100 (16 GB VRAM)
├── run_cross_gpu.py         # Cross-architecture comparison plots
│
├── results/                 # Benchmark outputs (CSV + PDF per GPU)
│   ├── gtx1650/
│   ├── a5000/
│   ├── cross_gpu/
│   └── validation/
│
├── docs/                    # Architecture & pipeline documentation
│   └── project_pipeline.md
│
├── titan_height_cells_0to2s.npz   # TITAN2D reference data
└── main2.tex                       # LaTeX research manuscript
```

---

## 📝 Citation

If you use this solver in your research, please cite the accompanying manuscript:

```bibtex
@article{savage_hutter_cuda_2026,
  title   = {GPU-Accelerated Savage-Hutter Avalanche Simulation: 
             An Empirical Benchmarking Study Across NVIDIA Architectures},
  year    = {2026},
  note    = {Manuscript in preparation}
}
```

---

## 📄 License

This project is developed for academic research purposes. Please contact the authors for licensing inquiries.

---

<div align="center">
<sub>Built with CUDA • CuPy • NumPy • Numba • Matplotlib</sub>
</div>
