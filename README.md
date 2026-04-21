# High-Performance Avalanche Dynamics Solver

A computational physics suite designed to simulate granular avalanche flows and shallow-water equations. This repository features both a 1D Lagrangian approach and a highly optimized 2D Eulerian solver utilizing the Rusanov flux scheme. The 2D solver is fully parallelized using CUDA and CuPy, designed specifically for High-Performance Computing (HPC) analysis across diverse NVIDIA GPU architectures.

## Features

* **1D Lagrangian Solver:** Tracks individual fluid/granular parcels over time, ideal for fundamental conservation verification and rapid prototyping.
* **2D Eulerian Solver (Dynamic-k Rusanov):** Implements a first-order Rusanov scheme on a fixed spatial grid. Features a dynamic Earth pressure coefficient (k_act and k_pass) based on the local velocity divergence field.
* **CUDA/GPU Acceleration:** * Replaces standard nested loops with custom C++ CUDA kernels (`cupy.RawKernel`).
  * Utilizes a Structure of Arrays (SoA) memory layout and Single Precision (FP32) math to maximize memory bandwidth and compute throughput.
  * Implements highly optimized global reductions for CFL time-stepping.
* **HPC Profiling & Analytics:** Built-in tracking for Mega-Cell-Updates per Second (MCUPS) and Effective Memory Bandwidth (GB/s). Designed to be profiled via NVIDIA Nsight Compute.
* **TITAN2D Validation:** Integrated data loaders and plotting functions to validate centerline profiles against TITAN2D benchmark data.

## Tech Stack

* **Language:** Python, C++ (CUDA Kernels)
* **Libraries:** CuPy, NumPy, Matplotlib, Numba (Legacy CPU support)
* **Hardware Target:** NVIDIA GPUs (Turing, Ada, Ampere, Volta)
