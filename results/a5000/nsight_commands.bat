@echo off
REM Nsight Compute profiling -- A5000
REM Run this file from the d:\cuda directory as Administrator.

ncu --section SpeedOfLight --section MemoryWorkloadAnalysis --metrics sm__warps_active.avg.pct_of_peak_sustained_active,smsp__inst_executed.sum --kernel-name compute_k_grid_kernel --launch-skip 2 --launch-count 3 --target-processes all --csv --log-file "/home/teaching/avalanchesim/cuda/results/a5000/ncu_compute_k_grid_kernel.csv" --export "/home/teaching/avalanchesim/cuda/results/a5000/ncu_compute_k_grid_kernel" python "/home/teaching/avalanchesim/cuda/benchmarks/_nsight_snippet.py"

ncu --section SpeedOfLight --section MemoryWorkloadAnalysis --metrics sm__warps_active.avg.pct_of_peak_sustained_active,smsp__inst_executed.sum --kernel-name compute_wave_speeds_kernel --launch-skip 2 --launch-count 3 --target-processes all --csv --log-file "/home/teaching/avalanchesim/cuda/results/a5000/ncu_compute_wave_speeds_kernel.csv" --export "/home/teaching/avalanchesim/cuda/results/a5000/ncu_compute_wave_speeds_kernel" python "/home/teaching/avalanchesim/cuda/benchmarks/_nsight_snippet.py"

ncu --section SpeedOfLight --section MemoryWorkloadAnalysis --metrics sm__warps_active.avg.pct_of_peak_sustained_active,smsp__inst_executed.sum --kernel-name compute_fluxes_global_kernel --launch-skip 2 --launch-count 3 --target-processes all --csv --log-file "/home/teaching/avalanchesim/cuda/results/a5000/ncu_compute_fluxes_global_kernel.csv" --export "/home/teaching/avalanchesim/cuda/results/a5000/ncu_compute_fluxes_global_kernel" python "/home/teaching/avalanchesim/cuda/benchmarks/_nsight_snippet.py"

ncu --section SpeedOfLight --section MemoryWorkloadAnalysis --metrics sm__warps_active.avg.pct_of_peak_sustained_active,smsp__inst_executed.sum --kernel-name update_state_kernel --launch-skip 2 --launch-count 3 --target-processes all --csv --log-file "/home/teaching/avalanchesim/cuda/results/a5000/ncu_update_state_kernel.csv" --export "/home/teaching/avalanchesim/cuda/results/a5000/ncu_update_state_kernel" python "/home/teaching/avalanchesim/cuda/benchmarks/_nsight_snippet.py"

