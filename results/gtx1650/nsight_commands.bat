@echo off
REM Nsight Compute profiling -- GTX 1650
REM Run this file from the d:\cuda directory as Administrator.

ncu --section SpeedOfLight --section MemoryWorkloadAnalysis --metrics sm__warps_active.avg.pct_of_peak_sustained_active,smsp__inst_executed.sum --kernel-name compute_k_grid_kernel --launch-skip 2 --launch-count 3 --target-processes all --csv --log-file "D:\cuda\results\gtx1650\ncu_compute_k_grid_kernel.csv" --export "D:\cuda\results\gtx1650\ncu_compute_k_grid_kernel" python "D:\cuda\benchmarks\_nsight_snippet.py"

ncu --section SpeedOfLight --section MemoryWorkloadAnalysis --metrics sm__warps_active.avg.pct_of_peak_sustained_active,smsp__inst_executed.sum --kernel-name compute_wave_speeds_kernel --launch-skip 2 --launch-count 3 --target-processes all --csv --log-file "D:\cuda\results\gtx1650\ncu_compute_wave_speeds_kernel.csv" --export "D:\cuda\results\gtx1650\ncu_compute_wave_speeds_kernel" python "D:\cuda\benchmarks\_nsight_snippet.py"

ncu --section SpeedOfLight --section MemoryWorkloadAnalysis --metrics sm__warps_active.avg.pct_of_peak_sustained_active,smsp__inst_executed.sum --kernel-name compute_fluxes_global_kernel --launch-skip 2 --launch-count 3 --target-processes all --csv --log-file "D:\cuda\results\gtx1650\ncu_compute_fluxes_global_kernel.csv" --export "D:\cuda\results\gtx1650\ncu_compute_fluxes_global_kernel" python "D:\cuda\benchmarks\_nsight_snippet.py"

ncu --section SpeedOfLight --section MemoryWorkloadAnalysis --metrics sm__warps_active.avg.pct_of_peak_sustained_active,smsp__inst_executed.sum --kernel-name update_state_kernel --launch-skip 2 --launch-count 3 --target-processes all --csv --log-file "D:\cuda\results\gtx1650\ncu_update_state_kernel.csv" --export "D:\cuda\results\gtx1650\ncu_update_state_kernel" python "D:\cuda\benchmarks\_nsight_snippet.py"

