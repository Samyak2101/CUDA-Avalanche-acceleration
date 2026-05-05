"""Run all benchmarks and validation for NVIDIA RTX A5000."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from gpu_profiles import GPU_PROFILES
from utils import ensure_results_dir

GPU_KEY = "a5000"
profile = GPU_PROFILES[GPU_KEY]
output_dir = ensure_results_dir(GPU_KEY)

print(f"Target GPU: {profile['name']}")
print(f"VRAM: {profile['vram_gb']} GB | Peak BW: {profile['mem_bw_gbs']} GB/s")
print(f"FP32: {profile['fp32_tflops']} TFLOPS | FP64: {profile['fp64_tflops']} TFLOPS ({profile['fp64_ratio']})")
print(f"Max grid (FP32): {profile['max_grid_fp32']}")
print()

# 1. Run validation first
print("=" * 60)
print("VALIDATION")
print("=" * 60)
from validation.convergence import run_convergence
from validation.titan2d_compare import run_titan2d_compare
run_convergence()
run_titan2d_compare()

# 2. Run all benchmarks
print("\n" + "=" * 60)
print("BENCHMARKS")
print("=" * 60)
from benchmarks import run_all_benchmarks
run_all_benchmarks(GPU_KEY, output_dir)

print(f"\nAll results saved to: {output_dir}")
