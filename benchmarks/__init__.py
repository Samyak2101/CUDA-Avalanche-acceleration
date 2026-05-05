"""Benchmark suite for CUDA avalanche solver HPC analysis."""

def run_all_benchmarks(gpu_key, output_dir=None):
    """Run all benchmarks for a given GPU profile.

    Args:
        gpu_key: one of 'gtx1650', 'a5000', 'rtx5070', 'v100'
        output_dir: results directory (default: results/<gpu_key>/)
    """
    from gpu_profiles import GPU_PROFILES
    from utils import ensure_results_dir

    profile = GPU_PROFILES[gpu_key]
    if output_dir is None:
        output_dir = ensure_results_dir(gpu_key)

    print(f"\n{'='*60}")
    print(f"Running all benchmarks for {profile['name']}")
    print(f"{'='*60}\n")

    from .strong_scaling import run_strong_scaling
    from .weak_scaling import run_weak_scaling
    from .roofline import run_roofline
    from .nsight_profile import run_nsight_profile
    from .data_layout import run_data_layout
    from .precision_study import run_precision_study
    from .block_size_sweep import run_block_size_sweep
    from .shared_mem_impact import run_shared_mem_impact

    run_strong_scaling(gpu_key, output_dir)
    run_weak_scaling(gpu_key, output_dir)
    run_roofline(gpu_key, output_dir)
    run_nsight_profile(gpu_key, output_dir)
    run_data_layout(gpu_key, output_dir)
    run_precision_study(gpu_key, output_dir)
    run_block_size_sweep(gpu_key, output_dir)
    run_shared_mem_impact(gpu_key, output_dir)
