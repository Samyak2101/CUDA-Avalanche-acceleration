"""
gpu_profiles.py — Hardware profiles for the four target GPUs and grid-ladder helpers.
"""

# ---------------------------------------------------------------------------
# Per-GPU hardware specs
# ---------------------------------------------------------------------------

GPU_PROFILES = {
    "gtx1650": {
        "name":                 "NVIDIA GeForce GTX 1650",
        "short_name":           "GTX 1650",
        "arch":                 "Turing",
        "compute_capability":   "7.5",
        "sm_count":             14,
        "cuda_cores":           896,
        "vram_gb":              4,
        "mem_bw_gbs":           128,          # GB/s peak memory bandwidth
        "fp32_tflops":          2.98,
        "fp64_tflops":          0.047,        # 1:64 ratio
        "fp64_ratio":           "1:64",
        "shared_mem_per_sm_kb": 48,
        "default_block":        (16, 16),
        "max_grid_fp32":        (3200, 1040), # VRAM-limited
        "max_grid_fp64":        (1600, 520),
        "preferred_precision":  "float32",
        "sweep_block_sizes":    [(8, 8), (16, 8), (16, 16), (32, 8)],
    },
    "a5000": {
        "name":                 "NVIDIA RTX A5000",
        "short_name":           "A5000",
        "arch":                 "Ampere",
        "compute_capability":   "8.6",
        "sm_count":             64,
        "cuda_cores":           8192,
        "vram_gb":              24,
        "mem_bw_gbs":           768,
        "fp32_tflops":          27.8,
        "fp64_tflops":          0.43,
        "fp64_ratio":           "1:64",
        "shared_mem_per_sm_kb": 100,
        "default_block":        (16, 16),
        "max_grid_fp32":        (6400, 2080),
        "max_grid_fp64":        (4800, 1560),
        "preferred_precision":  "float32",
        "sweep_block_sizes":    [(8, 8), (16, 8), (16, 16), (32, 8), (32, 16), (32, 32)],
    },
    "rtx5070": {
        "name":                 "NVIDIA GeForce RTX 5070",
        "short_name":           "RTX 5070",
        "arch":                 "Blackwell",
        "compute_capability":   "12.0",
        "sm_count":             48,
        "cuda_cores":           6144,
        "vram_gb":              12,
        "mem_bw_gbs":           672,
        "fp32_tflops":          31.0,
        "fp64_tflops":          0.48,
        "fp64_ratio":           "1:64",
        "shared_mem_per_sm_kb": 100,
        "default_block":        (16, 16),
        "max_grid_fp32":        (4800, 1560),
        "max_grid_fp64":        (3200, 1040),
        "preferred_precision":  "float32",
        "sweep_block_sizes":    [(8, 8), (16, 8), (16, 16), (32, 8), (32, 16), (32, 32)],
    },
    "v100": {
        "name":                 "NVIDIA Tesla V100 PCIe",
        "short_name":           "V100",
        "arch":                 "Volta",
        "compute_capability":   "7.0",
        "sm_count":             80,
        "cuda_cores":           5120,
        "vram_gb":              16,
        "mem_bw_gbs":           900,
        "fp32_tflops":          14.0,
        "fp64_tflops":          7.0,
        "fp64_ratio":           "1:2",
        "shared_mem_per_sm_kb": 96,
        "default_block":        (16, 16),
        "max_grid_fp32":        (6400, 2080),
        "max_grid_fp64":        (6400, 2080),
        "preferred_precision":  "float64",
        "sweep_block_sizes":    [(8, 8), (16, 8), (16, 16), (32, 8), (32, 16), (32, 32)],
    },
}

# ---------------------------------------------------------------------------
# Standard grid ladder (each benchmark clips to its GPU's VRAM limit)
# ---------------------------------------------------------------------------

GRID_LADDER = [
    (100,  33),
    (200,  65),
    (400,  130),
    (800,  260),
    (1600, 520),
    (3200, 1040),
    (6400, 2080),
]


def get_grids_for_gpu(gpu_key, precision='float32'):
    """Return the GRID_LADDER entries that fit in the GPU's VRAM.

    Parameters
    ----------
    gpu_key   : str  — key in GPU_PROFILES (e.g. 'gtx1650')
    precision : str  — 'float32' or 'float64'

    Returns
    -------
    list of (nx, ny) tuples
    """
    profile  = GPU_PROFILES[gpu_key]
    prec_key = "fp32" if precision == 'float32' else "fp64"
    max_grid = profile[f"max_grid_{prec_key}"]
    return [(nx, ny) for nx, ny in GRID_LADDER
            if nx <= max_grid[0] and ny <= max_grid[1]]


def detect_gpu():
    """Auto-detect which GPU profile key matches the installed GPU.

    Tries CuPy first, then falls back to nvidia-smi.

    Returns
    -------
    str or None
        GPU profile key (e.g. 'gtx1650') or None if unrecognised.
    """
    name = None

    # --- Try CuPy ---
    try:
        import cupy as cp
        name = cp.cuda.runtime.getDeviceProperties(0)['name'].decode()
    except Exception:
        pass

    # --- Fall back to nvidia-smi ---
    if name is None:
        try:
            import subprocess
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                text=True, timeout=5,
            )
            name = out.strip().splitlines()[0]
        except Exception:
            pass

    if name is None:
        return None

    name_lower = name.lower()
    for key, profile in GPU_PROFILES.items():
        if key in name_lower or profile["short_name"].lower() in name_lower:
            return key
    return None
