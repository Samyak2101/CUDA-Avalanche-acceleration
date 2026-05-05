# -*- coding: utf-8 -*-
"""Minimal solver run for Nsight Compute profiling -- auto-generated."""
import sys
sys.path.insert(0, r'd:\cuda')
from solver.gpu_global import run_gpu_global
# 10 timesteps is enough for ncu to capture individual kernel launches
run_gpu_global(800, 260, [0.01] * 10, CFL=0.3, precision='float32')
