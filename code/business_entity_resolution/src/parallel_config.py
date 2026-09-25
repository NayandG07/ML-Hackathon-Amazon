"""
parallel_config.py
==================
Central configuration for parallel processing across the pipeline.

i5-13450HX specs:
  - 6 P-cores (12 threads) + 4 E-cores (4 threads) = 16 logical CPUs
  - 10 physical cores
  - 24 GB DDR5 RAM

Strategy per stage:
  - preprocessing  : loky backend (multi-process)  — pure Python regex, no shared state
  - blocking TF-IDF: threading backend              — scipy sparse releases the GIL
  - feature eng.   : threading backend              — rapidfuzz C++ releases the GIL
  - inference      : threading backend              — rapidfuzz C++ releases the GIL
  - LightGBM       : GPU + n_jobs=-1               — auto-detected
"""

import os
import psutil

# ── Core counts ──────────────────────────────────────────────────────────────
PHYSICAL_CORES = psutil.cpu_count(logical=False) or 6   # 10 for i5-13450HX
LOGICAL_CPUS   = psutil.cpu_count(logical=True)  or 16  # 16 for i5-13450HX

# ── Worker counts ─────────────────────────────────────────────────────────────
# EMPIRICALLY TUNED on this machine (i5-13450HX, 16 logical CPUs):
#
# Preprocessing benchmark results (loky, 300K rows, chunk=10K):
#   workers=8  →  70,835 rows/s  (4.0x)
#   workers=10 →  61,784 rows/s  (3.5x)   ← dips: HT contention on P-cores
#   workers=12 →  80,045 rows/s  (4.6x)   ← E-cores kick in fully
#   workers=14 →  79,727 rows/s  (4.5x)
#   workers=16 →  92,775 rows/s  (5.3x)   ← PEAK: all 16 logical CPUs busy
#
# Conclusion: use all 16 for loky (processes don't fight over the GIL).
# Threading (TF-IDF, features): same — more threads = more parallel GIL releases.
N_PROCESS_WORKERS  = LOGICAL_CPUS       # 16 ← empirically optimal for loky
N_THREAD_WORKERS   = LOGICAL_CPUS       # 16 ← for threading (scipy/rapidfuzz)
N_IO_WORKERS       = PHYSICAL_CORES     # 10 ← I/O bound tasks


# ── Chunk sizes ───────────────────────────────────────────────────────────────
# Small enough that each worker always has work to do.
# Rule of thumb: total_rows / chunk_size >= 4 × N_workers (enough queue depth)
CHUNK_PREPROCESS   = 12_000    # rows/chunk for normalisation (loky; 14 workers × 12K = 168K min queue)
CHUNK_FEATURES     = 10_000    # pairs/chunk for fuzzy features (threading)
CHUNK_INFERENCE    = 10_000    # candidate rows/chunk for inference (threading)
CHUNK_BLOCKING     = 50_000    # S1 rows/batch for TF-IDF queries (threading)


# ── Memory guard ─────────────────────────────────────────────────────────────
# If available RAM drops below this, reduce parallelism automatically
RAM_FLOOR_GB       = 4.0

def safe_n_workers(requested: int, backend: str = "loky") -> int:
    """
    Reduce worker count if available RAM is critically low.
    """
    vm = psutil.virtual_memory()
    avail_gb = vm.available / 1e9
    if avail_gb < RAM_FLOOR_GB:
        safe = max(1, requested // 2)
        print(f"[parallel_config] Low RAM ({avail_gb:.1f} GB) — reducing workers {requested}→{safe}")
        return safe
    return requested


# ── Quick diagnostics ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    vm = psutil.virtual_memory()
    print(f"Physical cores : {PHYSICAL_CORES}")
    print(f"Logical CPUs   : {LOGICAL_CPUS}")
    print(f"Process workers: {N_PROCESS_WORKERS}  (loky backend)")
    print(f"Thread workers : {N_THREAD_WORKERS}   (threading backend)")
    print(f"Available RAM  : {vm.available/1e9:.1f} / {vm.total/1e9:.1f} GB")
    print(f"Chunk (preproc): {CHUNK_PREPROCESS:,} rows")
    print(f"Chunk (features): {CHUNK_FEATURES:,} pairs")
