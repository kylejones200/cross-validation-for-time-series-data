#!/usr/bin/env python3
"""Python vs Rust kernel benchmark."""

from __future__ import annotations

import time
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from compute_kernel import rolling_window_cv_bounds  # noqa: E402

def main() -> None:
    n, w, st = 5000, 500, 10
    t0 = time.perf_counter()
    for _ in range(200):
        rolling_window_cv_bounds(n, w, st)
    py_s = time.perf_counter() - t0
    try:
        import cross_validation_for_time_series_data_rs as rs
    except ImportError:
        print("Build: maturin develop --release -m rust/py/Cargo.toml")
        print(f"Python {py_s:.3f}s")
        return
    rs_s = rs.bench_kernel_py(n, w, st, 5000)
    print(f"Python {py_s:.3f}s Rust {rs_s:.3f}s speedup {py_s / max(rs_s, 1e-9):.1f}x")
    np.testing.assert_allclose(
        rolling_window_cv_bounds(n, w, st),
        np.asarray(rs.rolling_window_cv_bounds_py(n, w, st)),
    )
    print("Correctness: OK")

if __name__ == "__main__":
    main()
