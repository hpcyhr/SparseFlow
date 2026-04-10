"""
SparseFlow shared configuration constants.

This module centralizes runtime/dispatch thresholds used across Core, Ops,
Kernels, and Benchmark code paths to reduce parameter drift.
"""

from __future__ import annotations

# -----------------------------------------------------------------------------
# Numeric epsilons
# -----------------------------------------------------------------------------

# Prescan activation threshold used by sparse kernels.
PRESCAN_ACTIVITY_EPS = 1e-6

# Runtime fallback comparison epsilon for speedup/ratio divisions.
NUMERIC_EPS = 1e-6

# Static-zero detection epsilon defaults by spike input mode.
STATICZERO_EPS_BY_SPIKE_MODE = {
    "normalized_bernoulli": 1e-7,
    "raw_bernoulli": 1e-8,
    "raw_repeat": 1e-8,
}
STATICZERO_EPS_DEFAULT = 1e-6


def staticzero_eps_for_mode(spike_mode: str) -> float:
    return float(STATICZERO_EPS_BY_SPIKE_MODE.get(str(spike_mode), STATICZERO_EPS_DEFAULT))


# -----------------------------------------------------------------------------
# Sparse runtime policy thresholds
# -----------------------------------------------------------------------------

# Shared dense-fallback ratio threshold for sparse kernels and wrappers.
SPARSE_DENSE_RATIO_THRESHOLD = 0.85


# -----------------------------------------------------------------------------
# Dispatch model thresholds (execution-grounded, empirically calibrated)
# -----------------------------------------------------------------------------

DISPATCH_TAU_3X3 = 100_000
DISPATCH_TAU_1X1 = 200_000
DISPATCH_TAU_LINEAR = 800_000
DISPATCH_TAU_ATTN_LINEAR = 900_000
DISPATCH_TAU_ATTN_MATMUL = 1_200_000

DISPATCH_R_MIN = 0.15
DISPATCH_R_MARGIN = 0.0
DISPATCH_TAU_MARGIN = 0.0

# Confidence gate for estimated/fallback metadata decisions.
DISPATCH_MIN_CONFIDENCE = 0.60


# -----------------------------------------------------------------------------
# Benchmark reporting defaults
# -----------------------------------------------------------------------------

# Energy is a latency-derived proxy unless externally measured power is sampled.
REPORT_ENERGY_PROXY_DEFAULT = False

# Unified benchmark banner/version tag.
BENCH_VERSION = "v29-observability"
