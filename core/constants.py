"""
core/constants.py
Application-wide constants and configuration values.
"""
from typing import List
# ============================================================================
# UI OPTIONS
# ============================================================================
POLARITY_OPTIONS: List[str] = [
    'BOTH',
    'CD ON (polarity=1)',
    'CD OFF (polarity=0)',
    'SIGNED (ON - OFF)'
]
BIAS_NAMES: List[str] = [
    'bias_diff',
    'bias_diff_off',
    'bias_diff_on',
    'bias_fo',
    'bias_hpf',
    'bias_refr'
]
# ============================================================================
# DISPLAY LIMITS
# ============================================================================
MAX_TIMETRACE_POINTS: int = 10000
MAX_IEI_POINTS: int = 10000
MAX_DISPLAY_FRAMES: int = 1000
RECONNECT_TIMEOUT: int = 120
# ============================================================================
# ANALYSIS PARAMETERS
# ============================================================================
IEI_HISTOGRAM_NBINS: int = 100
POWER_SPECTRUM_BIN_WIDTH_US: int = 100
MIN_FREQUENCY_HZ: float = 0.1
MAX_FREQUENCY_HZ: float = 2000
# ============================================================================
# VISUALIZATION PARAMETERS
# ============================================================================
TIMETRACE_JITTER: float = 0.5
TIMETRACE_MARGIN_RATIO: float = 0.01