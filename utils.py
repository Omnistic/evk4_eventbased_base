"""
utils.py
Utility functions for event-based camera data processing.
Provides core functionality for working with event camera data from Prophesee EVK4
sensors, including file format conversion, event filtering, histogram computation,
frame generation, and Plotly figure creation.
Event Data Format:
    Events are stored as structured NumPy arrays with fields:
        - 't': timestamp in microseconds (uint64)
        - 'x': horizontal pixel coordinate (uint16)
        - 'y': vertical pixel coordinate (uint16)
        - 'p': polarity (0=OFF, 1=ON) (uint8)
"""
import warnings
import numpy as np
import numpy.typing as npt
import plotly.graph_objects as go
from typing import Tuple, Optional
from pathlib import Path
from tqdm import tqdm
from metavision_sdk_stream import Camera, CameraStreamSlicer
# ============================================================================
# FILE I/O
# ============================================================================
def raw_to_npz(file_path: Path, overwrite: bool = False) -> None:
    """
    Convert .raw event file to compressed .npz format.
    Reads event data using Prophesee SDK and saves to NumPy compressed format
    for faster loading. Also extracts and saves sensor dimensions and bias
    settings if available.
    Args:
        file_path: Path to input .raw file
        overwrite: If True, overwrite existing .npz file; if False, skip conversion
                   if .npz already exists
    Raises:
        FileNotFoundError: If input file does not exist
        ValueError: If input file is not a .raw file
    Note:
        Output .npz file is created in the same directory as input with .npz extension.
        Bias settings are read from companion .bias file if present.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() != '.raw':
        raise ValueError(f'Expected .raw file, got: {path.suffix}')
    npz_path = path.with_suffix('.npz')
    if npz_path.exists() and not overwrite:
        warnings.warn(f'{npz_path} already exists, skipping (use overwrite=True to replace)')
        return
    camera = Camera.from_file(str(path))
    slicer = CameraStreamSlicer(camera.move())
    width = slicer.camera().width()
    height = slicer.camera().height()
    chunks = []
    for slice in tqdm(slicer, desc='Converting'):
        if slice.events.size > 0:
            chunks.append(slice.events.copy())
    all_events = np.concatenate(chunks)
    bias_path = path.with_suffix('.bias')
    biases = {}
    if bias_path.exists():
        with open(bias_path, 'r') as f:
            for line in f:
                parts = line.strip().split('%')
                if len(parts) == 2:
                    value = int(parts[0].strip())
                    name = parts[1].strip()
                    biases[name] = value
    np.savez(npz_path, events=all_events, width=width, height=height, **biases)
# ============================================================================
# POLARITY MODE CONVERSION
# ============================================================================
def get_polarity_mode_from_string(polarity_string: str) -> str:
    """
    Convert UI polarity selection string to internal mode identifier.
    Args:
        polarity_string: One of 'BOTH', 'CD ON (polarity=1)', 'CD OFF (polarity=0)',
                         or 'SIGNED (ON - OFF)'
    Returns:
        Mode string: 'all', 'on', 'off', or 'signed'
    """
    if polarity_string == 'CD ON (polarity=1)':
        return 'on'
    elif polarity_string == 'CD OFF (polarity=0)':
        return 'off'
    elif polarity_string == 'SIGNED (ON - OFF)':
        return 'signed'
    return 'all'
# ============================================================================
# EVENT FILTERING
# ============================================================================
def filter_events_by_polarity(
    events: npt.NDArray[np.void],
    mode: str
) -> npt.NDArray[np.void]:
    """
    Filter events based on polarity mode.
    Args:
        events: Event array with 'p' field containing polarity values
        mode: 'on' (p=1 only), 'off' (p=0 only), 'all' or 'signed' (unchanged)
    Returns:
        Filtered event array. For 'all' and 'signed' modes, returns input unchanged.
    """
    if mode == 'on':
        return events[events['p'] == 1]
    elif mode == 'off':
        return events[events['p'] == 0]
    return events
def filter_events_by_roi(
    events: npt.NDArray[np.void],
    roi: Optional[Tuple[int, int, int, int]]
) -> npt.NDArray[np.void]:
    """
    Filter events to those within region of interest (ROI) bounds.
    Args:
        events: Event array with 'x' and 'y' coordinate fields
        roi: ROI bounds as (x_min, x_max, y_min, y_max), inclusive on both ends,
             or None for no filtering.
    Returns:
        Filtered event array. If roi is None, returns input unchanged.
    """
    if roi is None:
        return events
    x_min, x_max, y_min, y_max = roi
    mask = (
        (events['x'] >= x_min) & (events['x'] <= x_max) &
        (events['y'] >= y_min) & (events['y'] <= y_max)
    )
    return events[mask]
def filter_events_by_time_range(
    events: npt.NDArray[np.void],
    time_range: Optional[Tuple[float, float]]
) -> npt.NDArray[np.void]:
    """
    Filter events to those within a time range.
    Args:
        events: Event array with 't' timestamp field in microseconds
        time_range: Time range as (t_min_s, t_max_s) in seconds, or None for no filtering.
    Returns:
        Filtered event array. If time_range is None, returns input unchanged.
    """
    if time_range is None:
        return events
    t_min_us = int(time_range[0] * 1e6)
    t_max_us = int(time_range[1] * 1e6)
    mask = (events['t'] >= t_min_us) & (events['t'] <= t_max_us)
    return events[mask]
# ============================================================================
# HISTOGRAM COMPUTATION
# ============================================================================
def compute_event_histogram(
    events: npt.NDArray[np.void],
    width: int,
    height: int,
    mode: str = 'all'
) -> npt.NDArray[np.int32] | npt.NDArray[np.uint32]:
    """
    Compute 2D histogram of event locations (spatial event density).
    Counts events at each pixel location. For signed mode, computes the difference
    between ON and OFF event counts. Uses np.bincount on flattened pixel coordinates
    for fast accumulation.
    Args:
        events: Event array with 'x', 'y', and 'p' fields
        width: Sensor width in pixels
        height: Sensor height in pixels
        mode: 'all', 'on', 'off', or 'signed' (ON minus OFF)
    Returns:
        2D array of shape (height, width). int32 for signed mode, uint32 otherwise.
    """
    width = int(width)
    height = int(height)
    n_pixels = height * width
    if mode == 'signed':
        on_events = events[events['p'] == 1]
        off_events = events[events['p'] == 0]
        histogram = np.zeros(n_pixels, dtype=np.int32)
        on_coords = on_events['y'].astype(np.int32) * width + on_events['x'].astype(np.int32)
        histogram += np.bincount(on_coords, minlength=n_pixels).astype(np.int32)
        off_coords = off_events['y'].astype(np.int32) * width + off_events['x'].astype(np.int32)
        histogram -= np.bincount(off_coords, minlength=n_pixels).astype(np.int32)
        return histogram.reshape(height, width)
    if mode == 'on':
        events = events[events['p'] == 1]
    elif mode == 'off':
        events = events[events['p'] == 0]
    coords = events['y'].astype(np.int32) * width + events['x'].astype(np.int32)
    histogram = np.bincount(coords, minlength=n_pixels).astype(np.uint32)
    return histogram.reshape(height, width)
# ============================================================================
# PLOTLY FIGURE CREATION
# ============================================================================
def create_signed_heatmap(
    histogram: npt.NDArray[np.int32],
    signed_colorscale: list,
    zmin: Optional[float] = None,
    zmax: Optional[float] = None,
) -> go.Figure:
    """
    Create Plotly heatmap for signed (ON - OFF) event data.
    Uses a diverging colorscale centered at zero. By default, limits are set
    symmetrically around zero based on the data range.
    Args:
        histogram: 2D signed event count array
        signed_colorscale: Plotly colorscale list for diverging data
        zmin: Minimum colorscale value. None = auto symmetric.
        zmax: Maximum colorscale value. None = auto symmetric.
    Returns:
        Plotly Figure object
    """
    max_abs = max(abs(histogram.min()), abs(histogram.max()), 1)
    return go.Figure(go.Heatmap(
        z=histogram,
        colorscale=signed_colorscale,
        zmid=0,
        zmin=zmin if zmin is not None else -max_abs,
        zmax=zmax if zmax is not None else max_abs,
        colorbar=dict(title='ON - OFF')
    ))
def create_regular_heatmap(
    histogram: npt.NDArray[np.uint32],
    zmin: Optional[float] = None,
    zmax: Optional[float] = None,
) -> go.Figure:
    """
    Create Plotly heatmap for standard (non-signed) event data.
    Args:
        histogram: 2D event count array
        zmin: Minimum colorscale value. None = Plotly auto-scale.
        zmax: Maximum colorscale value. None = Plotly auto-scale.
    Returns:
        Plotly Figure object
    """
    return go.Figure(go.Heatmap(
        z=histogram,
        colorscale='Viridis',
        zmin=zmin,
        zmax=zmax,
        colorbar=dict(title='Count')
    ))
# ============================================================================
# FRAME GENERATION
# ============================================================================
def generate_frames(
    events: npt.NDArray[np.void],
    width: int,
    height: int,
    delta_t_ms: float,
    mode: str = 'all',
    clip_value: Optional[int] = 65535,
    scale_to_fit: bool = False
) -> Tuple[npt.NDArray[np.int32] | npt.NDArray[np.uint16], npt.NDArray[np.float64]]:
    """
    Generate video frames from events by accumulating events in time windows.
    Events are binned into temporal windows of size delta_t_ms. For non-signed modes,
    event counts per pixel are accumulated. For signed mode, ON events are added and
    OFF events subtracted to produce signed frames.
    Args:
        events: Structured array of events with fields 't', 'x', 'y', 'p'
        width: Frame width in pixels
        height: Frame height in pixels
        delta_t_ms: Time window for each frame in milliseconds
        mode: 'all', 'on', 'off', or 'signed'
        clip_value: Saturate pixel values at this threshold (default 65535 for uint16).
                    Set to None to keep raw uint32 accumulation.
        scale_to_fit: If True and clip_value is set, scale linearly so the maximum
                      value maps to clip_value instead of hard-clipping.
    Returns:
        Tuple of (frames, timestamps):
            - frames: (n_frames, height, width) array.
              int32 for signed mode, uint16 when clipped, uint32 otherwise.
            - timestamps: frame start times in microseconds (uint64)
    """
    width = int(width)
    height = int(height)
    if len(events) == 0:
        return np.array([]), np.array([])
    delta_t_us = delta_t_ms * 1000
    t_start = events['t'][0]
    t_end = events['t'][-1]
    # floor division so the requested frame count matches UI calculation
    n_frames = max(1, int((t_end - t_start) / delta_t_us))
    timestamps = np.arange(n_frames) * delta_t_us + t_start
    if mode == 'signed':
        on_events = events[events['p'] == 1]
        off_events = events[events['p'] == 0]
        frames = np.zeros((n_frames, height, width), dtype=np.int32)
        if len(on_events) > 0:
            frame_idx = ((on_events['t'] - t_start) / delta_t_us).astype(np.int64)
            frame_idx = np.clip(frame_idx, 0, n_frames - 1)
            flat_coords = (
                frame_idx * (height * width) +
                on_events['y'].astype(np.int64) * width +
                on_events['x'].astype(np.int64)
            )
            unique_coords, counts = np.unique(flat_coords, return_counts=True)
            frames.ravel()[unique_coords] += counts.astype(np.int32)
        if len(off_events) > 0:
            frame_idx = ((off_events['t'] - t_start) / delta_t_us).astype(np.int64)
            frame_idx = np.clip(frame_idx, 0, n_frames - 1)
            flat_coords = (
                frame_idx * (height * width) +
                off_events['y'].astype(np.int64) * width +
                off_events['x'].astype(np.int64)
            )
            unique_coords, counts = np.unique(flat_coords, return_counts=True)
            frames.ravel()[unique_coords] -= counts.astype(np.int32)
        # Clipping/scaling not applied to signed frames (contain negative values)
        return frames, timestamps
    # Non-signed modes
    if mode == 'on':
        events = events[events['p'] == 1]
    elif mode == 'off':
        events = events[events['p'] == 0]
    if len(events) == 0:
        return np.array([]), np.array([])
    frame_idx = ((events['t'] - t_start) / delta_t_us).astype(np.int64)
    frame_idx = np.clip(frame_idx, 0, n_frames - 1)
    # Start with uint32 to avoid overflow during accumulation
    frames = np.zeros((n_frames, height, width), dtype=np.uint32)
    flat_coords = (
        frame_idx * (height * width) +
        events['y'].astype(np.int64) * width +
        events['x'].astype(np.int64)
    )
    unique_coords, counts = np.unique(flat_coords, return_counts=True)
    frames.ravel()[unique_coords] = counts
    if clip_value is not None:
        if scale_to_fit:
            # Scale linearly so the maximum value equals clip_value
            max_val = frames.max()
            if max_val > clip_value:
                frames = (frames.astype(np.float64) * clip_value / max_val).astype(np.uint16)
            else:
                frames = frames.astype(np.uint16)
        else:
            # Hard clip at clip_value (default: saturate at 65535)
            frames = np.minimum(frames, clip_value).astype(np.uint16)
    return frames, timestamps