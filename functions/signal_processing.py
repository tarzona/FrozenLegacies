import numpy as np
import cv2
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import os
import sys

# Add the parent directory to sys.path to allow imports from sibling modules
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)


def detect_signal_in_frame(frame, config):
    h, w = frame.shape
    processing_params = config.get("processing_params", {})

    # Enhance contrast
    clip_limit = processing_params.get("grid_enhance_clip_limit", 2.0)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    enhanced = clahe.apply(frame)

    signal_x, signal_y = [], []
    band_frac = processing_params.get("signal_detect_band_frac", [0.10, 0.85])
    signal_min_row = int(h * band_frac[0])
    signal_max_row = int(h * band_frac[1])

    # Improved: Use adaptive thresholding for each column
    for x in range(w):
        col = enhanced[signal_min_row:signal_max_row, x]

        # Use multiple thresholds to capture different intensity levels
        percentile_low = processing_params.get("signal_detect_percentile_low", 15)
        percentile_high = processing_params.get("signal_detect_percentile_high", 25)

        threshold_low = np.percentile(col, percentile_low)
        threshold_high = np.percentile(col, percentile_high)

        # Find darkest points with two thresholds
        dark_indices_low = np.where(col < threshold_low)[0]
        dark_indices_high = np.where(col < threshold_high)[0]

        # Prioritize the darkest points, but use higher threshold if needed
        if len(dark_indices_low) > 0:
            # Find the absolute darkest point
            darkest_idx = np.argmin(col[dark_indices_low])
            y = signal_min_row + dark_indices_low[darkest_idx]
        elif len(dark_indices_high) > 0:
            # Fall back to less strict threshold
            darkest_idx = np.argmin(col[dark_indices_high])
            y = signal_min_row + dark_indices_high[darkest_idx]
        else:
            # Skip this column if no dark points found
            continue

        signal_x.append(x)
        signal_y.append(y)

    if len(signal_x) < w * 0.2:  # Require signal to cover at least 20% of width
        print("Warning: Signal trace too short or not detected.")
        return None, None

    # Smooth the detected trace
    smooth_sigma = processing_params.get("signal_detect_smooth_sigma", 2)
    signal_y_smooth = gaussian_filter1d(np.array(signal_y), sigma=smooth_sigma)
    return np.array(signal_x), signal_y_smooth


def trim_signal_trace(frame_img, signal_x, signal_y, config):
    """Trims ends of the signal trace and keeps the longest valid run."""
    if signal_x is None or len(signal_x) == 0:
        return np.array([]), np.array([])

    h, w = frame_img.shape
    x = np.array(signal_x)
    y = np.array(signal_y)

    # Get configuration parameters with fallbacks
    processing_params = config.get("processing_params", {})

    min_run_frac = processing_params.get("signal_trim_min_run_frac", 0.4)
    trim_frac = processing_params.get("signal_trim_frac", 0.17)

    min_run = int(w * min_run_frac)  # Min length relative to frame width
    trim_px = int(w * trim_frac)  # Pixels to trim from each end

    if len(x) < trim_px * 2 + min_run:
        print("Warning: Signal too short to trim effectively.")
        return x, y  # Return untrimmed if too short

    # Initial trim based on pixel distance from ends
    mask = (x >= trim_px) & (x <= (x.max() - trim_px))
    if not np.any(mask):
        print("Warning: Initial trimming removed all signal points.")
        return np.array([]), np.array([])

    x_trim = x[mask]
    y_trim = y[mask]

    # Validate remaining points by checking for dark pixels in original frame
    valid_indices_in_trim = []
    black_thresh = processing_params.get("signal_trim_black_thresh", 90)
    for i, xi in enumerate(x_trim):
        col_idx = int(np.round(xi))
        if 0 <= col_idx < w:
            col = frame_img[:, col_idx]
            if np.min(col) < black_thresh:  # Check if there's actual dark ink
                valid_indices_in_trim.append(i)

    if not valid_indices_in_trim:
        print("Warning: No valid dark signal found in trimmed region.")
        return np.array([]), np.array([])

    # Find the longest contiguous run of valid indices
    if len(valid_indices_in_trim) == 1:
        longest_run_indices_in_trim = valid_indices_in_trim
    else:
        diffs = np.diff(valid_indices_in_trim)
        breaks = np.where(diffs > 1)[0]
        if not breaks.size:  # Only one contiguous run
            longest_run_indices_in_trim = valid_indices_in_trim
        else:
            run_starts = [0] + (breaks + 1).tolist()
            run_ends = breaks.tolist() + [
                len(valid_indices_in_trim)
            ]  # Index after last element
            run_lengths = [run_ends[k] - run_starts[k] for k in range(len(run_starts))]
            max_run_idx = np.argmax(run_lengths)
            start_index = run_starts[max_run_idx]
            end_index = run_ends[max_run_idx]  # Exclusive index
            longest_run_indices_in_trim = valid_indices_in_trim[start_index:end_index]

    if len(longest_run_indices_in_trim) < min_run:
        print(
            f"Warning: Longest valid signal run ({len(longest_run_indices_in_trim)}px) < min ({min_run}px)."
        )
        return np.array([]), np.array([])

    # Get the final x and y values corresponding to the longest valid run
    final_x = x_trim[longest_run_indices_in_trim]
    final_y = y_trim[longest_run_indices_in_trim]

    return final_x, final_y


# Adaptive smoothing that preserves peaks
def adaptive_peak_preserving_smooth(signal_y, config):
    processing_params = config.get("processing_params", {})
    smooth_sigma = processing_params.get("signal_detect_smooth_sigma", 3)

    # Find potential peaks before smoothing
    peaks, _ = find_peaks(-np.array(signal_y), prominence=5)  # Negative to find minima

    # Apply standard smoothing
    signal_y_smooth = gaussian_filter1d(np.array(signal_y), sigma=smooth_sigma)

    # Restore peak values where significant
    if len(peaks) > 0:
        for peak in peaks:
            if peak < len(signal_y) and peak < len(signal_y_smooth):
                # Only restore if the smoothing significantly changed the peak
                if (
                    signal_y_smooth[peak] > signal_y[peak] + 2
                ):  # If smoothing raised value by >2 pixels
                    signal_y_smooth[peak] = signal_y[peak]

    return signal_y_smooth


def refine_signal_trace(frame_img, signal_x, signal_y, config):
    """Second pass to refine the signal trace around potential echo regions."""
    processing_params = config.get("processing_params", {})
    h, w = frame_img.shape

    # Convert to numpy arrays if not already
    signal_x = np.array(signal_x)
    signal_y = np.array(signal_y)

    # Find potential echo regions (local minima in y values)
    # Negative because lower y pixel value = higher on image = stronger echo
    peaks, _ = find_peaks(-signal_y, prominence=3, distance=20)

    # For each potential echo, refine the trace
    for peak_idx in peaks:
        if peak_idx >= len(signal_x):
            continue

        x_pos = signal_x[peak_idx]

        # Define a small window around the peak
        window_size = processing_params.get("signal_refine_window_size", 5)
        x_start = max(0, x_pos - window_size)
        x_end = min(w - 1, x_pos + window_size)

        # For each column in the window, find the absolute darkest point
        for x in range(int(x_start), int(x_end) + 1):
            col_idx = np.where(signal_x == x)[0]
            if len(col_idx) == 0:
                continue

            # Get the column from the original image
            col = frame_img[:, x]

            # Find the absolute darkest point
            darkest_y = np.argmin(col)

            # Update the signal trace if the darkest point is significantly darker
            if (
                col[darkest_y] < col[int(signal_y[col_idx[0]])] - 10
            ):  # At least 10 intensity units darker
                signal_y[col_idx[0]] = darkest_y

    return signal_x, signal_y


def verify_trace_quality(frame_img, signal_x, signal_y):
    """Verify the quality of the trace by checking pixel intensity along the trace."""
    h, w = frame_img.shape
    quality_scores = []

    for i in range(len(signal_x)):
        x, y = int(signal_x[i]), int(signal_y[i])
        if 0 <= x < w and 0 <= y < h:
            # Get a small window around the trace point
            window_size = 3
            y_min = max(0, y - window_size)
            y_max = min(h - 1, y + window_size)

            # Check if the trace point is close to the darkest point in the window
            col_segment = frame_img[y_min : y_max + 1, x]
            darkest_y_local = y_min + np.argmin(col_segment)

            # Calculate quality score (0 = perfect, higher = worse)
            quality_scores.append(abs(y - darkest_y_local))

    # Return average quality score
    return np.mean(quality_scores) if quality_scores else float("inf")
