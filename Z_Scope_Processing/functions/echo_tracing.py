# zscope/functions/echo_tracing.py

import numpy as np
import cv2
from scipy.signal import find_peaks
from scipy.interpolate import interp1d


def enhance_image(image, clahe_clip=2.0, clahe_tile=(8, 8), blur_ksize=(5, 5)):
    """Enhance image contrast and reduce noise for better echo tracing."""
    if image.dtype != np.uint8:
        img_uint8 = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    else:
        img_uint8 = image.copy()

    clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=clahe_tile)
    enhanced = clahe.apply(img_uint8)

    if blur_ksize and blur_ksize[0] > 0 and blur_ksize[1] > 0:
        enhanced = cv2.GaussianBlur(enhanced, blur_ksize, 0)
    return enhanced


def extend_boundaries(
    data,
    extension_size=100,
    dampen_factor=0.9,
    trend_points=10,
    use_reflect_padding=True,
):
    """
    Enhanced boundary extension using trend-based extrapolation.

    Args:
        data (np.ndarray): 1D array to extend
        extension_size (int): Number of points to add at each end
        dampen_factor (float): Factor to reduce oscillation in mirrored values
        trend_points (int): Number of points to use for trend estimation
        use_reflect_padding (bool): Whether to use reflection padding for right edge

    Returns:
        tuple: (extended_data, extension_size)
    """
    if len(data) < 3:
        return data, 0

    # Find valid indices
    valid_indices = np.where(np.isfinite(data))[0]
    if len(valid_indices) < 3:
        return data, 0

    # Get edge values
    left_valid = valid_indices[0]
    right_valid = valid_indices[-1]

    # Create extended array
    extended = np.full(len(data) + 2 * extension_size, np.nan)

    # Copy original data to center
    extended[extension_size : extension_size + len(data)] = data

    # Mirror left boundary with dampening
    if left_valid > 0:
        left_values = data[left_valid : left_valid + extension_size]
        for i in range(min(extension_size, len(left_values))):
            mirror_idx = extension_size - i - 1
            if mirror_idx >= 0 and i < len(left_values):
                # Apply dampening factor (reduces oscillation)
                dampen = dampen_factor ** (i + 1)
                delta = (left_values[i] - data[left_valid]) * dampen
                extended[mirror_idx] = data[left_valid] - delta

    # For right edge, use reflection padding if requested
    if right_valid < len(data) - 1:
        if use_reflect_padding:
            # Use reflection padding for right edge
            right_reflection_size = min(extension_size, right_valid)
            for i in range(right_reflection_size):
                mirror_idx = extension_size + len(data) + i
                reflect_idx = right_valid - i
                if 0 <= mirror_idx < len(extended) and 0 <= reflect_idx < len(data):
                    extended[mirror_idx] = data[reflect_idx]
        else:
            # Use dampened mirroring for right edge (original method)
            right_values = data[
                max(0, right_valid - extension_size + 1) : right_valid + 1
            ]
            for i in range(min(extension_size, len(right_values))):
                mirror_idx = extension_size + len(data) + i
                if mirror_idx < len(extended) and i < len(right_values):
                    # Apply dampening factor
                    dampen = dampen_factor ** (i + 1)
                    delta = (right_values[-i - 1] - data[right_valid]) * dampen
                    extended[mirror_idx] = data[right_valid] + delta

    # Add trend-based extrapolation for more natural continuation
    if len(valid_indices) > trend_points:
        # Left boundary - use linear trend from first few valid points
        left_trend_indices = valid_indices[: min(trend_points, len(valid_indices) // 4)]
        if len(left_trend_indices) >= 2:
            try:
                left_slope, left_intercept = np.polyfit(
                    left_trend_indices, data[left_trend_indices], 1
                )
                for i in range(left_valid):
                    # Blend between trend-based extrapolation and dampened mirror
                    weight = min(1.0, (left_valid - i) / (extension_size / 2))
                    trend_value = left_intercept + left_slope * (i - left_valid)
                    mirror_idx = extension_size - left_valid + i
                    if 0 <= mirror_idx < len(extended) and np.isfinite(
                        extended[mirror_idx]
                    ):
                        extended[mirror_idx] = (
                            trend_value * (1 - weight) + extended[mirror_idx] * weight
                        )
            except:
                pass  # Fall back to current method if fit fails

        # Right boundary - use linear trend from last few valid points
        right_trend_indices = valid_indices[
            max(0, len(valid_indices) - min(trend_points, len(valid_indices) // 4)) :
        ]
        if len(right_trend_indices) >= 2:
            try:
                right_slope, right_intercept = np.polyfit(
                    right_trend_indices, data[right_trend_indices], 1
                )
                for i in range(len(data) - right_valid - 1):
                    # Blend between trend-based extrapolation and dampened mirror
                    weight = min(1.0, i / (extension_size / 2))
                    trend_value = right_intercept + right_slope * (i + 1)
                    mirror_idx = extension_size + right_valid + i
                    if 0 <= mirror_idx < len(extended) and np.isfinite(
                        extended[mirror_idx]
                    ):
                        extended[mirror_idx] = (
                            trend_value * (1 - weight) + extended[mirror_idx] * weight
                        )
            except:
                pass  # Fall back to current method if fit fails

    return extended, extension_size


def bilateral_filter_1d(signal, diameter=5, sigma_color=10.0, sigma_space=2.0):
    """
    Apply a bilateral filter to a 1D signal to preserve edges while smoothing.

    Args:
        signal (np.ndarray): 1D array to filter
        diameter (int): Window size (should be odd)
        sigma_color (float): Filter sigma in the intensity/color space
        sigma_space (float): Filter sigma in the spatial domain

    Returns:
        np.ndarray: Filtered signal
    """
    if diameter % 2 == 0:
        diameter += 1
    half_d = diameter // 2
    filtered = np.zeros_like(signal)
    length = len(signal)

    valid_indices = np.where(np.isfinite(signal))[0]
    if len(valid_indices) < 3:
        return signal

    # Fill NaNs for processing
    working_signal = signal.copy()
    if len(valid_indices) < len(signal):
        # Interpolate NaNs
        x_valid = valid_indices
        y_valid = signal[valid_indices]
        x_all = np.arange(len(signal))
        if len(valid_indices) >= 2:
            interp_func = interp1d(
                x_valid,
                y_valid,
                kind="linear",
                bounds_error=False,
                fill_value=(y_valid[0], y_valid[-1]),
            )
            working_signal = interp_func(x_all)

    for i in range(length):
        w_sum = 0.0
        val_sum = 0.0
        for j in range(max(0, i - half_d), min(length, i + half_d + 1)):
            spatial_dist = abs(i - j)
            color_dist = abs(working_signal[i] - working_signal[j])
            w = np.exp(-(spatial_dist**2) / (2 * sigma_space**2)) * np.exp(
                -(color_dist**2) / (2 * sigma_color**2)
            )
            w_sum += w
            val_sum += w * working_signal[j]
        filtered[i] = val_sum / w_sum if w_sum > 0 else working_signal[i]

    return filtered


def bilateral_filter_with_edge_emphasis(
    signal, diameter=25, edge_fraction=0.05, edge_sigma_factor=1.5
):
    """
    Apply bilateral filter with stronger smoothing at edges.

    Args:
        signal (np.ndarray): Signal to filter
        diameter (int): Window size for main filtering
        edge_fraction (float): Fraction of signal length to consider as edge
        edge_sigma_factor (float): Factor to increase sigma at edges

    Returns:
        np.ndarray: Filtered signal
    """
    # Regular bilateral filtering for most of the signal
    filtered = bilateral_filter_1d(signal, diameter)

    # Apply stronger filtering at edges
    n = len(signal)
    edge_width = int(n * edge_fraction)

    valid_indices = np.where(np.isfinite(signal))[0]
    if len(valid_indices) < edge_width * 2:
        return filtered

    # For right edge, use stronger parameters
    right_valid = valid_indices[-1]
    right_edge_start = max(0, right_valid - edge_width)

    # Apply stronger bilateral filter to right edge region
    right_edge_signal = signal[right_edge_start : right_valid + 1]
    if len(right_edge_signal) > 3:
        right_edge_filtered = bilateral_filter_1d(
            right_edge_signal,
            diameter=diameter,
            sigma_color=np.nanstd(right_edge_signal) * edge_sigma_factor,
            sigma_space=diameter / 4,
        )
        filtered[right_edge_start : right_valid + 1] = right_edge_filtered

    return filtered


def apply_edge_constraints(
    trace,
    left_width_fraction=0.02,
    right_width_fraction=0.04,
    left_strength=0.7,
    right_strength=0.9,
):
    """
    Apply special constraints at image edges to ensure smooth transitions,
    with different parameters for left and right edges.

    Args:
        trace (np.ndarray): Trace to constrain
        left_width_fraction (float): Fraction of trace length to consider as left edge
        right_width_fraction (float): Fraction of trace length to consider as right edge
        left_strength (float): Strength of constraint at left edge (0-1)
        right_strength (float): Strength of constraint at right edge (0-1)

    Returns:
        np.ndarray: Trace with edge constraints applied
    """
    result = trace.copy()
    n = len(trace)
    left_edge_width = max(int(n * left_width_fraction), 3)
    right_edge_width = max(int(n * right_width_fraction), 3)

    valid_indices = np.where(np.isfinite(trace))[0]
    if len(valid_indices) < left_edge_width + right_edge_width:
        return trace

    # Get stable regions just inside the edges
    left_stable_idx = valid_indices[min(left_edge_width, len(valid_indices) // 10)]
    right_stable_idx = valid_indices[
        max(0, len(valid_indices) - min(right_edge_width, len(valid_indices) // 10) - 1)
    ]

    left_stable_val = trace[left_stable_idx]
    right_stable_val = trace[right_stable_idx]

    # Apply constraints to left edge
    for i in range(valid_indices[0], left_stable_idx):
        # Calculate weight that increases as we move away from the edge
        weight = min(1.0, (i - valid_indices[0]) / left_edge_width)
        # Blend between stable value and actual value
        result[i] = left_stable_val * (1 - weight * left_strength) + trace[i] * (
            weight * left_strength
        )

    # Apply stronger constraints to right edge
    for i in range(right_stable_idx + 1, valid_indices[-1] + 1):
        # Calculate weight that increases as we move away from the edge
        weight = min(1.0, (valid_indices[-1] - i) / right_edge_width)
        # Blend between stable value and actual value
        result[i] = right_stable_val * (1 - weight * right_strength) + trace[i] * (
            weight * right_strength
        )

    return result


def apply_right_edge_window(trace, window_fraction=0.03):
    """
    Apply a tapered window to smooth the right edge transition.

    Args:
        trace (np.ndarray): Trace to window
        window_fraction (float): Fraction of trace length to apply window to

    Returns:
        np.ndarray: Trace with window applied to right edge
    """
    result = trace.copy()
    n = len(trace)
    window_width = max(int(n * window_fraction), 3)

    valid_indices = np.where(np.isfinite(trace))[0]
    if len(valid_indices) < window_width:
        return trace

    right_valid = valid_indices[-1]
    window_start = max(0, right_valid - window_width)

    # Create Hann window for smooth tapering
    window_size = right_valid - window_start + 1
    window = 0.5 * (1 - np.cos(np.pi * np.arange(window_size) / window_size))

    # Apply window to right edge
    right_edge_values = trace[window_start : right_valid + 1]
    right_edge_mean = np.mean(right_edge_values)
    result[window_start : right_valid + 1] = (
        right_edge_mean + (right_edge_values - right_edge_mean) * window
    )

    return result


def gradient_limited_smoothing(trace, max_gradient=0.5):
    """
    Apply smoothing that limits maximum gradient changes.

    Args:
        trace (np.ndarray): The trace to smooth
        max_gradient (float): Maximum allowed gradient between adjacent points

    Returns:
        np.ndarray: Smoothed trace with limited gradients
    """
    result = trace.copy()
    valid_indices = np.where(np.isfinite(trace))[0]

    if len(valid_indices) < 3:
        return trace

    # Calculate gradients
    gradients = np.diff(trace[valid_indices])

    # Identify excessive gradients
    excessive = np.abs(gradients) > max_gradient

    if np.any(excessive):
        # Process points with excessive gradients
        for i in range(len(gradients)):
            if excessive[i]:
                idx = valid_indices[i + 1]
                # Limit the change to max_gradient
                if gradients[i] > 0:
                    result[idx] = result[idx - 1] + max_gradient
                else:
                    result[idx] = result[idx - 1] - max_gradient

    return result


def apply_consistency_constraints(trace, max_deviation=5, window_size=5):
    """
    Apply constraints to ensure consistency between adjacent columns.

    Args:
        trace (np.ndarray): The trace to constrain
        max_deviation (float): Maximum allowed deviation from local median
        window_size (int): Size of window for calculating local median

    Returns:
        np.ndarray: Trace with consistency constraints applied
    """
    result = trace.copy()
    valid_indices = np.where(np.isfinite(trace))[0]

    if len(valid_indices) < window_size:
        return trace

    for i in range(len(valid_indices) - 1):
        idx = valid_indices[i]
        next_idx = valid_indices[i + 1]

        if next_idx - idx == 1:  # Adjacent columns
            # Check if deviation exceeds threshold
            if abs(trace[idx] - trace[next_idx]) > max_deviation:
                # Get local neighborhood for context
                start = max(0, idx - window_size // 2)
                end = min(len(trace), idx + window_size // 2 + 1)
                neighborhood = trace[start:end]
                valid_neighborhood = neighborhood[np.isfinite(neighborhood)]

                if len(valid_neighborhood) > 0:
                    # Use median of neighborhood to constrain outlier
                    median_value = np.median(valid_neighborhood)
                    # Apply constraint to the outlier
                    if abs(trace[idx] - median_value) > abs(
                        trace[next_idx] - median_value
                    ):
                        result[idx] = median_value + max_deviation * np.sign(
                            trace[idx] - median_value
                        )
                    else:
                        result[next_idx] = median_value + max_deviation * np.sign(
                            trace[next_idx] - median_value
                        )

    return result


def multi_scale_peak_detection(profile, prominence_range=(10, 30), scales=3):
    """
    Detect peaks at multiple scales and combine results.

    Args:
        profile (np.ndarray): 1D signal profile to analyze
        prominence_range (tuple): Min and max prominence values
        scales (int): Number of scales to analyze

    Returns:
        np.ndarray: Array of peak indices that appear in multiple scales
    """
    if len(profile) < 3:
        return np.array([])

    all_peaks = []
    prominences = np.linspace(prominence_range[0], prominence_range[1], scales)

    for prom in prominences:
        peaks, _ = find_peaks(profile, prominence=prom)
        all_peaks.extend(peaks)

    # Count occurrences of each peak across scales
    if len(all_peaks) == 0:
        return np.array([])

    unique_peaks, counts = np.unique(all_peaks, return_counts=True)

    # Return peaks that appear in multiple scales, sorted by prominence
    multi_scale_peaks = unique_peaks[counts > 1]

    # If no peaks appear in multiple scales, return the peaks from the middle scale
    if len(multi_scale_peaks) == 0 and len(all_peaks) > 0:
        mid_scale = scales // 2
        mid_prom = prominences[mid_scale]
        multi_scale_peaks, _ = find_peaks(profile, prominence=mid_prom)

    return multi_scale_peaks


def adaptive_smooth_trace(
    trace,
    min_window=11,
    max_window=101,
    polyorder=3,
    interp_kind="linear",
    max_gradient=0.5,
    max_deviation=5,
    use_bilateral=False,
    bilateral_diameter=25,
    bilateral_sigma_color_factor=2.0,
    bilateral_sigma_space_factor=0.5,
    use_edge_constraints=False,
    left_width_fraction=0.02,
    right_width_fraction=0.04,
    left_strength=0.7,
    right_strength=0.9,
    use_reflect_padding=True,
    use_right_edge_window=False,
    right_edge_window_fraction=0.03,
    use_edge_emphasis=False,
    edge_emphasis_fraction=0.05,
    edge_emphasis_factor=1.5,
):
    """
    Adaptive smoothing of a 1D trace using variable window sizes based on local polynomial regression.
    Window size is adjusted based on local variability.
    """
    # Extend boundaries to reduce edge effects
    extended_trace, extension_size = extend_boundaries(
        trace,
        extension_size=max(100, int(len(trace) * 0.05)),
        dampen_factor=0.9,
        trend_points=10,
        use_reflect_padding=use_reflect_padding,
    )

    # Process the extended trace
    x_coords_ext = np.arange(len(extended_trace))
    valid_indices_ext = np.where(np.isfinite(extended_trace))[0]

    if len(valid_indices_ext) < 2:  # Need at least 2 points for interpolation
        return trace

    # Interpolate missing values in extended trace
    interpolated_ext = np.copy(extended_trace)
    if len(valid_indices_ext) > 0:
        fill_val_start = extended_trace[valid_indices_ext[0]]
        fill_val_end = extended_trace[valid_indices_ext[-1]]
        f_interp = interp1d(
            x_coords_ext[valid_indices_ext],
            extended_trace[valid_indices_ext],
            kind=interp_kind,
            bounds_error=False,
            fill_value=(fill_val_start, fill_val_end),
        )
        interpolated_ext = f_interp(x_coords_ext)

    # Apply gradient limiting to prevent large jumps
    interpolated_ext = gradient_limited_smoothing(interpolated_ext, max_gradient)

    # Apply consistency constraints to ensure smooth transitions
    interpolated_ext = apply_consistency_constraints(
        interpolated_ext, max_deviation, window_size=min_window
    )

    # Calculate local variability
    variability_estimation_window = min_window
    if variability_estimation_window % 2 == 0:
        variability_estimation_window += 1  # Ensure odd

    if len(interpolated_ext) < variability_estimation_window:
        variability_estimation_window = (
            len(interpolated_ext)
            if len(interpolated_ext) % 2 != 0
            else max(1, len(interpolated_ext) - 1)
        )

    local_std = np.zeros_like(interpolated_ext)
    if variability_estimation_window > 0:
        half_var_win = variability_estimation_window // 2
        for i in range(len(interpolated_ext)):
            start_idx = max(0, i - half_var_win)
            end_idx = min(len(interpolated_ext), i + half_var_win + 1)
            local_std[i] = np.std(interpolated_ext[start_idx:end_idx])
    else:
        return trace[:]  # Cannot calculate local_std

    # Normalize local std to [0,1]
    std_min, std_max = np.min(local_std), np.max(local_std)
    if std_max - std_min > 1e-6:  # Avoid division by zero if std is constant
        norm_std = (local_std - std_min) / (std_max - std_min)
    else:
        norm_std = np.zeros_like(
            local_std
        )  # If no variability, use min_window everywhere

    # Map normalized std to smoothing window sizes
    window_sizes = (max_window - min_window) * (1 - norm_std) + min_window
    window_sizes = np.clip(window_sizes, min_window, max_window).astype(int)
    window_sizes += 1 - (window_sizes % 2)  # Ensure odd

    # Apply adaptive smoothing using local polynomial regression
    smoothed_ext = np.copy(interpolated_ext)

    for i in range(len(interpolated_ext)):
        current_poly_order = polyorder
        w = window_sizes[i]

        if w <= current_poly_order:
            current_poly_order = w - 1
            if current_poly_order < 1:
                current_poly_order = 1  # Min polyorder 1 (line)

        if current_poly_order < 0:  # Should not happen if w >=1
            smoothed_ext[i] = interpolated_ext[i]
            continue

        half_w = w // 2
        start_idx = max(0, i - half_w)
        end_idx = min(len(interpolated_ext), i + half_w + 1)
        segment_data = interpolated_ext[start_idx:end_idx]
        segment_x_coords = np.arange(len(segment_data))

        if len(segment_data) > current_poly_order and len(segment_data) > 0:
            try:
                coeffs = np.polyfit(segment_x_coords, segment_data, current_poly_order)
                smoothed_ext[i] = np.polyval(coeffs, i - start_idx)
            except (np.linalg.LinAlgError, ValueError):
                smoothed_ext[i] = interpolated_ext[i]  # Fallback
        else:
            smoothed_ext[i] = interpolated_ext[i]

    # Extract the original portion from the extended and smoothed trace
    result = smoothed_ext[extension_size : extension_size + len(trace)]

    # Apply final consistency constraints
    result = apply_consistency_constraints(
        result, max_deviation, window_size=min_window
    )

    # Apply bilateral filtering if requested
    if use_bilateral:
        sigma_color = np.nanstd(result) * bilateral_sigma_color_factor
        sigma_space = min_window * bilateral_sigma_space_factor
        result = bilateral_filter_1d(
            result,
            diameter=bilateral_diameter,
            sigma_color=sigma_color,
            sigma_space=sigma_space,
        )

    # Apply stronger bilateral filtering at edges if requested
    if use_edge_emphasis:
        result = bilateral_filter_with_edge_emphasis(
            result,
            diameter=bilateral_diameter,
            edge_fraction=edge_emphasis_fraction,
            edge_sigma_factor=edge_emphasis_factor,
        )

    # Apply edge constraints if requested
    if use_edge_constraints:
        result = apply_edge_constraints(
            result,
            left_width_fraction=left_width_fraction,
            right_width_fraction=right_width_fraction,
            left_strength=left_strength,
            right_strength=right_strength,
        )

    # Apply right edge window if requested
    if use_right_edge_window:
        result = apply_right_edge_window(
            result, window_fraction=right_edge_window_fraction
        )

    return result


def detect_surface_echo(image, tx_pulse_y, config):
    """Detect surface echo starting just below Tx pulse."""
    print("INFO: detect_surface_echo called.")
    crop_height, crop_width = image.shape

    cfg_search_start_offset = config.get("search_start_offset_px", 20)
    cfg_search_depth = config.get("search_depth_px", image.shape[0] // 3)

    y_start = tx_pulse_y + cfg_search_start_offset
    y_end = y_start + cfg_search_depth
    y_start = max(0, y_start)
    y_end = min(image.shape[0], y_end)

    if y_start >= y_end:
        print(
            f"WARNING (detect_surface_echo): Invalid search window [{y_start}, {y_end}]. Returning NaNs."
        )
        return np.full(crop_width, np.nan)

    enhanced = enhance_image(
        image,
        clahe_clip=config.get("enhancement_clahe_clip", 2.0),
        clahe_tile=tuple(config.get("enhancement_clahe_tile", (8, 8))),
        blur_ksize=tuple(config.get("enhancement_blur_ksize", (3, 3))),
    )

    raw_trace = np.full(crop_width, np.nan)
    polarity = config.get("echo_polarity", "bright")

    # Get multi-scale peak detection parameters
    use_multi_scale = config.get("use_multi_scale", True)
    prominence_min = config.get("prominence_min", 15)
    prominence_max = config.get("prominence_max", 35)
    scales = config.get("scales", 3)

    for x_col in range(crop_width):
        col_profile_data = enhanced[y_start:y_end, x_col]
        if col_profile_data.size == 0:
            continue

        if polarity == "dark":
            profile_to_search = 255 - col_profile_data
        else:
            profile_to_search = col_profile_data

        if use_multi_scale:
            # Use multi-scale peak detection
            peaks = multi_scale_peak_detection(
                profile_to_search,
                prominence_range=(prominence_min, prominence_max),
                scales=scales,
            )
        else:
            # Use standard peak detection
            prominence = config.get("peak_prominence", 20)
            peaks, _ = find_peaks(profile_to_search, prominence=prominence)

        if len(peaks) > 0:
            raw_trace[x_col] = y_start + peaks[0]

    # Get edge processing parameters
    edge_handling = config.get("edge_handling", {})
    use_bilateral = edge_handling.get("use_bilateral", True)
    bilateral_diameter = edge_handling.get("bilateral_diameter", 25)
    bilateral_sigma_color_factor = edge_handling.get(
        "bilateral_sigma_color_factor", 2.0
    )
    bilateral_sigma_space_factor = edge_handling.get(
        "bilateral_sigma_space_factor", 0.5
    )
    use_edge_constraints = edge_handling.get("use_edge_constraints", True)
    left_width_fraction = edge_handling.get("left_width_fraction", 0.02)
    right_width_fraction = edge_handling.get("right_width_fraction", 0.04)
    left_strength = edge_handling.get("left_strength", 0.7)
    right_strength = edge_handling.get("right_strength", 0.9)
    use_reflect_padding = edge_handling.get("use_reflect_padding", True)
    use_right_edge_window = edge_handling.get("use_right_edge_window", False)
    right_edge_window_fraction = edge_handling.get("right_edge_window_fraction", 0.03)
    use_edge_emphasis = edge_handling.get("use_edge_emphasis", False)
    edge_emphasis_fraction = edge_handling.get("edge_emphasis_fraction", 0.05)
    edge_emphasis_factor = edge_handling.get("edge_emphasis_factor", 1.5)

    # Apply adaptive smoothing with all enhancements
    max_gradient = config.get("max_gradient", 0.5)
    max_deviation = config.get("max_deviation", 5)

    smoothed_trace = adaptive_smooth_trace(
        raw_trace,
        min_window=config.get("adaptive_min_window", 11),
        max_window=config.get("adaptive_max_window", 101),
        polyorder=config.get("adaptive_polyorder", 3),
        interp_kind=config.get("adaptive_interp_kind", "linear"),
        max_gradient=max_gradient,
        max_deviation=max_deviation,
        use_bilateral=use_bilateral,
        bilateral_diameter=bilateral_diameter,
        bilateral_sigma_color_factor=bilateral_sigma_color_factor,
        bilateral_sigma_space_factor=bilateral_sigma_space_factor,
        use_edge_constraints=use_edge_constraints,
        left_width_fraction=left_width_fraction,
        right_width_fraction=right_width_fraction,
        left_strength=left_strength,
        right_strength=right_strength,
        use_reflect_padding=use_reflect_padding,
        use_right_edge_window=use_right_edge_window,
        right_edge_window_fraction=right_edge_window_fraction,
        use_edge_emphasis=use_edge_emphasis,
        edge_emphasis_fraction=edge_emphasis_fraction,
        edge_emphasis_factor=edge_emphasis_factor,
    )

    print("INFO: Ice surface echo detection attempt complete.")
    return smoothed_trace


def detect_bed_echo(
    image_data_region,
    surface_y_coords_relative,
    z_boundary_y_relative,
    config_params=None,
):
    """Detects the ice bed echo, searching below the detected surface for each column."""
    if config_params is None:
        config_params = {}
    print("INFO: detect_bed_echo called.")
    crop_height, crop_width = image_data_region.shape
    raw_bed_y_relative = np.full(crop_width, np.nan)

    enh_clip_bed = config_params.get("enhancement_clahe_clip", 3.0)
    enh_tile_list_bed = config_params.get("enhancement_clahe_tile", [4, 4])
    enh_tile_bed = (
        tuple(enh_tile_list_bed)
        if isinstance(enh_tile_list_bed, list) and len(enh_tile_list_bed) == 2
        else (4, 4)
    )
    enh_blur_list_bed = config_params.get("enhancement_blur_ksize", [5, 5])
    enh_blur_bed = (
        tuple(enh_blur_list_bed)
        if isinstance(enh_blur_list_bed, list) and len(enh_blur_list_bed) == 2
        else (5, 5)
    )

    enhanced_image_for_bed = enhance_image(
        image_data_region, enh_clip_bed, enh_tile_bed, enh_blur_bed
    )

    cfg_offset_from_surface = config_params.get(
        "search_start_offset_from_surface_px", 100
    )
    cfg_offset_from_z_boundary = config_params.get(
        "search_end_offset_from_z_boundary_px", 20
    )

    # Get multi-scale peak detection parameters
    use_multi_scale = config_params.get("use_multi_scale", True)
    prominence_min = config_params.get("prominence_min", 10)
    prominence_max = config_params.get("prominence_max", 25)
    scales = config_params.get("scales", 3)

    cfg_polarity = config_params.get("echo_polarity", "bright")
    cfg_search_dir = config_params.get("search_direction", "bottom_up")

    # Optional minimum absolute depth for bed search
    min_abs_bed_search_start = config_params.get("min_absolute_bed_search_start", 0)

    for x_col in range(crop_width):
        if (
            surface_y_coords_relative is None
            or x_col >= len(surface_y_coords_relative)
            or np.isnan(surface_y_coords_relative[x_col])
        ):
            continue

        # Calculate search window for this column
        search_y_start_col = int(
            surface_y_coords_relative[x_col] + cfg_offset_from_surface
        )

        # Apply minimum absolute depth constraint if configured
        search_y_start_col = max(search_y_start_col, min_abs_bed_search_start)

        search_y_end_col = int(z_boundary_y_relative - cfg_offset_from_z_boundary)

        # Ensure search window is within image bounds
        search_y_start_col = max(0, search_y_start_col)
        search_y_end_col = min(crop_height, search_y_end_col)

        if search_y_start_col >= search_y_end_col:
            continue

        column_profile_data_for_bed = enhanced_image_for_bed[
            search_y_start_col:search_y_end_col, x_col
        ]
        if column_profile_data_for_bed.size == 0:
            continue

        if cfg_polarity == "dark":
            profile_to_search_for_peaks = 255 - column_profile_data_for_bed
        else:
            profile_to_search_for_peaks = column_profile_data_for_bed

        if use_multi_scale:
            # Use multi-scale peak detection
            peaks_found_in_profile = multi_scale_peak_detection(
                profile_to_search_for_peaks,
                prominence_range=(prominence_min, prominence_max),
                scales=scales,
            )
        else:
            # Use standard peak detection
            cfg_peak_prom = config_params.get("peak_prominence", 15)
            peaks_found_in_profile, _ = find_peaks(
                profile_to_search_for_peaks, prominence=cfg_peak_prom
            )

        if len(peaks_found_in_profile) > 0:
            if cfg_search_dir == "top_down":
                chosen_peak_local_y_in_profile = peaks_found_in_profile[0]
            elif cfg_search_dir == "bottom_up":
                chosen_peak_local_y_in_profile = peaks_found_in_profile[-1]
            else:
                chosen_peak_local_y_in_profile = peaks_found_in_profile[0]

            raw_bed_y_relative[x_col] = (
                search_y_start_col + chosen_peak_local_y_in_profile
            )

    # Get edge processing parameters
    edge_handling = config_params.get("edge_handling", {})
    use_bilateral = edge_handling.get("use_bilateral", True)
    bilateral_diameter = edge_handling.get("bilateral_diameter", 35)
    bilateral_sigma_color_factor = edge_handling.get(
        "bilateral_sigma_color_factor", 2.5
    )
    bilateral_sigma_space_factor = edge_handling.get(
        "bilateral_sigma_space_factor", 0.5
    )
    use_edge_constraints = edge_handling.get("use_edge_constraints", True)
    left_width_fraction = edge_handling.get("left_width_fraction", 0.02)
    right_width_fraction = edge_handling.get("right_width_fraction", 0.04)
    left_strength = edge_handling.get("left_strength", 0.7)
    right_strength = edge_handling.get("right_strength", 0.9)
    use_reflect_padding = edge_handling.get("use_reflect_padding", True)
    use_right_edge_window = edge_handling.get("use_right_edge_window", True)
    right_edge_window_fraction = edge_handling.get("right_edge_window_fraction", 0.03)
    use_edge_emphasis = edge_handling.get("use_edge_emphasis", True)
    edge_emphasis_fraction = edge_handling.get("edge_emphasis_fraction", 0.05)
    edge_emphasis_factor = edge_handling.get("edge_emphasis_factor", 1.5)

    # Apply adaptive smoothing with all enhancements
    max_gradient = config_params.get("max_gradient", 0.3)
    max_deviation = config_params.get("max_deviation", 5)

    smoothed_bed_y_relative = adaptive_smooth_trace(
        raw_bed_y_relative,
        min_window=config_params.get("adaptive_min_window", 21),
        max_window=config_params.get("adaptive_max_window", 201),
        polyorder=config_params.get("adaptive_polyorder", 3),
        interp_kind=config_params.get("adaptive_interp_kind", "linear"),
        max_gradient=max_gradient,
        max_deviation=max_deviation,
        use_bilateral=use_bilateral,
        bilateral_diameter=bilateral_diameter,
        bilateral_sigma_color_factor=bilateral_sigma_color_factor,
        bilateral_sigma_space_factor=bilateral_sigma_space_factor,
        use_edge_constraints=use_edge_constraints,
        left_width_fraction=left_width_fraction,
        right_width_fraction=right_width_fraction,
        left_strength=left_strength,
        right_strength=right_strength,
        use_reflect_padding=use_reflect_padding,
        use_right_edge_window=use_right_edge_window,
        right_edge_window_fraction=right_edge_window_fraction,
        use_edge_emphasis=use_edge_emphasis,
        edge_emphasis_fraction=edge_emphasis_fraction,
        edge_emphasis_factor=edge_emphasis_factor,
    )

    print("INFO: Ice bed echo detection attempt complete.")
    return smoothed_bed_y_relative
