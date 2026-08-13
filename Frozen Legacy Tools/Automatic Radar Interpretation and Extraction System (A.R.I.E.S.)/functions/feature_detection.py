import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from pathlib import Path


def refine_peaks(profile, peaks, window_size=5):
    """
    Refine peak positions using center-of-mass in a window.

    Args:
        profile (np.ndarray): 1D array (e.g., intensity profile).
        peaks (np.ndarray or list): Array of detected peak indices.
        window_size (int): Size of the window around each peak for refinement.

    Returns:
        np.ndarray: Array of refined peak positions.
    """
    refined = []
    for peak_idx in peaks:
        start = max(0, peak_idx - window_size // 2)
        end = min(len(profile), peak_idx + window_size // 2 + 1)
        window_data = profile[start:end]
        indices = np.arange(start, end)

        if np.sum(window_data) > 0:
            refined_pos = np.sum(indices * window_data) / np.sum(window_data)
            refined.append(refined_pos)
        else:
            refined.append(peak_idx)  # Fallback to original peak if window sum is zero

    return np.array(refined)


def detect_transmitter_pulse(
    image,
    base_filename,
    top_boundary,
    bottom_boundary,
    tx_pulse_params=None,
):
    """
    Detect transmitter pulse in Z-scope radar data, excluding film artifacts.

    Args:
        image (np.ndarray): 2D grayscale image array.
        base_filename (str): Base name for output files (for visualization).
        top_boundary (int): Top boundary of the valid data area.
        bottom_boundary (int): Bottom boundary of the valid data area.
        tx_pulse_params (dict, optional): Parameters for transmitter pulse detection.
            Expected keys:
            - "search_height_ratio" (float): Ratio of valid height to search (default 0.25).
            - "smoothing_kernel_size" (int): Kernel size for smoothing profile (default 15).
            - "peak_prominence" (float): Prominence for find_peaks (default 0.3).
            - "peak_distance" (int): Min distance between peaks (default 15).
            - "position_weight" (float): Weight for peak position in scoring (default 0.7).
            - "prominence_weight" (float): Weight for peak prominence in scoring (default 0.3).
            - "fallback_depth_ratio" (float): Ratio of valid height for fallback if no peaks (default 0.1).
            - "visualize_tx_pulse_detection" (bool): If True, plot diagnostic figure.

    Returns:
        int: Pixel row of the detected transmitter pulse.
    """
    if tx_pulse_params is None:
        tx_pulse_params = {}

    search_height_ratio = tx_pulse_params.get("search_height_ratio", 0.25)
    smoothing_kernel_size = tx_pulse_params.get("smoothing_kernel_size", 15)
    peak_prominence = tx_pulse_params.get("peak_prominence", 0.3)
    peak_distance = tx_pulse_params.get("peak_distance", 15)
    position_weight_factor = tx_pulse_params.get("position_weight", 0.7)
    prominence_weight_factor = tx_pulse_params.get("prominence_weight", 0.3)
    fallback_depth_ratio = tx_pulse_params.get("fallback_depth_ratio", 0.1)
    visualize = tx_pulse_params.get("visualize_tx_pulse_detection", False)

    valid_height = bottom_boundary - top_boundary
    if valid_height <= 0:
        print("Warning: Invalid data boundaries for Tx pulse detection.")
        return top_boundary  # Fallback

    search_height = int(valid_height * search_height_ratio)
    search_area = image[top_boundary : top_boundary + search_height, :]

    if search_area.size == 0:
        print("Warning: Search area for Tx pulse is empty.")
        return top_boundary + int(valid_height * fallback_depth_ratio)

    vertical_profile = np.mean(search_area, axis=1)

    if np.max(vertical_profile) == np.min(vertical_profile):  # Avoid division by zero
        normalized_profile = np.zeros_like(vertical_profile)
    else:
        normalized_profile = (vertical_profile - np.min(vertical_profile)) / (
            np.max(vertical_profile) - np.min(vertical_profile)
        )

    # Smooth profile to reduce noise
    kernel = np.ones(smoothing_kernel_size) / smoothing_kernel_size
    smoothed_profile = np.convolve(normalized_profile, kernel, mode="same")

    # Find peaks - looking for the strong initial transmitter pulse
    peaks, properties = find_peaks(
        smoothed_profile, prominence=peak_prominence, distance=peak_distance
    )

    if len(peaks) > 0:
        # Weight peaks by both prominence and early position
        # Earlier peaks (smaller index) get higher position_weight
        position_score = 1 - (peaks / search_height)

        max_prominence = np.max(properties["prominences"])
        if max_prominence == 0:  # Avoid division by zero if all prominences are 0
            prominence_score = np.zeros_like(properties["prominences"])
        else:
            prominence_score = properties["prominences"] / max_prominence

        combined_weight = (position_score * position_weight_factor) + (
            prominence_score * prominence_weight_factor
        )

        best_peak_idx_in_peaks_array = np.argmax(combined_weight)
        transmitter_pulse_pixel = top_boundary + peaks[best_peak_idx_in_peaks_array]
    else:
        # Default fallback - typically transmitter pulse is ~10% into valid area
        transmitter_pulse_pixel = top_boundary + int(
            valid_height * fallback_depth_ratio
        )

    if visualize:
        plt.figure(figsize=(12, 8))
        plt.subplot(2, 1, 1)
        plt.imshow(search_area, cmap="gray", aspect="auto")
        if len(peaks) > 0:
            plt.axhline(
                y=peaks[best_peak_idx_in_peaks_array],
                color="r",
                linestyle="-",
                linewidth=2,
            )
        plt.title("Search Area for Transmitter Pulse")

        plt.subplot(2, 1, 2)
        plt.plot(normalized_profile, label="Normalized Intensity Profile")
        plt.plot(smoothed_profile, label="Smoothed Profile")
        if len(peaks) > 0:
            plt.plot(peaks, smoothed_profile[peaks], "rx", label="Detected Peaks")
            plt.plot(
                peaks[best_peak_idx_in_peaks_array],
                smoothed_profile[peaks[best_peak_idx_in_peaks_array]],
                "go",
                markersize=10,
                label="Selected Pulse",
            )
        plt.legend()
        plt.title("Transmitter Pulse Detection Profile Analysis")
        output_dir = Path("debug_output")
        output_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(
            output_dir / f"{base_filename}_transmitter_pulse_detection.png", dpi=300
        )
        plt.close()

    return transmitter_pulse_pixel


def detect_calibration_pip(
    image,
    base_filename,
    approx_x_position,
    data_top,
    data_bottom,
    z_boundary_y,
    pip_detection_params=None,
):
    """
    Detect calibration pip and its tick marks in Z-scope radar image.
    This version assumes z_boundary_y is passed as an argument.

    Args:
        image (np.ndarray): 2D grayscale image array.
        base_filename (str): Base name for output files.
        approx_x_position (int): Approximate x-coordinate of the calibration pip from user click.
        data_top (int): Top boundary of the valid data area.
        data_bottom (int): Bottom boundary of the valid data area.
        z_boundary_y (int): Pre-detected Y-coordinate of the Z-scope boundary.
        pip_detection_params (dict, optional): Parameters for pip detection.
            Expected keys: "approach_1", "approach_2_aggressive", "ranking_proximity_weight_contribution".

    Returns:
        dict or None: Details of the best detected pip, or None if no suitable pip is found.
                      Structure: {'x_position', 'y_start', 'y_end', 'tick_count',
                                  'mean_spacing', 'tick_positions', 'z_boundary', 'match_score'}
    """
    if pip_detection_params is None:
        pip_detection_params = {}

    params_approach_1 = pip_detection_params.get("approach_1", {})
    params_approach_2 = pip_detection_params.get("approach_2_aggressive", {})
    ranking_prox_weight = pip_detection_params.get(
        "ranking_proximity_weight_contribution", 0.8
    )

    height, width = image.shape
    candidates = []
    output_dir = Path("debug_output")  # Or get from config
    output_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"Detecting calibration pip focused around x={approx_x_position}, with Z-boundary at y={z_boundary_y}"
    )

    # --- Approach 1: Based on user-selected region ---
    max_dist_click = params_approach_1.get(
        "max_distance_from_click_px", 1000
    )  # Used for final ranking
    strip_w_app1 = params_approach_1.get("strip_width_px", 2000)
    clahe_clip_app1 = params_approach_1.get("clahe_clip_limit", 3.0)
    clahe_tile_app1 = tuple(params_approach_1.get("clahe_tile_grid_size", [8, 8]))
    vert_ksize_app1 = tuple(params_approach_1.get("vertical_kernel_size", [1, 25]))
    horiz_ksize_app1 = tuple(params_approach_1.get("horizontal_kernel_size", [15, 1]))
    combined_w_vert_app1 = params_approach_1.get(
        "combined_features_vertical_weight", 0.3
    )
    combined_w_horiz_app1 = params_approach_1.get(
        "combined_features_horizontal_weight", 0.7
    )
    binary_thresh_app1 = params_approach_1.get("binary_threshold", 10)
    profile_roi_margin_app1 = params_approach_1.get("profile_roi_margin_px", 50)
    expected_spacing_app1 = params_approach_1.get("expected_tick_spacing_approx_px", 30)
    spacing_tolerance_app1 = params_approach_1.get("tick_spacing_tolerance_factor", 0.8)
    tick_prominence_app1 = params_approach_1.get("tick_prominence", 30)
    tick_offset_app1 = params_approach_1.get("tick_vertical_offset_px", 1.5)
    z_safety_margin_app1 = params_approach_1.get("z_boundary_safety_margin_px", 50)
    min_ticks_app1 = params_approach_1.get("min_valid_ticks", 3)
    spacing_std_thresh_app1 = params_approach_1.get(
        "spacing_std_dev_factor_threshold", 0.4
    )
    score_base_app1 = params_approach_1.get("match_score_base", 0.9)
    refine_window_size = pip_detection_params.get("peak_refinement_params", {}).get(
        "window_size", 5
    )

    start_x_app1 = max(0, approx_x_position - strip_w_app1 // 2)
    end_x_app1 = min(width, approx_x_position + strip_w_app1 // 2)

    # y_start and y_end for strip are data_top and data_bottom
    # but ticks are filtered by z_boundary_y
    user_strip_app1 = image[data_top:data_bottom, start_x_app1:end_x_app1].copy()

    if user_strip_app1.size > 0:
        print(
            f"Approach 1: Processing user-selected region: x={start_x_app1}-{end_x_app1}"
        )
        clahe = cv2.createCLAHE(clipLimit=clahe_clip_app1, tileGridSize=clahe_tile_app1)
        enhanced_app1 = clahe.apply(user_strip_app1)
        if pip_detection_params.get("output_params", {}).get(
            "save_intermediate_plots", False
        ):
            cv2.imwrite(
                str(output_dir / f"{base_filename}_app1_strip_enhanced.png"),
                enhanced_app1,
            )

        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, vert_ksize_app1)
        vertical_enhanced_app1 = cv2.morphologyEx(
            enhanced_app1, cv2.MORPH_TOPHAT, vertical_kernel
        )

        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, horiz_ksize_app1)
        horizontal_enhanced_app1 = cv2.morphologyEx(
            enhanced_app1, cv2.MORPH_TOPHAT, horizontal_kernel
        )

        combined_app1 = cv2.addWeighted(
            vertical_enhanced_app1,
            combined_w_vert_app1,
            horizontal_enhanced_app1,
            combined_w_horiz_app1,
            0,
        )
        _, binary_app1 = cv2.threshold(
            combined_app1, binary_thresh_app1, 255, cv2.THRESH_BINARY
        )

        center_x_strip_app1 = user_strip_app1.shape[1] // 2
        profile_x_start = max(0, center_x_strip_app1 - profile_roi_margin_app1)
        profile_x_end = min(
            user_strip_app1.shape[1], center_x_strip_app1 + profile_roi_margin_app1
        )

        roi_for_profile_app1 = enhanced_app1[:, profile_x_start:profile_x_end]
        profile_app1 = np.sum(255 - roi_for_profile_app1, axis=1)  # Inverted

        peaks_app1, _ = find_peaks(
            profile_app1,
            distance=expected_spacing_app1 * spacing_tolerance_app1,
            prominence=tick_prominence_app1,
        )
        refined_peaks_app1_relative = refine_peaks(
            profile_app1, peaks_app1, window_size=refine_window_size
        )
        refined_peaks_app1_relative += tick_offset_app1  # Apply vertical offset

        if pip_detection_params.get("output_params", {}).get(
            "save_intermediate_plots", False
        ):
            plt.figure(figsize=(8, 6))
            plt.plot(profile_app1, label="Profile (Sum of Inverted ROI)")
            plt.plot(
                peaks_app1, profile_app1[peaks_app1], "x", label="Detected Raw Peaks"
            )
            plt.plot(
                refined_peaks_app1_relative,
                profile_app1[refined_peaks_app1_relative.astype(int)],
                "o",
                label="Refined Peaks",
            )
            plt.title("Approach 1: Vertical Profile with Ticks")
            plt.legend()
            plt.savefig(output_dir / f"{base_filename}_app1_vertical_profile.png")
            plt.close()

        if len(refined_peaks_app1_relative) >= min_ticks_app1:
            # Filter ticks by Z-boundary
            valid_peaks_indices = [
                i
                for i, p_rel in enumerate(refined_peaks_app1_relative)
                if (data_top + p_rel) < (z_boundary_y - z_safety_margin_app1)
            ]

            if len(valid_peaks_indices) >= min_ticks_app1:
                valid_refined_peaks_rel = refined_peaks_app1_relative[
                    valid_peaks_indices
                ]
                diffs = np.diff(valid_refined_peaks_rel)
                mean_diff = np.mean(diffs)
                std_diff = np.std(diffs)

                if std_diff < spacing_std_thresh_app1 * mean_diff and mean_diff > 0:
                    abs_x_pos = start_x_app1 + center_x_strip_app1
                    candidates.append(
                        {
                            "x_position": abs_x_pos,
                            "y_start_abs": data_top
                            + np.min(
                                valid_refined_peaks_rel
                            ),  # y relative to full image
                            "y_end_abs": min(
                                z_boundary_y - z_safety_margin_app1,
                                data_top + np.max(valid_refined_peaks_rel),
                            ),
                            "tick_count": len(valid_refined_peaks_rel),
                            "mean_spacing": mean_diff,
                            "tick_positions_abs": valid_refined_peaks_rel
                            + data_top,  # absolute Y in full image
                            "z_boundary_abs": z_boundary_y,
                            "match_score": score_base_app1,
                            "method": "Approach 1",
                        }
                    )
                    print(
                        f"Approach 1: Found candidate pip with {len(valid_refined_peaks_rel)} ticks."
                    )

    # --- Approach 2: Aggressive search if Approach 1 fails ---
    if not candidates:
        print(
            "Approach 1 failed or found no candidates. Trying Approach 2 (aggressive)..."
        )
        clahe_clip_app2 = params_approach_2.get("clahe_clip_limit", 4.0)
        clahe_tile_app2 = tuple(params_approach_2.get("clahe_tile_grid_size", [4, 4]))
        vert_ksize_app2 = tuple(params_approach_2.get("vertical_kernel_size", [1, 30]))
        canny_low = params_approach_2.get("canny_edge_low_threshold", 20)
        canny_high = params_approach_2.get("canny_edge_high_threshold", 70)
        hough_thresh = params_approach_2.get("hough_lines_threshold", 20)
        hough_min_len_ratio = params_approach_2.get(
            "hough_lines_min_length_ratio_of_strip", 0.1666
        )
        hough_max_gap = params_approach_2.get("hough_lines_max_gap_px", 30)
        hough_max_xdiff_vert = params_approach_2.get(
            "hough_lines_max_x_diff_for_vertical", 15
        )
        profile_roi_margin_app2 = params_approach_2.get("profile_roi_margin_px", 50)
        tick_dist_app2 = params_approach_2.get("tick_peak_distance_px", 5)
        z_safety_margin_app2 = params_approach_2.get("z_boundary_safety_margin_px", 50)
        min_ticks_app2 = params_approach_2.get("min_valid_ticks", 3)
        spacing_std_thresh_app2 = params_approach_2.get(
            "spacing_std_dev_factor_threshold", 0.4
        )
        score_base_app2 = params_approach_2.get("match_score_base", 0.6)

        # Use the same strip as approach 1 or define a new one
        strip_w_app2 = strip_w_app1
        start_x_app2 = start_x_app1
        end_x_app2 = end_x_app1

        aggressive_strip = image[data_top:data_bottom, start_x_app2:end_x_app2].copy()

        if aggressive_strip.size > 0:
            clahe = cv2.createCLAHE(
                clipLimit=clahe_clip_app2, tileGridSize=clahe_tile_app2
            )
            enhanced_app2 = clahe.apply(aggressive_strip)

            vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, vert_ksize_app2)
            vertical_enhanced_app2 = cv2.morphologyEx(
                enhanced_app2, cv2.MORPH_TOPHAT, vertical_kernel
            )
            edges_app2 = cv2.Canny(vertical_enhanced_app2, canny_low, canny_high)
            if pip_detection_params.get("output_params", {}).get(
                "save_intermediate_plots", False
            ):
                cv2.imwrite(
                    str(output_dir / f"{base_filename}_app2_edges.png"), edges_app2
                )

            lines = cv2.HoughLinesP(
                edges_app2,
                1,
                np.pi / 180,
                threshold=hough_thresh,
                minLineLength=aggressive_strip.shape[0] * hough_min_len_ratio,
                maxLineGap=hough_max_gap,
            )

            if lines is not None:
                for line in lines:
                    x1, y1_rel, x2, y2_rel = line[0]
                    if abs(x2 - x1) < hough_max_xdiff_vert:  # Vertical line
                        line_x_strip = (x1 + x2) // 2

                        # Extract profile around this line
                        profile_x_start = max(0, line_x_strip - profile_roi_margin_app2)
                        profile_x_end = min(
                            aggressive_strip.shape[1],
                            line_x_strip + profile_roi_margin_app2,
                        )

                        # Use edges for profile in this aggressive approach
                        profile_region_app2 = edges_app2[
                            :, profile_x_start:profile_x_end
                        ]
                        profile_app2 = np.sum(profile_region_app2, axis=1)

                        peaks_app2_rel, _ = find_peaks(
                            profile_app2, distance=tick_dist_app2
                        )  # Prominence might be too strict here

                        # Filter by Z-boundary
                        valid_peaks_indices_app2 = [
                            i
                            for i, p_rel in enumerate(peaks_app2_rel)
                            if (data_top + p_rel)
                            < (z_boundary_y - z_safety_margin_app2)
                        ]

                        if len(valid_peaks_indices_app2) >= min_ticks_app2:
                            valid_peaks_rel_app2 = peaks_app2_rel[
                                valid_peaks_indices_app2
                            ]
                            # Refine these raw peaks
                            valid_refined_peaks_rel_app2 = refine_peaks(
                                profile_app2,
                                valid_peaks_rel_app2,
                                window_size=refine_window_size,
                            )

                            diffs = np.diff(valid_refined_peaks_rel_app2)
                            if len(diffs) > 0:  # Need at least 2 diffs for mean/std
                                mean_diff = np.mean(diffs)
                                std_diff = np.std(diffs)
                                if (
                                    mean_diff > 0
                                    and std_diff < spacing_std_thresh_app2 * mean_diff
                                ):
                                    abs_x_pos = start_x_app2 + line_x_strip
                                    candidates.append(
                                        {
                                            "x_position": abs_x_pos,
                                            "y_start_abs": data_top
                                            + np.min(valid_refined_peaks_rel_app2),
                                            "y_end_abs": min(
                                                z_boundary_y - z_safety_margin_app2,
                                                data_top
                                                + np.max(valid_refined_peaks_rel_app2),
                                            ),
                                            "tick_count": len(
                                                valid_refined_peaks_rel_app2
                                            ),
                                            "mean_spacing": mean_diff,
                                            "tick_positions_abs": valid_refined_peaks_rel_app2
                                            + data_top,
                                            "z_boundary_abs": z_boundary_y,
                                            "match_score": score_base_app2,
                                            "method": "Approach 2",
                                        }
                                    )
                                    print(
                                        f"Approach 2: Found candidate pip at x={abs_x_pos} with {len(valid_refined_peaks_rel_app2)} ticks."
                                    )

    # --- Final Ranking ---
    if not candidates:
        print("\nNo calibration pip candidates found by any approach.")
        return None

    for cand in candidates:
        distance = abs(cand["x_position"] - approx_x_position)
        # Proximity score: 1 if at click, 0 if at max_distance or further
        cand["proximity_score"] = max(0, 1 - (distance / max_dist_click))
        cand["final_score"] = cand["match_score"] * (
            (1 - ranking_prox_weight) + ranking_prox_weight * cand["proximity_score"]
        )

    sorted_candidates = sorted(candidates, key=lambda x: x["final_score"], reverse=True)
    best_pip = sorted_candidates[0]

    print(
        f"\nBest calibration pip selected (Method: {best_pip['method']}) at x={best_pip['x_position']}"
    )
    print(
        f"  Distance from user click: {abs(best_pip['x_position'] - approx_x_position)} px"
    )
    print(
        f"  Number of tick marks: {best_pip['tick_count']}, Mean spacing: {best_pip['mean_spacing']:.2f} px"
    )
    print(f"  Final Score: {best_pip['final_score']:.3f}")

    # Adjust returned dict keys to match original script if needed for visualization
    final_pip_dict = {
        "x_position": best_pip["x_position"],
        "y_start": best_pip["y_start_abs"],
        "y_end": best_pip["y_end_abs"],
        "tick_count": best_pip["tick_count"],
        "mean_spacing": best_pip["mean_spacing"],
        "tick_positions": best_pip["tick_positions_abs"],
        "z_boundary": best_pip["z_boundary_abs"],
        "match_score": best_pip["final_score"],
        "method": best_pip["method"],
    }
    return final_pip_dict
