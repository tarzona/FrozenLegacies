import numpy as np
import cv2
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import os
import sys
from skimage.transform import rotate

# Add the parent directory to sys.path to allow imports from sibling modules
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from functions.utils import save_plot, get_param


def detect_grid_lines_and_dotted(frame, config, qa_plot_path=None, ref_row_for_qa=None):
    """
    Enhanced grid line detection that handles tilted grids, enforces known physical grid spacing,
    and dynamically determines grid range based on signal extent.
    """
    h, w = frame.shape
    processing_params = config.get("processing_params", {})
    physical_params = config.get("physical_params", {})
    debug_images = {"original": frame.copy()}

    # STEP 1: Detect and correct grid tilt
    clip_limit = get_param(config, "processing_params", "grid_enhance_clip_limit", 2.0)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    enhanced = clahe.apply(frame)
    debug_images["enhanced"] = enhanced.copy()

    low_threshold = get_param(
        config, "processing_params", "grid_canny_low_threshold", 30
    )
    high_threshold = get_param(
        config, "processing_params", "grid_canny_high_threshold", 100
    )
    edges = cv2.Canny(enhanced, low_threshold, high_threshold)

    lines = cv2.HoughLines(edges, 1, np.pi / 180, 100)
    horizontal_angles = []
    if lines is not None:
        for line in lines:
            rho, theta = line[0]
            if abs(theta - np.pi / 2) < np.pi / 18:  # Within ±10 degrees of horizontal
                horizontal_angles.append(theta - np.pi / 2)

    tilt_angle = 0
    if horizontal_angles:
        tilt_angle = np.median(horizontal_angles) * 180 / np.pi
        print(f"Detected grid tilt angle: {tilt_angle:.2f} degrees")

    if abs(tilt_angle) > 0.5:  # Only correct if tilt is more than 0.5 degrees
        rotated_frame = rotate(frame, tilt_angle, preserve_range=True).astype(np.uint8)
        debug_images["rotated"] = rotated_frame.copy()
        enhanced = clahe.apply(rotated_frame)
        debug_images["enhanced_rotated"] = enhanced.copy()
    else:
        rotated_frame = frame.copy()
        print("No significant grid tilt detected")

    # STEP 2: Detect signal extent
    _, signal_mask = cv2.threshold(rotated_frame, 150, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(
        signal_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    signal_x_min, signal_y_min = w, h
    signal_x_max, signal_y_max = 0, 0
    for contour in contours:
        if cv2.contourArea(contour) > 100:  # Filter out small noise
            x, y, width, height = cv2.boundingRect(contour)
            signal_x_min = min(signal_x_min, x)
            signal_y_min = min(signal_y_min, y)
            signal_x_max = max(signal_x_max, x + width)
            signal_y_max = max(signal_y_max, y + height)

    if signal_x_min >= signal_x_max or signal_y_min >= signal_y_max:
        signal_x_min, signal_y_min = 0, 0
        signal_x_max, signal_y_max = w, h

    margin_x = get_param(
        config, "processing_params", "signal_extent_margin_frac_x", 0.05
    )
    margin_y = get_param(
        config, "processing_params", "signal_extent_margin_frac_y", 0.05
    )
    margin_x_px = int(w * margin_x)
    margin_y_px = int(h * margin_y)
    signal_x_min = max(0, signal_x_min - margin_x_px)
    signal_y_min = max(0, signal_y_min - margin_y_px)
    signal_x_max = min(w, signal_x_max + margin_x_px)
    signal_y_max = min(h, signal_y_max + margin_y_px)

    # STEP 3: Multi-scale preprocessing to enhance grid lines
    blur_size = get_param(config, "processing_params", "grid_blur_size", 3)
    blurred = cv2.GaussianBlur(enhanced, (blur_size, blur_size), 0)

    scales = get_param(config, "processing_params", "grid_scales", [1.0, 0.75, 0.5])
    multi_scale_binaries = []

    for scale in scales:
        if scale != 1.0:
            scaled_img = cv2.resize(
                blurred, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR
            )
        else:
            scaled_img = blurred.copy()

        try:
            block_size = get_param(
                config, "processing_params", "grid_adaptive_thresh_blocksize", 51
            )
            if block_size % 2 == 0:
                block_size += 1  # Must be odd
            C = get_param(config, "processing_params", "grid_adaptive_thresh_C", -10)

            thresh1 = cv2.adaptiveThreshold(
                scaled_img,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                block_size,
                C,
            )
            thresh2 = cv2.adaptiveThreshold(
                scaled_img,
                255,
                cv2.ADAPTIVE_THRESH_MEAN_C,
                cv2.THRESH_BINARY,
                block_size,
                C - 5,
            )
            thresh_combined = cv2.bitwise_or(thresh1, thresh2)
            edges = cv2.Canny(scaled_img, low_threshold, high_threshold)
            binary = cv2.bitwise_or(thresh_combined, edges)

            if scale != 1.0:
                binary = cv2.resize(
                    binary,
                    (rotated_frame.shape[1], rotated_frame.shape[0]),
                    interpolation=cv2.INTER_LINEAR,
                )

            multi_scale_binaries.append(binary)
        except Exception as e:
            print(f"Error during preprocessing at scale {scale}: {e}")

    if multi_scale_binaries:
        combined_binary = multi_scale_binaries[0].copy()
        for binary in multi_scale_binaries[1:]:
            combined_binary = cv2.bitwise_or(combined_binary, binary)
    else:
        binary_thresh = get_param(
            config, "processing_params", "grid_binary_thresh", 190
        )
        _, combined_binary = cv2.threshold(
            enhanced, binary_thresh, 255, cv2.THRESH_BINARY
        )

    debug_images["binary"] = combined_binary.copy()

    # STEP 4: Apply Gabor filters to enhance grid lines
    h_gabor = apply_gabor_filter(enhanced, 0, config)  # 0 degrees for horizontal
    v_gabor = apply_gabor_filter(enhanced, 90, config)  # 90 degrees for vertical

    _, h_binary = cv2.threshold(h_gabor, 30, 255, cv2.THRESH_BINARY)
    _, v_binary = cv2.threshold(v_gabor, 30, 255, cv2.THRESH_BINARY)

    combined_binary = cv2.bitwise_or(combined_binary, h_binary)
    combined_binary = cv2.bitwise_or(combined_binary, v_binary)
    debug_images["gabor"] = cv2.bitwise_or(h_binary, v_binary)

    # STEP 5: Enhance horizontal and vertical lines separately
    kernel_size = get_param(config, "processing_params", "grid_morph_kernel_size", 5)
    h_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (kernel_size, 1)
    )  # Horizontal kernel
    v_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (1, kernel_size)
    )  # Vertical kernel

    h_lines = cv2.morphologyEx(combined_binary, cv2.MORPH_OPEN, h_kernel, iterations=1)
    v_lines = cv2.morphologyEx(combined_binary, cv2.MORPH_OPEN, v_kernel, iterations=1)

    h_lines = cv2.dilate(h_lines, h_kernel, iterations=1)
    v_lines = cv2.dilate(v_lines, v_kernel, iterations=1)

    debug_images["h_lines"] = h_lines.copy()
    debug_images["v_lines"] = v_lines.copy()

    # STEP 6: Detect grid lines using multiple methods
    h_peaks = []
    v_peaks = []

    # Method 1: Line Segment Detector (LSD)
    try:
        lsd = cv2.createLineSegmentDetector(0)
        h_lines_lsd = lsd.detect(h_lines)[0]
        if h_lines_lsd is not None:
            for line in h_lines_lsd:
                x1, y1, x2, y2 = line[0]
                if abs(y2 - y1) < 10 and abs(x2 - x1) > rotated_frame.shape[1] / 5:
                    h_peaks.append(int((y1 + y2) / 2))  # Average y-coordinate

        v_lines_lsd = lsd.detect(v_lines)[0]
        if v_lines_lsd is not None:
            for line in v_lines_lsd:
                x1, y1, x2, y2 = line[0]
                if abs(x2 - x1) < 10 and abs(y2 - y1) > rotated_frame.shape[0] / 5:
                    v_peaks.append(int((x1 + x2) / 2))  # Average x-coordinate
    except Exception as e:
        print(f"Error during LSD detection: {e}. Falling back to Hough transform.")

    # Method 2: Hough transform (as fallback)
    if len(h_peaks) < 3 or len(v_peaks) < 3:
        try:
            min_line_length = get_param(
                config,
                "processing_params",
                "grid_min_line_length",
                rotated_frame.shape[1] // 4,
            )
            max_line_gap = get_param(
                config, "processing_params", "grid_max_line_gap", 20
            )
            hough_threshold = get_param(
                config, "processing_params", "grid_hough_threshold", 30
            )

            h_lines_hough = cv2.HoughLinesP(
                h_lines,
                rho=1,
                theta=np.pi / 180,
                threshold=hough_threshold,
                minLineLength=min_line_length,
                maxLineGap=max_line_gap,
            )
            if h_lines_hough is not None:
                for line in h_lines_hough:
                    x1, y1, x2, y2 = line[0]
                    if abs(y2 - y1) < 10:
                        h_peaks.append(int((y1 + y2) / 2))

            min_line_length_v = get_param(
                config,
                "processing_params",
                "grid_min_line_length_v",
                rotated_frame.shape[0] // 4,
            )
            v_lines_hough = cv2.HoughLinesP(
                v_lines,
                rho=1,
                theta=np.pi / 180,
                threshold=hough_threshold,
                minLineLength=min_line_length_v,
                maxLineGap=max_line_gap,
            )
            if v_lines_hough is not None:
                for line in v_lines_hough:
                    x1, y1, x2, y2 = line[0]
                    if abs(x2 - x1) < 10:
                        v_peaks.append(int((x1 + x2) / 2))
        except Exception as e:
            print(f"Error during Hough transform: {e}")

    # Method 3: Projection-based detection
    h_peaks_proj = detect_grid_lines_projection(h_lines, True, config)
    v_peaks_proj = detect_grid_lines_projection(v_lines, False, config)

    h_peaks.extend(h_peaks_proj)
    v_peaks.extend(v_peaks_proj)

    h_peaks_clustered = cluster_nearby_lines(h_peaks, 10)
    v_peaks_clustered = cluster_nearby_lines(v_peaks, 10)

    # STEP 7: Generate enforced grid based on physical parameters
    y_ref_dB = get_param(config, "physical_params", "y_ref_dB", -60)
    y_major_dB = get_param(config, "physical_params", "y_major_dB", 10)
    y_minor_per_major = get_param(config, "physical_params", "y_minor_per_major", 4)

    x_range_factor = get_param(config, "physical_params", "x_range_factor", 10)
    x_major_us = get_param(config, "physical_params", "x_major_us", 3)
    x_range_us = x_range_factor * x_major_us

    y_range_factor = get_param(config, "physical_params", "y_range_factor", 3)
    y_range_dB = y_range_factor * y_major_dB

    x_minor_per_major = get_param(config, "physical_params", "x_minor_per_major", 4)

    # Calculate expected grid spacing in pixels
    expected_y_major_count = y_range_dB / y_major_dB
    expected_x_major_count = x_range_us / x_major_us

    signal_width = signal_x_max - signal_x_min
    signal_height = signal_y_max - signal_y_min

    expected_px_per_y_major = signal_height / expected_y_major_count
    expected_px_per_x_major = signal_width / expected_x_major_count

    # Generate enforced grid with reference line as anchor
    if ref_row_for_qa is not None:
        h_peaks_enforced = generate_enforced_grid(
            ref_row_for_qa, expected_px_per_y_major, rotated_frame.shape[0], True
        )
        tx_anchor = signal_x_min  # Use left edge of signal as anchor
        v_peaks_enforced = generate_enforced_grid(
            tx_anchor, expected_px_per_x_major, rotated_frame.shape[1], False
        )

        h_peaks_final = combine_detected_and_enforced(
            h_peaks_clustered, h_peaks_enforced, 15
        )
        v_peaks_final = combine_detected_and_enforced(
            v_peaks_clustered, v_peaks_enforced, 15
        )
    else:
        h_peaks_final = h_peaks_clustered
        v_peaks_final = v_peaks_clustered

    # Generate minor grid lines
    h_minor_peaks = generate_minor_grid_lines(h_peaks_final, y_minor_per_major)
    v_minor_peaks = generate_minor_grid_lines(v_peaks_final, x_minor_per_major)

    h_peaks_final = np.array(sorted(h_peaks_final))
    v_peaks_final = np.array(sorted(v_peaks_final))
    h_minor_peaks = np.array(sorted(h_minor_peaks))
    v_minor_peaks = np.array(sorted(v_minor_peaks))

    print(
        f"Detected {len(h_peaks_final)} horizontal major peaks and {len(h_minor_peaks)} minor peaks."
    )
    print(
        f"Detected {len(v_peaks_final)} vertical major peaks and {len(v_minor_peaks)} minor peaks."
    )

    # Create QA visualization if requested
    if qa_plot_path is not None:
        plt.figure(figsize=(15, 12))

        # Plot original image
        plt.subplot(2, 2, 1)
        plt.imshow(debug_images["original"], cmap="gray")
        plt.title(
            "Original Image"
            + (f" (Tilt: {tilt_angle:.2f}°)" if abs(tilt_angle) > 0.5 else "")
        )
        plt.axis("on")

        # Plot enhanced binary result
        plt.subplot(2, 2, 2)
        if "gabor" in debug_images:
            plt.imshow(debug_images["gabor"], cmap="gray")
            plt.title("Gabor Filtered Image")
        else:
            plt.imshow(debug_images.get("binary", combined_binary), cmap="gray")
            plt.title("Enhanced Binary Image")
        plt.axis("on")

        # Plot detected lines overlay
        plt.subplot(2, 2, 3)
        plt.imshow(debug_images["original"], cmap="gray")
        plotted_labels = set()  # To avoid duplicate legend entries

        # Plot detected horizontal major lines
        for y in h_peaks_final:
            label = "H Major Grid Line"
            unique_label = label if label not in plotted_labels else ""
            plt.axhline(
                y,
                color="magenta",
                linestyle="-",
                linewidth=1.0,
                alpha=0.6,
                label=unique_label,
            )
            plotted_labels.add(label)

        # Plot detected horizontal minor lines
        for y in h_minor_peaks:
            label = "H Minor Grid Line"
            unique_label = label if label not in plotted_labels else ""
            plt.axhline(
                y,
                color="magenta",
                linestyle=":",
                linewidth=0.8,
                alpha=0.4,
                label=unique_label,
            )
            plotted_labels.add(label)

        # Plot detected vertical major lines
        for x in v_peaks_final:
            label = "V Major Grid Line"
            unique_label = label if label not in plotted_labels else ""
            plt.axvline(
                x,
                color="cyan",
                linestyle="-",
                linewidth=1.0,
                alpha=0.6,
                label=unique_label,
            )
            plotted_labels.add(label)

        # Plot detected vertical minor lines
        for x in v_minor_peaks:
            label = "V Minor Grid Line"
            unique_label = label if label not in plotted_labels else ""
            plt.axvline(
                x,
                color="cyan",
                linestyle=":",
                linewidth=0.8,
                alpha=0.4,
                label=unique_label,
            )
            plotted_labels.add(label)

        # Highlight the identified reference line if provided
        if ref_row_for_qa is not None:
            label = f"Ref Line ({y_ref_dB} dB)"
            unique_label = label if label not in plotted_labels else ""
            plt.axhline(
                ref_row_for_qa,
                color="lime",
                linestyle="--",
                linewidth=2.5,
                label=unique_label,
                zorder=5,
            )
            plotted_labels.add(label)

        plt.title("Detected Grid Lines")
        plt.ylim(rotated_frame.shape[0], 0)  # Image coordinates
        plt.xlim(0, rotated_frame.shape[1])

        # Plot final combined view
        plt.subplot(2, 2, 4)
        plt.imshow(debug_images["original"], cmap="gray")

        # Draw major and minor grid lines
        for y in h_peaks_final:
            plt.axhline(y, color="magenta", linestyle="-", alpha=0.6, linewidth=1.0)
        for x in v_peaks_final:
            plt.axvline(x, color="cyan", linestyle="-", alpha=0.6, linewidth=1.0)
        for y in h_minor_peaks:
            plt.axhline(y, color="magenta", linestyle=":", alpha=0.4, linewidth=0.8)
        for x in v_minor_peaks:
            plt.axvline(x, color="cyan", linestyle=":", alpha=0.4, linewidth=0.8)

        # Highlight reference line
        if ref_row_for_qa is not None:
            plt.axhline(ref_row_for_qa, color="lime", linestyle="--", linewidth=2.5)

        plt.title("Grid Line Detection QA (Tilt Corrected)")
        plt.ylim(rotated_frame.shape[0], 0)  # Image coordinates
        plt.xlim(0, rotated_frame.shape[1])

        # Add legend to the final plot
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys(), loc="upper right", fontsize=8)

        plt.tight_layout()

        # Get plot DPI from config or use default
        output_config = config.get("output", {})
        dpi = output_config.get("plot_dpi", 150)
        save_plot(plt.gcf(), os.path.basename(qa_plot_path), dpi=dpi)

    # Return detected peaks for grid interpolation
    return h_peaks_final, v_peaks_final, h_minor_peaks, v_minor_peaks


def apply_gabor_filter(image, orientation, config):
    """Apply Gabor filter to enhance grid lines at specified orientation."""
    ksize = get_param(config, "processing_params", "gabor_ksize", 31)
    sigma = get_param(config, "processing_params", "gabor_sigma", 4.0)
    lambda_val = get_param(config, "processing_params", "gabor_lambda", 10.0)
    gamma = get_param(config, "processing_params", "gabor_gamma", 0.5)
    psi = 0  # Phase offset

    # Convert orientation to radians
    theta = np.radians(orientation)

    # Create Gabor kernel
    kernel = cv2.getGaborKernel(
        (ksize, ksize), sigma, theta, lambda_val, gamma, psi, ktype=cv2.CV_32F
    )

    # Normalize kernel
    kernel /= 1.5 * kernel.sum()

    # Apply filter
    filtered = cv2.filter2D(image, cv2.CV_8UC3, kernel)
    return filtered


def detect_grid_lines_projection(binary_img, is_horizontal, config):
    """Detect grid lines using projection profiles."""
    smooth_sigma = get_param(config, "processing_params", "grid_proj_smooth_sigma", 5)
    height_percentile = get_param(
        config, "processing_params", "grid_peak_height_percentile", 80
    )
    min_distance = get_param(
        config, "processing_params", "grid_peak_min_distance_px", 15
    )
    prominence_factor = get_param(
        config, "processing_params", "grid_peak_prominence_std_factor", 0.4
    )

    if is_horizontal:
        # Detect horizontal lines using row-wise projection
        projection = np.sum(binary_img, axis=1)
        smoothed = gaussian_filter1d(projection, sigma=smooth_sigma)
        height_thresh = np.percentile(smoothed, height_percentile)

        try:
            peaks, _ = find_peaks(
                smoothed,
                height=height_thresh,
                distance=min_distance,
                prominence=np.std(smoothed) * prominence_factor,
            )
        except Exception as e:
            print(f"Error during peak finding: {e}")
            peaks = np.array([])
    else:
        # Detect vertical lines using column-wise projection
        projection = np.sum(binary_img, axis=0)
        smoothed = gaussian_filter1d(projection, sigma=smooth_sigma)
        height_thresh = np.percentile(smoothed, height_percentile)

        try:
            peaks, _ = find_peaks(
                smoothed,
                height=height_thresh,
                distance=min_distance,
                prominence=np.std(smoothed) * prominence_factor,
            )
        except Exception as e:
            print(f"Error during peak finding: {e}")
            peaks = np.array([])

    return peaks.tolist()


def cluster_nearby_lines(lines, threshold):
    """Group lines that are close to each other and return the average position."""
    if not lines:
        return []

    # Sort lines
    sorted_lines = sorted(lines)
    clusters = []
    current_cluster = [sorted_lines[0]]

    # Group lines that are within threshold distance
    for i in range(1, len(sorted_lines)):
        if sorted_lines[i] - sorted_lines[i - 1] <= threshold:
            current_cluster.append(sorted_lines[i])
        else:
            # Calculate average position for current cluster
            clusters.append(int(np.mean(current_cluster)))
            current_cluster = [sorted_lines[i]]

    # Add the last cluster
    if current_cluster:
        clusters.append(int(np.mean(current_cluster)))

    return clusters


def generate_enforced_grid(anchor, spacing, axis_length, is_y_axis):
    """Generate grid lines with enforced regular spacing based on physical parameters."""
    if spacing <= 0:
        return []

    # Generate lines outward from anchor
    lines = [anchor]

    # Generate lines in negative direction (up/left)
    pos = anchor - spacing
    while pos > axis_length * 0.2:  # Stop near edge
        lines.append(pos)
        pos -= spacing

    # Generate lines in positive direction (down/right)
    pos = anchor + spacing
    while pos < axis_length * 0.8:  # Stop near edge
        lines.append(pos)
        pos += spacing

    return sorted([int(line) for line in lines])


def generate_minor_grid_lines(major_lines, minor_per_major):
    """Generate minor grid lines between major grid lines."""
    minor_lines = []
    if len(major_lines) < 2:
        return minor_lines

    # Sort major lines
    sorted_lines = sorted(major_lines)

    # Generate minor lines between each pair of major lines
    for i in range(len(sorted_lines) - 1):
        start = sorted_lines[i]
        end = sorted_lines[i + 1]
        spacing = (end - start) / (minor_per_major + 1)
        for j in range(1, minor_per_major + 1):
            minor_pos = int(start + j * spacing)
            minor_lines.append(minor_pos)

    return minor_lines


def combine_detected_and_enforced(detected_lines, enforced_lines, tolerance):
    """Combine detected and enforced grid lines, prioritizing detected lines when close."""
    if not detected_lines:
        return enforced_lines
    if not enforced_lines:
        return detected_lines

    final_lines = []

    # For each enforced line, check if there's a detected line nearby
    for enforced in enforced_lines:
        # Find closest detected line
        closest_detected = min(detected_lines, key=lambda x: abs(x - enforced))

        # If a detected line is close enough, use it instead of the enforced line
        if abs(closest_detected - enforced) <= tolerance:
            if closest_detected not in final_lines:
                final_lines.append(closest_detected)
        else:
            # Otherwise use the enforced line
            final_lines.append(enforced)

    # Add any detected lines that aren't close to enforced lines
    for detected in detected_lines:
        if all(abs(detected - enforced) > tolerance for enforced in enforced_lines):
            final_lines.append(detected)

    return sorted(list(set(final_lines)))


def find_reference_line_blackhat(frame, base_filename, frame_idx, config):
    """
    Finds the reference line using black hat morphology, adaptive thresholding,
    median filtering, and peak scoring on the horizontal projection.
    """
    h, w = frame.shape

    # Get band fractions with fallbacks
    band_frac = get_param(
        config, "processing_params", "ref_line_signal_band_frac", [0.57, 0.65]
    )
    band_start = int(h * band_frac[0])
    band_end = int(h * band_frac[1])

    if band_start >= band_end or band_start >= h or band_end <= 0:
        print(
            f"Error: Invalid reference line search band {band_start}-{band_end}. Frame height {h}. Using fallback."
        )
        return int(h * 0.95)  # Fallback near bottom

    search_region = frame[band_start:band_end, :]
    if search_region.size == 0:
        print(
            f"Error: Reference line search region is empty ({band_start}-{band_end}). Using fallback."
        )
        return int(h * 0.95)  # Fallback near bottom

    print(f"Searching for reference line in region y={band_start} to y={band_end}")

    # Black Hat Transform
    kernel_size = get_param(
        config, "processing_params", "ref_line_blackhat_kernel_size", 51
    )
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, 1))
    blackhat = cv2.morphologyEx(search_region, cv2.MORPH_BLACKHAT, kernel)

    # Adaptive Thresholding
    block_size = get_param(
        config, "processing_params", "ref_line_adaptive_thresh_blocksize", 51
    )
    if block_size % 2 == 0:
        block_size += 1
    C = get_param(config, "processing_params", "ref_line_adaptive_thresh_C", -8)

    try:
        thresh = cv2.adaptiveThreshold(
            blackhat, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, block_size, C
        )
    except cv2.error as e:
        print(f"Error during adaptive thresholding: {e}. Using simple threshold.")
        _, thresh = cv2.threshold(blackhat, 30, 255, cv2.THRESH_BINARY)

    # Median Filtering
    median_ksize = get_param(
        config, "processing_params", "ref_line_median_filter_ksize", 3
    )
    if median_ksize % 2 == 0:
        median_ksize += 1  # Ensure odd
    if median_ksize > 1:
        thresh_filtered = cv2.medianBlur(thresh, median_ksize)
        print(f"Applied median filter with ksize={median_ksize}")
    else:
        thresh_filtered = thresh  # Skip if ksize is 1 or less

    # Horizontal Projection of Filtered Thresholded Image
    h_proj = np.sum(thresh_filtered, axis=1) / 255.0  # Sum bright pixels along rows
    if len(h_proj) == 0:
        print("Error: Horizontal projection is empty.")
        return int(h * 0.95)  # Fallback

    # Peak Detection in Projection
    try:
        max_proj_val = np.max(h_proj) if h_proj.size > 0 else 0
        min_peak_height_factor = get_param(
            config, "processing_params", "ref_line_min_peak_height_factor", 0.15
        )
        min_peak_height = max(1, max_proj_val * min_peak_height_factor)
        peak_distance = get_param(
            config, "processing_params", "ref_line_max_peak_distance_px", 30
        )
        h_proj_smooth = gaussian_filter1d(h_proj, sigma=1)  # Smooth slightly
        peaks, properties = find_peaks(
            h_proj_smooth, height=min_peak_height, distance=peak_distance, prominence=1
        )
    except Exception as e:
        print(f"Error during peak finding in projection: {e}")
        peaks = np.array([])
        properties = {}  # Ensure properties exists

    ref_line = None
    selected_peak_offset = None  # Store the offset of the chosen peak for QA plot

    if len(peaks) > 0:
        # Peak Location Filtering & Scoring
        proj_len = len(h_proj_smooth)
        peak_search_band_frac = get_param(
            config, "processing_params", "ref_line_peak_search_band_frac", [0.5, 1.0]
        )
        peak_search_start = int(proj_len * peak_search_band_frac[0])
        peak_search_end = int(proj_len * peak_search_band_frac[1])

        # Filter peaks to keep only those within the desired lower band
        valid_peak_indices_in_peaks = np.where(
            (peaks >= peak_search_start) & (peaks < peak_search_end)
        )[0]

        if len(valid_peak_indices_in_peaks) > 0:
            valid_peaks_offsets = peaks[valid_peak_indices_in_peaks]
            valid_peaks_heights = h_proj_smooth[valid_peaks_offsets]

            # Get prominences if available, otherwise use height as proxy
            if "prominences" in properties:
                if len(properties["prominences"]) == len(peaks):
                    valid_peaks_prominences = properties["prominences"][
                        valid_peak_indices_in_peaks
                    ]
                else:
                    print(
                        "Warning: Mismatch between peaks and prominences length. Using height for scoring."
                    )
                    valid_peaks_prominences = valid_peaks_heights  # Fallback to height
            else:
                valid_peaks_prominences = (
                    valid_peaks_heights  # Fallback if prominences not calculated
                )

            # Normalize heights and offsets within the valid range for scoring
            min_h, max_h = np.min(valid_peaks_heights), np.max(valid_peaks_heights)
            norm_heights = (valid_peaks_heights - min_h) / (max_h - min_h + 1e-6)

            # Normalize offsets so that higher offset (lower in image) gets higher score
            min_o, max_o = np.min(valid_peaks_offsets), np.max(valid_peaks_offsets)

            # Handle case where min_o == max_o (only one valid peak)
            if max_o - min_o > 0:
                norm_offsets = (valid_peaks_offsets - min_o) / (max_o - min_o + 1e-6)
            else:
                norm_offsets = np.ones_like(
                    valid_peaks_offsets
                )  # Score 1 if only one peak

            # Combine scores: weight height and location (offset)
            location_weight = get_param(
                config, "processing_params", "ref_line_peak_score_location_weight", 0.7
            )
            height_weight = 1.0 - location_weight
            scores = height_weight * norm_heights + location_weight * norm_offsets

            best_score_index = np.argmax(scores)
            selected_peak_offset = valid_peaks_offsets[
                best_score_index
            ]  # Offset within the search region
            ref_line = (
                band_start + selected_peak_offset
            )  # Absolute row position in the image

            print(
                f"Initial peaks at offsets: {peaks}. Filtered offsets: {valid_peaks_offsets}"
            )
            print(
                f"Scores: {np.round(scores, 2)}. Selected peak offset: {selected_peak_offset} (abs row: {ref_line}) with score {scores[best_score_index]:.2f}"
            )
        else:
            print("No peaks found in the valid lower projection band. Falling back.")
            ref_line = int(h * 0.95)  # Fallback
    else:
        print(
            "No reference line peaks found using blackhat and projection. Falling back."
        )
        ref_line = int(h * 0.95)

    # Optional QA Saving
    save_intermediate_qa = get_param(
        config, "processing_params", "ref_line_save_intermediate_qa", True
    )
    if save_intermediate_qa:
        output_dir = get_param(config, "output", "output_dir", "output")
        qa_output_dir = os.path.join(output_dir, "ref_line_qa")
        os.makedirs(qa_output_dir, exist_ok=True)
        qa_base = f"{qa_output_dir}/{base_filename}_frame{frame_idx:02d}"

        cv2.imwrite(f"{qa_base}_01_search_region.png", search_region)
        cv2.imwrite(f"{qa_base}_02_blackhat.png", blackhat)
        cv2.imwrite(f"{qa_base}_03_threshold_raw.png", thresh)
        if median_ksize > 1:
            cv2.imwrite(f"{qa_base}_04_threshold_filtered.png", thresh_filtered)

        # Plot Projection and Peaks
        plt.figure(figsize=(8, 4))
        plt.plot(h_proj_smooth, label="Smoothed Projection")
        plt.plot(peaks, h_proj_smooth[peaks], "x", label="All Detected Peaks")
        plt.axhline(
            min_peak_height,
            color="r",
            linestyle="--",
            label=f"Min Height ({min_peak_height:.1f})",
        )
        plt.axvspan(
            peak_search_start,
            peak_search_end,
            color="gray",
            alpha=0.2,
            label="Peak Search Band",
        )

        if ref_line is not None and selected_peak_offset is not None:
            # Plot marker only if the selected offset exists
            if selected_peak_offset in peaks:
                plt.plot(
                    selected_peak_offset,
                    h_proj_smooth[selected_peak_offset],
                    "o",
                    color="lime",
                    ms=8,
                    label=f"Selected Peak (Offset {selected_peak_offset})",
                )

        plt.title(f"Reference Line Horizontal Projection (Frame {frame_idx})")
        plt.xlabel("Row offset within search band")
        plt.ylabel("Summed Pixel Value (Normalized)")
        plt.legend(fontsize=8)
        plt.grid(True)
        plt.savefig(f"{qa_base}_05_projection.png", dpi=150)
        plt.close()

        print(f"Saved reference line intermediate QA images to {qa_output_dir}")

    # Ensure returned ref_line is an integer
    return int(round(ref_line)) if ref_line is not None else int(h * 0.95)


def interpolate_regular_grid(
    axis_length,
    detected_peaks,
    anchor,
    major_spacing_physical,
    minor_per_major,
    physical_range,
    is_y_axis,
    config,
):
    """
    Interpolates a regular grid based on the known physical parameters and anchor point.
    For A-scope radar, we expect:
    - X-axis: 3 μs per major division with 4 minor divisions (0.75 μs each)
    - Y-axis: 10 dB per major division with 4 minor divisions (2.5 dB each)
    """
    if axis_length <= 0:
        print(f"Error: Axis length ({axis_length}) is invalid.")
        return np.array([]), np.array([])

    # Default anchor if none provided
    if anchor is None:
        # Default anchor near start (X) or near expected ref line position (Y)
        anchor = axis_length * 0.15 if not is_y_axis else axis_length * 0.85
        print(
            f"Warning: Anchor not provided for {'Y' if is_y_axis else 'X'} axis, using default {anchor:.1f}."
        )

    # Use dynamically determined physical range
    if is_y_axis:
        # Y-axis range
        if physical_range <= 0:
            physical_range = 1 * 10
    else:
        # X-axis range
        if physical_range <= 0:
            physical_range = 10 * 3

    # Calculate expected number of major intervals based on physical parameters
    num_major_intervals = 0
    if major_spacing_physical > 0:
        num_major_intervals = physical_range / major_spacing_physical
    else:
        print(
            f"Warning: Invalid physical major spacing ({major_spacing_physical}). Using default intervals ({num_major_intervals})."
        )

    # Estimate usable length and expected pixel spacing
    usable_length_factor = 0.75 if is_y_axis else 0.80
    usable_length = axis_length * usable_length_factor
    expected_spacing_px = (
        usable_length / num_major_intervals
        if num_major_intervals > 0
        else axis_length / (6 if is_y_axis else 9)
    )

    # Generate regular grid lines outwards from the anchor
    major = [anchor]

    # Generate upwards/leftwards from anchor
    pos = anchor - expected_spacing_px
    while pos > axis_length * 0.02:  # Stop near edge
        major.append(pos)
        pos -= expected_spacing_px

    # Generate downwards/rightwards from anchor
    pos = anchor + expected_spacing_px
    while pos < axis_length * 0.98:  # Stop near edge
        major.append(pos)
        pos += expected_spacing_px

    major = np.sort(np.array(major))

    # Interpolate minor ticks between the major lines
    minor = []
    for i in range(len(major) - 1):
        spacing = major[i + 1] - major[i]  # Spacing between major lines
        for j in range(1, minor_per_major + 1):
            minor_pos = int(major[i] + j * spacing / (minor_per_major + 1))
            if 0 <= minor_pos < axis_length:  # Only add if within image bounds
                minor.append(minor_pos)

    print(
        f"Interpolated {len(major)} major lines and {len(minor)} minor lines for {'Y' if is_y_axis else 'X'} axis."
    )
    return major.astype(int), np.array(minor).astype(int)
