import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d, label
import matplotlib.pyplot as plt
import os
import sys

# Add the parent directory to sys.path to allow imports from sibling modules
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from functions.utils import save_plot


def mask_sprocket_holes(image, config):
    """Creates a mask to exclude sprocket holes and applies it."""
    height, width = image.shape
    mask = np.ones_like(image, dtype=np.uint8)

    # Get configuration parameters with fallbacks to avoid KeyError
    processing_params = config.get("processing_params", {})

    # Use get() with default values to avoid KeyError
    upper_frac = processing_params.get("sprocket_mask_upper_frac", 0.08)
    lower_frac = processing_params.get("sprocket_mask_lower_frac", 0.10)

    # Mask top and bottom margins
    upper_margin = int(height * upper_frac)
    lower_margin = int(height * lower_frac)
    mask[:upper_margin, :] = 0
    mask[-lower_margin:, :] = 0

    # Attempt to find sprocket holes based on brightness and geometry
    _, bright_thresh = cv2.threshold(image, 220, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(
        bright_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        # Heuristic conditions for sprocket holes (near edges, specific size range)
        if (
            (y < height * 0.2 or y + h > height * 0.8)  # Near top/bottom edges
            and (20 < w < 100)  # Width range
            and (10 < h < 50)  # Height range
        ):
            mask[y : y + h, x : x + w] = 0  # Mask out the detected contour area

    return cv2.bitwise_and(image, image, mask=mask), mask


def detect_ascope_frames(image, config, expected_frames=12):
    """Detects individual A-scope frame boundaries based on wide white gaps."""
    height, width = image.shape

    # Get configuration parameters with fallbacks
    processing_params = config.get("processing_params", {})

    # Get band fractions with fallbacks
    band_frac = processing_params.get("frame_detect_band_frac", [0.2, 0.8])

    # Analyze a central band to find gaps
    band_start = int(height * band_frac[0])
    band_end = int(height * band_frac[1])
    central_band = image[band_start:band_end, :]
    col_means = np.mean(central_band, axis=0)

    # Smooth the column means to identify broad peaks (gaps)
    smooth_sigma = processing_params.get("frame_detect_smooth_sigma", 20)
    smoothed = gaussian_filter1d(col_means, sigma=smooth_sigma)
    try:  # Avoid error if smoothed is constant
        normalized = (smoothed - np.min(smoothed)) / (
            np.max(smoothed) - np.min(smoothed)
        )
    except ZeroDivisionError:
        normalized = np.zeros_like(smoothed)

    # Find regions above the gap threshold
    gap_threshold = processing_params.get("frame_detect_gap_threshold", 0.90)
    gap_mask = normalized > gap_threshold
    labeled_gaps, num_gaps = label(gap_mask)

    gap_centers = []
    min_gap_width = processing_params.get("frame_detect_min_gap_width_px", 80)

    # Find center of sufficiently wide gaps
    for i in range(1, num_gaps + 1):
        indices = np.where(labeled_gaps == i)[0]
        if len(indices) >= min_gap_width:
            gap_centers.append(int(np.mean(indices)))

    gap_centers = sorted(gap_centers)

    # Fallback if not enough gaps detected
    if len(gap_centers) < min(
        3, expected_frames - 1
    ):  # Require at least 3 gaps if expected > 4
        print(
            f"Warning: Only {len(gap_centers)} gaps detected. Using estimated frame width."
        )
        # Estimate frame width, maybe excluding first/last partial frames if possible
        est_frame_width = width / expected_frames if expected_frames > 0 else width
        # Place gaps assuming equal spacing
        gap_centers = [
            int((i + 0.5) * est_frame_width) for i in range(expected_frames - 1)
        ]

    frames = []
    frame_edges = [0] + gap_centers + [width]
    min_frame_width = processing_params.get("frame_detect_min_frame_width_px", 200)

    # Define frame boundaries slightly inside the gaps
    for i in range(len(frame_edges) - 1):
        left, right = int(frame_edges[i]), int(frame_edges[i + 1])
        if right - left > min_frame_width:
            buffer = 5  # Small buffer inward from gap centers
            frames.append((max(0, left + buffer), min(width, right - buffer)))

    # Optional: Adjust first/last frame if they seem like partials (heuristic)
    if (
        frames and frames[0][0] < min_frame_width / 4
    ):  # If first frame starts very close to edge
        frames[0] = (0, frames[0][1])  # Start from edge
    if (
        frames and (width - frames[-1][1]) < min_frame_width / 4
    ):  # If last frame ends very close to edge
        frames[-1] = (frames[-1][0], width)  # End at edge

    print(f"Detected {len(frames)} potential frames.")
    return frames


def verify_frames_visually(image, frames, base_filename, config):
    """Saves an image visualizing the detected frame boundaries."""
    height, width = image.shape
    plt.figure(figsize=(20, 6))
    plt.imshow(image, cmap="gray")

    for i, (left, right) in enumerate(frames):
        plt.axvline(x=left, color="r", linestyle="-")
        plt.axvline(x=right, color="r", linestyle="-")
        plt.text(
            (left + right) / 2,
            height * 0.1,
            f"{i + 1}",
            color="yellow",
            fontsize=12,
            ha="center",
        )

    plt.title(f"Verification: {len(frames)} Detected Frames")
    plt.ylabel("Row (pixels)")
    plt.xlabel("Column (pixels)")

    # Get output configuration
    output_config = config.get("output", {})
    dpi = output_config.get("plot_dpi", 200)

    filename = f"{base_filename}_frame_verification.png"
    save_plot(plt.gcf(), filename, dpi=dpi)

    return frames
