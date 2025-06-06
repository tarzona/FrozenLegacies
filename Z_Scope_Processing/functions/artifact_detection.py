import numpy as np
import cv2
import matplotlib.pyplot as plt


def detect_film_artifact_boundaries(
    image,
    base_filename,
    top_exclude_ratio=0.05,
    bottom_exclude_ratio=0.05,
    gradient_smooth_kernel=15,
    gradient_threshold_factor=1.5,
    safety_margin=20,
    visualize=False,
):
    """
    Detect top and bottom boundaries to exclude camera film artifacts (sprocket holes, edge markings).

    Args:
        image (np.ndarray): 2D grayscale image array.
        base_filename (str): Base name for output files.
        top_exclude_ratio (float): Fraction of image height at top to exclude from search (default 0.05).
        bottom_exclude_ratio (float): Fraction of image height at bottom to exclude from search (default 0.05).
        gradient_smooth_kernel (int): Kernel size for smoothing gradient (default 15).
        gradient_threshold_factor (float): Multiplier for mean+std threshold (default 1.5).
        safety_margin (int): Pixels to add inside detected boundary to avoid artifacts (default 20).
        visualize (bool): If True, plot diagnostic figures.

    Returns:
        tuple: (top_boundary, bottom_boundary) pixel indices defining valid data region.
    """
    height, width = image.shape

    # Compute horizontal intensity profile (mean pixel intensity per row)
    horizontal_profile = np.mean(image, axis=1)

    # Normalize profile
    norm_profile = (horizontal_profile - np.min(horizontal_profile)) / (
        np.max(horizontal_profile) - np.min(horizontal_profile)
    )

    # Compute gradient magnitude
    gradient = np.abs(np.diff(norm_profile))

    # Smooth gradient vertically
    smooth_gradient = cv2.GaussianBlur(
        gradient.reshape(-1, 1), (gradient_smooth_kernel, 1), 0
    ).flatten()

    # Define search regions excluding top and bottom film artifact areas
    top_search_limit = int(height * top_exclude_ratio)
    bottom_search_start = int(height * (1 - bottom_exclude_ratio))

    # Search for top boundary edge in region below top excluded area
    top_region = smooth_gradient[
        top_search_limit : top_search_limit + int(height * 0.15)
    ]  # next 15% after excluded top
    top_threshold = np.mean(top_region) + gradient_threshold_factor * np.std(top_region)
    top_edges = np.where(top_region > top_threshold)[0]

    if len(top_edges) > 0:
        top_edge_idx = np.argmax(top_region[top_edges])
        top_boundary = top_search_limit + top_edges[top_edge_idx] + safety_margin
        if top_boundary > height:
            top_boundary = height
    else:
        top_boundary = top_search_limit + safety_margin

    # Search for bottom boundary edge in region above bottom excluded area
    bottom_region = smooth_gradient[
        bottom_search_start - int(height * 0.15) : bottom_search_start
    ]
    bottom_threshold = np.mean(bottom_region) + gradient_threshold_factor * np.std(
        bottom_region
    )
    bottom_edges = np.where(bottom_region > bottom_threshold)[0]

    if len(bottom_edges) > 0:
        bottom_edge_idx = np.argmax(bottom_region[bottom_edges])
        bottom_boundary = (
            bottom_search_start
            - int(height * 0.15)
            + bottom_edges[bottom_edge_idx]
            - safety_margin
        )
        if bottom_boundary < 0:
            bottom_boundary = 0
    else:
        bottom_boundary = bottom_search_start - safety_margin

    if visualize:
        plt.figure(figsize=(12, 8))
        plt.subplot(2, 1, 1)
        plt.imshow(image, cmap="gray")
        plt.axhline(
            y=top_boundary, color="r", linestyle="--", linewidth=2, label="Top Boundary"
        )
        plt.axhline(
            y=bottom_boundary,
            color="r",
            linestyle="--",
            linewidth=2,
            label="Bottom Boundary",
        )
        plt.title("Detected Film Artifact Boundaries")
        plt.legend()

        plt.subplot(2, 1, 2)
        plt.plot(norm_profile, label="Normalized Intensity Profile")
        plt.plot(np.arange(len(gradient)) + 0.5, gradient, label="Gradient")
        plt.plot(
            np.arange(len(smooth_gradient)) + 0.5,
            smooth_gradient,
            label="Smoothed Gradient",
        )
        plt.axvline(x=top_boundary, color="r", linestyle="--", label="Top Boundary")
        plt.axvline(
            x=bottom_boundary, color="r", linestyle="--", label="Bottom Boundary"
        )
        plt.legend()
        plt.title("Profile and Gradient Analysis")
        plt.tight_layout()
        plt.savefig(f"{base_filename}_film_artifact_boundaries.png", dpi=300)
        plt.close()

    return top_boundary, bottom_boundary


def detect_zscope_boundary(img, y_start, y_end):
    """
    Detect the Z-scope boundary between radar data and metadata text.

    Args:
        img (np.ndarray): 2D grayscale image (vertical slice).
        y_start (int): Starting y-coordinate for search.
        y_end (int): Ending y-coordinate for search.

    Returns:
        int: y-coordinate of detected boundary.
    """
    # Extract vertical profile (mean intensity per row)
    profile = np.mean(img[y_start:y_end, :], axis=1)

    # Calculate gradient magnitude
    gradient = np.abs(np.diff(profile))

    # Smooth gradient to avoid local maxima
    smooth_gradient = cv2.GaussianBlur(gradient.reshape(-1, 1), (31, 1), 0).flatten()

    # Focus on middle third of the search region
    middle_start = y_start + (y_end - y_start) * 1 // 3
    middle_end = y_start + (y_end - y_start) * 2 // 3
    middle_idx_start = middle_start - y_start
    middle_idx_end = middle_end - y_start

    # Threshold to find strong edges
    threshold = np.mean(smooth_gradient) + 2 * np.std(smooth_gradient)
    candidate_edges = np.where(
        smooth_gradient[middle_idx_start:middle_idx_end] > threshold
    )[0]

    if len(candidate_edges) > 0:
        max_grad_idx = (
            middle_idx_start
            + candidate_edges[
                np.argmax(smooth_gradient[middle_idx_start + candidate_edges])
            ]
        )
    else:
        max_grad_idx = np.argmax(smooth_gradient[middle_idx_start:]) + middle_idx_start

    # Apply safety offset to move boundary up slightly (above CBD)
    safety_offset = 50
    boundary = y_start + max_grad_idx - safety_offset

    return boundary
