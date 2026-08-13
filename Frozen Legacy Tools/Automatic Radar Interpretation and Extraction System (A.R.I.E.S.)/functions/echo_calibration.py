import matplotlib.pyplot as plt
import numpy as np
import cv2


def analyze_echo_points(image, surface_points, bed_points):
    """
    Analyze user-selected points to determine optimal detection parameters.

    Args:
        image: Full radar image
        surface_points: List of (x, y) tuples for surface echo points
        bed_points: List of (x, y) tuples for bed echo points

    Returns:
        dict: Optimized parameters for surface and bed detection
    """

    # Analyze surface echo characteristics
    surface_params = _analyze_echo_points_type(image, surface_points, "surface")

    # Analyze bed echo characteristics
    bed_params = _analyze_echo_points_type(image, bed_points, "bed")

    return {"surface_detection": surface_params, "bed_detection": bed_params}


def _analyze_echo_points_type(image, points, echo_type):
    """Analyze characteristics of selected echo points - SEARCH PARAMETERS ONLY."""

    if not points:
        return {}

    # Define analysis window around each point
    window_size = 15  # 15x15 pixel window around each point
    half_window = window_size // 2

    # Collect Y-positions of selected points for search parameter optimization
    y_positions = []

    for x, y in points:
        # Validate point is within image bounds
        if 0 <= y < image.shape[0] and 0 <= x < image.shape[1]:
            y_positions.append(y)

    # Calculate search parameters based on actual echo locations
    if y_positions:
        min_y = min(y_positions)
        max_y = max(y_positions)
        avg_y = np.mean(y_positions)
        y_spread = max_y - min_y if len(y_positions) > 1 else 50

        # Calculate search parameters based on echo type and actual positions
        if echo_type == "surface":
            # Surface echoes: search starts close to transmitter pulse
            search_offset = max(
                10, min(100, min_y - 20)
            )  # Start slightly above highest surface point
            search_depth = max(
                100, min(300, y_spread + 100)
            )  # Cover spread plus buffer
        else:  # bed
            # Bed echoes: search starts after surface region
            search_offset = max(
                100, min(400, avg_y - 50)
            )  # Start before average bed position
            search_depth = max(
                200, min(500, y_spread + 200)
            )  # Larger search window for bed

        # ONLY return search window parameters - no peak detection or enhancement params
        return {
            "search_start_offset_px": int(search_offset),
            "search_depth_px": int(search_depth),
        }

    return {}
