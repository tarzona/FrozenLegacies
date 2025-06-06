import numpy as np


def calculate_pixels_per_microsecond(mean_pip_spacing_px, pip_interval_us):
    """
    Calculates the spatial calibration factor in pixels per microsecond.

    This factor is essential for converting distances in pixels (on the Z-scope image)
    to time intervals in microseconds.

    Args:
        mean_pip_spacing_px (float): The average spacing between calibration tick marks
                                     on the Z-scope image, measured in pixels.
        pip_interval_us (float): The known time interval that each calibration pip
                                 spacing represents, in microseconds (e.g., 2 µs).

    Returns:
        float: The calibration factor in pixels per microsecond.
               Returns None if pip_interval_us is zero to prevent division by zero.

    Raises:
        ValueError: If mean_pip_spacing_px is not positive or pip_interval_us is not positive.
    """
    if mean_pip_spacing_px <= 0:
        raise ValueError("Mean pip spacing must be positive.")
    if pip_interval_us <= 0:
        raise ValueError("Pip interval must be positive.")

    return mean_pip_spacing_px / pip_interval_us


def convert_time_to_depth(
    one_way_travel_time_us,
    speed_of_light_mps,
    ice_relative_permittivity,
    firn_correction_m=0.0,
    surface_elevation_m=None,
):
    """
    Converts one-way radar travel time in microseconds to depth in meters through ice.

    Args:
        one_way_travel_time_us (float or np.ndarray): One-way travel time in microseconds.
        speed_of_light_mps (float): Speed of light in vacuum in m/s.
        ice_relative_permittivity (float): Relative permittivity of ice.
        firn_correction_m (float, optional): Firn correction in meters.
        surface_elevation_m (float, optional): Surface elevation in meters above WGS-84.
            If provided, returns elevation relative to WGS-84 instead of depth below surface.

    Returns:
        float or np.ndarray: Calculated depth in meters (or elevation if surface_elevation_m provided).
    """
    if speed_of_light_mps <= 0:
        raise ValueError("Speed of light must be positive.")
    if ice_relative_permittivity <= 0:
        raise ValueError("Ice relative permittivity must be positive.")

    # Convert one-way travel time from microseconds to seconds
    travel_time_s = one_way_travel_time_us * 1e-6

    # Calculate the velocity of the radar wave in ice
    velocity_in_ice_mps = speed_of_light_mps / np.sqrt(ice_relative_permittivity)

    # Calculate depth below surface
    depth_m = velocity_in_ice_mps * travel_time_s + firn_correction_m

    # If surface elevation is provided, convert depth to WGS-84 referenced elevation
    if surface_elevation_m is not None:
        return surface_elevation_m - depth_m  # Elevation decreases with depth
    else:
        return depth_m  # Return depth below surface


def convert_depth_to_time(
    depth_m,
    speed_of_light_mps,
    ice_relative_permittivity,
    firn_correction_m=0.0,
    surface_elevation_m=None,
):
    """
    Converts depth in meters through ice to one-way radar travel time in microseconds.

    Args:
        depth_m (float or np.ndarray): Depth in meters or elevation if surface_elevation_m provided.
        speed_of_light_mps (float): Speed of light in vacuum in m/s.
        ice_relative_permittivity (float): Relative permittivity of ice.
        firn_correction_m (float, optional): Firn correction in meters.
        surface_elevation_m (float, optional): Surface elevation in meters above WGS-84.
            If provided, depth_m is interpreted as elevation relative to WGS-84.

    Returns:
        float or np.ndarray: Calculated one-way travel time in microseconds.
    """
    if speed_of_light_mps <= 0:
        raise ValueError("Speed of light must be positive.")
    if ice_relative_permittivity <= 0:
        raise ValueError("Ice relative permittivity must be positive.")

    # If surface elevation is provided, convert elevation to depth
    if surface_elevation_m is not None:
        actual_depth_m = surface_elevation_m - depth_m
    else:
        actual_depth_m = depth_m

    # Adjust depth for firn correction
    depth_adjusted_m = actual_depth_m - firn_correction_m

    # Ensure adjusted depth is not negative
    if isinstance(depth_adjusted_m, np.ndarray):
        depth_adjusted_m[depth_adjusted_m < 0] = 0
    elif depth_adjusted_m < 0:
        depth_adjusted_m = 0

    velocity_in_ice_mps = speed_of_light_mps / np.sqrt(ice_relative_permittivity)

    if velocity_in_ice_mps == 0:
        return (
            0.0
            if not isinstance(depth_adjusted_m, np.ndarray)
            else np.zeros_like(depth_adjusted_m)
        )

    travel_time_s = depth_adjusted_m / velocity_in_ice_mps
    one_way_travel_time_us = travel_time_s * 1e6

    return one_way_travel_time_us
