# functions/data_export.py

import numpy as np
import csv
from pathlib import Path
from datetime import datetime


def export_ice_measurements(
    processor, output_filepath, include_metadata=True, coordinate_system="pixel"
):
    """
    Export ice surface and bed measurements from Z-scope radar analysis.

    This function exports the detected ice surface and bed depths, along with their
    corresponding one-way travel times, for each pixel location in the processed echogram.

    Args:
        processor (ZScopeProcessor): The processor instance containing analysis results
        output_filepath (str or Path): Path for the output CSV file
        include_metadata (bool): Whether to include metadata header in the output file
        coordinate_system (str): Coordinate system for X positions ("pixel" or "distance")
                               - "pixel": X values as pixel indices (0, 1, 2, ...)
                               - "distance": X values as distance if spatial scale is available

    Returns:
        bool: True if export was successful, False otherwise

    Raises:
        ValueError: If required processor data is missing
        IOError: If file cannot be written
    """

    # Validate input processor has required data
    required_attrs = [
        "detected_surface_y_abs",
        "detected_bed_y_abs",
        "pixels_per_microsecond",
        "transmitter_pulse_y_abs",
        "physics_constants",
        "base_filename",
    ]

    for attr in required_attrs:
        if not hasattr(processor, attr) or getattr(processor, attr) is None:
            raise ValueError(f"Processor missing required attribute: {attr}")

    # Get physical constants for ice
    # Wave velocity in ice: v = c / sqrt(dielectric_constant)
    # where c = speed of light in vacuum (m/s)
    c = processor.physics_constants.get("speed_of_light_vacuum_m_per_s", 299792458.0)
    dielectric_constant = processor.physics_constants.get(
        "ice_dielectric_constant", 3.15
    )

    # Calculate radar wave velocity in ice
    wave_velocity_ice = c / np.sqrt(dielectric_constant)  # m/s

    print(f"INFO: Using wave velocity in ice: {wave_velocity_ice:.0f} m/s")
    print(f"INFO: Using dielectric constant: {dielectric_constant}")

    # Get dimensions and create coordinate arrays
    surface_y_abs = processor.detected_surface_y_abs
    bed_y_abs = processor.detected_bed_y_abs
    n_pixels = len(surface_y_abs)

    # Create X coordinates based on coordinate system choice
    if coordinate_system == "pixel":
        x_coordinates = np.arange(n_pixels)
        x_unit = "pixel"
        x_description = "Pixel index (column number in echogram)"
    elif coordinate_system == "distance":
        # If processor has spatial calibration info, use it
        # Otherwise fall back to pixel indices
        if (
            hasattr(processor, "horizontal_resolution_m")
            and processor.horizontal_resolution_m is not None
        ):
            x_coordinates = np.arange(n_pixels) * processor.horizontal_resolution_m
            x_unit = "m"
            x_description = "Distance along track"
        else:
            print("WARNING: No spatial calibration available, using pixel coordinates")
            x_coordinates = np.arange(n_pixels)
            x_unit = "pixel"
            x_description = "Pixel index (column number in echogram)"
    else:
        raise ValueError(f"Unknown coordinate_system: {coordinate_system}")

    # Calculate two-way travel times (in microseconds) using calpip-based calibration like TERRA
    # Two-way travel time = (echo_y_abs - transmitter_pulse_y_abs) * us_per_pixel
    # where us_per_pixel is derived from calpip spacing (2 μs between calpip lines)
    
    # Use calpip-based calibration like TERRA (2 μs spacing between calibration pip lines)
    if hasattr(processor, 'calpip_pixel_distance') and processor.calpip_pixel_distance and processor.calpip_pixel_distance > 0:
        us_per_pixel = 2.0 / processor.calpip_pixel_distance  # TERRA's method: 2 μs between calpip lines
    else:
        # Fallback to old method if no calpip data available
        us_per_pixel = 1.0 / processor.pixels_per_microsecond
        print("WARNING: No calpip calibration available, using fallback pixels_per_microsecond")
    
    surface_travel_time_us = (
        surface_y_abs - processor.transmitter_pulse_y_abs
    ) * us_per_pixel
    bed_travel_time_us = (
        bed_y_abs - processor.transmitter_pulse_y_abs
    ) * us_per_pixel

    # Calculate depths (in meters) using two-way travel time like TERRA
    # For radar: depth = (two_way_travel_time / 2) * wave_velocity
    # Travel times are already two-way, so divide by 2 to get one-way, then multiply by velocity
    surface_depth_m = (surface_travel_time_us * 1e-6) / 2 * wave_velocity_ice
    bed_depth_m = (bed_travel_time_us * 1e-6) / 2 * wave_velocity_ice

    # Handle invalid/missing data
    # Set negative travel times to NaN (invalid echoes above transmitter pulse)
    surface_travel_time_us = np.where(
        surface_travel_time_us < 0, np.nan, surface_travel_time_us
    )
    bed_travel_time_us = np.where(bed_travel_time_us < 0, np.nan, bed_travel_time_us)
    surface_depth_m = np.where(surface_depth_m < 0, np.nan, surface_depth_m)
    bed_depth_m = np.where(bed_depth_m < 0, np.nan, bed_depth_m)

    # Ice thickness calculation using two-way travel time like TERRA
    # Method 1: Direct thickness from travel time difference (TERRA's approach)
    travel_time_diff_us = bed_travel_time_us - surface_travel_time_us  # Two-way travel time through ice
    ice_thickness_direct_m = (travel_time_diff_us * 1e-6) / 2 * wave_velocity_ice  # Convert to one-way then to depth
    
    # Method 2: Bed depth minus surface depth (should be equivalent)
    ice_thickness_alternate_m = bed_depth_m - surface_depth_m
    
    # Use direct method as primary (matches TERRA's approach exactly)
    ice_thickness_m = ice_thickness_direct_m

    # Create output file path
    output_path = Path(output_filepath)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        with open(output_path, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)

            # Write metadata header if requested
            if include_metadata:
                writer.writerow(["# Z-Scope Ice Measurements Export"])
                writer.writerow(
                    [f"# Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"]
                )
                writer.writerow([f"# Source file: {processor.base_filename}"])
                writer.writerow(
                    [f"# Wave velocity in ice: {wave_velocity_ice:.0f} m/s"]
                )
                writer.writerow([f"# Ice dielectric constant: {dielectric_constant}"])
                if hasattr(processor, 'calpip_pixel_distance') and processor.calpip_pixel_distance and processor.calpip_pixel_distance > 0:
                    writer.writerow(
                        [
                            f"# Calpip calibration: 2.0 μs between calpip lines, {processor.calpip_pixel_distance:.1f} pixels spacing"
                        ]
                    )
                    writer.writerow(
                        [
                            f"# Microseconds per pixel: {2.0/processor.calpip_pixel_distance:.4f} (calpip-based)"
                        ]
                    )
                else:
                    writer.writerow(
                        [
                            f"                if hasattr(processor, 'calpip_pixel_distance') and processor.calpip_pixel_distance and processor.calpip_pixel_distance > 0:
                    writer.writerow(
                        [
                            f"# Calpip calibration: 2.0 μs between calpip lines, {processor.calpip_pixel_distance:.1f} pixels spacing"
                        ]
                    )
                    writer.writerow(
                        [
                            f"# Microseconds per pixel: {2.0/processor.calpip_pixel_distance:.4f} (calpip-based)"
                        ]
                    )
                else:
                    writer.writerow(
                        [
                            f"# Pixels per microsecond: {processor.pixels_per_microsecond:.3f} (fallback method)"
                        ]
                    )"
                        ]
                    )
                writer.writerow(
                    [
                        f"# Transmitter pulse Y position: {processor.transmitter_pulse_y_abs:.1f} pixels"
                    ]
                )
                writer.writerow(
                    [f"# X coordinate system: {coordinate_system} ({x_unit})"]
                )
                writer.writerow([f"# X description: {x_description}"])
                writer.writerow(["# Data quality notes:"])
                writer.writerow(["#   - Travel times are TWO-WAY (like TERRA methodology)"])
                writer.writerow(["#   - Calibration uses calpip spacing (2 μs between lines)"])
                writer.writerow(["#   - NaN values indicate no valid echo detected"])
                writer.writerow(
                    ["#   - Negative travel times set to NaN (invalid echoes)"]
                )
                writer.writerow(["#   - Surface depth: depth from air-ice interface"])
                writer.writerow(
                    [
                        "#   - Bed depth: depth from air-ice interface to ice-bedrock interface"
                    ]
                )
                writer.writerow(["#   - Ice thickness: (bed_twt - surface_twt) / 2 * ice_velocity"])
                writer.writerow(["#"])

            # Write column headers
            headers = [
                f"x_position_{x_unit}",
                "surface_twt_us",  # Two-way travel time like TERRA
                "surface_depth_m",
                "bed_twt_us",      # Two-way travel time like TERRA
                "bed_depth_m",
                "ice_thickness_m",
            ]
            writer.writerow(headers)

            # Write data rows
            for i in range(n_pixels):
                row = [
                    f"{x_coordinates[i]:.3f}"
                    if x_unit == "m"
                    else f"{x_coordinates[i]}",
                    f"{surface_travel_time_us[i]:.3f}"
                    if np.isfinite(surface_travel_time_us[i])
                    else "NaN",
                    f"{surface_depth_m[i]:.3f}"
                    if np.isfinite(surface_depth_m[i])
                    else "NaN",
                    f"{bed_travel_time_us[i]:.3f}"
                    if np.isfinite(bed_travel_time_us[i])
                    else "NaN",
                    f"{bed_depth_m[i]:.3f}" if np.isfinite(bed_depth_m[i]) else "NaN",
                    f"{ice_thickness_m[i]:.3f}"
                    if np.isfinite(ice_thickness_m[i])
                    else "NaN",
                ]
                writer.writerow(row)

        # Print summary statistics
        valid_surface = np.isfinite(surface_depth_m)
        valid_bed = np.isfinite(bed_depth_m)
        valid_thickness = np.isfinite(ice_thickness_m)

        print(f"\nINFO: Export completed successfully to: {output_path}")
        print(f"INFO: Total pixel locations: {n_pixels}")
        print(
            f"INFO: Valid surface detections: {np.sum(valid_surface)} ({100 * np.sum(valid_surface) / n_pixels:.1f}%)"
        )
        print(
            f"INFO: Valid bed detections: {np.sum(valid_bed)} ({100 * np.sum(valid_bed) / n_pixels:.1f}%)"
        )

        if np.any(valid_surface):
            print(
                f"INFO: Surface depth range: {np.nanmin(surface_depth_m):.1f} - {np.nanmax(surface_depth_m):.1f} m"
            )
        if np.any(valid_bed):
            print(
                f"INFO: Bed depth range: {np.nanmin(bed_depth_m):.1f} - {np.nanmax(bed_depth_m):.1f} m"
            )
        if np.any(valid_thickness):
            print(
                f"INFO: Ice thickness range: {np.nanmin(ice_thickness_m):.1f} - {np.nanmax(ice_thickness_m):.1f} m"
            )
            print(f"INFO: Mean ice thickness: {np.nanmean(ice_thickness_m):.1f} m")

        return True

    except IOError as e:
        print(f"ERROR: Could not write to file {output_path}: {e}")
        return False
    except Exception as e:
        print(f"ERROR: Unexpected error during export: {e}")
        return False


def export_ice_measurements_with_config(processor, export_config=None):
    """
    Export ice measurements using configuration parameters.

    This wrapper function uses configuration settings to determine export parameters,
    making it easy to integrate with the existing config-driven architecture.

    Args:
        processor (ZScopeProcessor): The processor instance containing analysis results
        export_config (dict): Configuration dictionary with export settings

    Returns:
        bool: True if export was successful, False otherwise
    """

    if export_config is None:
        export_config = {}

    # Default configuration
    default_config = {
        "output_filename_suffix": "_ice_measurements.csv",
        "include_metadata": True,
        "coordinate_system": "pixel",  # "pixel" or "distance"
        "output_directory": None,  # If None, uses processor.output_dir
    }

    # Merge with provided config
    config = {**default_config, **export_config}

    # Determine output filepath
    if config["output_directory"] is not None:
        output_dir = Path(config["output_directory"])
    elif hasattr(processor, "output_dir") and processor.output_dir is not None:
        output_dir = Path(processor.output_dir)
    else:
        output_dir = Path(".")

    output_filename = processor.base_filename + config["output_filename_suffix"]
    output_filepath = output_dir / output_filename

    return export_ice_measurements(
        processor,
        output_filepath,
        include_metadata=config["include_metadata"],
        coordinate_system=config["coordinate_system"],
    )
