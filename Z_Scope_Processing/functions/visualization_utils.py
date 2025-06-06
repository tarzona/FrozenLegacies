import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
import numpy as np
import cv2  # For image normalization and CLAHE if used within visualization
from pathlib import Path

# Import calibration utilities for conversions within plotting
from .calibration_utils import convert_time_to_depth, convert_depth_to_time


def visualize_calibration_pip_detection(
    image_full,
    base_filename,
    best_pip,
    approx_x_click=None,
    visualization_params=None,
    output_params=None,
):
    """
    Generates a 3-panel visualization of the calibration pip detection process.

    This function helps users understand how the calibration pip was identified by showing:
    1.  **Context Panel**: The broader area around the user's click or the detected pip,
        highlighting the region processed.
    2.  **Results Panel**: A zoomed-in view of the detected pip, showing the individual tick
        marks and the determined Z-scope boundary.
    3.  **Detail Zoom Panel**: An even closer, contrast-enhanced view of the ticks to
        assess their clarity and spacing.

    Args:
        image_full (np.ndarray): The full, original 2D grayscale Z-scope image.
        base_filename (str): Base name for saving the output plot (e.g., "image_01").
                             The plot will be saved as "<base_filename>_calibration_pip_detection_overview.png".
        best_pip (dict or None): A dictionary containing details of the best detected pip.
            Expected keys:
                'x_position' (int): Absolute X-coordinate of the detected pip's vertical line.
                'y_start' (int): Absolute Y-coordinate of the first detected tick mark.
                'y_end' (int): Absolute Y-coordinate of the last detected tick mark (or Z-boundary cutoff).
                'tick_count' (int): Number of valid tick marks found.
                'mean_spacing' (float): Average spacing between tick marks in pixels.
                'tick_positions' (list/np.ndarray): List of absolute Y-coordinates of each tick mark.
                'z_boundary' (int): Absolute Y-coordinate of the detected Z-scope data/metadata boundary.
            If None, a simple plot indicating no pip was found/provided is generated.
        approx_x_click (int, optional): The approximate X-coordinate on the full image where the user
                                        clicked to indicate the pip location. Used for context.
        visualization_params (dict, optional): Parameters to control the visual appearance.
            Example keys:
                "context_panel_width_px" (int): Width of the context panel view.
                "results_panel_y_padding_px" (int): Vertical padding around ticks in the results panel.
                "results_panel_x_margin_px" (int): Horizontal margin for the results panel.
                "zoom_panel_height_px" (int): Height of the detailed zoom panel.
                "zoom_panel_clahe_clip_limit" (float): CLAHE clip limit for contrast in zoom panel.
                "zoom_panel_clahe_tile_grid_size" (list): CLAHE tile grid size for zoom panel.
                "pip_strip_display_width_px" (int): Width of the red rectangle in context view.
        output_params (dict, optional): Parameters for saving the output.
            Example keys:
                "debug_output_directory" (str): Folder to save the plot.
                "figure_save_dpi" (int): DPI for the saved image.

    Returns:
        None. The function saves the plot to a file and may display it.
    """
    if visualization_params is None:
        visualization_params = {}  # Ensure it's a dict to allow .get()
    if (
        output_params is None
    ):  # This default should ideally not be hit if called from ZScopeProcessor
        output_params = {
            "debug_output_directory": "debug_output",
            "figure_save_dpi": 300,
        }

    output_dir_name = output_params.get("debug_output_directory", "debug_output")
    output_dir = Path(output_dir_name)
    save_dpi = output_params.get("figure_save_dpi", 300)

    if not best_pip:
        print(
            "Visualization Info: No valid 'best_pip' data provided for calibration pip visualization."
        )
        # Plot original image with click mark if no pip data
        fig_no_pip, ax_no_pip = plt.subplots(figsize=(12, 6))
        ax_no_pip.imshow(image_full, cmap="gray", aspect="auto")
        if approx_x_click is not None:
            ax_no_pip.axvline(
                x=approx_x_click,
                color="r",
                linestyle="--",
                label=f"User Click (x={approx_x_click})",
            )
            ax_no_pip.legend()
        ax_no_pip.set_title(
            f"Original Image: {base_filename} - No calibration pip data for visualization"
        )
        plt.savefig(
            output_dir / f"{base_filename}_no_pip_data_for_visualization.png",
            dpi=save_dpi,
        )
        plt.close(fig_no_pip)
        return

    fig, (ax1, ax2, ax3) = plt.subplots(
        1, 3, figsize=visualization_params.get("figure_size_inches", (20, 6))
    )
    fig.suptitle(f"Calibration Pip Detection Overview: {base_filename}", fontsize=14)

    # --- Panel 1: Context Panel ---
    # Shows a wide view of the Z-scope image around the approximate pip location.
    # A red rectangle highlights the specific vertical strip that was analyzed for pips.
    context_width = visualization_params.get("context_panel_width_px", 10000)
    # Define vertical extent for context: from a bit above first tick to below Z-boundary.
    y_start_context = int(best_pip["y_start"] - 100)
    y_end_context = int(best_pip["z_boundary"] + 50)
    y_start_context = max(0, y_start_context)  # Ensure within image bounds
    y_end_context = min(image_full.shape[0], y_end_context)

    # Define horizontal extent for context, centered on user click or detected pip.
    center_x_context = (
        approx_x_click if approx_x_click is not None else best_pip["x_position"]
    )
    context_x_start_abs = max(0, center_x_context - context_width // 2)
    context_x_end_abs = min(image_full.shape[1], context_x_start_abs + context_width)

    context_image_crop = image_full[
        y_start_context:y_end_context, context_x_start_abs:context_x_end_abs
    ]
    # Normalize for better display if image has low contrast.
    enhanced_context = cv2.normalize(context_image_crop, None, 0, 255, cv2.NORM_MINMAX)
    ax1.imshow(
        enhanced_context,
        cmap="gray",
        aspect="auto",
        extent=[context_x_start_abs, context_x_end_abs, y_end_context, y_start_context],
    )  # Set extent for correct coord display
    ax1.set_title("1. Context: Pip Location")
    ax1.set_xlabel("X-pixel (Full Image)")
    ax1.set_ylabel("Y-pixel (Full Image)")

    # Highlight the actual strip analyzed for the pip within the context view.
    pip_strip_display_width = visualization_params.get(
        "pip_strip_display_width_px", 400
    )
    rect_x_abs_start = best_pip["x_position"] - pip_strip_display_width // 2

    rect = plt.Rectangle(
        (
            rect_x_abs_start,
            y_start_context,
        ),  # Using absolute coordinates based on extent
        pip_strip_display_width,
        y_end_context - y_start_context,
        linewidth=1,
        edgecolor="red",
        facecolor="none",
        linestyle="--",
        label="Analyzed Region for Pip",
    )
    ax1.add_patch(rect)
    if approx_x_click is not None:
        ax1.axvline(
            x=approx_x_click,
            color="lime",
            linestyle=":",
            linewidth=1.5,
            label=f"User Click (x={approx_x_click})",
        )
    ax1.legend(loc="upper right", fontsize="small")

    # --- Panel 2: Results Panel ---
    # Zooms into the detected calibration pip's vertical line.
    # Shows the individual tick marks (red lines) and the Z-scope boundary (yellow line).
    results_y_padding = visualization_params.get("results_panel_y_padding_px", 50)
    results_x_margin = visualization_params.get("results_panel_x_margin_px", 200)

    # Define ROI for results panel based on detected pip features.
    y_min_roi_abs = max(0, int(best_pip["y_start"]) - results_y_padding)
    y_max_roi_abs = min(image_full.shape[0], int(best_pip["y_end"]) + results_y_padding)
    x_start_roi_abs = max(0, best_pip["x_position"] - results_x_margin)
    x_end_roi_abs = min(image_full.shape[1], best_pip["x_position"] + results_x_margin)

    roi_image_crop = image_full[
        y_min_roi_abs:y_max_roi_abs, x_start_roi_abs:x_end_roi_abs
    ]
    roi_enhanced_display = cv2.normalize(roi_image_crop, None, 0, 255, cv2.NORM_MINMAX)
    ax2.imshow(
        roi_enhanced_display,
        cmap="gray",
        aspect="auto",
        extent=[x_start_roi_abs, x_end_roi_abs, y_max_roi_abs, y_min_roi_abs],
    )
    ax2.set_title(f"2. Detected Pip ({best_pip['tick_count']} ticks)")
    ax2.set_xlabel("X-pixel (Full Image)")
    # ax2.set_ylabel("Y-pixel (Full Image)") # Redundant if aligned with ax1

    # Draw vertical line at the detected pip's x-position.
    ax2.axvline(
        x=best_pip["x_position"],
        color="cyan",
        linestyle="-",
        linewidth=1.5,
        label="Pip Centerline",
    )
    # Draw Z-scope boundary.
    ax2.axhline(
        y=best_pip["z_boundary"],
        color="yellow",
        linestyle="-",
        linewidth=1.5,
        label="Z-Boundary",
    )
    # Draw detected tick marks.
    for tick_y_abs in best_pip["tick_positions"]:
        ax2.axhline(y=tick_y_abs, color="red", linestyle="-", alpha=0.7, linewidth=1)
    ax2.legend(loc="upper right", fontsize="small")

    # --- Panel 3: Detail Zoom Panel ---
    zoom_height_px = visualization_params.get("zoom_panel_height_px", 500)
    clahe_clip = visualization_params.get("zoom_panel_clahe_clip_limit", 2.0)
    clahe_tile_list = visualization_params.get(
        "zoom_panel_clahe_tile_grid_size", [8, 8]
    )
    clahe_tile = tuple(clahe_tile_list) if isinstance(clahe_tile_list, list) else (8, 8)

    # Center this zoom panel around the vertical middle of the detected ticks in the ROI.
    if len(best_pip["tick_positions"]) > 0:
        mid_tick_y_abs = np.mean(
            [min(best_pip["tick_positions"]), max(best_pip["tick_positions"])]
        )
    else:  # Fallback if no ticks (should not happen if best_pip is valid)
        mid_tick_y_abs = best_pip["y_start"]

    # Define zoom region relative to the full image coordinates.
    zoom_y_start_abs = max(0, int(mid_tick_y_abs - zoom_height_px // 2))
    zoom_y_end_abs = min(image_full.shape[0], zoom_y_start_abs + zoom_height_px)
    # Use the same X-range as the results panel (roi_image_crop) for horizontal extent.

    zoomed_crop_full_image = image_full[
        zoom_y_start_abs:zoom_y_end_abs, x_start_roi_abs:x_end_roi_abs
    ]

    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) for better local contrast.
    if zoomed_crop_full_image.size > 0:  # Ensure crop is not empty
        clahe_zoom = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=clahe_tile)
        # Ensure image is uint8 for CLAHE
        if zoomed_crop_full_image.dtype != np.uint8:
            zoomed_crop_uint8 = cv2.normalize(
                zoomed_crop_full_image, None, 0, 255, cv2.NORM_MINMAX
            ).astype(np.uint8)
        else:
            zoomed_crop_uint8 = zoomed_crop_full_image
        zoomed_final_display = clahe_zoom.apply(zoomed_crop_uint8)
    else:
        zoomed_final_display = np.zeros(
            (10, 10), dtype=np.uint8
        )  # Placeholder if crop failed

    ax3.imshow(
        zoomed_final_display,
        cmap="gray",
        aspect="auto",
        extent=[x_start_roi_abs, x_end_roi_abs, zoom_y_end_abs, zoom_y_start_abs],
    )
    ax3.set_title(f"3. Zoom (Avg Spacing: {best_pip['mean_spacing']:.1f}px)")
    ax3.set_xlabel("X-pixel (Full Image)")
    # ax3.set_ylabel("Y-pixel (Full Image)")

    # Draw ticks in this zoomed panel using their absolute Y-coordinates.
    for tick_y_abs in best_pip["tick_positions"]:
        # Only draw if the tick falls within the vertical range of this zoomed panel.
        if zoom_y_start_abs <= tick_y_abs <= zoom_y_end_abs:
            ax3.axhline(
                y=tick_y_abs, color="red", linestyle="-", alpha=0.8, linewidth=1.5
            )

    # Final adjustments and save.
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust for suptitle
    plot_path = output_dir / f"{base_filename}_calibration_pip_detection_overview.png"
    plt.savefig(plot_path, dpi=save_dpi)
    print(f"INFO: Calibration pip detection overview plot saved to {plot_path}")
    # plt.show(block=False) # Display plot non-blockingly
    plt.close(fig)  # Close figure to free memory

    # Print summary to console.
    print("\nCalibration Pip Detection Summary (for visualization):")
    print(f"  Number of tick marks detected: {best_pip['tick_count']}")
    print(f"  Average spacing between ticks: {best_pip['mean_spacing']:.2f} pixels")
    if len(best_pip["tick_positions"]) > 1:
        spacing_std = np.std(np.diff(best_pip["tick_positions"]))
        print(f"  Standard deviation of spacing: {spacing_std:.2f} pixels")


def apply_publication_style(
    fig,
    ax,
    time_ax,
    surface_y_abs=None,
    bed_y_abs=None,
    data_top_abs=None,
    transmitter_pulse_y_abs=None,
    pixels_per_microsecond=None,
    physics_constants=None,
):
    """
    Apply publication-quality styling to the figure.

    Args:
        fig (matplotlib.figure.Figure): The figure object
        ax (matplotlib.axes.Axes): The main plot axis
        time_ax (matplotlib.axes.Axes): The time axis
        surface_y_abs (np.ndarray, optional): Surface echo Y-coordinates
        bed_y_abs (np.ndarray, optional): Bed echo Y-coordinates
        data_top_abs (int, optional): Top of data region
        transmitter_pulse_y_abs (int, optional): Transmitter pulse position
        pixels_per_microsecond (float, optional): Calibration factor
        physics_constants (dict, optional): Physical constants for depth calculation
    """
    # Typography settings
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Arial"]
    plt.rcParams["axes.titlesize"] = 12
    plt.rcParams["axes.labelsize"] = 10
    plt.rcParams["xtick.labelsize"] = 9
    plt.rcParams["ytick.labelsize"] = 9

    # Enhance axis labels
    ax.set_xlabel(
        "Horizontal Distance along Flight Path (pixels)", fontsize=10, fontweight="bold"
    )
    ax.set_ylabel("Depth (m)", fontsize=10, fontweight="bold")
    time_ax.set_ylabel("Two-way Travel Time (µs)", fontsize=10, fontweight="bold")

    # Adjust tick labels
    ax.tick_params(axis="both", labelsize=9)
    time_ax.tick_params(axis="both", labelsize=9)

    # Reduce grid line visibility - make them much more subtle
    for line in ax.get_lines():
        if line.get_linestyle() == "--":  # Minor grid lines
            line.set_alpha(0.15)  # Reduced from 0.2
        elif (
            line.get_linestyle() == "-" and line.get_color() == "white"
        ):  # Major grid lines
            line.set_alpha(0.3)  # Reduced from 0.4

    # Update colors and line styles for existing traces
    for line in ax.get_lines():
        # Transmitter pulse line (blue)
        if line.get_color() == "blue":
            line.set_color("#4477AA")  # Professional blue
            line.set_linewidth(1.5)
            line.set_alpha(0.9)

        # Calibration pip column (green)
        if line.get_color() == "g":
            line.set_color("#999933")  # Muted gold
            line.set_linewidth(1.0)
            line.set_alpha(0.6)
            line.set_linestyle(":")  # Dotted line

    # Update the legend
    legend = ax.get_legend()
    if legend:
        legend.set_frame_on(True)
        legend.get_frame().set_alpha(0.8)
        for text in legend.get_texts():
            text.set_fontsize(8)

        # Update legend title
        if hasattr(legend, "set_title"):
            legend.set_title("Features", prop={"size": 9, "weight": "bold"})

    # If we have surface and bed data, add depth annotations
    if (
        surface_y_abs is not None
        and bed_y_abs is not None
        and data_top_abs is not None
        and transmitter_pulse_y_abs is not None
        and pixels_per_microsecond is not None
        and physics_constants is not None
    ):
        # Helper function to convert from absolute Y to depth
        def abs_y_to_depth(y_abs):
            y_rel = y_abs - transmitter_pulse_y_abs
            time_us = y_rel / pixels_per_microsecond
            one_way_time_us = time_us / 2.0
            c0 = physics_constants.get("speed_of_light_vacuum_mps")
            epsilon_r_ice = physics_constants.get("ice_relative_permittivity_real")
            firn_corr_m = physics_constants.get("firn_correction_meters")
            return convert_time_to_depth(
                one_way_time_us, c0, epsilon_r_ice, firn_corr_m
            )

        # Calculate average depths
        valid_surface = surface_y_abs[np.isfinite(surface_y_abs)]
        valid_bed = bed_y_abs[np.isfinite(bed_y_abs)]

        if len(valid_surface) > 0 and len(valid_bed) > 0:
            avg_surface_depth = np.mean([abs_y_to_depth(y) for y in valid_surface])
            avg_bed_depth = np.mean([abs_y_to_depth(y) for y in valid_bed])
            ice_thickness = avg_bed_depth - avg_surface_depth

            # Get the x-coordinate for annotations (95% of the way across)
            x_pos = 0.95 * ax.get_xlim()[1]

            # Add annotations for surface, bed, and ice thickness
            ax.annotate(
                f"Surface: {avg_surface_depth:.1f} m",
                xy=(x_pos, (valid_surface[0] - data_top_abs)),
                xytext=(10, -5),
                textcoords="offset points",
                color="#117733",
                fontsize=9,
                fontweight="bold",
                bbox=dict(
                    boxstyle="round,pad=0.3", fc="white", ec="#117733", alpha=0.8
                ),
            )

            ax.annotate(
                f"Bed: {avg_bed_depth:.1f} m",
                xy=(x_pos, (valid_bed[0] - data_top_abs)),
                xytext=(10, 5),
                textcoords="offset points",
                color="#CC6677",
                fontsize=9,
                fontweight="bold",
                bbox=dict(
                    boxstyle="round,pad=0.3", fc="white", ec="#CC6677", alpha=0.8
                ),
            )

            ax.annotate(
                f"Ice thickness: {ice_thickness:.1f} m",
                xy=(x_pos, (valid_surface[0] + valid_bed[0]) / 2 - data_top_abs),
                xytext=(10, 0),
                textcoords="offset points",
                color="black",
                fontsize=9,
                fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", alpha=0.8),
            )

    # Add to apply_publication_style around line 390
    # Add horizontal scale bar if pixels_per_microsecond is available
    if pixels_per_microsecond is not None:
        # Estimate horizontal scale (assuming similar horizontal/vertical scaling)
        km_per_pixel = 0.169 / pixels_per_microsecond  # Example conversion factor
        bar_length_km = 5  # 5 km scale bar
        bar_length_px = bar_length_km / km_per_pixel

        # Get the axes dimensions
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()

        # Position the scale bar in the lower right corner
        bar_x_start = xlim[1] - bar_length_px - 100
        bar_y_position = ylim[1] - 50

        # Draw the scale bar
        ax.plot(
            [bar_x_start, bar_x_start + bar_length_px],
            [bar_y_position, bar_y_position],
            "k-",
            linewidth=2,
        )

        # Add label
        ax.text(
            bar_x_start + bar_length_px / 2,
            bar_y_position - 20,
            f"{bar_length_km} km",
            ha="center",
            va="top",
            fontsize=9,
        )

    # Improve the title
    title = ax.get_title()
    if title:
        ax.set_title(title, fontsize=12, fontweight="bold")

    return fig, ax, time_ax


def create_time_calibrated_zscope(
    image_full,
    base_filename,
    best_pip,
    transmitter_pulse_y_abs,
    data_top_abs,
    data_bottom_abs,
    pixels_per_microsecond,
    time_vis_params=None,
    physics_constants=None,
    output_params=None,
    surface_y_abs=None,
    bed_y_abs=None,
):
    """
    Creates the primary time-calibrated Z-scope visualization with dynamic cropping.

    Args:
        image_full (np.ndarray): The full 2D grayscale Z-scope image.
        base_filename (str): Base name for saving the plot (e.g., "image_01").
        best_pip (dict): Pip detection details (used for marking the pip location).
        transmitter_pulse_y_abs (int): Absolute Y-coordinate (full image) of the transmitter pulse.
        data_top_abs (int): Absolute Y-coordinate (full image) of the top of the valid data region.
        data_bottom_abs (int): Absolute Y-coordinate (full image) of the bottom of the valid data region.
        pixels_per_microsecond (float): The calibration factor (pixels / µs).
        time_vis_params (dict, optional): Parameters for controlling the visualization.
        physics_constants (dict, optional): Physical constants needed for the ice thickness scale.
        output_params (dict, optional): Parameters for saving the output.
        surface_y_abs (np.ndarray, optional): Detected surface echo Y-coordinates.
        bed_y_abs (np.ndarray, optional): Detected bed echo Y-coordinates.

    Returns:
        tuple: (fig, ax, primary_time_ax) Matplotlib figure and axes objects,
               or (None, None, None) if an error occurs.
    """
    if time_vis_params is None:
        time_vis_params = {}
    if output_params is None:
        output_params = {}

    output_dir_name = output_params.get("debug_output_directory", "debug_output")
    output_dir = Path(output_dir_name)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_dpi = output_params.get("figure_save_dpi", 300)

    fig_size_list = time_vis_params.get("figure_size_inches", [24, 10])
    fig_size = tuple(fig_size_list) if isinstance(fig_size_list, list) else (24, 10)
    major_grid_interval_us = time_vis_params.get("major_grid_time_interval_us", 10)
    minor_grid_interval_us = time_vis_params.get("minor_grid_time_interval_us", 2)
    label_x_offset = time_vis_params.get("label_x_offset_px", 50)
    label_fontsize = time_vis_params.get("label_font_size", 9)
    aspect_divisor = time_vis_params.get("aspect_ratio_divisor", 5.0)
    legend_loc = time_vis_params.get("legend_location", "upper right")

    try:
        fig, ax = plt.subplots(figsize=fig_size)

        # Determine dynamic bottom boundary if bed echo is detected
        dynamic_bottom = data_bottom_abs
        if bed_y_abs is not None and np.any(np.isfinite(bed_y_abs)):
            # Find the maximum valid bed echo Y-coordinate and add a margin
            valid_bed_indices = np.where(np.isfinite(bed_y_abs))[0]
            if len(valid_bed_indices) > 0:
                max_bed_y = np.max(bed_y_abs[valid_bed_indices])
                # Add a margin below the deepest bed point (e.g., 20% of the distance from Tx to bed)
                margin_below_bed = (
                    time_vis_params.get("margin_below_bed_percent", 20) / 100
                )
                bed_margin_px = int(
                    (max_bed_y - transmitter_pulse_y_abs) * margin_below_bed
                )
                dynamic_bottom = min(data_bottom_abs, max_bed_y + bed_margin_px)

        dynamic_bottom = int(dynamic_bottom)

        # Use the dynamic bottom for cropping
        valid_data_crop = image_full[data_top_abs:dynamic_bottom, :]

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced_data = clahe.apply(
            cv2.normalize(valid_data_crop, None, 0, 255, cv2.NORM_MINMAX).astype(
                np.uint8
            )
        )
        ax.imshow(enhanced_data, cmap="gray", aspect="auto")

        # Helper function: Maps absolute Y-coordinate (full image) to Y-coordinate in the cropped view.
        def abs_to_cropped_y(y_abs_coord):
            return y_abs_coord - data_top_abs

        # Mark the location of the detected calibration pip.
        if best_pip and "x_position" in best_pip:
            ax.axvline(
                x=best_pip["x_position"],
                color="g",
                linestyle="-",
                linewidth=1.5,
                alpha=0.8,
                label="Calibration Pip Column",
            )

        # Mark the transmitter pulse as the 0µs time reference.
        tx_y_cropped = abs_to_cropped_y(transmitter_pulse_y_abs)
        ax.axhline(y=tx_y_cropped, color="blue", linestyle="-", linewidth=2, alpha=0.8)
        ax.text(
            label_x_offset,
            tx_y_cropped,
            "0µs (Tx Pulse)",
            color="blue",
            fontsize=label_fontsize,
            fontweight="bold",
            va="center",
            path_effects=[
                path_effects.withStroke(linewidth=2, foreground="white", alpha=0.5)
            ],
        )

        # Calculate total two-way travel time visible in the valid data area.
        total_time_range_us = (
            dynamic_bottom - transmitter_pulse_y_abs
        ) / pixels_per_microsecond

        # Draw time grid lines (major and minor).
        for t_us in np.arange(
            0, total_time_range_us + minor_grid_interval_us, minor_grid_interval_us
        ):
            pixel_y_abs_coord = transmitter_pulse_y_abs + t_us * pixels_per_microsecond
            # Only draw if the line is within the displayed cropped data.
            if data_top_abs <= pixel_y_abs_coord <= dynamic_bottom:
                pixel_y_cropped_coord = abs_to_cropped_y(pixel_y_abs_coord)
                is_major_grid = round(t_us, 6) % major_grid_interval_us == 0
                ax.axhline(
                    y=pixel_y_cropped_coord,
                    color="white",
                    linestyle="-" if is_major_grid else "--",
                    alpha=0.7 if is_major_grid else 0.4,
                    linewidth=1.0 if is_major_grid else 0.7,
                )
                if is_major_grid:
                    ax.text(
                        label_x_offset,
                        pixel_y_cropped_coord,
                        f"{int(round(t_us))}µs",
                        color="white",
                        fontsize=label_fontsize,
                        alpha=0.9,
                        va="center",
                        path_effects=[
                            path_effects.withStroke(
                                linewidth=2, foreground="black", alpha=0.5
                            )
                        ],
                    )

        # --- Define conversion functions ---
        def cropped_y_to_time_us(y_cropped_coord):
            return (
                y_cropped_coord + data_top_abs - transmitter_pulse_y_abs
            ) / pixels_per_microsecond

        def time_us_to_cropped_y(t_us_val):
            return (
                (t_us_val * pixels_per_microsecond)
                + transmitter_pulse_y_abs
                - data_top_abs
            )

        # --- Define depth conversion functions ---
        c0 = physics_constants.get("speed_of_light_vacuum_mps")
        epsilon_r_ice = physics_constants.get("ice_relative_permittivity_real")
        firn_corr_m = physics_constants.get("firn_correction_meters")

        def cropped_y_to_depth_m(y_cropped_coord):
            time_us = cropped_y_to_time_us(y_cropped_coord)
            one_way_time_us = time_us / 2.0
            return convert_time_to_depth(
                one_way_time_us, c0, epsilon_r_ice, firn_corr_m
            )

        def depth_m_to_cropped_y(depth_m):
            one_way_time_us = convert_depth_to_time(
                depth_m, c0, epsilon_r_ice, firn_corr_m
            )
            two_way_time_us = one_way_time_us * 2.0
            return time_us_to_cropped_y(two_way_time_us)

        # --- Set up the axes ---
        ax.yaxis.set_visible(True)  # Ensure the main y-axis is visible
        ax.set_ylabel("Depth (m)")  # Set the label for the main axis

        # Calculate depth at transmitter pulse (should be 0)
        tx_depth = cropped_y_to_depth_m(tx_y_cropped)

        # Calculate maximum depth based on the bottom of the visible area
        max_depth = cropped_y_to_depth_m(valid_data_crop.shape[0])

        # Create depth ticks at regular intervals - ensure we start at 0 (transmitter pulse)
        num_ticks = 7  # Adjust for desired tick density
        depth_ticks = np.linspace(0, max_depth, num_ticks)
        y_tick_positions = [depth_m_to_cropped_y(d) for d in depth_ticks]

        # Filter out any invalid positions (might happen if depth conversion has issues)
        valid_ticks = [
            (pos, depth)
            for pos, depth in zip(y_tick_positions, depth_ticks)
            if 0 <= pos < valid_data_crop.shape[0]
        ]

        if valid_ticks:
            valid_positions, valid_depths = zip(*valid_ticks)
            ax.set_yticks(valid_positions)
            ax.set_yticklabels([f"{int(d)}" for d in valid_depths])

        # Create time axis on the right
        time_ax = ax.secondary_yaxis(
            "right", functions=(cropped_y_to_time_us, time_us_to_cropped_y)
        )
        time_ax.set_ylabel("Two-way Travel Time (µs)")

        # Set plot aspect ratio, labels, title, and legend.
        data_height_px = dynamic_bottom - data_top_abs
        if data_height_px > 0:
            aspect_val = valid_data_crop.shape[1] / data_height_px / aspect_divisor
            ax.set_aspect(aspect_val)
        else:
            ax.set_aspect("auto")

        ax.set_xlabel("Horizontal Distance along Flight Path (pixels)")
        ax.set_title(f"Time-Calibrated Z-scope: {base_filename}")
        ax.legend(loc=legend_loc, fontsize="small")

        plt.tight_layout()
        # Apply publication-quality styling
        fig, ax, time_ax = apply_publication_style(
            fig,
            ax,
            time_ax,
            surface_y_abs=surface_y_abs,
            bed_y_abs=bed_y_abs,
            data_top_abs=data_top_abs,
            transmitter_pulse_y_abs=transmitter_pulse_y_abs,
            pixels_per_microsecond=pixels_per_microsecond,
            physics_constants=physics_constants,
        )

        plot_path = output_dir / f"{base_filename}_time_calibrated_zscope.png"
        plt.savefig(plot_path, dpi=save_dpi, bbox_inches="tight")

        return fig, ax, time_ax

    except Exception as e:
        print(f"ERROR in create_time_calibrated_zscope for {base_filename}: {e}")
        import traceback

        traceback.print_exc()
        return None, None, None


def add_ice_thickness_scale(
    fig,
    main_ax,
    main_ax_y_to_time_us_func,
    time_us_to_main_ax_y_func,
    physics_constants,
    ice_scale_params=None,
):
    """
    Adds a third y-axis to the plot for depth in meters below transmitter pulse.

    Args:
        fig (matplotlib.figure.Figure): The figure object.
        main_ax (matplotlib.axes.Axes): The main plot axis (displaying the image).
        main_ax_y_to_time_us_func (callable): Function converting main_ax Y-pixels to two-way time (µs).
        time_us_to_main_ax_y_func (callable): Function converting two-way time (µs) to main_ax Y-pixels.
        physics_constants (dict): Dictionary with physical constants.
        ice_scale_params (dict, optional): Parameters for the ice scale appearance.
    """
    if ice_scale_params is None:
        ice_scale_params = {}

    c0 = physics_constants.get("speed_of_light_vacuum_mps")
    epsilon_r_ice = physics_constants.get("ice_relative_permittivity_real")
    firn_corr_m = physics_constants.get("firn_correction_meters")

    if c0 is None or epsilon_r_ice is None or firn_corr_m is None:
        print("WARNING: Missing physical constants. Cannot add depth scale.")
        return None

    # 1. From cropped Y-pixel on main_ax to depth (m)
    def final_main_ax_y_to_depth_m_func(y_cropped_val):
        time_us = main_ax_y_to_time_us_func(y_cropped_val)
        one_way_time_us = time_us / 2.0
        return convert_time_to_depth(one_way_time_us, c0, epsilon_r_ice, firn_corr_m)

    # 2. From depth (m) back to cropped Y-pixel on main_ax
    def final_depth_m_to_main_ax_y_func(d_m_val):
        one_way_time_us = convert_depth_to_time(d_m_val, c0, epsilon_r_ice, firn_corr_m)
        two_way_time_us = one_way_time_us * 2.0
        return time_us_to_main_ax_y_func(two_way_time_us)

    axis_offset = ice_scale_params.get("axis_offset", -0.12)
    label_offset_points = ice_scale_params.get("label_offset_points", 10)

    # Create the depth axis
    depth_ax = main_ax.secondary_yaxis(
        location=axis_offset,
        functions=(final_main_ax_y_to_depth_m_func, final_depth_m_to_main_ax_y_func),
    )

    # Set label to indicate depth below transmitter
    depth_ax.set_ylabel("Depth (m)", labelpad=label_offset_points)

    # Set y-axis limits to start at 0
    min_depth = 0
    max_depth = final_main_ax_y_to_depth_m_func(main_ax.get_ylim()[1])
    depth_ax.set_ylim(min_depth, max_depth)

    print("INFO: Added depth scale.")
    return depth_ax


def annotate_radar_features(
    ax,
    feature_annotations,
    data_top_abs,
    pixels_per_microsecond,
    transmitter_pulse_y_abs,
):
    """
    Adds horizontal lines and text labels for key radar features on the Z-scope plot.

    Features are defined by their absolute Y-pixel coordinate in the full image.
    This function converts them to the cropped view and calculates their time for the label.

    Args:
        ax (matplotlib.axes.Axes): The Matplotlib axes object to annotate.
        feature_annotations (dict): A dictionary where keys are unique feature identifiers (e.g., 'i' for ice surface)
                                    and values are dictionaries containing:
                                        'pixel_abs' (int): Absolute Y-coordinate of the feature in the full image.
                                        'name' (str): Display name of the feature (e.g., "Ice Surface").
                                        'color' (str): Matplotlib color for the line and text.
        data_top_abs (int): Absolute Y-coordinate of the top of the displayed (cropped) data area.
                            Used to convert absolute feature pixels to relative display pixels.
        pixels_per_microsecond (float): Calibration factor for calculating time.
        transmitter_pulse_y_abs (int): Absolute Y-coordinate of the transmitter pulse (0 µs reference).
    """
    if not feature_annotations:
        print("INFO: No features provided for annotation.")
        return

    print("INFO: Annotating radar features on the plot...")
    for key, feature_details in feature_annotations.items():
        pixel_abs_coord = feature_details.get("pixel_abs")
        name_label = feature_details.get("name", f"Feature {key.upper()}")
        line_color = feature_details.get(
            "color", "red"
        )  # Default to red if no color specified

        if pixel_abs_coord is None:
            print(
                f"WARNING: No 'pixel_abs' provided for feature '{name_label}'. Skipping annotation."
            )
            continue

        # Convert absolute Y-pixel (full image) to Y-pixel in the cropped view displayed on 'ax'.
        y_cropped_coord = pixel_abs_coord - data_top_abs

        # Calculate two-way travel time for the label.
        time_us_val = (
            pixel_abs_coord - transmitter_pulse_y_abs
        ) / pixels_per_microsecond

        # Draw a horizontal line for the feature.
        ax.axhline(
            y=y_cropped_coord, color=line_color, linestyle="-", linewidth=1.5, alpha=0.9
        )
        # Add a text label for the feature.
        ax.text(
            50,  # X-offset for the label from the left edge of the plot.
            y_cropped_coord + 5,  # Y-offset for the label (slightly above the line).
            f"{name_label}: {time_us_val:.1f} µs",
            color=line_color,
            fontsize=9,
            fontweight="bold",
            # Path effects add a slight stroke around text for better readability on varying backgrounds.
            path_effects=[
                path_effects.withStroke(linewidth=2, foreground="white", alpha=0.5)
            ],
        )
    print(f"INFO: Annotated {len(feature_annotations)} features.")
