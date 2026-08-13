import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
import numpy as np
import cv2
import pandas as pd
import re
from pathlib import Path
from scipy.signal import find_peaks
from scipy import ndimage
from .calibration_utils import convert_time_to_depth, convert_depth_to_time


def extract_cbd_count_from_filename(filename):
    """
    Extract the expected CBD count from a filename like 'F125-C0565_C0578'.
    
    Args:
        filename: Filename containing CBD range pattern
        
    Returns:
        int: Expected number of CBD ticks, or 13 as default if parsing fails
    """
    # Try pattern with C prefix on both numbers: C0565_C0578
    cbd_match = re.search(r"C(\d+)_C(\d+)", filename)
    if not cbd_match:
        # Try pattern with C prefix only on first number: C0565_0578  
        cbd_match = re.search(r"C(\d+)_(\d+)", filename)
    
    if cbd_match:
        cbd_start = int(cbd_match.group(1))
        cbd_end = int(cbd_match.group(2))
        cbd_count = abs(cbd_end - cbd_start) + 1
        print(f"INFO: Extracted CBD range {cbd_start}-{cbd_end} from filename, expecting {cbd_count} CBD ticks")
        return cbd_count
    else:
        print(f"WARNING: Could not extract CBD range from filename '{filename}', using default count of 13")
        return 13


class CBDTickSelector:
    """Enhanced interactive CBD tick mark selector with local image recognition refinement."""

    def __init__(
        self,
        image_full,
        expected_count=13,
        sprocket_removal_ratio=0.08,
        search_height_ratio=0.12,
    ):
        self.image_full = image_full
        self.expected_count = expected_count
        self.sprocket_removal_ratio = sprocket_removal_ratio
        self.search_height_ratio = search_height_ratio
        self.selected_points = []
        self.calculated_ticks = []
        self.refined_ticks = []
        self.fig = None
        self.ax = None

    def _refine_tick_positions_with_local_detection(
        self, approximate_positions, search_radius=25
    ):
        """
        Refine tick positions using local image recognition around approximate locations.

        Args:
            approximate_positions: List of approximate x-coordinates from uniform spacing
            search_radius: Pixel radius to search around each approximate position

        Returns:
            List of refined x-coordinates for actual tick marks
        """
        height, width = self.image_full.shape

        # Define the same search region used for manual selection
        sprocket_height = int(height * self.sprocket_removal_ratio)
        search_start = sprocket_height
        search_height = int(height * self.search_height_ratio)
        search_end = search_start + search_height

        # Extract the clean region for analysis
        clean_region = self.image_full[search_start:search_end, :]

        # Apply same preprocessing as in manual selection
        clahe = cv2.createCLAHE(clipLimit=8.0, tileGridSize=(4, 4))
        enhanced_region = clahe.apply(clean_region)

        refined_positions = []

        for i, approx_x in enumerate(approximate_positions):
            # Skip the first two positions (manually selected)
            if i < 2:
                refined_positions.append(approx_x)
                continue

            # Define local search window
            x_start = max(0, approx_x - search_radius)
            x_end = min(width, approx_x + search_radius)

            # Extract local region
            local_region = enhanced_region[:, x_start:x_end]

            if local_region.shape[1] == 0:
                # Fallback to approximate position if region is invalid
                refined_positions.append(approx_x)
                continue

            # Method 1: Vertical line detection using morphology
            kernel_height = min(local_region.shape[0] // 2, 20)
            if kernel_height >= 3:
                vertical_kernel = cv2.getStructuringElement(
                    cv2.MORPH_RECT, (1, kernel_height)
                )
                morph_result = cv2.morphologyEx(
                    255 - local_region, cv2.MORPH_OPEN, vertical_kernel
                )
                vertical_profile = np.mean(morph_result, axis=0)
            else:
                vertical_profile = np.mean(255 - local_region, axis=0)

            # Method 2: Gradient-based edge detection
            if local_region.shape[1] > 3:
                sobel_x = cv2.Sobel(local_region, cv2.CV_64F, 1, 0, ksize=3)
                gradient_magnitude = np.abs(sobel_x)
                gradient_profile = np.mean(gradient_magnitude, axis=0)
            else:
                gradient_profile = np.zeros(local_region.shape[1])

            # Method 3: Template matching for vertical lines
            if local_region.shape[0] >= 5 and local_region.shape[1] >= 3:
                template_height = min(local_region.shape[0] // 2, 15)
                template = np.ones((template_height, 3), dtype=np.uint8) * 128
                template[:, 1] = 0  # Dark center line

                try:
                    template_result = cv2.matchTemplate(
                        local_region, template, cv2.TM_CCOEFF_NORMED
                    )
                    if template_result.size > 0:
                        template_profile = np.mean(template_result, axis=0)
                        # Pad to match other profiles
                        if len(template_profile) < len(vertical_profile):
                            pad_width = len(vertical_profile) - len(template_profile)
                            template_profile = np.pad(
                                template_profile, (0, pad_width), mode="constant"
                            )
                    else:
                        template_profile = np.zeros(len(vertical_profile))
                except:
                    template_profile = np.zeros(len(vertical_profile))
            else:
                template_profile = np.zeros(len(vertical_profile))

            # Normalize profiles
            def safe_normalize(profile):
                if len(profile) == 0 or np.max(profile) == 0:
                    return profile
                return profile / np.max(profile)

            vertical_norm = safe_normalize(vertical_profile)
            gradient_norm = safe_normalize(gradient_profile)
            template_norm = safe_normalize(template_profile)

            # Ensure all profiles have the same length
            min_length = min(len(vertical_norm), len(gradient_norm), len(template_norm))
            if min_length > 0:
                vertical_norm = vertical_norm[:min_length]
                gradient_norm = gradient_norm[:min_length]
                template_norm = template_norm[:min_length]

                # Combine profiles with weights
                combined_profile = (
                    0.4 * vertical_norm + 0.4 * gradient_norm + 0.2 * template_norm
                )

                # Find the best peak in the local region
                if len(combined_profile) > 0:
                    try:
                        peaks, _ = find_peaks(
                            combined_profile,
                            distance=max(3, len(combined_profile) // 10),
                            prominence=np.std(combined_profile) * 0.2,
                        )

                        if len(peaks) > 0:
                            # Choose the peak closest to the center (original approximate position)
                            center_idx = len(combined_profile) // 2
                            best_peak = peaks[np.argmin(np.abs(peaks - center_idx))]
                            refined_x = x_start + best_peak
                        else:
                            # No peaks found, use approximate position
                            refined_x = approx_x
                    except:
                        refined_x = approx_x
                else:
                    refined_x = approx_x
            else:
                refined_x = approx_x

            # Ensure refined position is within image bounds
            refined_x = max(0, min(width - 1, refined_x))
            refined_positions.append(int(refined_x))

        return refined_positions

    def start_selection(self, title="Select Six CBD Tick Marks for Maximum Accuracy"):
        """Start interactive selection with crosshair cursor."""
        height, width = self.image_full.shape

        # REMOVE the top portion where sprocket holes are located
        sprocket_height = int(height * self.sprocket_removal_ratio)
        search_start = sprocket_height
        search_height = int(height * self.search_height_ratio)
        search_end = search_start + search_height

        # Extract the clean region BELOW the sprocket holes
        clean_region = self.image_full[search_start:search_end, :]

        # Create figure optimized for the focused view
        self.fig, self.ax = plt.subplots(figsize=(15, 8))  # Match TERRA's 1.5:1 width ratio

        # Enhanced preprocessing specifically for tick mark visibility
        clahe = cv2.createCLAHE(clipLimit=8.0, tileGridSize=(4, 4))
        enhanced_region = clahe.apply(clean_region)

        # Apply strong unsharp masking to make tick marks very prominent
        gaussian_blur = cv2.GaussianBlur(enhanced_region, (0, 0), 3.0)
        unsharp_mask = cv2.addWeighted(enhanced_region, 2.0, gaussian_blur, -1.0, 0)
        final_region = np.clip(unsharp_mask, 0, 255).astype(np.uint8)

        # Display with correct coordinate mapping
        self.ax.imshow(
            final_region,
            cmap="gray",
            aspect="auto",
            extent=[0, width, search_end, search_start],
        )

        # Initialize crosshair lines
        self.crosshair_v = self.ax.axvline(
            x=0,
            color="yellow",
            linestyle="-",
            linewidth=1,
            alpha=0.8,
            visible=False,
            zorder=20,
        )
        self.crosshair_h = self.ax.axhline(
            y=0,
            color="yellow",
            linestyle="-",
            linewidth=1,
            alpha=0.8,
            visible=False,
            zorder=20,
        )

        # Connect mouse motion event for crosshair
        self.motion_cid = self.fig.canvas.mpl_connect(
            "motion_notify_event", self._on_mouse_move
        )

        # Enhanced title with crosshair instructions
        self.ax.set_title(
            f"{title}\nClick to select first CBD tick mark\n",
            fontsize=16,
            fontweight="bold",
            pad=20,
        )

        self.ax.set_xlabel("X Position (pixels)", fontsize=14)
        self.ax.set_ylabel("Y Position (pixels)", fontsize=14)

        # Enhanced grid for precision
        self.ax.grid(True, alpha=0.6, linestyle="-", linewidth=0.8, color="cyan")

        # Set limits to match the clean view
        self.ax.set_xlim(0, width)
        self.ax.set_ylim(search_end, search_start)

        # Updated instructions with crosshair info
        self.ax.text(
            0.98,
            0.98,
            "CBD TICK MARK SELECTION:\n"
            "• Click on leftmost CBD tick mark\n"
            "• Click on next tick mark to the right\n"
            "• Close window when done",
            transform=self.ax.transAxes,
            fontsize=12,
            verticalalignment="top",
            horizontalalignment="right",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgreen", alpha=0.95),
        )

        # Connect click event
        self.cid = self.fig.canvas.mpl_connect("button_press_event", self._on_click)

        # Store parameters for coordinate adjustment
        self.search_start = search_start
        self.search_end = search_end
        self.sprocket_height = sprocket_height

        plt.tight_layout()
        plt.show()

        return self.selected_points

    def _on_mouse_move(self, event):
        """Handle mouse movement to update crosshair position."""
        if event.inaxes == self.ax:
            # Update crosshair position
            self.crosshair_v.set_xdata([event.xdata])
            self.crosshair_h.set_ydata([event.ydata])
            self.crosshair_v.set_visible(True)
            self.crosshair_h.set_visible(True)
            self.fig.canvas.draw_idle()
        else:
            self.crosshair_v.set_visible(False)
            self.crosshair_h.set_visible(False)
            self.fig.canvas.draw_idle()

    def _on_click(self, event):
        """Handle mouse click events with improved visual markers."""
        if event.inaxes != self.ax:
            return

        if len(self.selected_points) < self.expected_count:
            x, y = event.xdata, event.ydata
            self.selected_points.append((int(x), int(y)))

            # Improved visual markers - cycle through colors for any number of CBDs
            colors = ["red", "blue", "green", "orange", "purple", "cyan", "magenta", "brown", "pink", "gray", "olive", "navy", "teal", "lime", "indigo", "coral"]
            color = colors[(len(self.selected_points) - 1) % len(colors)]
            label = f"CBD Tick #{len(self.selected_points)}"

            self.ax.plot(
                x,
                y,
                "o",
                color=color,
                markersize=12,
                markeredgewidth=2,
                markeredgecolor="white",
                markerfacecolor=color,
                label=label,
                zorder=15,
                alpha=0.8,
            )

            self.ax.axvline(
                x=x,
                color=color,
                linestyle="-",
                alpha=0.6,
                linewidth=2,
                zorder=14,
            )

            self.ax.annotate(
                f"{label}\nX: {int(x)}",
                xy=(x, y),
                xytext=(15, 15),
                textcoords="offset points",
                bbox=dict(boxstyle="round,pad=0.3", fc=color, alpha=0.7),
                fontsize=12,
                color="white",
                fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=color, lw=2),
                zorder=16,
            )

            if len(self.selected_points) == self.expected_count:
                self._calculate_all_ticks()
                self._show_calculated_ticks()

            self.ax.legend(fontsize=12, loc="upper right")
            self.fig.canvas.draw()

    def _calculate_all_ticks(self):
        """Use all manually selected CBD positions directly (no spacing calculation needed)."""
        if len(self.selected_points) != self.expected_count:
            print(f"Warning: Expected {self.expected_count} points but got {len(self.selected_points)}")
            return

        # Sort points by x-coordinate to ensure proper order
        sorted_points = sorted(self.selected_points, key=lambda p: p[0])
        
        # Use the exact manually selected positions directly
        calculated_ticks = [int(point[0]) for point in sorted_points]
        
        print(f"Using {len(calculated_ticks)} manually selected CBD positions (no spacing calculation needed)")
        print(f"Selected CBD positions: {calculated_ticks}")

        # Use exact calculated positions without any refinement
        self.calculated_ticks = calculated_ticks
        self.refined_ticks = calculated_ticks.copy()  # Same as calculated - no refinement

        print(
            f"Using {len(self.calculated_ticks)} manually selected CBD tick positions"
        )
        print(f"CBD tick positions: {self.calculated_ticks}")

    def _show_calculated_ticks(self):
        """Show the exact calculated tick positions (no refinement)."""
        for i, tick_x in enumerate(self.calculated_ticks):
            # Show exact calculated position - draw ALL ticks regardless of image bounds
            self.ax.axvline(
                x=tick_x,
                color="green",
                linestyle="-",
                alpha=0.8,
                linewidth=2,
                zorder=13,
            )
            print(f"DEBUG: Drew green line {i+1} at x={tick_x}")

            # Tick labels - show labels for even indices AND the last tick
            if i % 2 == 0 or i == len(self.calculated_ticks) - 1:
                self.ax.text(
                    tick_x,
                    self.search_start + (self.search_end - self.search_start) * 0.15,
                    f"T{i + 1}",
                    ha="center",
                    va="bottom",
                    fontsize=10,
                    color="green",
                    fontweight="bold",
                    zorder=17,
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.8),
                )
                print(f"DEBUG: Drew label T{i+1} at x={tick_x}")
            else:
                print(f"DEBUG: Skipped label T{i+1} (odd index)")

        # Update title with results
        self.ax.set_title(
            f"CBD Tick Mark Selection Complete \n"
            f"{len(self.calculated_ticks)} ticks: manually selected positions\n"
            f"Green lines show exact selected positions\n"
            f"Yellow crosshair provides precise alignment feedback",
            fontsize=16,
            fontweight="bold",
            pad=20,
        )

        self.fig.canvas.draw()

    def get_tick_positions(self):
        """Return the calculated tick positions (based on manual selection without local refinement)."""
        return self.calculated_ticks if self.calculated_ticks else []

    def cleanup(self):
        """Clean up event connections to prevent memory leaks."""
        if hasattr(self, "cid") and self.cid is not None:
            self.fig.canvas.mpl_disconnect(self.cid)
            self.cid = None

        if hasattr(self, "motion_cid") and self.motion_cid is not None:
            self.fig.canvas.mpl_disconnect(self.motion_cid)
            self.motion_cid = None


def manual_cbd_tick_selection(
    image_full,
    expected_count=13,  # Keep 13 as default for backward compatibility
    sprocket_removal_ratio=0.08,
    search_height_ratio=0.12,
    debug=False,
):
    """
    Enhanced manual CBD tick mark selection with local image recognition refinement.

    Args:
        image_full: Full radar image array
        expected_count: Expected number of CBD tick marks (default: 13)
        sprocket_removal_ratio: Ratio of top region to remove (sprocket holes) (default: 0.08)
        search_height_ratio: Height ratio for CBD search region below sprockets (default: 0.12)
        debug: Enable debug output

    Returns:
        List of x-coordinates for all CBD tick marks (refined positions)
    """
    print("Starting ENHANCED manual CBD tick mark selection...")
    print(
        "This version uses uniform spacing as a guide + local image recognition for refinement."
    )
    print(f"Please select all {expected_count} CBD tick marks in sequence from left to right for maximum accuracy.")

    selector = CBDTickSelector(
        image_full, expected_count, sprocket_removal_ratio, search_height_ratio
    )
    selected_points = selector.start_selection()

    if len(selected_points) < 2:
        print("Warning: Less than 2 points selected. Using fallback method.")
        return []

    tick_positions = selector.get_tick_positions()

    if debug:
        print(f"Manual selection complete:")
        print(f"  Selected points: {selected_points}")
        print(
            f"  Calculated spacing: {abs(selected_points[1][0] - selected_points[0][0])} pixels"
        )
        print(
            f"  Generated {len(tick_positions)} refined tick positions: {tick_positions}"
        )
        if hasattr(selector, "refined_ticks") and selector.refined_ticks:
            adjustments = [
                abs(r - a)
                for r, a in zip(selector.refined_ticks, selector.calculated_ticks)
            ]
            print(f"  Average refinement adjustment: {np.mean(adjustments):.1f} pixels")

    return tick_positions


def validate_manual_selection_enhanced(
    image_full, tick_positions, base_filename, output_dir
):
    """Create enhanced validation plot with complete sprocket hole removal and focused view."""
    if not tick_positions:
        return

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))  # Match TERRA's 1.5:1 width ratio

    # Show clean, focused region with detected ticks
    height, width = image_full.shape
    search_height = max(30, int(height * 0.12))
    sprocket_height = int(height * 0.08)

    # Extract clean region
    clean_region = image_full[sprocket_height : sprocket_height + search_height, :]

    # Apply same enhancement as in selection interface
    clahe = cv2.createCLAHE(clipLimit=8.0, tileGridSize=(4, 4))
    enhanced_region = clahe.apply(clean_region)

    gaussian_blur = cv2.GaussianBlur(enhanced_region, (0, 0), 3.0)
    final_region = cv2.addWeighted(enhanced_region, 2.0, gaussian_blur, -1.0, 0)
    final_region = np.clip(final_region, 0, 255).astype(np.uint8)

    ax1.imshow(
        final_region,
        cmap="gray",
        aspect="auto",
        extent=[0, width, sprocket_height + search_height, sprocket_height],
    )

    # Plot tick marks with enhanced visibility
    for i, tick_x in enumerate(tick_positions):
        if 0 <= tick_x < width:
            ax1.axvline(x=tick_x, color="red", linewidth=4, alpha=0.9, zorder=10)
            if i % 2 == 0:  # Label every other tick
                ax1.text(
                    tick_x,
                    sprocket_height + search_height * 0.2,
                    f"CBD{i + 1}",
                    ha="center",
                    va="bottom",
                    fontsize=14,
                    color="red",
                    fontweight="bold",
                    zorder=11,
                    bbox=dict(boxstyle="round,pad=0.4", fc="white", alpha=0.95),
                )

    ax1.set_title(
        f"Manual CBD Tick Selection Results: {base_filename}\n"
        f"{len(tick_positions)} ticks detected with local image recognition refinement\n",
        fontsize=16,
    )
    ax1.set_xlabel("X Position (pixels)", fontsize=14)
    ax1.set_ylabel("Y Position (pixels)", fontsize=14)
    ax1.grid(True, alpha=0.4, color="cyan")

    # Enhanced spacing analysis
    if len(tick_positions) > 1:
        spacings = np.diff(tick_positions)
        bars = ax2.bar(
            range(len(spacings)),
            spacings,
            alpha=0.8,
            color="steelblue",
            edgecolor="navy",
            linewidth=2,
        )

        # Add value labels on bars
        for i, (bar, spacing) in enumerate(zip(bars, spacings)):
            height = bar.get_height()
            ax2.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 1,
                f"{spacing:.0f}",
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold",
            )

        ax2.axhline(
            y=np.mean(spacings),
            color="red",
            linestyle="--",
            linewidth=3,
            label=f"Mean Spacing: {np.mean(spacings):.1f} pixels",
        )
        ax2.set_title(
            f"Spacing Between Adjacent CBD Ticks\n"
            f"Standard Deviation: {np.std(spacings):.1f} pixels (Lower is better)",
            fontsize=16,
        )
        ax2.set_xlabel("Tick Pair Index", fontsize=14)
        ax2.set_ylabel("Spacing (pixels)", fontsize=14)
        ax2.legend(fontsize=14)
        ax2.grid(True, alpha=0.4)

        # Enhanced statistics
        uniformity = (
            (1 - np.std(spacings) / np.mean(spacings)) * 100
            if np.mean(spacings) > 0
            else 0
        )
        ax2.text(
            0.02,
            0.98,
            f"ENHANCED SPACING STATISTICS:\n"
            f"Min: {np.min(spacings):.1f} px\n"
            f"Max: {np.max(spacings):.1f} px\n"
            f"Mean: {np.mean(spacings):.1f} px\n"
            f"Std Dev: {np.std(spacings):.1f} px\n"
            f"Uniformity: {uniformity:.1f}%\n"
            f"Range: {np.max(spacings) - np.min(spacings):.1f} px\n"
            f"Method: Local Image Recognition",
            transform=ax2.transAxes,
            fontsize=12,
            verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.95),
        )
    else:
        ax2.text(
            0.5,
            0.5,
            "Not enough ticks for spacing analysis",
            ha="center",
            va="center",
            transform=ax2.transAxes,
            fontsize=18,
        )
        ax2.set_title("Spacing Analysis", fontsize=16)

    plt.tight_layout()
    plt.savefig(
        output_dir / f"{base_filename}_enhanced_local_refinement_validation.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


def align_cbd_labels_with_manual_selection(
    ax, cbd_tick_xs, cbd_list, nav_df, base_filename
):
    """Enhanced alignment function for manual selection results with local refinement."""

    if len(cbd_tick_xs) != len(cbd_list):
        print(f"Info: Adjusting CBD list to match {len(cbd_tick_xs)} detected ticks")

        if len(cbd_list) > len(cbd_tick_xs):
            # Take the first N CBDs
            cbd_list = cbd_list[: len(cbd_tick_xs)]
        elif len(cbd_list) < len(cbd_tick_xs):
            # Extend CBD list if needed (though this should be rare)
            last_cbd = cbd_list[-1]
            direction = -1 if len(cbd_list) > 1 and cbd_list[1] < cbd_list[0] else 1

            while len(cbd_list) < len(cbd_tick_xs):
                last_cbd += direction
                cbd_list.append(last_cbd)

    # Get navigation data for each CBD
    cbd_data = []
    print(f"DEBUG: Looking for CBD numbers: {cbd_list}")
    print(f"DEBUG: Navigation file contains {len(nav_df['CBD'].unique())} unique CBD entries")
    
    for cbd in cbd_list:
        row = nav_df[nav_df["CBD"] == cbd]
        if not row.empty:
            # Use only basic CBD, LAT, LON columns
            print(f"DEBUG: Found CBD {cbd} in navigation data: LAT={row['LAT'].values[0]}, LON={row['LON'].values[0]}")
            cbd_data.append(
                {
                    "cbd": cbd,
                    "lat": row["LAT"].values[0],
                    "lon": row["LON"].values[0],
                }
            )
        else:
            print(f"WARNING: CBD {cbd} not found in navigation data")
            cbd_data.append(
                {
                    "cbd": cbd,
                    "lat": np.nan,
                    "lon": np.nan,
                }
            )

    # Create formatted labels
    labels = []
    for data in cbd_data:
        if np.isnan(data["lat"]):
            labels.append(f"{data['cbd']}\nN/A")
        else:
            labels.append(
                f"{data['cbd']}\n"
                f"{data['lat']:.3f},{data['lon']:.3f}"
            )

    # Set ticks and labels with precise alignment
    ax.set_xticks(cbd_tick_xs)
    ax.set_xticklabels(labels, rotation=0, fontsize=8, ha="center")

    # Add subtle vertical lines for visual confirmation
    for x_pos in cbd_tick_xs:
        ax.axvline(x=x_pos, color="blue", linestyle=":", alpha=0.4, linewidth=1)

    return cbd_tick_xs, labels


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
    approx_x_click=None,
    visualization_params=None,
    surface_y_abs=None,
    bed_y_abs=None,
    nav_df=None,
    nav_path=None,
    main_output_dir=None,
    processor_ref=None,
    calpip_y_lines=None,
    calpip_method=None,
):
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "axes.labelsize": 16,
            "axes.titlesize": 18,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "legend.fontsize": 13,
        }
    )

    if time_vis_params is None:
        time_vis_params = {}
    if output_params is None:
        output_params = {}

    output_dir_name = output_params.get("debug_output_directory", "debug_output")
    if main_output_dir is not None:
        output_dir = Path(main_output_dir) / Path(output_dir_name).name
    else:
        output_dir = Path(output_dir_name)
    output_dir.mkdir(parents=True, exist_ok=True)

    save_dpi = output_params.get("figure_save_dpi", 600)
    legend_loc = time_vis_params.get("legend_location", "upper right")

    img_height = data_bottom_abs - data_top_abs
    img_width = image_full.shape[1]

    # Margin above transmitter pulse for context
    margin_above_tx_us = 10
    margin_above_tx_px = int(margin_above_tx_us * pixels_per_microsecond)
    cropped_top = max(0, transmitter_pulse_y_abs - margin_above_tx_px)

    # Dynamic bottom based on bed echo
    if bed_y_abs is not None and np.any(np.isfinite(bed_y_abs)):
        max_bed_y = np.nanpercentile(bed_y_abs, 95)
        margin_us = 20
        margin_px = int(margin_us * pixels_per_microsecond)
        cropped_bottom = min(data_bottom_abs, int(max_bed_y + margin_px))
    else:
        cropped_bottom = data_bottom_abs

    valid_data_crop = image_full[cropped_top:cropped_bottom, :]

    colormap = "Greys_r"  # Fixed: Changed from "Grays_r" to "Greys_r"
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_data = clahe.apply(
        cv2.normalize(valid_data_crop, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    )

    # Match TERRA's fixed window proportions (1200x800 ratio = 1.5:1)
    fig, ax = plt.subplots(figsize=(15, 10))  # 15:10 = 1.5:1 ratio like TERRA

    # Store processor reference for CBD tick storage
    if processor_ref is not None:
        ax._processor_ref = processor_ref

    ax.imshow(
        enhanced_data,
        cmap=colormap,
        aspect="auto",  # Keep auto but constrain figure to TERRA proportions
        extent=[0, img_width, cropped_bottom, cropped_top],
    )

    ax.set_ylim(cropped_bottom, cropped_top)
    ax.yaxis.set_ticks([])
    ax.set_ylabel("")

    # Calibration pip and transmitter pulse overlays
    # Only show vertical "Picked calpip" line for ARIES method
    if best_pip and "x_position" in best_pip and calpip_method == "ARIES":
        ax.axvline(
            x=best_pip["x_position"],
            color="#DCC462",
            linestyle=":",
            linewidth=2.5,
            alpha=0.9,
            label="Picked calpip",
            zorder=5,
        )

    ax.axhline(
        y=transmitter_pulse_y_abs,
        color="#0072B2",
        linestyle="-",
        linewidth=3,
        alpha=0.9,
        label="Transmitter Pulse",
        zorder=5,
    )

    ax.text(
        60,
        transmitter_pulse_y_abs,
        "0 µs (Tx Pulse)",
        color="#0072B2",
        fontsize=14,
        fontweight="bold",
        va="center",
        path_effects=[
            path_effects.withStroke(linewidth=3, foreground="white", alpha=0.7)
        ],
        zorder=6,
    )

    # Grid lines (time) - Use TERRA-style positioning if available, otherwise mathematical
    # Determine calpip spacing (pixels per 2 μs) from processor if available
    if hasattr(ax, '_processor_ref') and hasattr(ax._processor_ref, 'calpip_pixel_distance') and ax._processor_ref.calpip_pixel_distance:
        calpip_distance = ax._processor_ref.calpip_pixel_distance  # px per 2μs
    else:
        # fallback: derive pixels-per-2μs from pixels_per_microsecond
        calpip_distance = pixels_per_microsecond * 2.0

    # Use provided calpip_y_lines when available; otherwise, if we have a calpip spacing
    # from the processor (e.g., reusing calibration), synthesize a uniform grid anchored
    # to the transmitter pulse so the visual axes remain consistent between images.
    calpip_lines_for_display = None
    if calpip_y_lines and len(calpip_y_lines) > 0:
        calpip_lines_for_display = calpip_y_lines
        is_terra_method = calpip_method == "TERRA"
        line_color = "grey" if is_terra_method else "white"
        method_name = "TERRA manual picks" if is_terra_method else "ARIES auto-detection"
        print(f"Using {method_name} grid positioning with {len(calpip_lines_for_display)} lines")
    else:
        # No explicit calpip y-lines provided; synthesize grid if spacing available
        calpip_lines_for_display = []
        # Generate calpip lines below transmitter pulse at multiples of calpip_distance
        y = transmitter_pulse_y_abs + calpip_distance
        while y <= cropped_bottom:
            calpip_lines_for_display.append(float(y))
            y += calpip_distance
        method_name = "synthesized from calpip_spacing" if len(calpip_lines_for_display) > 0 else "none"
        line_color = "white"
        is_terra_method = False
        if len(calpip_lines_for_display) > 0:
            print(f"INFO: Synthesized {len(calpip_lines_for_display)} calpip lines from spacing {calpip_distance:.2f}px")

    # Single concise runtime calibration info line (safe/noisy) — useful for validation
    try:
        print(f"INFO: calibration: calpip_distance={calpip_distance:.3f} px/2μs, pixels_per_microsecond={pixels_per_microsecond:.6f} px/μs")
    except Exception:
        pass

    # Make a working copy used for axis computations. Prefer measured spacing
    # from visible calpip lines (below TX) when available to ensure labels match grid.
    calpip_distance_used = calpip_distance
    try:
        visible_lines_all = [y for y in calpip_lines_for_display if cropped_top <= y <= cropped_bottom]
        visible_below_tx = [y for y in visible_lines_all if y > transmitter_pulse_y_abs]
        if len(visible_below_tx) >= 3:
            spacings = np.diff(visible_below_tx)
            measured_spacing = float(np.mean(spacings))
            # Override calpip_distance_used with measured spacing to match displayed grid
            if measured_spacing > 0:
                print(f"INFO: Overriding calpip_distance with measured spacing {measured_spacing:.3f}px (from {len(visible_below_tx)} visible lines)")
                calpip_distance_used = measured_spacing
    except Exception:
        pass

    # If we have calpip positions (either provided or synthesized), draw those; otherwise
    # fall back to the original ARIES mathematical approach based on pixels_per_microsecond.
    if calpip_lines_for_display and len(calpip_lines_for_display) > 0:
        # Draw grid lines at the chosen calpip positions
        for i, y_line in enumerate(calpip_lines_for_display):
            if cropped_top <= y_line <= cropped_bottom:
                # Calculate time (μs) using calpip_distance_used as px per 2μs
                time_us = (y_line - transmitter_pulse_y_abs) / calpip_distance_used * 2.0

                # For ARIES method or synthesized grid, use white lines with major/minor styling
                # For TERRA manual method we intentionally skip drawing the grey grid lines.
                if not is_terra_method:
                    # Major grid every 10μs, minor every 2μs for ARIES
                    is_major_grid = (abs(time_us) % 10) < 1  # Approximately every 10μs

                    ax.axhline(
                        y=y_line,
                        color=line_color,
                        linestyle="-" if is_major_grid else "--",
                        alpha=0.8 if is_major_grid else 0.3,
                        linewidth=2 if is_major_grid else 1.5,
                        zorder=2,
                    )
    else:
        # Original ARIES mathematical approach
        total_time_range_us = (
            cropped_bottom - transmitter_pulse_y_abs
        ) / pixels_per_microsecond
        min_time_us = (cropped_top - transmitter_pulse_y_abs) / pixels_per_microsecond

        major_grid_interval_us = time_vis_params.get("major_grid_time_interval_us", 10)
        minor_grid_interval_us = time_vis_params.get("minor_grid_time_interval_us", 2)

        t_us_vals = np.arange(
            min_time_us,
            total_time_range_us + minor_grid_interval_us,
            minor_grid_interval_us,
        )

        for t_us in t_us_vals:
            pixel_y_abs_coord = transmitter_pulse_y_abs + t_us * pixels_per_microsecond
            if cropped_top <= pixel_y_abs_coord <= cropped_bottom:
                is_major_grid = round(t_us, 6) % major_grid_interval_us == 0

                ax.axhline(
                    y=pixel_y_abs_coord,
                    color="white",
                    linestyle="-" if is_major_grid else "--",
                    alpha=0.8 if is_major_grid else 0.2,
                    linewidth=2 if is_major_grid else 1.5,
                    zorder=2,
                )

                # Skip text labels on grid lines - time axis will handle labeling
                pass

    # --- MANUAL CBD TICK MARK SELECTION WITH LOCAL REFINEMENT ---
    detection_method = output_params.get("cbd_detection_method", "manual")

    if detection_method == "manual":
        print("Using manual CBD tick mark selection with local image recognition...")
        # Dynamically calculate expected CBD count from filename
        expected_cbd_count = extract_cbd_count_from_filename(base_filename)
        
        cbd_tick_xs = manual_cbd_tick_selection(
            image_full,
            expected_count=expected_cbd_count,
            sprocket_removal_ratio=output_params.get("sprocket_removal_ratio", 0.08),
            search_height_ratio=output_params.get("search_height_ratio", 0.12),
            debug=output_params.get("debug_tick_detection", False),
        )
    else:
        # Fallback to automated methods if needed
        print("Manual selection not specified, using automated detection...")
        cbd_tick_xs = []

    # Load navigation data
    print(f"DEBUG: nav_df is None: {nav_df is None}, nav_path: {nav_path}")
    if nav_df is None and nav_path is not None:
        print(f"DEBUG: Loading navigation data from: {nav_path}")
        nav_df = pd.read_csv(nav_path)
        print(f"DEBUG: Loaded nav_df with shape: {nav_df.shape}")

    if nav_df is None:
        flight_match = re.search(r"F(\d+)", base_filename)
        if flight_match:
            flight_num = flight_match.group(1)
            nav_file_guess = f"merged_{flight_num}_nav.csv"
            print(f"DEBUG: Trying fallback nav file: {nav_file_guess}")
            if Path(nav_file_guess).exists():
                nav_df = pd.read_csv(nav_file_guess)
                print(f"DEBUG: Loaded fallback nav_df with shape: {nav_df.shape}")
            else:
                print(f"DEBUG: Fallback nav file not found: {nav_file_guess}")
        else:
            print(f"DEBUG: Could not extract flight number from: {base_filename}")
    
    print(f"DEBUG: CBD tick detection found {len(cbd_tick_xs)} ticks: {cbd_tick_xs}")

    print(f"DEBUG: Checking CBD annotation conditions - nav_df exists: {nav_df is not None}, cbd_tick_xs count: {len(cbd_tick_xs)}")
    
    if nav_df is not None and len(cbd_tick_xs) > 0:
        print("DEBUG: Entering CBD annotation section")
        # Try pattern with C prefix on both numbers: C0565_C0578
        cbd_match = re.search(r"C(\d+)_C(\d+)", base_filename)
        if not cbd_match:
            # Try pattern with C prefix only on first number: C0565_0578  
            cbd_match = re.search(r"C(\d+)_(\d+)", base_filename)
        
        print(f"DEBUG: Regex search result: {cbd_match}")
        if cbd_match:
            cbd_start = int(cbd_match.group(1))
            cbd_end = int(cbd_match.group(2))
            print(f"DEBUG: Extracted CBD range from filename: {cbd_start} to {cbd_end}")

            # Create CBD list in descending order (largest to smallest, left to right)
            if cbd_start > cbd_end:
                cbd_list = list(range(cbd_start, cbd_end - 1, -1))
            else:
                cbd_list = list(range(cbd_start, cbd_end + 1))
                cbd_list.reverse()  # Reverse to get largest first
            
            print(f"DEBUG: Generated CBD list: {cbd_list}")

            # Use enhanced alignment with local refinement
            print("DEBUG: Calling align_cbd_labels_with_manual_selection")
            cbd_tick_xs, cbd_labels = align_cbd_labels_with_manual_selection(
                ax, cbd_tick_xs, cbd_list, nav_df, base_filename
            )
            print(f"DEBUG: CBD alignment completed, labels: {cbd_labels}")
            ax.set_xlabel(
                "CBD\nLat, Lon",
                fontsize=14,
                fontweight="bold",
            )

            # Create enhanced validation plot
            if output_params.get("create_validation_plot", True):
                validate_manual_selection_enhanced(
                    image_full, cbd_tick_xs, base_filename, output_dir
                )
        else:
            print(f"DEBUG: Could not extract CBD range from filename: {base_filename}")
    else:
        print("DEBUG: Skipping CBD annotation - either no nav_df or no CBD ticks detected")

    # Surface, Bed, and Ice Thickness Labels (existing code remains the same)
    if surface_y_abs is not None and np.any(np.isfinite(surface_y_abs)):
        x_coords = np.arange(len(surface_y_abs))
        valid = np.isfinite(surface_y_abs)
        ax.plot(
            x_coords[valid],
            surface_y_abs[valid],
            color="#009E73",
            linestyle="--",
            linewidth=3,
            label="Surface",
            zorder=6,
        )

    if bed_y_abs is not None and np.any(np.isfinite(bed_y_abs)):
        x_coords = np.arange(len(bed_y_abs))
        valid = np.isfinite(bed_y_abs)
        ax.plot(
            x_coords[valid],
            bed_y_abs[valid],
            color="#D55E00",
            linestyle="--",
            linewidth=3,
            label="Bed",
            zorder=6,
        )

    # Ice thickness calculations
    if (
        surface_y_abs is not None
        and bed_y_abs is not None
        and np.any(np.isfinite(surface_y_abs))
        and np.any(np.isfinite(bed_y_abs))
        and physics_constants is not None
    ):
        valid_surface = surface_y_abs[np.isfinite(surface_y_abs)]
        valid_bed = bed_y_abs[np.isfinite(bed_y_abs)]

        if len(valid_surface) > 0 and len(valid_bed) > 0:
            avg_surface_y = np.mean(valid_surface)
            avg_bed_y = np.mean(valid_bed)

            c0 = physics_constants.get("speed_of_light_vacuum_mps")
            epsilon_r_ice = physics_constants.get("ice_relative_permittivity_real")
            firn_corr_m = physics_constants.get("firn_correction_meters")

            def abs_y_to_depth_annotation(y_abs):
                y_rel = y_abs - transmitter_pulse_y_abs
                # Use TERRA's exact methodology: depth = (y_pixel/calpip_distance) * 168
                if hasattr(ax, '_processor_ref') and hasattr(ax._processor_ref, 'calpip_pixel_distance') and ax._processor_ref.calpip_pixel_distance:
                    calpip_distance = ax._processor_ref.calpip_pixel_distance
                    velocity_ice = 168.0  # m/μs - TERRA's exact velocity
                    return (y_rel / calpip_distance) * velocity_ice
                else:
                    # Fallback: use two-way travel time calculation
                    two_way_time_us = y_rel / pixels_per_microsecond
                    velocity_ice = 168.0  # m/μs - TERRA's exact velocity
                    return (two_way_time_us / 2) * velocity_ice

            avg_surface_depth = abs_y_to_depth_annotation(avg_surface_y)
            avg_bed_depth = abs_y_to_depth_annotation(avg_bed_y)
            
            # TERRA's ice thickness: h_ice = (pixel_difference / calpip_distance) * 168
            pixel_difference = avg_bed_y - avg_surface_y
            if hasattr(ax, '_processor_ref') and hasattr(ax._processor_ref, 'calpip_pixel_distance') and ax._processor_ref.calpip_pixel_distance:
                calpip_distance = ax._processor_ref.calpip_pixel_distance
                ice_thickness = (pixel_difference / calpip_distance) * 168.0
            else:
                # Fallback: use depth difference
                ice_thickness = avg_bed_depth - avg_surface_depth

            x_pos = 0.80 * img_width

            ax.annotate(
                f"Surface: {avg_surface_depth:.1f} m",
                xy=(x_pos, avg_surface_y),
                xytext=(10, -5),
                textcoords="offset points",
                color="#009E73",
                fontsize=14,
                fontweight="bold",
                bbox=dict(
                    boxstyle="round,pad=0.3", fc="white", ec="#009E73", alpha=0.8
                ),
                zorder=10,
            )

            ax.annotate(
                f"Bed: {avg_bed_depth:.1f} m",
                xy=(x_pos, avg_bed_y),
                xytext=(10, 5),
                textcoords="offset points",
                color="#D55E00",
                fontsize=14,
                fontweight="bold",
                bbox=dict(
                    boxstyle="round,pad=0.3", fc="white", ec="#D55E00", alpha=0.8
                ),
                zorder=10,
            )

            ax.annotate(
                f"Ice thickness: {ice_thickness:.1f} m",
                xy=(x_pos, (avg_surface_y + avg_bed_y) / 2),
                xytext=(10, 0),
                textcoords="offset points",
                color="black",
                fontsize=14,
                fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", alpha=0.8),
                zorder=10,
            )

    ax.legend(
        loc=legend_loc,
        frameon=True,
        facecolor="white",
        edgecolor="black",
        fontsize=14,
        fancybox=True,
        borderpad=1.2,
    )

    # Depth and time axes
    def abs_y_to_depth(y_abs_coord):
        # Explicit TERRA mapping: calpip_distance is pixels per 2μs. 2 μs -> 168 m.
        y_rel = y_abs_coord - transmitter_pulse_y_abs

        # Determine calpip spacing in pixels corresponding to 2 μs
        if hasattr(ax, '_processor_ref') and hasattr(ax._processor_ref, 'calpip_pixel_distance') and ax._processor_ref.calpip_pixel_distance:
            calpip_distance = ax._processor_ref.calpip_pixel_distance  # px per 2μs
        else:
            calpip_distance = pixels_per_microsecond * 2.0  # fallback px per 2μs

        velocity_ice = 168.0  # m per μs (TERRA exact)

        # Depth (m) = (y_rel / calpip_distance) * velocity_ice
        # Reason: y_rel == calpip_distance -> corresponds to 2 μs TWT -> depth = (2/2)*V = V
        depth_m = (y_rel / calpip_distance) * velocity_ice
        return depth_m

    def depth_to_abs_y(depth_m):
        # Inverse of the explicit TERRA mapping used above.
        velocity_ice = 168.0  # m per μs

        if hasattr(ax, '_processor_ref') and hasattr(ax._processor_ref, 'calpip_pixel_distance') and ax._processor_ref.calpip_pixel_distance:
            calpip_distance = ax._processor_ref.calpip_pixel_distance  # px per 2μs
        else:
            calpip_distance = pixels_per_microsecond * 2.0

        # y_rel = (depth / V_ICE) * calpip_distance
        y_rel = (depth_m / velocity_ice) * calpip_distance
        return transmitter_pulse_y_abs + y_rel

    # Use the established calpip y-lines for depth axis labeling (TERRA methodology)
    if calpip_lines_for_display and len(calpip_lines_for_display) > 0:
        print(f"Using TERRA calpip lines for depth axis: {len(calpip_lines_for_display)} lines")
        
        # Filter calpip lines to only those visible in the current view
        visible_calpip_lines = [y_line for y_line in calpip_lines_for_display 
                      if cropped_top <= y_line <= cropped_bottom]

        # For depth and time axis labeling we must only use calpip lines below the
        # transmitter pulse (positive y_rel). This ensures depth values are
        # positive and align with the time axis (which also only uses lines below TX).
        visible_calpip_lines_below_tx = [y for y in visible_calpip_lines if y > transmitter_pulse_y_abs]

        # Small diagnostic: print the first visible calpip mapping (below TX) so user can validate
        try:
            if len(visible_calpip_lines_below_tx) > 0:
                first_y = visible_calpip_lines_below_tx[0]
                y_rel_first = first_y - transmitter_pulse_y_abs
                # use existing abs_y_to_depth to compute depth for the first visible line
                first_depth = abs_y_to_depth(first_y)
                print(
                    f"INFO: tx_y={transmitter_pulse_y_abs:.1f}, first_calpip_y={first_y:.1f}, y_rel={y_rel_first:.1f}, calpip_distance={calpip_distance:.3f}, depth_for_first={first_depth:.1f} m"
                )
                # If possible, report measured spacing between visible calpip lines (below TX) vs calpip_distance
                if len(visible_calpip_lines_below_tx) > 1:
                    try:
                        spacings = np.diff(visible_calpip_lines_below_tx)
                        measured_spacing = float(np.mean(spacings))
                        ratio = measured_spacing / float(calpip_distance) if calpip_distance != 0 else float('nan')
                        print(f"INFO: measured_spacing={measured_spacing:.3f} px, calpip_distance={calpip_distance:.3f} px, ratio={ratio:.3f}")
                    except Exception:
                        pass
        except Exception:
            pass
        
        # Calculate depth values for each visible calpip line
        depth_values = []
        tick_positions = []
        
        # Determine whether measured spacing between visible calpip lines (below TX)
        # is a better calibration than the stored `calpip_distance`. If so, use it.
        calpip_distance_used = calpip_distance
        try:
            if len(visible_calpip_lines_below_tx) > 1:
                spacings = np.diff(visible_calpip_lines_below_tx)
                measured_spacing = float(np.mean(spacings))
                # If measured spacing deviates from stored calibration, prefer measured
                if calpip_distance > 0 and abs(measured_spacing / calpip_distance - 1.0) > 0.01:
                    calpip_distance_used = measured_spacing
                    print(f"INFO: Using measured calpip spacing {measured_spacing:.3f}px instead of stored {calpip_distance:.3f}px")
        except Exception:
            measured_spacing = None

        # Use only calpip lines below the transmitter pulse for depth ticks
        # Compute depth from the pixel mapping so labels reflect pixel->time mapping:
        # depth_m = (y_rel / calpip_distance_used) * velocity_ice
        velocity_ice = 168.0
        for y_line in visible_calpip_lines_below_tx:
            y_rel = y_line - transmitter_pulse_y_abs
            try:
                depth_m = (y_rel / float(calpip_distance_used)) * velocity_ice
            except Exception:
                depth_m = np.nan
            depth_values.append(depth_m)
            tick_positions.append(y_line)
        
        # Create secondary y-axis and use the calpip line positions directly
        depth_ax = ax.secondary_yaxis("left")
        depth_ax.set_ylabel("Depth (m)", fontsize=16, fontweight="bold")
        
        # Set ticks at the exact calpip line positions
        depth_ax.set_yticks(tick_positions)
        depth_ax.set_yticklabels([f"{int(depth)}" for depth in depth_values])
        
        # Set axis limits to match main axis
        depth_ax.set_ylim(cropped_bottom, cropped_top)
        
        print(f"TERRA method: Placed {len(tick_positions)} depth labels at calpip line positions")
        print(f"TERRA method: Depth range from {min(depth_values):.0f}m to {max(depth_values):.0f}m")
        
    else:
        print("No calpip lines available, using mathematical grid for depth axis")
        # Fallback to mathematical approach when no TERRA calpip lines available
        max_depth = abs_y_to_depth(cropped_bottom)
        min_depth = abs_y_to_depth(cropped_top)
        
        # Create ticks every 500m as fallback
        start_tick = int(min_depth / 500) * 500
        if start_tick > min_depth:
            start_tick -= 500
        end_tick = int(max_depth / 500 + 1) * 500
        
        depth_ticks = list(range(start_tick, end_tick + 1, 500))
        visible_depth_ticks = [tick for tick in depth_ticks if min_depth - 200 <= tick <= max_depth + 200]
        tick_y_positions = [depth_to_abs_y(tick) for tick in visible_depth_ticks]
        
        depth_ax = ax.secondary_yaxis("left")
        depth_ax.set_ylabel("Depth (m)", fontsize=16, fontweight="bold")
        depth_ax.set_yticks(tick_y_positions)
        depth_ax.set_yticklabels([f"{int(tick)}" for tick in visible_depth_ticks])
        depth_ax.set_ylim(cropped_bottom, cropped_top)
    
    depth_ax.tick_params(
        axis="y",
        which="both",
        colors="black",
        labelcolor="black",
        direction="out",
        length=8,
        width=2,
        labelsize=14,
    )
    depth_ax.yaxis.label.set_color("black")

    # Create right-side time axis using the calpip lines directly
    time_ax = ax.twinx()
    time_ax.set_ylabel("Two-way Travel Time (μs)", fontsize=16, fontweight="bold")
    time_ax.set_ylim(ax.get_ylim())
    
    # Compute time axis ticks using the calpip_distance_used so labels reflect
    # the actual pixel→time mapping (2 μs per calpip interval).
    time_positions = [transmitter_pulse_y_abs]
    time_labels = ["0"]

    # Add only calpip lines that are BELOW the transmitter pulse (greater y-value)
    # Compute time labels from pixel-derived mapping: time_us = (y_rel / calpip_distance_used)*2.0
    if calpip_lines_for_display and len(calpip_lines_for_display) > 0:
        for calpip_y in calpip_lines_for_display:
            if calpip_y > transmitter_pulse_y_abs:  # Only lines below transmitter pulse
                try:
                    y_rel = calpip_y - transmitter_pulse_y_abs
                    time_us = (y_rel / float(calpip_distance_used)) * 2.0
                except Exception:
                    time_us = np.nan
                time_positions.append(calpip_y)
                # Show whole-number μs labels only
                if np.isfinite(time_us):
                    time_labels.append(str(int(round(time_us))))
                else:
                    time_labels.append("")
    else:
        # Fallback: Create mathematical time grid when no calpip lines available
        print("Creating mathematical time axis fallback (no calpip lines)")
        # Add time markers every 2μs below transmitter pulse
        for i in range(1, 11):  # Add up to 20μs worth of markers
            time_us = i * 2
            time_y = transmitter_pulse_y_abs + (time_us * pixels_per_microsecond)
            if time_y <= cropped_bottom:  # Only if within visible area
                time_positions.append(time_y)
                time_labels.append(str(time_us))
    
    # Set the time axis ticks and labels
    time_ax.set_yticks(time_positions)
    time_ax.set_yticklabels(time_labels)
    
    # Style the time axis
    time_ax.tick_params(
        axis="y",
        which="both",
        colors="black",
        labelcolor="black",
        direction="out",
        length=8,
        width=2,
        labelsize=14,
        right=True,
        labelright=True
    )
    time_ax.yaxis.label.set_color("black")
    time_ax.yaxis.set_label_position('right')

    # Adjust layout to accommodate both y-axes
    plt.subplots_adjust(left=0.12, right=0.85, bottom=0.10, top=0.95)

    ax.set_title(
        f"Picked Z-scope: {base_filename}",
        fontsize=20,
        fontweight="bold",
    )

    if main_output_dir is not None:
        main_plot_path = Path(main_output_dir) / f"{base_filename}_picked.png"
    else:
        main_plot_path = output_dir / f"{base_filename}_picked.png"

    fig.savefig(main_plot_path, dpi=save_dpi)
    plt.close(fig)

    # Store CBD tick positions for export
    if "cbd_tick_xs" in locals() and cbd_tick_xs and hasattr(ax, "_processor_ref"):
        ax._processor_ref.calculated_ticks = cbd_tick_xs

    return fig, ax, time_ax
