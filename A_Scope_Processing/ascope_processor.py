import os
import sys
import numpy as np
import cv2
import matplotlib.pyplot as plt
import json
from pathlib import Path

# Add the functions directory to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
functions_dir = os.path.join(current_dir, "functions")
sys.path.append(functions_dir)

# Import from functions directory
from utils import load_config, ensure_output_dirs, load_and_preprocess_image
from preprocessing import (
    mask_sprocket_holes,
    detect_ascope_frames,
    verify_frames_visually,
)
from signal_processing import (
    detect_signal_in_frame,
    trim_signal_trace,
    adaptive_peak_preserving_smooth,
    # refine_signal_trace,
    verify_trace_quality,
)
from grid_detection import (
    detect_grid_lines_and_dotted,
    find_reference_line_blackhat,
    interpolate_regular_grid,
)
from echo_detection import find_tx_pulse, detect_surface_echo, detect_bed_echo


class AScope:
    """
    A-scope radar data processing class that orchestrates the entire processing pipeline.
    """

    def __init__(self, config_path=None):
        # Set up configuration
        self.debug_mode = False  # Default until loaded from config

        if config_path is None:
            # Use default config path relative to the ascope directory
            script_dir = os.path.dirname(os.path.abspath(__file__))
            config_path = os.path.join(script_dir, "config", "default_config.json")

        # Resolve the path (handle relative paths)
        resolved_config_path = os.path.abspath(os.path.expanduser(config_path))

        print(f"Loading configuration from: {resolved_config_path}")

        try:
            with open(resolved_config_path, "r") as f:
                self.config = json.load(f)
        except FileNotFoundError:
            print(
                f"ERROR: Processing configuration file not found at {resolved_config_path}"
            )
            raise

        # Initialize configuration sections
        self.processing_params = self.config.get("processing_params", {})
        self.physical_params = self.config.get("physical_params", {})
        self.output_config = self.config.get("output", {})
        self.debug_mode = self.output_config.get("debug_mode", False)
        self.output_dir = ensure_output_dirs(self.config)

        if self.debug_mode:
            print(f"Debug mode enabled. Using config from: {resolved_config_path}")

    def set_output_directory(self, output_dir):
        """Override the output directory."""
        self.output_config["output_dir"] = output_dir
        self.output_dir = ensure_output_dirs(self.config)

    def set_debug_mode(self, debug_mode):
        """Set debug mode for additional outputs."""
        self.debug_mode = debug_mode
        self.output_config["debug_mode"] = debug_mode

    def process_interactive(self):
        """Process an image with interactive input for the file path."""
        try:
            image, base_filename = load_and_preprocess_image()
            self._process_image_data(image, base_filename)
        except Exception as e:
            print(f"Error during interactive processing: {e}")
            import traceback

            traceback.print_exc()

    def process_image(self, file_path, output_dir=None):
        """
        Process a specific image file.

        Args:
            file_path (str): Path to the input image file
            output_dir (str, optional): Path to output directory
        """
        try:
            if output_dir:
                self.set_output_directory(output_dir)

            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Image file not found: {file_path}")

            image, base_filename = load_and_preprocess_image(file_path)
            self._process_image_data(image, base_filename)
        except Exception as e:
            print(f"Error processing image {file_path}: {e}")
            import traceback

            traceback.print_exc()

    def _process_image_data(self, image, base_filename):
        """Internal method to process image data through the pipeline."""
        # Step 1: Mask sprocket holes
        masked_image, mask = mask_sprocket_holes(image, self.config)

        # Step 2: Detect A-scope frames
        frames = detect_ascope_frames(masked_image, self.config)

        # Step 3: Verify frames visually
        verify_frames_visually(masked_image, frames, base_filename, self.config)

        # Process each frame
        for idx, (left, right) in enumerate(frames):
            print(
                f"\n--- Processing frame {idx + 1}/{len(frames)}: cols {left}-{right} ---"
            )
            self._process_frame(masked_image, left, right, base_filename, idx + 1)

    def _process_frame(self, masked_image, left, right, base_filename, frame_idx):
        """Process an individual A-scope frame."""
        # Extract the frame
        frame_img = masked_image[:, left:right].copy()
        h, w = frame_img.shape

        if w <= 0 or h <= 0:
            print("Warning: Frame has zero width or height. Skipping.")
            return

        # 1. Detect Signal Trace
        signal_x, signal_y = detect_signal_in_frame(frame_img, self.config)
        if signal_x is None:
            print(f"Signal detection failed for frame {frame_idx}. Skipping.")
            return

        signal_y = adaptive_peak_preserving_smooth(signal_y, self.config)

        # 2. Clean Signal Trace
        signal_x_clean, signal_y_clean = trim_signal_trace(
            frame_img, signal_x, signal_y, self.config
        )
        if len(signal_x_clean) == 0:
            print(
                f"No valid signal trace left after cleaning for frame {frame_idx}. Skipping."
            )
            return
        print(f"Cleaned signal length: {len(signal_x_clean)} points")

        # Verify the signal trace
        trace_quality_score = verify_trace_quality(
            frame_img, signal_x_clean, signal_y_clean
        )
        print(f"Trace quality score for frame {frame_idx}: {trace_quality_score:.2f}")

        # 3. Find Tx Pulse (for X-axis anchor)
        tx_pulse_col, tx_idx_in_clean = find_tx_pulse(
            signal_x_clean, signal_y_clean, self.config
        )
        if tx_pulse_col is None:
            print("Warning: Tx pulse detection failed. Using default X-anchor.")
            tx_anchor = w * 0.15  # Fallback anchor near start
        else:
            tx_anchor = tx_pulse_col
            print(
                f"Tx pulse estimated at col {tx_pulse_col:.1f} (index {tx_idx_in_clean} in clean signal)"
            )

        # 4. Find Reference Line (Y-axis anchor)
        ref_row = find_reference_line_blackhat(
            frame_img, base_filename, frame_idx, self.config
        )
        if ref_row is None:
            print("Error: Reference line detection failed critically. Skipping frame.")
            return
        print(
            f"Reference row ({self.physical_params['y_ref_dB']} dB) estimated at y={ref_row}"
        )

        # 5. Detect Grid Lines (for interpolation)
        # Define qa_path before using it
        qa_path = f"{self.output_dir}/{base_filename}_frame{frame_idx:02d}_grid_QA.png"

        # Call detect_grid_lines_and_dotted with four return values
        h_peaks_initial, v_peaks_initial, h_minor_peaks, v_minor_peaks = (
            detect_grid_lines_and_dotted(
                frame_img, self.config, qa_plot_path=qa_path, ref_row_for_qa=ref_row
            )
        )

        # When calling interpolate_regular_grid, use the dynamic range values
        # For Y-axis (power in dB), use range of 55 dB (-60 to -5 dB)
        y_range_dB = 8.25 * 10  # Dynamic range based on signal extent
        y_major, y_minor = interpolate_regular_grid(
            h,
            h_peaks_initial,
            ref_row,
            self.physical_params["y_major_dB"],
            self.physical_params["y_minor_per_major"],
            y_range_dB,  # Use dynamic range
            is_y_axis=True,
            config=self.config,
        )

        # For X-axis (time in μs), use range of 17.5 μs (0 to 17.5 μs)
        x_range_us = self.physical_params.get(
            "x_range_factor"
        ) * self.physical_params.get("x_major_us")

        x_major, x_minor = interpolate_regular_grid(
            w,
            v_peaks_initial,
            tx_anchor,
            self.physical_params["x_major_us"],
            self.physical_params["x_minor_per_major"],
            x_range_us,  # Use dynamic range
            is_y_axis=False,
            config=self.config,
        )

        # 6. Generate Grid QA Plot
        qa_path = f"{self.output_dir}/{base_filename}_frame{frame_idx:02d}_grid_QA.png"
        detect_grid_lines_and_dotted(
            frame_img, self.config, qa_plot_path=qa_path, ref_row_for_qa=ref_row
        )
        print(f"Saved grid QA plot: {qa_path}")

        # 8. Calculate Calibration Factors
        px_per_us_echo, px_per_db_echo = self._calculate_calibration_factors(
            x_major, y_major, w, h
        )

        # 9. Calibrate Signal
        power_vals, time_vals = None, None
        if tx_pulse_col is not None and ref_row is not None:
            power_vals = (
                self.physical_params["y_ref_dB"]
                - (signal_y_clean - ref_row) / px_per_db_echo
            )
            time_vals = (signal_x_clean - tx_pulse_col) / px_per_us_echo
        else:
            print(
                "Warning: Cannot calibrate signal power due to missing anchors or factors."
            )

        # 10. Detect Echoes
        surf_idx_in_clean, bed_idx_in_clean = None, None
        if power_vals is not None and len(power_vals) > 0:
            surf_idx_in_clean = detect_surface_echo(
                power_vals, tx_idx_in_clean, self.config
            )
            bed_idx_in_clean = detect_bed_echo(
                power_vals, time_vals, surf_idx_in_clean, px_per_us_echo, self.config
            )
        else:
            print("Skipping echo detection due to calibration failure.")

        # 11. Plot Combined Results
        self._plot_combined_results(
            frame_img,
            signal_x_clean,
            signal_y_clean,
            x_major,
            y_major,
            x_minor,
            y_minor,
            ref_row,
            base_filename,
            frame_idx,
            tx_pulse_col,
            tx_idx_in_clean,
            surf_idx_in_clean,
            bed_idx_in_clean,
            power_vals,
            time_vals,
            px_per_us_echo,
            px_per_db_echo,
        )

    def _calculate_calibration_factors(self, x_major, y_major, w, h):
        """Calculate px_per_us and px_per_db calibration factors."""
        px_per_us, px_per_db = None, None

        # Calculate px_per_us
        if len(x_major) >= 2:
            x_spacings = np.diff(x_major)
            positive_spacings = x_spacings[x_spacings > 0]
            if len(positive_spacings) > 0:
                median_x_spacing = np.median(positive_spacings)
                if median_x_spacing > 0 and self.physical_params["x_major_us"] > 0:
                    px_per_us = abs(
                        median_x_spacing / self.physical_params["x_major_us"]
                    )

        # Calculate px_per_db
        if len(y_major) >= 2:
            y_spacings = np.diff(y_major)
            positive_spacings = y_spacings[y_spacings > 0]
            if len(positive_spacings) > 0:
                median_y_spacing = np.median(positive_spacings)
                if median_y_spacing > 0 and self.physical_params["y_major_dB"] > 0:
                    px_per_db = abs(
                        median_y_spacing / self.physical_params["y_major_dB"]
                    )

        # Use fallbacks if calculation failed
        if px_per_us is None or px_per_us <= 0:
            usable_width_fraction = self.physical_params.get(
                "usable_width_fraction", 0.8
            )
            num_x_intervals = (
                self.physical_params.get("x_range_us", 30)
                / self.physical_params.get("x_major_us", 3)
                if self.physical_params.get("x_major_us", 3) > 0
                else 5
            )
            px_per_us = (
                (w * usable_width_fraction) / num_x_intervals
                if num_x_intervals > 0
                else w / 5.0
            )

        if px_per_db is None or px_per_db <= 0:
            usable_height_fraction = self.physical_params.get(
                "usable_height_fraction", 0.8
            )
            num_y_intervals = (
                self.physical_params.get("y_range_dB", 60)
                / self.physical_params.get("y_major_dB", 10)
                if self.physical_params.get("y_major_dB", 10) > 0
                else 6
            )
            px_per_db = (
                (h * usable_height_fraction) / num_y_intervals
                if num_y_intervals > 0
                else h / 6.0
            )

        return px_per_us, px_per_db

    def _plot_combined_results(
        self,
        frame_img,
        signal_x_clean,
        signal_y_clean,
        x_major,
        y_major,
        x_minor,
        y_minor,
        ref_row,
        base_filename,
        frame_idx,
        tx_pulse_col,
        tx_idx_in_clean,
        surf_idx_in_clean,
        bed_idx_in_clean,
        power_vals,
        time_vals,
        px_per_us,
        px_per_db,
    ):
        """Generate and save a combined plot with debug view and calibrated view."""
        h, w = frame_img.shape
        plot_filename = f"{self.output_dir}/{base_filename}_frame{frame_idx:02d}_combined_annotated.png"

        # Create the figure with two subplots
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # --- Plot 1: Debug View (Image + Overlays) ---
        ax_debug = axes[0]
        ax_debug.imshow(frame_img, cmap="gray", aspect="auto")

        # Plot signal trace
        if signal_x_clean is not None and len(signal_x_clean) > 0:
            ax_debug.plot(
                signal_x_clean,
                signal_y_clean,
                "r-",
                linewidth=1,
                label="Detected Trace",
            )

        # Plot grid lines
        grid_line_color = "#00BFFF"  # Deep sky blue
        major_alpha, minor_alpha = 0.6, 0.3

        for y in y_major:
            ax_debug.axhline(
                y,
                color=grid_line_color,
                linestyle="-",
                alpha=major_alpha,
                linewidth=0.8,
            )
        for y in y_minor:
            ax_debug.axhline(
                y,
                color=grid_line_color,
                linestyle=":",
                alpha=minor_alpha,
                linewidth=0.8,
            )
        for x in x_major:
            ax_debug.axvline(
                x,
                color=grid_line_color,
                linestyle="-",
                alpha=major_alpha,
                linewidth=0.8,
            )
        for x in x_minor:
            ax_debug.axvline(
                x,
                color=grid_line_color,
                linestyle=":",
                alpha=minor_alpha,
                linewidth=1,
            )

        # Highlight Reference Line and Tx Column
        if ref_row is not None:
            ax_debug.axhline(
                ref_row,
                color="lime",
                linestyle=":",
                linewidth=0.8,
                label=f"Ref Line ({self.physical_params['y_ref_dB']} dB)",
            )
        if tx_pulse_col is not None:
            ax_debug.axvline(
                tx_pulse_col,
                color="orange",
                linestyle=":",
                linewidth=1.5,
                label="Tx Time Zero",
            )

        # Mark detected echoes
        valid_signal = (
            signal_x_clean is not None
            and len(signal_x_clean) > 0
            and signal_y_clean is not None
            and len(signal_y_clean) == len(signal_x_clean)
        )

        if valid_signal:
            if tx_idx_in_clean is not None and tx_idx_in_clean < len(signal_x_clean):
                ax_debug.plot(
                    signal_x_clean[tx_idx_in_clean],
                    signal_y_clean[tx_idx_in_clean],
                    "bo",
                    ms=6,
                    label="Tx",
                )
            if surf_idx_in_clean is not None and surf_idx_in_clean < len(
                signal_x_clean
            ):
                ax_debug.plot(
                    signal_x_clean[surf_idx_in_clean],
                    signal_y_clean[surf_idx_in_clean],
                    "go",
                    ms=6,
                    label="Surf",
                )
            if bed_idx_in_clean is not None and bed_idx_in_clean < len(signal_x_clean):
                ax_debug.plot(
                    signal_x_clean[bed_idx_in_clean],
                    signal_y_clean[bed_idx_in_clean],
                    "mo",
                    ms=6,
                    label="Bed",
                )

        ax_debug.set_title(f"A-scope Frame {frame_idx} (Debug View)")
        ax_debug.set_ylim(h, 0)
        ax_debug.set_xlim(0, w)
        ax_debug.axis("on")
        ax_debug.set_xticks([])
        ax_debug.set_yticks([])
        ax_debug.legend(fontsize=8, loc="lower left", bbox_to_anchor=(0, -0.15))

        # --- Plot 2: Calibrated View (Time vs Power) ---
        ax_calib = axes[1]

        if time_vals is not None and power_vals is not None and len(time_vals) > 0:
            ax_calib.plot(time_vals, power_vals, "r-", linewidth=1.2)
        else:
            ax_calib.text(
                0.5,
                0.5,
                "Calibration Failed",
                horizontalalignment="center",
                verticalalignment="center",
                transform=ax_calib.transAxes,
            )

        # Plot physical grid
        plot_min_db = -65
        plot_max_db = 2.5  # Extend slightly above 0 dB

        # Define major and minor grid lines
        major_db_ticks = np.arange(
            self.physical_params["y_ref_dB"],
            plot_max_db + 1,
            self.physical_params["y_major_dB"],
        )

        minor_db_per_major = self.physical_params["y_major_dB"] / (
            self.physical_params["y_minor_per_major"] + 1
        )
        minor_db_ticks = np.arange(
            plot_min_db, plot_max_db + minor_db_per_major, minor_db_per_major
        )

        # Draw horizontal grid lines
        for db in major_db_ticks:
            ax_calib.axhline(
                db,
                color=grid_line_color,
                linestyle="-",
                alpha=major_alpha,
                linewidth=1.0,
            )
        for db in minor_db_ticks:
            ax_calib.axhline(
                db,
                color=grid_line_color,
                linestyle=":",
                alpha=minor_alpha,
                linewidth=0.8,
            )

        # Define time range
        plot_min_time = -1  # Start slightly before Tx
        plot_max_time = self.physical_params["x_range_us"] + 2

        if time_vals is not None and len(time_vals) > 0:
            plot_max_time = max(plot_max_time, np.ceil(time_vals.max()) + 2)

        major_time_ticks = np.arange(
            0, plot_max_time, self.physical_params["x_major_us"]
        )
        minor_time_per_major = self.physical_params["x_major_us"] / (
            self.physical_params["x_minor_per_major"] + 1
        )
        minor_time_ticks = np.arange(plot_min_time, plot_max_time, minor_time_per_major)

        # Draw vertical grid lines
        for t in major_time_ticks:
            ax_calib.axvline(
                t,
                color=grid_line_color,
                linestyle="-",
                alpha=major_alpha,
                linewidth=1.0,
            )
        for t in minor_time_ticks:
            ax_calib.axvline(
                t,
                color=grid_line_color,
                linestyle=":",
                alpha=minor_alpha,
                linewidth=0.8,
            )

        # Highlight reference line
        ax_calib.axhline(
            self.physical_params["y_ref_dB"],
            color="lime",
            linestyle=":",
            linewidth=1,
            label=f"Reference ({self.physical_params['y_ref_dB']} dB)",
        )

        # Set ticks and labels
        from matplotlib.ticker import FixedLocator

        # Y-axis labels (every 20 dB)
        y_label_step = 20
        y_major_labels = np.arange(
            self.physical_params["y_ref_dB"], plot_max_db + 1, y_label_step
        )
        ax_calib.set_yticks(y_major_labels)
        ax_calib.set_yticklabels([f"{int(db)}" for db in y_major_labels], fontsize=10)
        ax_calib.yaxis.set_minor_locator(FixedLocator(minor_db_ticks))

        # X-axis labels
        ax_calib.set_xticks(major_time_ticks)
        ax_calib.set_xticklabels([f"{int(t)}" for t in major_time_ticks], fontsize=10)
        ax_calib.xaxis.set_minor_locator(FixedLocator(minor_time_ticks))

        ax_calib.set_xlabel("One-way travel time (µs)")
        ax_calib.set_ylabel("Power (dB)")
        ax_calib.set_title(f"Calibrated A-scope Frame {frame_idx}")

        # Set plot limits
        ax_calib.set_ylim(plot_min_db, plot_max_db)
        ax_calib.set_xlim(
            plot_min_time,
            plot_max_time - 1 if plot_max_time > plot_min_time + 1 else plot_max_time,
        )

        # Annotate echoes on calibrated plot
        if valid_signal and time_vals is not None and len(time_vals) > 0:
            if tx_idx_in_clean is not None and tx_idx_in_clean < len(time_vals):
                ax_calib.plot(
                    time_vals[tx_idx_in_clean],
                    power_vals[tx_idx_in_clean],
                    "o",
                    color="blue",
                    label="Tx",
                    markersize=6,
                )

            if surf_idx_in_clean is not None and surf_idx_in_clean < len(time_vals):
                ax_calib.plot(
                    time_vals[surf_idx_in_clean],
                    power_vals[surf_idx_in_clean],
                    "o",
                    color="green",
                    label="Surface",
                    markersize=6,
                )

            if bed_idx_in_clean is not None and bed_idx_in_clean < len(time_vals):
                ax_calib.plot(
                    time_vals[bed_idx_in_clean],
                    power_vals[bed_idx_in_clean],
                    "o",
                    color="magenta",
                    label="Bed",
                    markersize=6,
                )

        # Add legend to calibrated plot
        handles, labels = ax_calib.get_legend_handles_labels()
        if handles:
            by_label = dict(zip(labels, handles))
            ax_calib.legend(
                by_label.values(), by_label.keys(), loc="upper right", fontsize=9
            )

        # Add annotations with arrows for detected points
        if valid_signal and time_vals is not None and len(time_vals) > 0:
            # Format for annotation: "Point (~X.X dB at Y.Y μs)"

            # Annotate Tx point
            if tx_idx_in_clean is not None and tx_idx_in_clean < len(time_vals):
                tx_time = time_vals[tx_idx_in_clean]
                tx_power = power_vals[tx_idx_in_clean]
                tx_label = (
                    f"transmitter pulse\n(~{tx_power:.1f} dB at {tx_time:.1f} μs)"
                )
                ax_calib.annotate(
                    tx_label,
                    xy=(tx_time, tx_power),  # Point to annotate
                    xytext=(-50, -40),  # Offset text position
                    textcoords="offset points",
                    ha="center",
                    va="top",
                    fontsize=9,
                    bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.7),
                    arrowprops=dict(
                        arrowstyle="->", connectionstyle="arc3,rad=0.3", color="blue"
                    ),
                )

            # Annotate Surface echo point
            if surf_idx_in_clean is not None and surf_idx_in_clean < len(time_vals):
                surf_time = time_vals[surf_idx_in_clean]
                surf_power = power_vals[surf_idx_in_clean]
                surf_label = f"surface\n(~{surf_power:.1f} dB at {surf_time:.1f} μs)"
                ax_calib.annotate(
                    surf_label,
                    xy=(surf_time, surf_power),  # Point to annotate
                    xytext=(-60, -15),  # Offset: 60 points left, 15 points down
                    textcoords="offset points",
                    ha="right",  # Text aligned to the right of the offset point
                    va="top",  # Top of the text box aligned with the offset point's y-coordinate
                    fontsize=9,
                    bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.7),
                    arrowprops=dict(
                        arrowstyle="->",
                        connectionstyle="arc3,rad=0.3",  # Adjusted curve for leftward arrow
                        color="green",
                    ),
                )

            # Annotate Bed echo point
            if bed_idx_in_clean is not None and bed_idx_in_clean < len(time_vals):
                bed_time = time_vals[bed_idx_in_clean]
                bed_power = power_vals[bed_idx_in_clean]
                bed_label = f"bed\n(~{bed_power:.1f} dB at {bed_time:.1f} μs)"
                ax_calib.annotate(
                    bed_label,
                    xy=(bed_time, bed_power),  # Point to annotate
                    xytext=(60, -15),  # Offset: 60 points right, 15 points down
                    textcoords="offset points",
                    ha="left",  # Text aligned to the left of the offset point
                    va="top",  # Top of the text box aligned with the offset point's y-coordinate
                    fontsize=9,
                    bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.7),
                    arrowprops=dict(
                        arrowstyle="->",
                        connectionstyle="arc3,rad=-0.3",  # Adjusted curve for rightward arrow
                        color="magenta",
                    ),
                )
        plt.tight_layout(pad=1.5)
        plt.savefig(plot_filename, dpi=self.output_config.get("plot_dpi", 200))
        print(f"Saved combined plot: {plot_filename}")
        plt.close(fig)
