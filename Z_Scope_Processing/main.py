# zscope_processor/main.py

import argparse
import sys
from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# --- Matplotlib Backend Configuration ---
# For ClickSelector to work, an interactive backend is needed.
# Common choices are 'Qt5Agg', 'TkAgg', 'WXAgg', 'GTK3Agg', 'macosx'.

try:
    matplotlib.use("Qt5Agg")
    print("INFO: Using Matplotlib backend: Qt5Agg")
except ImportError:
    print("WARNING: Qt5Agg backend for Matplotlib not found or failed to load.")
    try:
        matplotlib.use("TkAgg")  # Fallback to TkAgg
        print("INFO: Using Matplotlib backend: TkAgg")
    except ImportError:
        print("WARNING: TkAgg backend for Matplotlib not found or failed to load.")
        print(
            "INFO: Matplotlib will use its default backend. Interactive features may not work if headless."
        )

# --- Module Imports ---
from functions.image_utils import (
    load_and_preprocess_image,
)
from functions.interactive_tools import ClickSelector
from zscope_processor import ZScopeProcessor


def run_processing():
    """
    Main function to parse arguments and run the Z-scope processing workflow.
    """
    parser = argparse.ArgumentParser(
        description="Process Z-scope radar film images from raw image to calibrated data display.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,  # Shows default values in help
    )
    parser.add_argument(
        "image_path",
        type=str,
        help="Path to the Z-scope image file (e.g., .tif, .png, .jpg).",
    )
    parser.add_argument(
        "output_dir",
        type=str,
        nargs="?",  # Makes it optional, see 'default'
        default="output",
        help="Directory where all output files (plots, data) will be saved.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/default_config.json",
        help="Path to the JSON file containing processing parameters.",
    )
    parser.add_argument(
        "--physics",
        type=str,
        default="config/physical_constants.json",
        help="Path to the JSON file containing physical constants for calibration.",
    )
    parser.add_argument(
        "--non_interactive_pip_x",
        type=int,
        default=None,
        help="Specify the approximate X-coordinate for the calibration pip non-interactively. "
        "If provided, the ClickSelector GUI will be skipped.",
    )

    args = parser.parse_args()

    # --- Define Project Root or Script's Directory for Path Resolution ---
    SCRIPT_DIR = Path(__file__).resolve().parent
    output_path_obj = Path(args.output_dir)
    if not output_path_obj.is_absolute():
        final_output_dir = SCRIPT_DIR / output_path_obj
    else:
        final_output_dir = output_path_obj

    final_output_dir.mkdir(parents=True, exist_ok=True)
    print(f"INFO: Output will be saved to: {final_output_dir.resolve()}")

    # --- Initialize the ZScopeProcessor ---
    try:
        processor = ZScopeProcessor(config_path=args.config, physics_path=args.physics)
    except Exception as e:
        print(f"ERROR: Failed to initialize ZScopeProcessor: {e}")
        sys.exit(1)

    # --- Get Approximate X-coordinate for Calibration Pip ---
    approx_x_pip_selected = args.non_interactive_pip_x

    if approx_x_pip_selected is None:
        # Interactive mode: Load image temporarily for ClickSelector
        print("\nINFO: Preparing for interactive calibration pip selection...")
        temp_image_for_selector = load_and_preprocess_image(
            args.image_path,
            processor.config.get(
                "preprocessing_params", {}
            ),  # Use relevant params for display
        )
        if temp_image_for_selector is None:
            print(
                f"ERROR: Failed to load image '{args.image_path}' for pip selection. Exiting."
            )
            sys.exit(1)

        print(
            "INFO: Please click on the approximate vertical location of the calibration pip ticks in the displayed image."
        )
        selector_title = processor.config.get("click_selector_params", {}).get(
            "title", "Click on the calibration pip column"
        )
        selector = ClickSelector(temp_image_for_selector, title=selector_title)
        approx_x_pip_selected = selector.selected_x

        if approx_x_pip_selected is None:
            print(
                "ERROR: No location selected for calibration pip via ClickSelector. Exiting."
            )
            sys.exit(1)
        print(
            f"INFO: User selected approximate X-coordinate for calibration pip: {approx_x_pip_selected}"
        )
    else:
        print(
            f"INFO: Using non-interactive X-coordinate for calibration pip: {approx_x_pip_selected}"
        )
    pass

    # --- Run the main processing pipeline ---
    print(f"\nINFO: Starting main processing for image: {args.image_path}")
    processing_successful = processor.process_image(
        args.image_path,
        str(final_output_dir.resolve()),  # Pass the correct absolute path
        approx_x_pip_selected,
    )

    if not processing_successful:
        print("ERROR: Z-scope image processing failed. Check logs for details.")
        sys.exit(1)

    print("\nINFO: Core processing completed successfully.")

    # --- Plotting Automatically Detected Echoes ---
    if processor.calibrated_fig and processor.calibrated_ax:
        print(
            "\nINFO: Plotting automatically detected echoes on the calibrated Z-scope image..."
        )

        # Define X-coordinates for the plot (number of columns in the valid_data_crop)
        # Ensure image_np and data_top_abs/data_bottom_abs are valid before trying to get shape
        if (
            processor.image_np is not None
            and processor.data_top_abs is not None
            and processor.data_bottom_abs is not None
            and processor.data_top_abs < processor.data_bottom_abs
        ):
            num_cols = processor.image_np[
                processor.data_top_abs : processor.data_bottom_abs, :
            ].shape[1]
            x_plot_coords = np.arange(num_cols)

            # Get echo tracing plot parameters from config
            echo_plot_config = processor.config.get("echo_tracing_params", {})
            surface_plot_params = echo_plot_config.get("surface_detection", {})
            bed_plot_params = echo_plot_config.get("bed_detection", {})

            # Plot Surface Echo
            if processor.detected_surface_y_abs is not None and np.any(
                np.isfinite(processor.detected_surface_y_abs)
            ):
                # Convert absolute Y trace to Y relative to the cropped image on the axes
                surface_y_cropped = (
                    processor.detected_surface_y_abs - processor.data_top_abs
                )

                valid_indices = np.isfinite(
                    surface_y_cropped
                )  # Plot only where trace is valid
                if np.any(valid_indices):  # Check if there's anything to plot
                    processor.calibrated_ax.plot(
                        x_plot_coords[valid_indices],
                        surface_y_cropped[valid_indices],
                        color=surface_plot_params.get("plot_color", "cyan"),
                        linestyle=surface_plot_params.get("plot_linestyle", "-"),
                        linewidth=1.5,
                        label="Auto Surface Echo",
                    )
                    print("INFO: Plotted automatically detected surface echo.")
            else:
                print("INFO: No valid automatic surface echo trace to plot.")

            # Plot Bed Echo
            if processor.detected_bed_y_abs is not None and np.any(
                np.isfinite(processor.detected_bed_y_abs)
            ):
                bed_y_cropped = processor.detected_bed_y_abs - processor.data_top_abs

                valid_indices = np.isfinite(
                    bed_y_cropped
                )  # Plot only where trace is valid
                if np.any(valid_indices):
                    processor.calibrated_ax.plot(
                        x_plot_coords[valid_indices],
                        bed_y_cropped[valid_indices],
                        color=bed_plot_params.get("plot_color", "lime"),
                        linestyle=bed_plot_params.get("plot_linestyle", "-"),
                        linewidth=1.5,
                        label="Auto Bed Echo",
                    )
                    print("INFO: Plotted automatically detected bed echo.")
            else:
                print("INFO: No valid automatic bed echo trace to plot.")

            # Update the legend to include the new traces
            time_vis_params = processor.config.get(
                "time_calibration_visualization_params", {}
            )
            processor.calibrated_ax.legend(
                loc=time_vis_params.get("legend_location", "upper right"),
                fontsize="small",  # Or fetch from config if available
            )

            # Save the figure with these new auto-detected echoes plotted
            auto_echo_plot_filename = (
                f"{processor.base_filename}_time_calibrated_auto_echoes.png"
            )
            auto_echo_plot_path = final_output_dir / auto_echo_plot_filename

            output_params_config = processor.config.get("output_params", {})
            save_dpi = output_params_config.get(
                "annotated_figure_save_dpi",
                output_params_config.get("figure_save_dpi", 300),
            )

            try:
                processor.calibrated_fig.savefig(
                    auto_echo_plot_path, dpi=save_dpi, bbox_inches="tight"
                )
                print(
                    f"INFO: Plot with auto-detected echoes saved to: {auto_echo_plot_path}"
                )
            except Exception as e:
                print(f"ERROR: Could not save plot with auto-detected echoes: {e}")
        else:
            print(
                "WARNING: Cannot plot echoes because prerequisite image data is missing from processor."
            )

        # Display the final plot
        print("\nINFO: Displaying final plot (close window to exit script).")
        plt.show()

    elif (
        processing_successful
    ):  # Processing was successful but no calibrated_fig/ax (should not happen)
        print(
            "WARNING: Core processing completed, but calibrated plot figure/axes are not available for final display."
        )

    print("\n--- Z-scope Processing Script Finished ---")
    sys.exit(0)


if __name__ == "__main__":
    run_processing()
