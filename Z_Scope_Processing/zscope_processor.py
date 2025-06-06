import json
from pathlib import Path
import numpy as np

from functions.image_utils import load_and_preprocess_image
from functions.artifact_detection import (
    detect_film_artifact_boundaries,
    detect_zscope_boundary,
)
from functions.feature_detection import (
    detect_transmitter_pulse,
    detect_calibration_pip,
)
from functions.calibration_utils import calculate_pixels_per_microsecond
from functions.visualization_utils import (
    visualize_calibration_pip_detection,
    create_time_calibrated_zscope,
)
from functions.echo_tracing import detect_surface_echo, detect_bed_echo


class ZScopeProcessor:
    def __init__(
        self,
        config_path="config/default_config.json",
        physics_path="config/physical_constants.json",
    ):
        """
        Initializes the ZScopeProcessor with configuration files.

        Args:
            config_path (str, optional): Path to the default processing configuration JSON file,
                                         expected to be relative to this script's directory if not absolute.
            physics_path (str, optional): Path to the physical constants JSON file,
                                          expected to be relative to this script's directory if not absolute.

        Raises:
            FileNotFoundError: If configuration files are not found.
            JSONDecodeError: If configuration files are not valid JSON.
        """
        # Get the directory where zscope_processor.py is located
        processor_script_dir = Path(__file__).resolve().parent

        # Convert to Path objects to handle potential absolute paths correctly as well
        config_path_obj = Path(config_path)
        physics_path_obj = Path(physics_path)

        if not config_path_obj.is_absolute():
            resolved_config_path = processor_script_dir / config_path_obj
        else:
            resolved_config_path = config_path_obj

        if not physics_path_obj.is_absolute():
            resolved_physics_path = processor_script_dir / physics_path_obj
        else:
            resolved_physics_path = physics_path_obj

        try:
            with open(resolved_config_path, "r") as f:  # Use the resolved path
                self.config = json.load(f)
            print(
                f"INFO: Successfully loaded processing configuration from {resolved_config_path.resolve()}"
            )
        except FileNotFoundError:
            print(
                f"ERROR: Processing configuration file not found at {resolved_config_path.resolve()}"
            )
            raise
        except json.JSONDecodeError:
            print(
                f"ERROR: Invalid JSON in processing configuration file: {resolved_config_path.resolve()}"
            )
            raise

        try:
            with open(resolved_physics_path, "r") as f:  # Use the resolved path
                self.physics_constants = json.load(f)
            print(
                f"INFO: Successfully loaded physical constants from {resolved_physics_path.resolve()}"
            )
        except FileNotFoundError:
            print(
                f"ERROR: Physical constants file not found at {resolved_physics_path.resolve()}"
            )
            raise
        except json.JSONDecodeError:
            print(
                f"ERROR: Invalid JSON in physical constants file: {resolved_physics_path.resolve()}"
            )
            raise
        # Initialize instance attributes for storing processing results
        self.image_np = None
        self.base_filename = None
        self.data_top_abs = None
        self.data_bottom_abs = None
        self.transmitter_pulse_y_abs = None
        self.best_pip_details = None
        self.pixels_per_microsecond = None
        self.calibrated_fig = None
        self.calibrated_ax = None
        self.detected_surface_y_abs = None
        self.detected_bed_y_abs = None
        self.time_axis = None
        self.output_dir = None

    def process_image(self, image_path_str, output_dir_str, approx_x_pip):
        """
        Processes a single Z-scope image through the entire pipeline.

        Args:
            image_path_str (str): Path to the Z-scope image file.
            output_dir_str (str): Path to the directory where output files will be saved.
            approx_x_pip (int): Approximate X-coordinate of the calibration pip,
                                typically obtained from user interaction (e.g., ClickSelector).

        Returns:
            bool: True if processing was successful, False otherwise.
        """
        image_path_obj = Path(image_path_str)
        self.base_filename = image_path_obj.stem
        self.output_dir = Path(output_dir_str)

        # Consolidate output parameters for utility functions
        output_params_config = self.config.get("output_params", {})
        debug_subdir_name = output_params_config.get(
            "debug_output_directory", "debug_output"
        )
        current_output_params = {
            # Construct the full path to the debug subdirectory
            "debug_output_directory": str(self.output_dir / debug_subdir_name),
            "figure_save_dpi": output_params_config.get("figure_save_dpi", 300),
        }
        # Ensure the specific debug output directory (sub-directory) also exists
        Path(current_output_params["debug_output_directory"]).mkdir(
            parents=True, exist_ok=True
        )

        print(f"\n--- Processing Z-scope Image: {self.base_filename} ---")

        # Step 1: Load and preprocess image
        print("\nStep 1: Loading and preprocessing image...")
        self.image_np = load_and_preprocess_image(
            image_path_str, self.config.get("preprocessing_params", {})
        )
        if self.image_np is None:
            print(
                f"ERROR: Failed to load or preprocess image {image_path_str}. Aborting."
            )
            return False

        img_height, img_width = self.image_np.shape
        print(f"INFO: Image dimensions: {img_width}x{img_height}")

        # Step 2: Detect film artifact boundaries
        print("\nStep 2: Detecting film artifact boundaries...")
        artifact_params = self.config.get("artifact_detection_params", {})
        self.data_top_abs, self.data_bottom_abs = detect_film_artifact_boundaries(
            self.image_np,
            self.base_filename,  # For saving debug plot if visualize=True in params
            top_exclude_ratio=artifact_params.get("top_exclude_ratio", 0.05),
            bottom_exclude_ratio=artifact_params.get("bottom_exclude_ratio", 0.05),
            gradient_smooth_kernel=artifact_params.get("gradient_smooth_kernel", 15),
            gradient_threshold_factor=artifact_params.get(
                "gradient_threshold_factor", 1.5
            ),
            safety_margin=artifact_params.get("safety_margin", 20),
            visualize=artifact_params.get("visualize_film_artifact_boundaries", False),
        )
        print(
            f"INFO: Film artifact boundaries determined: Top={self.data_top_abs}, Bottom={self.data_bottom_abs}"
        )

        # Step 3: Detect transmitter pulse
        print("\nStep 3: Detecting transmitter pulse...")
        tx_pulse_params_config = self.config.get("transmitter_pulse_params", {})
        self.transmitter_pulse_y_abs = detect_transmitter_pulse(
            self.image_np,
            self.base_filename,  # For saving debug plot
            self.data_top_abs,
            self.data_bottom_abs,
            tx_pulse_params=tx_pulse_params_config,  # Pass the whole sub-dictionary
        )
        print(
            f"INFO: Transmitter pulse detected at Y-pixel (absolute): {self.transmitter_pulse_y_abs}"
        )

        # Step 4: Detect calibration pip (requires approx_x_pip from user)
        # Before calling detect_calibration_pip, we need to determine z_boundary_y for its strip.
        print(f"\nStep 4: Detecting calibration pip around X-pixel {approx_x_pip}...")
        if approx_x_pip is None:
            print(
                "ERROR: Approximate X-position for calibration pip not provided. Cannot detect pip."
            )
            return False

        # Determine Z-scope boundary for the vertical strip where pip is expected.
        # This replicates logic from original script's detect_calibration_pip.
        # Define a narrow vertical slice around the approx_x_pip for Z-boundary detection.
        pip_detection_strip_config = self.config.get("pip_detection_params", {}).get(
            "approach_1", {}
        )
        strip_center_for_z_boundary = (
            approx_x_pip  # Or best_pip['x_position'] if iterative.
        )
        z_boundary_vslice_width = pip_detection_strip_config.get(
            "z_boundary_vslice_width_px", 10
        )
        v_slice_x_start = max(
            0, strip_center_for_z_boundary - z_boundary_vslice_width // 2
        )
        v_slice_x_end = min(
            img_width, strip_center_for_z_boundary + z_boundary_vslice_width // 2
        )

        if v_slice_x_start >= v_slice_x_end:  # Handle edge cases if image is too narrow
            print(
                f"WARNING: Cannot extract vertical slice for Z-boundary detection at X={strip_center_for_z_boundary}. Using full width."
            )
            vertical_slice_for_z = self.image_np  # Fallback or specific handling
        else:
            vertical_slice_for_z = self.image_np[:, v_slice_x_start:v_slice_x_end]

        z_boundary_params_config = self.config.get(
            "zscope_boundary_detection_params", {}
        )

        z_boundary_y_for_pip = detect_zscope_boundary(
            vertical_slice_for_z,  # Pass the narrow vertical strip
            self.data_top_abs,  # Search within these absolute Y bounds
            self.data_bottom_abs,
            # Parameters from config for detect_zscope_boundary would be passed here
            # For now, assuming detect_zscope_boundary has reasonable defaults or they are in its own config section
            # For example: z_boundary_params_config.get("gradient_smooth_kernel", 31)
        )
        print(
            f"INFO: Z-scope boundary for pip strip detected at Y-pixel (absolute): {z_boundary_y_for_pip}"
        )

        pip_detection_main_config = self.config.get("pip_detection_params", {})
        self.best_pip_details = detect_calibration_pip(
            self.image_np,
            self.base_filename,  # For saving debug plots
            approx_x_pip,
            self.data_top_abs,
            self.data_bottom_abs,
            z_boundary_y_for_pip,  # Pass the detected Z-boundary for this specific strip
            pip_detection_params=pip_detection_main_config,  # Pass the whole sub-dictionary
        )

        # Step 5: Visualize calibration pip detection
        print("\nStep 5: Visualizing calibration pip detection results...")
        pip_visualization_params_config = pip_detection_main_config.get(
            "visualization_params", {}
        )
        visualize_calibration_pip_detection(
            self.image_np,
            self.base_filename,
            self.best_pip_details,
            approx_x_click=approx_x_pip,
            visualization_params=pip_visualization_params_config,
            output_params=current_output_params,  # Pass consolidated output params
        )

        if not self.best_pip_details:
            print(
                "ERROR: Calibration pip detection failed. Cannot perform time calibration."
            )
            return False

        # Step 6: Calculate pixels per microsecond
        print("\nStep 6: Calculating pixels per microsecond...")
        pip_interval_us = self.physics_constants.get(
            "calibration_pip_interval_microseconds", 2.0
        )
        try:
            self.pixels_per_microsecond = calculate_pixels_per_microsecond(
                self.best_pip_details["mean_spacing"], pip_interval_us
            )
        except ValueError as e:
            print(f"ERROR calculating pixels_per_microsecond: {e}")
            return False

        print(
            f"INFO: Calculated pixels per microsecond: {self.pixels_per_microsecond:.3f}"
        )

        print("\nStep 6.5: Automatic echo tracing...")
        if (
            self.image_np is not None
            and self.data_top_abs is not None
            and self.data_bottom_abs is not None
            and self.transmitter_pulse_y_abs is not None
            and self.best_pip_details is not None
            and self.pixels_per_microsecond is not None
        ):  # Ensure calibration is done
            valid_data_crop = self.image_np[self.data_top_abs : self.data_bottom_abs, :]
            crop_height, crop_width = valid_data_crop.shape

            tx_pulse_y_rel = self.transmitter_pulse_y_abs - self.data_top_abs

            z_boundary_y_abs_for_echo_search = self.data_bottom_abs
            z_boundary_y_rel = z_boundary_y_abs_for_echo_search - self.data_top_abs

            echo_tracing_config = self.config.get("echo_tracing_params", {})
            surface_config = echo_tracing_config.get("surface_detection", {})

            print(
                f"DEBUG: data_top_abs: {self.data_top_abs}, data_bottom_abs: {self.data_bottom_abs}"
            )
            print(f"DEBUG: transmitter_pulse_y_abs: {self.transmitter_pulse_y_abs}")
            print(f"DEBUG: tx_pulse_y_rel (Tx pulse Y within crop): {tx_pulse_y_rel}")

            surf_search_start_offset = surface_config.get("search_start_offset_px", 20)
            surf_search_depth = surface_config.get("search_depth_px", crop_height // 3)

            actual_surf_search_y_start_in_crop = (
                tx_pulse_y_rel + surf_search_start_offset
            )
            actual_surf_search_y_end_in_crop = (
                actual_surf_search_y_start_in_crop + surf_search_depth
            )

            # Ensure these are within the bounds of valid_data_crop
            actual_surf_search_y_start_in_crop = max(
                0, actual_surf_search_y_start_in_crop
            )
            actual_surf_search_y_end_in_crop = min(
                crop_height, actual_surf_search_y_end_in_crop
            )

            print(
                f"DEBUG: Surface search config offset: {surf_search_start_offset}, depth: {surf_search_depth}"
            )
            print(
                f"DEBUG: Surface search Y-window (within crop): {actual_surf_search_y_start_in_crop} to {actual_surf_search_y_end_in_crop}"
            )

            print(f"INFO: Detecting surface echo with config: {surface_config}")
            surface_y_rel = detect_surface_echo(
                valid_data_crop,
                tx_pulse_y_rel,  # This is the reference point for the offset inside detect_surface_echo
                surface_config,
            )

            if np.any(
                np.isfinite(surface_y_rel)
            ):  # Check if surface_y_rel has any valid numbers
                self.detected_surface_y_abs = surface_y_rel + self.data_top_abs
                print(
                    f"INFO: Surface echo detected. Example points (absolute Y): {self.detected_surface_y_abs[: min(5, len(self.detected_surface_y_abs))] if self.detected_surface_y_abs.size > 0 else 'N/A'}"
                )

                # Debug print for bed echo
                bed_config = echo_tracing_config.get("bed_detection", {})
                print(
                    f"DEBUG (ZScopeProcessor for Bed): Bed detection config to be used: {bed_config}"
                )
                print(f"  DEBUG (ZScopeProcessor for Bed): Passing to detect_bed_echo:")
                print(f"    - valid_data_crop shape: {valid_data_crop.shape}")
                print(
                    f"    - surface_y_rel (first 5, relative to crop): {surface_y_rel[: min(5, len(surface_y_rel))]}"
                )
                print(
                    f"    - z_boundary_y_rel (NOW BASED ON data_bottom_abs, relative to crop): {z_boundary_y_rel}"
                )  # Will be much larger

                bed_y_rel = detect_bed_echo(
                    valid_data_crop,
                    surface_y_rel,
                    z_boundary_y_rel,  # Pass the new, deeper z_boundary_y_rel
                    bed_config,
                )
                if np.any(
                    np.isfinite(bed_y_rel)
                ):  # Check if bed_y_rel has any valid numbers
                    self.detected_bed_y_abs = bed_y_rel + self.data_top_abs
                    print(
                        f"INFO: Bed echo detected. Example points (absolute Y): {self.detected_bed_y_abs[: min(5, len(self.detected_bed_y_abs))] if self.detected_bed_y_abs.size > 0 else 'N/A'}"
                    )
                else:
                    print(
                        "WARNING: Bed echo not reliably detected (all NaNs returned)."
                    )
                    # Ensure self.detected_bed_y_abs is an array of NaNs of correct width
                    if valid_data_crop is not None:
                        self.detected_bed_y_abs = np.full(
                            valid_data_crop.shape[1], np.nan
                        )
            else:
                print(
                    "WARNING: Surface echo not reliably detected. Skipping bed echo detection."
                )
                self.detected_surface_y_abs = np.full(
                    valid_data_crop.shape[1] if valid_data_crop is not None else 100,
                    np.nan,
                )  # Default width if crop failed
                self.detected_bed_y_abs = np.full(
                    valid_data_crop.shape[1] if valid_data_crop is not None else 100,
                    np.nan,
                )
        else:  # This 'else' corresponds to the "if self.image_np is not None ..." block
            print(
                "WARNING: Skipping automatic echo tracing due to missing prerequisite data (e.g., image not loaded)."
            )
            # Ensure attributes are initialized to prevent errors later
            width_for_nan_fallback = 100  # A default width if image_np is None
            if self.image_np is not None:
                width_for_nan_fallback = self.image_np.shape[1]
            self.detected_surface_y_abs = np.full(width_for_nan_fallback, np.nan)
            self.detected_bed_y_abs = np.full(width_for_nan_fallback, np.nan)

        # Step 7: Create time-calibrated Z-scope visualization
        print("\nStep 7: Creating time-calibrated Z-scope visualization...")
        time_vis_params_config = self.config.get(
            "time_calibration_visualization_params", {}
        )
        # In the process_image method of ZScopeProcessor class
        self.calibrated_fig, self.calibrated_ax, self.time_axis = (
            create_time_calibrated_zscope(
                self.image_np,
                self.base_filename,
                self.best_pip_details,
                self.transmitter_pulse_y_abs,
                self.data_top_abs,
                self.data_bottom_abs,
                self.pixels_per_microsecond,
                time_vis_params=time_vis_params_config,
                physics_constants=self.physics_constants,
                output_params=current_output_params,
                surface_y_abs=self.detected_surface_y_abs,  # Pass detected surface
                bed_y_abs=self.detected_bed_y_abs,  # Pass detected bed
            )
        )

        if self.calibrated_fig is None:
            print("ERROR: Failed to create time-calibrated Z-scope plot.")
            return False

        print(f"\n--- Processing for {self.base_filename} complete. ---")
        print(
            f"INFO: Main calibrated plot saved to {self.output_dir / (self.base_filename + '_time_calibrated_zscope.png')}"
        )

        return True
