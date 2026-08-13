import os
import sys
import numpy as np
import io
import cv2
import matplotlib.pyplot as plt
import json
import pandas as pd
import datetime
import re
from pathlib import Path

# Optional GUI: prefer PyQt5 for modal image dialogs; fall back to console/os viewer
HAVE_QT = False
try:
    from PyQt5 import QtWidgets, QtGui, QtCore
    HAVE_QT = True
except Exception as e:
    print(f"DEBUG: PyQt5 import failed: {e}")
    HAVE_QT = False

try:
    from PIL import Image
    HAVE_PIL = True
except Exception:
    HAVE_PIL = False

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
    verify_trace_quality,
)
from grid_detection import (
    detect_grid_lines_and_dotted,
    find_reference_line_blackhat,
    interpolate_regular_grid,
)
from echo_detection import (
    find_tx_pulse,
    detect_surface_echo,
    detect_bed_echo,
    detect_double_transmitter_pulse,
    detect_surface_echo_adaptive,
)


def export_ascope_database(base_filename, output_dir, frame_results, meta_info=None):
    """
    Export A-scope results as CSV and NPZ with metadata.
    """
    if not frame_results or len(frame_results) == 0:
        print("ERROR: No frame results to export.")
        return False

    try:
        # Create DataFrame from frame results
        df = pd.DataFrame(frame_results)
        csv_name = base_filename + "_pick.csv"
        npz_name = base_filename + "_pick.npz"
        csv_path = Path(output_dir) / csv_name
        npz_path = Path(output_dir) / npz_name

        # Export CSV with proper formatting
        df.to_csv(csv_path, index=False, float_format="%.6f", na_rep="NaN")
        print(f"INFO: Exported CSV database to {csv_path}")

        # Prepare metadata
        if meta_info is None:
            meta_info = {}
        meta_info["export_timestamp"] = str(datetime.datetime.now())
        meta_info["frame_count"] = len(df)
        meta_info["columns"] = list(df.columns)
        meta_info["filename"] = base_filename
        meta_info["ice_velocity_m_per_us"] = 168.0  # Standard ice velocity

        # Export NPZ with structured data
        np.savez(
            npz_path,
            frame=df["Frame"].values,
            cbd=df["CBD"].values,
            lat=df["LAT"].values if "LAT" in df.columns else np.full(len(df), np.nan),
            lon=df["LON"].values if "LON" in df.columns else np.full(len(df), np.nan),
            surface_time_us=df["Surface_Time_us"].values,
            bed_time_us=df["Bed_Time_us"].values,
            ice_thickness_m=df["Ice_Thickness_m"].values,
            surface_power_db=df["Surface_Power_dB"].values,
            bed_power_db=df["Bed_Power_dB"].values,
            transmitter_time_us=df["Transmitter_Time_us"].values,
            transmitter_power_db=df["Transmitter_Power_dB"].values,
            transmitter_x_pixel=df["Transmitter_X_pixel"].values,
            meta=meta_info,
        )
        print(f"INFO: Exported NPZ database to {npz_path}")

        # Display summary statistics
        valid_thickness = np.sum(~pd.isna(df["Ice_Thickness_m"]))
        valid_surface = np.sum(~pd.isna(df["Surface_Time_us"]))
        valid_bed = np.sum(~pd.isna(df["Bed_Time_us"]))

        print(
            f"INFO: Frame summary - Total: {len(df)}, Valid thickness: {valid_thickness}, Valid surface: {valid_surface}, Valid bed: {valid_bed}"
        )

        return True

    except Exception as e:
        print(f"ERROR: Database export failed: {e}")
        import traceback

        traceback.print_exc()
        return False

    except Exception as e:
        print(f"ERROR: Database export failed: {e}")
        import traceback

        traceback.print_exc()
        return False


class AScope:
    """
    A-scope radar data processing class that orchestrates the entire processing pipeline.
    Enhanced with ice thickness calculation and data export functionality.
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

        # Store resolved config path for interactive editing
        self.config_path_resolved = resolved_config_path

        # Initialize configuration sections
        self.processing_params = self.config.get("processing_params", {})
        self.physical_params = self.config.get("physical_params", {})
        self.output_config = self.config.get("output", {})
        self.debug_mode = self.output_config.get("debug_mode", False)
        self.output_dir = ensure_output_dirs(self.config)

        # Initialize frame results storage
        self.frame_results = []
        self.base_filename = None
        self.cbd_list = []
        self.nav_dict = {}  # CBD -> (LAT, LON) mapping

        # Interactive/deferred plotting state
        self.interactive_mode = False
        self.defer_frame_plots = False
        self._deferred_frame_images = []
        # deferred calibrated-only images (no surface/bed picks)
        self._deferred_calib_images = []

        if self.debug_mode:
            print(f"Debug mode enabled. Using config from: {resolved_config_path}")

        # Qt application instance (created lazily)
        self._qt_app = None

    def set_output_directory(self, output_dir):
        """Override the output directory."""
        if "output" not in self.config:
            self.config["output"] = {}
        self.config["output"]["output_dir"] = output_dir
        self.output_dir = ensure_output_dirs(self.config)

    def set_interactive_mode(self, enabled=True, defer_frame_plots=True):
        """Enable interactive confirmation flow. When enabled, per-frame plots will
        be deferred in memory until final approval.
        """
        self.interactive_mode = bool(enabled)
        self.defer_frame_plots = bool(defer_frame_plots)

    def set_debug_mode(self, debug_mode):
        """Set debug mode for additional outputs."""
        self.debug_mode = debug_mode
        self.output_config["debug_mode"] = debug_mode

    def _extract_cbd_sequence_from_filename(self, filename):
        """Extract CBD sequence from filename and validate frame count."""
        try:
            # Extract CBD range from filename (e.g., F103-C0467_C0479.tiff)
            match = re.search(r"C(\d+)_C(\d+)", filename)
            if not match:
                raise ValueError(
                    f"Could not extract CBD range from filename: {filename}"
                )

            cbd_start = int(match.group(1))  # 467
            cbd_end = int(match.group(2))  # 479
            cbd_list = list(range(cbd_start, cbd_end + 1))  # [467, 468, ..., 479]

            print(
                f"INFO: Extracted CBD sequence: {cbd_start} to {cbd_end} ({len(cbd_list)} CBDs expected)"
            )
            return cbd_list

        except Exception as e:
            print(f"ERROR: Failed to extract CBD sequence: {e}")
            return []

    def _calculate_ice_thickness(self, surface_time_us, bed_time_us):
        """
        Calculate ice thickness using two-way travel times.

        Args:
            surface_time_us (float): Surface echo time in microseconds (two-way)
            bed_time_us (float): Bed echo time in microseconds (two-way)

        Returns:
            float: Ice thickness in meters, or NaN if calculation not possible
        """
        if surface_time_us is None or bed_time_us is None:
            return np.nan

        if np.isnan(surface_time_us) or np.isnan(bed_time_us):
            return np.nan

        if bed_time_us <= surface_time_us:
            print(
                f"WARNING: Bed time ({bed_time_us:.2f} μs) <= surface time ({surface_time_us:.2f} μs)"
            )
            return np.nan

        # Ice thickness = [(bed_time - surface_time) / 2] × ice_velocity
        # Dividing by 2 because radar measures two-way travel time (down and back up)
        # Using standard ice velocity of 168 m/μs
        ice_velocity_m_per_us = 168.0
        travel_time_us = (bed_time_us - surface_time_us) / 2.0
        ice_thickness_m = travel_time_us * ice_velocity_m_per_us

        return ice_thickness_m

    def _export_results(self):
        """Export frame results as CSV and NPZ files."""
        if not self.frame_results:
            print("WARNING: No frame results to export")
            return False

        meta_info = {
            "ascope_file": self.base_filename,
            "processing_timestamp": str(datetime.datetime.now()),
            "ice_velocity_m_per_us": 168.0,
            "total_frames": len(self.frame_results),
            "cbd_sequence": self.cbd_list,
        }

        return export_ascope_database(
            self.base_filename, self.output_dir, self.frame_results, meta_info
        )

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
        self.base_filename = base_filename
        self.frame_results = []  # Reset frame results for new image

        # Clear deferred images state
        self._deferred_frame_images = []

        # Extract CBD sequence from filename
        self.cbd_list = self._extract_cbd_sequence_from_filename(base_filename)

        print(f"\n=== Processing A-scope Image: {base_filename} ===")

        # Step 1: Mask sprocket holes
        print("Step 1: Masking sprocket holes...")
        masked_image, mask = mask_sprocket_holes(image, self.config)
        # Keep masked image for possible re-runs / repicks
        self._last_masked_image = masked_image

        # Step 2: Detect A-scope frames
        print("Step 2: Detecting A-scope frames...")
        expected_frames = len(self.cbd_list) if self.cbd_list else None
        frames = detect_ascope_frames(masked_image, self.config, expected_frames)
        # persist detected frames for potential re-runs
        self.frames = frames

        # Validate frame count against CBD sequence
        if self.cbd_list and len(frames) != len(self.cbd_list):
            print(
                f"WARNING: Frame count ({len(frames)}) doesn't match expected CBD count ({len(self.cbd_list)})"
            )
            
            if len(frames) < len(self.cbd_list):
                # Adjust CBD list to match detected frames
                print(
                    f"INFO: Adjusting CBD list to match {len(frames)} detected frames"
                )
                self.cbd_list = self.cbd_list[: len(frames)]
            else:
                # Use detected frame count and extend CBD list if needed
                print(
                    f"INFO: Extending CBD list to match {len(frames)} detected frames"
                )
                while len(self.cbd_list) < len(frames):
                    self.cbd_list.append(
                        self.cbd_list[-1] + 1 if self.cbd_list else 467
                    )

        # Step 3: Verify frames visually
        print("Step 3: Creating frame verification plot...")
        # verify_frames_visually saves a verification PNG into output dir
        verify_frames_visually(masked_image, frames, base_filename, self.config)

        # Interactive confirmation after frame detection
        if self.interactive_mode:
            try:
                # Loop to allow config editing and re-detection
                while True:
                    verif_file = Path(self.output_dir) / f"{base_filename}_frame_verification.png"
                    if verif_file.exists():
                        print(f"\nFrame verification plot: {verif_file}")
                        
                        # Use simple PyQt5 Yes/No dialog
                        try:
                            from PyQt5.QtWidgets import QApplication, QLabel, QVBoxLayout, QHBoxLayout, QPushButton, QDialog
                            from PyQt5.QtGui import QPixmap
                            from PyQt5.QtCore import Qt
                            
                            app = QApplication.instance() or QApplication([])
                            
                            dialog = QDialog()
                            dialog.setWindowTitle("Frame Detection Verification")
                            dialog.setGeometry(100, 100, 1200, 500)
                            
                            layout = QVBoxLayout()
                            
                            # Load and display image
                            img_label = QLabel()
                            pixmap = QPixmap(str(verif_file))
                            img_label.setPixmap(pixmap.scaledToWidth(1100, Qt.SmoothTransformation))
                            layout.addWidget(img_label)
                            
                            # Buttons
                            button_layout = QHBoxLayout()
                            yes_btn = QPushButton("YES - Proceed")
                            no_btn = QPushButton("NO - Edit Config")
                            button_layout.addWidget(yes_btn)
                            button_layout.addWidget(no_btn)
                            layout.addLayout(button_layout)
                            
                            dialog.setLayout(layout)
                            
                            satisfied = [None]
                            yes_btn.clicked.connect(lambda: (satisfied.__setitem__(0, True), dialog.accept()))
                            no_btn.clicked.connect(lambda: (satisfied.__setitem__(0, False), dialog.accept()))
                            
                            dialog.exec_()
                            satisfied = satisfied[0]
                            
                        except Exception as e:
                            print(f"PyQt5 dialog failed: {e}. Using console input.")
                            resp = input("Are you satisfied with detected frames? (Y/N): ").strip().upper()
                            satisfied = resp == "Y"
                    else:
                        print(f"\nNo verification image found.")
                        resp = input("Are you satisfied with detected frames? (Y/N): ").strip().upper()
                        satisfied = resp == "Y"

                    if satisfied is False:
                        # Edit config manually
                        print(f"\nOpening config for editing: {self.config_path_resolved}")
                        try:
                            os.startfile(self.config_path_resolved)
                        except Exception:
                            print(f"Could not open config. Please open manually: {self.config_path_resolved}")
                        input("Edit config, SAVE it, then press ENTER to continue...")
                        
                        # Reload config after edit
                        try:
                            with open(self.config_path_resolved, "r") as f:
                                self.config = json.load(f)
                        except Exception as e:
                            print(f"Warning: could not reload config: {e}")
                        # Update dependent sections
                        self.processing_params = self.config.get("processing_params", {})
                        self.physical_params = self.config.get("physical_params", {})
                        self.output_config = self.config.get("output", {})
                        self.output_dir = ensure_output_dirs(self.config)
                        
                        # RE-DETECT FRAMES with new config
                        print(f"\n{'='*60}")
                        print(f"Step 2B: RE-DETECTING FRAMES - With updated config")
                        print(f"{'='*60}")
                        frames = detect_ascope_frames(masked_image, self.config, expected_frames=len(self.cbd_list))
                        print(f"✓ Re-detected {len(frames)} frames with updated config\n")
                        
                        # Regenerate verification plot
                        verify_frames_visually(masked_image, frames, base_filename, self.config)
                        
                        # Loop back to show dialog again
                        continue
                    elif satisfied is True:
                        print("✓ Proceeding with detected frames...")
                        break
            except Exception as e:
                print(f"Interactive verification skipped due to error: {e}")
                import traceback
                traceback.print_exc()

        # Centerize first and last A-scope frames by detecting actual signal and centering around it
        print("DEBUG: About to centerize frames...")
        try:
            from utils import centerize_frames
            print(f"DEBUG: centerize_frames imported successfully")
            frames = centerize_frames(frames, masked_image, masked_image.shape[1])
            print(f"DEBUG: Frames after centerize: {frames[:2]}")
        except Exception as e:
            print(f"ERROR: centerize_frames failed: {e}")
            import traceback
            traceback.print_exc()

        # Step 4: Generate quick trace detection for montage review (no echo picking yet)
        print(f"\n{'='*60}")
        print(f"Step 4: GENERATE TRACES - Quick detection for review")
        print(f"{'='*60}")
        print(f"Generating trace images for {len(frames)} frames (no echo picking)...")
        for idx, (left, right) in enumerate(frames):
            print(f"Frame {idx + 1}/{len(frames)}: cols {left}-{right}", end=" ... ")
            self._generate_trace_only(masked_image, left, right, base_filename, idx + 1, total_frames=len(frames))
            print("✓")

        print(f"\n✓ Generated {len(self._deferred_frame_images)} trace images.\n")
        
        # Step 4.5: Show trace montage for approval BEFORE full processing
        print(f"{'='*60}")
        print(f"Step 4.5: TRACE MONTAGE - Review detected traces")
        print(f"{'='*60}")
        if self.interactive_mode and self.defer_frame_plots and len(self._deferred_frame_images) > 0:
            from utils import show_trace_montage
            print(f"\nLaunching trace montage with {len(self._deferred_frame_images)} frames...")
            print("Use zoom controls to inspect. Click YES to proceed to full processing, or EDIT CONFIG to adjust parameters.\n")
            
            # Loop to allow config editing and re-detection
            while True:
                trace_result = show_trace_montage(
                    self._deferred_frame_images,
                    len(frames),
                    dpi=self.output_config.get("montage_dpi", 150)
                )
                # If user asked to show calibrated-only plots, build images and show them
                if trace_result == "calib":
                    print("\n=== Building calibrated-only plots (no picks) for montage review ===")
                    # Build calibrated images for all frames (replace existing list)
                    self._deferred_calib_images = []
                    for idx, (l, r) in enumerate(frames):
                        print(f"Calibrated plot: frame {idx+1}/{len(frames)}")
                        self._generate_calibrated_only(masked_image, l, r, base_filename, idx + 1, total_frames=len(frames))

                    from utils import show_calibrated_montage

                    calib_res = show_calibrated_montage(self._deferred_calib_images, len(frames), dpi=self.output_config.get("montage_dpi", 150))
                    if calib_res == "edit":
                        # Open config for editing then re-detect traces and calibrated plots
                        print(f"\n{'='*60}")
                        print(f"Opening config for editing: {self.config_path_resolved}")
                        print(f"{'='*60}")
                        try:
                            os.startfile(self.config_path_resolved)
                            print("✓ Config file opened in editor")
                        except Exception as e:
                            print(f"Could not open config automatically. Please open the file manually: {self.config_path_resolved}")
                            print(f"Error: {e}")

                        print("\nEdit the config file, SAVE it, and CLOSE the editor.")
                        print("Then press ENTER in THIS TERMINAL WINDOW to continue.\n")
                        sys.stdout.flush()
                        sys.stderr.flush()
                        os.system("pause")
                        try:
                            with open(self.config_path_resolved, 'r') as f:
                                self.config = json.load(f)
                        except Exception as e:
                            print(f"Warning: could not reload config: {e}")
                        # Update dependent sections
                        self.processing_params = self.config.get("processing_params", {})
                        self.physical_params = self.config.get("physical_params", {})
                        self.output_config = self.config.get("output", {})
                        self.output_dir = ensure_output_dirs(self.config)

                        # Rebuild trace images and calibrated images for re-review
                        print(f"\n{'='*60}")
                        print(f"Step 4: RE-DETECT TRACES & CALIBRATED PLOTS - With updated config")
                        print(f"{'='*60}")
                        self._deferred_frame_images = []
                        self.frame_results = []
                        self._deferred_calib_images = []
                        for idx, (left, right) in enumerate(frames):
                            self._generate_trace_only(masked_image, left, right, base_filename, idx + 1, total_frames=len(frames))
                            self._generate_calibrated_only(masked_image, left, right, base_filename, idx + 1, total_frames=len(frames))
                        print("\n✓ Trace + calibrated re-detection complete. Showing updated traces...\n")
                        # Loop back to the trace montage
                        continue
                    else:
                        # if user closed or saved, return to trace montage for next action
                        print("Returning to trace montage after calibrated-only review.")
                        continue
                
                if trace_result == "edit":
                    # User wants to edit config and re-detect traces
                    print(f"\n{'='*60}")
                    print(f"Opening config for editing: {self.config_path_resolved}")
                    print(f"{'='*60}")
                    try:
                        os.startfile(self.config_path_resolved)
                        print("✓ Config file opened in editor")
                    except Exception as e:
                        print(f"Could not open config automatically. Please open the file manually: {self.config_path_resolved}")
                        print(f"Error: {e}")
                    
                    print("\nEdit the config file, SAVE it, and CLOSE the editor.")
                    print("Then press ENTER in THIS TERMINAL WINDOW to continue.\n")
                    sys.stdout.flush()
                    sys.stderr.flush()
                    
                    # Use os.system() with 'pause' command on Windows - this WILL block and wait for user input
                    os.system("pause")
                    try:
                        with open(self.config_path_resolved, 'r') as f:
                            self.config = json.load(f)
                    except Exception as e:
                        print(f"Warning: could not reload config: {e}")
                    # Update dependent sections
                    self.processing_params = self.config.get("processing_params", {})
                    self.physical_params = self.config.get("physical_params", {})
                    self.output_config = self.config.get("output", {})
                    self.output_dir = ensure_output_dirs(self.config)
                    
                    # Re-detect traces only (Step 4 again)
                    print(f"\n{'='*60}")
                    print(f"Step 4: RE-DETECT TRACES - With updated config")
                    print(f"{'='*60}")
                    print(f"DEBUG: Re-detecting with {len(frames)} frames")
                    self._deferred_frame_images = []
                    self.frame_results = []  # ✓ CRITICAL: Clear cached frame results so Step 4.5 will reprocess with new config
                    for idx, (left, right) in enumerate(frames):
                        print(f"Re-detecting traces: frame {idx+1}/{len(frames)}...")
                        self._generate_trace_only(masked_image, left, right, base_filename, idx + 1, total_frames=len(frames))
                    print(f"\n✓ Trace re-detection complete. Showing updated traces...\n")
                    # Loop back to show trace montage again
                    continue
                
                elif trace_result is None:
                    print("⚠ Warning: Could not show trace montage, continuing anyway...")
                    break
                
                elif trace_result == "save":
                    print("✓ User approved traces. Proceeding to full processing...\n")
                    break
                
                else:
                    print("✗ Trace review cancelled. Processing aborted.")
                    return
        else:
            print("(Skipping trace montage)")

        # Step 4.5: Process frames for full echo detection and picking
        print(f"{'='*60}")
        print(f"Step 4.5: PROCESS FRAMES - Full echo detection & picking")
        print(f"{'='*60}")
        print(f"Processing {len(frames)} frames for surface/bed echo detection...")
        print(f"DEBUG: About to process frames, len(frames)={len(frames)}\n")
        for idx, (left, right) in enumerate(frames):
            print(
                f"--- Processing frame {idx + 1}/{len(frames)}: cols {left}-{right} ---"
            )
            self._process_frame(masked_image, left, right, base_filename, idx + 1, total_frames=len(frames))

        print(f"\n✓ All frames processed.\n")

        # If deferring frame plots, show combined montage for user approval before final export
        if self.interactive_mode and self.defer_frame_plots and len(self._deferred_frame_images) > 0:
            self._show_montage_and_handle_user_decisions()

        self._export_results()

        print(f"\n=== Processing Complete for {base_filename} ===")

    def _generate_trace_only(self, masked_image, left, right, base_filename, frame_idx, total_frames=None):
        """Generate trace image only (no echo detection/picking).
        
        For edge frames (first and last), use asymmetric trimming to preserve boundary signals.
        Frame 1: Trim left only (keep right edge)
        Last frame: Trim right only (keep left edge)
        """
        try:
            frame_img = masked_image[:, left:right].copy()
            h, w = frame_img.shape

            if w <= 0 or h <= 0:
                return

            # Detect signal
            from signal_processing import detect_signal_in_frame, trim_signal_trace
            signal_result = detect_signal_in_frame(frame_img, self.config)
            if signal_result is None:
                return
            
            signal_x, signal_y = signal_result
            if len(signal_x) < 10:
                return
            
            # Determine if this is an edge frame and apply asymmetric trimming
            is_first_frame = (frame_idx == 1)
            is_last_frame = (total_frames and frame_idx == total_frames)
            
            if is_first_frame or is_last_frame:
                # For edge frames, apply asymmetric trimming from config
                # left_frac and right_frac define the region to KEEP (as absolute positions)
                processing_params = self.config.get("processing_params", {})
                
                x = np.array(signal_x)
                y = np.array(signal_y)
                
                if is_first_frame:
                    left_pos_frac = processing_params.get("edge_frame_trim_first_left_frac", 0.09)
                    right_pos_frac = processing_params.get("edge_frame_trim_first_right_frac", 0.79)
                    frame_type = "FIRST"
                else:
                    left_pos_frac = processing_params.get("edge_frame_trim_last_left_frac", 0.23)
                    right_pos_frac = processing_params.get("edge_frame_trim_last_right_frac", 0.89)
                    frame_type = "LAST"
                
                print(f"DEBUG: is_first_frame={is_first_frame}, is_last_frame={is_last_frame}, frame_type={frame_type}, frame_idx={frame_idx}, total_frames={total_frames}")
                
                # Calculate trace region boundaries as absolute pixel positions
                left_boundary_px = int(w * left_pos_frac)
                right_boundary_px = int(w * right_pos_frac)
                
                # Create mask for the region to keep
                if left_boundary_px >= right_boundary_px:
                    print(f"[TRACE EDGE FRAME {frame_type}] Frame {frame_idx}: WARNING - invalid boundaries (left {left_boundary_px}px >= right {right_boundary_px}px)")
                    print(f"  Using full signal range as fallback")
                    mask = np.ones(len(x), dtype=bool)
                else:
                    mask = (x >= left_boundary_px) & (x <= right_boundary_px)
                    print(f"[TRACE EDGE FRAME {frame_type}] Frame {frame_idx}: Keeping region from {left_pos_frac*100:.1f}% ({left_boundary_px}px) to {right_pos_frac*100:.1f}% ({right_boundary_px}px)")
                    print(f"  Signal x range: {x.min():.1f} to {x.max():.1f}, keeping: {left_boundary_px} to {right_boundary_px}px")
                
                if np.any(mask):
                    signal_x_trim = x[mask]
                    signal_y_trim = y[mask]
                    print(f"  Trimmed x range: {signal_x_trim.min():.1f} to {signal_x_trim.max():.1f} ({len(signal_x_trim)} points)")
                else:
                    signal_x_trim, signal_y_trim = x, y
                    print(f"  No points within trim range, using original signal")
            else:
                # Normal trimming for non-edge frames
                trim_result = trim_signal_trace(frame_img, signal_x, signal_y, self.config)
                if trim_result is None:
                    return
                signal_x_trim, signal_y_trim = trim_result
            
            # Create simple trace plot
            fig, ax = plt.subplots(figsize=(12, 4))
            ax.imshow(frame_img, cmap='gray', aspect='auto')
            ax.plot(signal_x_trim, signal_y_trim, 'r-', linewidth=2, label='Detected Trace')
            ax.set_title(f"Frame {frame_idx}: Detected Trace (Preview)")
            ax.legend(fontsize=8)
            ax.set_xlabel("X (pixels)")
            ax.set_ylabel("Y (pixels)")
            
            # Store as image
            buf = io.BytesIO()
            fig.savefig(buf, format='png', dpi=self.output_config.get("montage_dpi", 150))
            buf.seek(0)
            from PIL import Image
            trace_img = Image.open(buf)
            self._deferred_frame_images.append(np.array(trace_img).copy())
            plt.close(fig)
            
        except Exception as e:
            print(f"  Error: {e}")

    def _generate_calibrated_only(self, masked_image, left, right, base_filename, frame_idx, total_frames=None, replace_index=None):
        """Generate a calibrated (time vs power) plot for this frame without surface/bed picks
        and store it in self._deferred_calib_images (or replace at replace_index if provided).
        """
        try:
            import matplotlib.pyplot as plt
            from PIL import Image as PILImage

            frame_img = masked_image[:, left:right].copy()
            h, w = frame_img.shape

            # Try to detect the signal trace
            signal_x, signal_y = detect_signal_in_frame(frame_img, self.config)
            if signal_x is None:
                # fallback: create a simple frame image
                fig, ax = plt.subplots(figsize=(10, 4))
                ax.imshow(frame_img, cmap='gray', aspect='auto')
                ax.set_title(f"Frame {frame_idx}: (no trace detected)")
                buf = io.BytesIO()
                fig.savefig(buf, format='png', dpi=self.output_config.get('montage_dpi',150), bbox_inches='tight')
                buf.seek(0)
                img = np.array(PILImage.open(buf).convert('RGB'))
                buf.close()
                if replace_index is None:
                    self._deferred_calib_images.append(img)
                else:
                    if 0 <= replace_index < len(self._deferred_calib_images):
                        self._deferred_calib_images[replace_index] = img
                    else:
                        while len(self._deferred_calib_images) < replace_index:
                            self._deferred_calib_images.append(img)
                        self._deferred_calib_images.append(img)
                plt.close(fig)
                return

            # Trim / clean trace (same as processing)
            is_first_frame = (frame_idx == 1)
            is_last_frame = False
            if total_frames is not None:
                is_last_frame = (frame_idx == total_frames)

            if is_first_frame or is_last_frame:
                processing_params = self.config.get('processing_params', {})
                x = np.array(signal_x)
                y = np.array(signal_y)
                h_frame, w_frame = frame_img.shape
                if is_first_frame:
                    left_pos_frac = processing_params.get('edge_frame_trim_first_left_frac', 0.09)
                    right_pos_frac = processing_params.get('edge_frame_trim_first_right_frac', 0.79)
                else:
                    left_pos_frac = processing_params.get('edge_frame_trim_last_left_frac', 0.23)
                    right_pos_frac = processing_params.get('edge_frame_trim_last_right_frac', 0.89)

                left_boundary_px = int(w_frame * left_pos_frac)
                right_boundary_px = int(w_frame * right_pos_frac)
                if left_boundary_px >= right_boundary_px:
                    mask = np.ones(len(x), dtype=bool)
                else:
                    mask = (x >= left_boundary_px) & (x <= right_boundary_px)

                if np.any(mask):
                    signal_x_clean = x[mask]
                    signal_y_clean = y[mask]
                else:
                    signal_x_clean, signal_y_clean = signal_x, signal_y
            else:
                res = trim_signal_trace(frame_img, signal_x, signal_y, self.config)
                if res is None:
                    signal_x_clean, signal_y_clean = signal_x, signal_y
                else:
                    signal_x_clean, signal_y_clean = res

            # Find TX pulse for calibration anchor
            tx_pulse_col, tx_idx_in_clean = find_tx_pulse(signal_x_clean, signal_y_clean, self.config)

            # Find reference line (y_ref)
            ref_row = find_reference_line_blackhat(frame_img, base_filename, frame_idx, self.config)

            # If we can't calibrate, fallback to just drawing trace image
            if tx_pulse_col is None or ref_row is None:
                fig, ax = plt.subplots(figsize=(10, 4))
                ax.imshow(frame_img, cmap='gray', aspect='auto')
                ax.plot(signal_x_clean, signal_y_clean, 'r-', linewidth=2)
                ax.set_title(f"Frame {frame_idx}: Trace (no calibration available)")
                buf = io.BytesIO()
                fig.savefig(buf, format='png', dpi=self.output_config.get('montage_dpi',150), bbox_inches='tight')
                buf.seek(0)
                img = np.array(PILImage.open(buf).convert('RGB'))
                buf.close()
                if replace_index is None:
                    self._deferred_calib_images.append(img)
                else:
                    if 0 <= replace_index < len(self._deferred_calib_images):
                        self._deferred_calib_images[replace_index] = img
                    else:
                        while len(self._deferred_calib_images) < replace_index:
                            self._deferred_calib_images.append(img)
                        self._deferred_calib_images.append(img)
                plt.close(fig)
                return

            # Detect grid and interpolate to compute calibration factors
            h_peaks_initial, v_peaks_initial, h_minor_peaks, v_minor_peaks = detect_grid_lines_and_dotted(frame_img, self.config)
            # Determine dynamic y/x ranges similar to full processing
            y_major, y_minor = interpolate_regular_grid(h, h_peaks_initial, ref_row, self.physical_params['y_major_dB'], self.physical_params['y_minor_per_major'], 8.25 * 10, is_y_axis=True, config=self.config)
            x_range_us = self.physical_params.get('x_range_factor') * self.physical_params.get('x_major_us')
            x_major, x_minor = interpolate_regular_grid(w, v_peaks_initial, tx_pulse_col, self.physical_params['x_major_us'], self.physical_params['x_minor_per_major'], x_range_us, is_y_axis=False, config=self.config)

            px_per_us_echo, px_per_db_echo = self._calculate_calibration_factors(x_major, y_major, w, h)

            # Calibrate values
            power_vals = self.physical_params['y_ref_dB'] - (signal_y_clean - ref_row) / px_per_db_echo
            time_vals = (signal_x_clean - tx_pulse_col) / px_per_us_echo

            # Plot calibrated power vs time WITHOUT surface/bed picks
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(time_vals, power_vals, color='red', linewidth=1.5)
            ax.set_title(f"Calibrated A-scope Frame {frame_idx}")
            ax.set_xlabel('One-way travel time (µs)')
            ax.set_ylabel('Power (dB)')
            ax.grid(True, alpha=0.3)

            # Plot Tx marker if available
            if tx_idx_in_clean is not None and tx_idx_in_clean < len(time_vals):
                ax.plot(time_vals[tx_idx_in_clean], power_vals[tx_idx_in_clean], 'o', color='blue', markersize=6, label='Tx')

            # Add legend for Tx only
            handles, labels = ax.get_legend_handles_labels()
            if handles:
                ax.legend(loc='upper right')

            buf = io.BytesIO()
            fig.savefig(buf, format='png', dpi=self.output_config.get('montage_dpi',150), bbox_inches='tight')
            buf.seek(0)
            img = np.array(PILImage.open(buf).convert('RGB'))
            buf.close()

            if replace_index is None:
                self._deferred_calib_images.append(img)
            else:
                if 0 <= replace_index < len(self._deferred_calib_images):
                    self._deferred_calib_images[replace_index] = img
                else:
                    while len(self._deferred_calib_images) < replace_index:
                        self._deferred_calib_images.append(img)
                    self._deferred_calib_images.append(img)

            plt.close(fig)

        except Exception as e:
            print(f"  Error (calib image): {e}")

    def _process_frame(self, masked_image, left, right, base_filename, frame_idx, total_frames=None):
        """Process an individual A-scope frame and store results.
        
        For edge frames (1st and last), uses asymmetric trimming to preserve boundary signals.
        """
        # Initialize frame result structure
        # Get CBD and navigation data
        cbd = self.cbd_list[frame_idx - 1] if frame_idx <= len(self.cbd_list) else np.nan
        lat, lon = np.nan, np.nan
        if not np.isnan(cbd) and cbd in self.nav_dict:
            lat, lon = self.nav_dict[cbd]
            
        frame_result = {
            "Frame": frame_idx,
            "CBD": cbd,
            "LAT": lat,
            "LON": lon,
            "Surface_Time_us": np.nan,
            "Bed_Time_us": np.nan,
            "Ice_Thickness_m": np.nan,
            "Surface_Power_dB": np.nan,
            "Bed_Power_dB": np.nan,
            "Transmitter_Time_us": np.nan,
            "Transmitter_Power_dB": np.nan,
            "Transmitter_X_pixel": np.nan,
            "Noise_Floor_Time_us": np.nan,
            "Noise_Floor_Power_dB": np.nan,
        }

        try:
            # Extract the frame
            frame_img = masked_image[:, left:right].copy()
            h, w = frame_img.shape

            if w <= 0 or h <= 0:
                print("Warning: Frame has zero width or height. Skipping.")
                self.frame_results.append(frame_result)
                return

            # 1. Detect Signal Trace
            signal_x, signal_y = detect_signal_in_frame(frame_img, self.config)
            if signal_x is None:
                print(f"Signal detection failed for frame {frame_idx}. Skipping.")
                self.frame_results.append(frame_result)
                return

            signal_y = adaptive_peak_preserving_smooth(signal_y, self.config)

            # 2. Clean Signal Trace
            # For edge frames, use asymmetric trimming
            is_first_frame = (frame_idx == 1)
            is_last_frame = (total_frames and frame_idx == total_frames)
            
            print(f"DEBUG: frame_idx={frame_idx}, total_frames={total_frames}, is_last_frame={is_last_frame}")
            
            if is_first_frame or is_last_frame:
                # Apply asymmetric trimming for edge frames
                # left_frac and right_frac define the region to KEEP (as absolute positions)
                processing_params = self.config.get("processing_params", {})
                x = np.array(signal_x)
                y = np.array(signal_y)
                h_frame, w_frame = frame_img.shape
                
                if is_first_frame:
                    left_pos_frac = processing_params.get("edge_frame_trim_first_left_frac", 0.09)
                    right_pos_frac = processing_params.get("edge_frame_trim_first_right_frac", 0.79)
                    frame_type = "FIRST"
                else:
                    left_pos_frac = processing_params.get("edge_frame_trim_last_left_frac", 0.23)
                    right_pos_frac = processing_params.get("edge_frame_trim_last_right_frac", 0.89)
                    frame_type = "LAST"
                
                # Calculate trace region boundaries as absolute pixel positions
                left_boundary_px = int(w_frame * left_pos_frac)
                right_boundary_px = int(w_frame * right_pos_frac)
                
                # Create mask for the region to keep
                if left_boundary_px >= right_boundary_px:
                    print(f"[EDGE FRAME {frame_type}] Frame {frame_idx}: WARNING - invalid boundaries (left {left_boundary_px}px >= right {right_boundary_px}px)")
                    print(f"  Using full signal range as fallback")
                    mask = np.ones(len(x), dtype=bool)
                else:
                    mask = (x >= left_boundary_px) & (x <= right_boundary_px)
                    print(f"[EDGE FRAME {frame_type}] Frame {frame_idx}: Keeping region from {left_pos_frac*100:.1f}% ({left_boundary_px}px) to {right_pos_frac*100:.1f}% ({right_boundary_px}px)")
                    print(f"  Signal x range: {x.min():.1f} to {x.max():.1f}, keeping: {left_boundary_px} to {right_boundary_px}px")
                
                if np.any(mask):
                    signal_x_clean = x[mask]
                    signal_y_clean = y[mask]
                else:
                    signal_x_clean, signal_y_clean = signal_x, signal_y
            else:
                # Normal trimming for non-edge frames
                signal_x_clean, signal_y_clean = trim_signal_trace(
                    frame_img, signal_x, signal_y, self.config
                )

            if len(signal_x_clean) == 0:
                print(
                    f"No valid signal trace left after cleaning for frame {frame_idx}. Skipping."
                )
                self.frame_results.append(frame_result)
                return

            print(f"Cleaned signal length: {len(signal_x_clean)} points")

            # Verify the signal trace
            trace_quality_score = verify_trace_quality(
                frame_img, signal_x_clean, signal_y_clean
            )
            print(
                f"Trace quality score for frame {frame_idx}: {trace_quality_score:.2f}"
            )

            # 3. Initial Tx Pulse Detection (for grid calibration)
            tx_pulse_col, tx_idx_in_clean = find_tx_pulse(
                signal_x_clean, signal_y_clean, self.config
            )

            if tx_pulse_col is None:
                print("Warning: Tx pulse detection failed. Using default X-anchor.")
                tx_pulse_col = w * 0.15  # Fallback anchor near start
                tx_idx_in_clean = (
                    int(len(signal_x_clean) * 0.15) if len(signal_x_clean) > 0 else 0
                )

            print(
                f"Initial Tx pulse estimated at col {tx_pulse_col:.1f} (index {tx_idx_in_clean} in clean signal)"
            )

            # 4. Find Reference Line (Y-axis anchor)
            ref_row = find_reference_line_blackhat(
                frame_img, base_filename, frame_idx, self.config
            )

            if ref_row is None:
                print(
                    "Error: Reference line detection failed critically. Skipping frame."
                )
                self.frame_results.append(frame_result)
                return

            print(
                f"Reference row ({self.physical_params['y_ref_dB']} dB) estimated at y={ref_row}"
            )

            # 5. Detect Grid Lines (for interpolation)
            # Only request grid QA plot when explicitly enabled in config or debug mode
            qa_path = None
            if self.output_config.get("save_grid_qa", False) or self.debug_mode:
                qa_path = f"{self.output_dir}/{base_filename}_frame{frame_idx:02d}_grid_QA.png"

            h_peaks_initial, v_peaks_initial, h_minor_peaks, v_minor_peaks = (
                detect_grid_lines_and_dotted(
                    frame_img, self.config, qa_plot_path=qa_path, ref_row_for_qa=ref_row
                )
            )

            # When calling interpolate_regular_grid, use the dynamic range values
            y_range_dB = 8.25 * 10  # Dynamic range based on signal extent
            y_major, y_minor = interpolate_regular_grid(
                h,
                h_peaks_initial,
                ref_row,
                self.physical_params["y_major_dB"],
                self.physical_params["y_minor_per_major"],
                y_range_dB,
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
                tx_pulse_col,  # ✅ Fixed: Use tx_pulse_col (number) not tx_analysis (dict)
                self.physical_params["x_major_us"],
                self.physical_params["x_minor_per_major"],
                x_range_us,
                is_y_axis=False,
                config=self.config,
            )

            print(f"Saved grid QA plot: {qa_path}")

            # 6. Calculate Calibration Factors
            px_per_us_echo, px_per_db_echo = self._calculate_calibration_factors(
                x_major, y_major, w, h
            )

            # 7. Calibrate Signal to get power_vals and time_vals
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
                self.frame_results.append(frame_result)
                return

            # 8. Detection with Double TX Pulse Support
            print("INFO: Starting enhanced echo detection...")

            # Call enhanced double transmitter pulse detection
            tx_analysis = detect_double_transmitter_pulse(
                signal_x_clean, signal_y_clean, power_vals, time_vals, self.config
            )

            if tx_analysis["is_double_pulse"]:
                print(
                    f"INFO: Double transmitter pulse detected (confidence: {tx_analysis['confidence']:.2f})"
                )
                print(
                    f"INFO: TX components at: {[time_vals[i] for i in tx_analysis['tx_peaks']]} μs"
                )

                # Update tx_idx_in_clean if enhanced detection found something better
                if tx_analysis.get("recommended_tx_idx") is not None:
                    tx_idx_in_clean = tx_analysis["recommended_tx_idx"]
                    print(
                        f"INFO: Updated TX index to {tx_idx_in_clean} based on enhanced detection"
                    )

            # Call surface detection
            surf_idx_in_clean = detect_surface_echo_adaptive(
                power_vals, time_vals, tx_analysis, self.config
            )

            # Call bed detection
            bed_idx_in_clean = detect_bed_echo(
                power_vals, time_vals, surf_idx_in_clean, px_per_us_echo, self.config
            )

            # 9. Store Frame Results with Ice Thickness Calculation
            if (
                surf_idx_in_clean is not None
                and surf_idx_in_clean < len(time_vals)
                and time_vals is not None
                and power_vals is not None
            ):
                surface_time = time_vals[surf_idx_in_clean]
                surface_power = power_vals[surf_idx_in_clean]
                frame_result["Surface_Time_us"] = surface_time
                frame_result["Surface_Power_dB"] = surface_power

                print(
                    f"Surface detected: {surface_time:.2f} μs, {surface_power:.1f} dB"
                )

            if (
                bed_idx_in_clean is not None
                and bed_idx_in_clean < len(time_vals)
                and time_vals is not None
                and power_vals is not None
            ):
                bed_time = time_vals[bed_idx_in_clean]
                bed_power = power_vals[bed_idx_in_clean]
                frame_result["Bed_Time_us"] = bed_time
                frame_result["Bed_Power_dB"] = bed_power

                print(f"Bed detected: {bed_time:.2f} μs, {bed_power:.1f} dB")

            # Calculate ice thickness if both surface and bed are available
            if not np.isnan(frame_result["Surface_Time_us"]) and not np.isnan(
                frame_result["Bed_Time_us"]
            ):
                ice_thickness = self._calculate_ice_thickness(
                    frame_result["Surface_Time_us"], frame_result["Bed_Time_us"]
                )
                frame_result["Ice_Thickness_m"] = ice_thickness

                if not np.isnan(ice_thickness):
                    print(f"Ice thickness calculated: {ice_thickness:.1f} m")

            # Store Transmitter Pulse Results
            if (
                tx_idx_in_clean is not None
                and tx_idx_in_clean < len(time_vals)
                and time_vals is not None
                and power_vals is not None
                and tx_pulse_col is not None
            ):
                tx_time = time_vals[tx_idx_in_clean]
                tx_power = power_vals[tx_idx_in_clean]

                frame_result["Transmitter_Time_us"] = tx_time
                frame_result["Transmitter_Power_dB"] = tx_power
                frame_result["Transmitter_X_pixel"] = tx_pulse_col

                print(
                    f"Transmitter pulse stored: {tx_time:.2f} μs, {tx_power:.1f} dB, X-pixel: {tx_pulse_col:.1f}"
                )

            # Store Noise Floor Results
            if time_vals is not None and power_vals is not None and len(time_vals) > 0:
                processing_params = self.config.get("processing_params", {})
                noise_floor_start_time = processing_params.get("noise_floor_window_start_us", 5.0)
                noise_floor_end_time = processing_params.get("noise_floor_window_end_us", 6.2)
                
                noise_floor_mask = (time_vals >= noise_floor_start_time) & (time_vals <= noise_floor_end_time)
                if np.any(noise_floor_mask):
                    noise_floor_region = power_vals[noise_floor_mask]
                    noise_floor_power = np.min(noise_floor_region)
                    min_idx_in_window = np.argmin(noise_floor_region)
                    noise_floor_time = time_vals[noise_floor_mask][min_idx_in_window]
                    
                    frame_result["Noise_Floor_Time_us"] = noise_floor_time
                    frame_result["Noise_Floor_Power_dB"] = noise_floor_power
                    
                    print(f"Noise floor detected: {noise_floor_time:.2f} μs, {noise_floor_power:.1f} dB")

            # 10. Plot Combined Results (renamed to "picked.png")
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
                replace_index=frame_idx-1,  # Replace the trace image with picked image
            )

        except Exception as e:
            print(f"ERROR: Frame {frame_idx} processing failed: {e}")
            import traceback

            traceback.print_exc()

        # Store frame result (even if processing failed)
        self.frame_results.append(frame_result)

    def _process_frame_for_override(
        self, masked_image, left, right, base_filename, frame_idx
    ):
        """
        Process frame up to automatic echo detection for interactive override.

        Returns:
            dict: Frame data containing all necessary components for override
        """
        try:
            # Extract the frame
            if masked_image is None:
                # When re-running per-frame processing without full image, try to load image from original
                # The calling code must ensure frame_img is available; here we cannot proceed
                print("ERROR: masked_image is None in _process_frame_for_override")
                return None
            frame_img = masked_image[:, left:right].copy()
            h, w = frame_img.shape

            if w <= 0 or h <= 0:
                print("Warning: Frame has zero width or height.")
                return None

            # 1-7: Same processing as normal frame processing up to signal calibration
            from functions.signal_processing import (
                detect_signal_in_frame,
                trim_signal_trace,
                adaptive_peak_preserving_smooth,
            )
            from functions.grid_detection import (
                detect_grid_lines_and_dotted,
                find_reference_line_blackhat,
                interpolate_regular_grid,
            )
            from functions.echo_detection import (
                find_tx_pulse,
                detect_surface_echo_adaptive,
                detect_bed_echo,
            )

            # Signal detection
            signal_x, signal_y = detect_signal_in_frame(frame_img, self.config)
            if signal_x is None:
                print(f"Signal detection failed for frame {frame_idx}")
                return None

            signal_y = adaptive_peak_preserving_smooth(signal_y, self.config)
            
            # Apply edge frame asymmetric trimming if applicable
            is_first_frame = (frame_idx == 1)
            is_last_frame = (frame_idx == len(self.frames))
            
            if is_first_frame or is_last_frame:
                # Apply asymmetric edge frame trimming
                # left_frac and right_frac define the region to KEEP (as absolute positions)
                processing_params = self.config.get("processing_params", {})
                x = np.array(signal_x)
                y = np.array(signal_y)
                h_frame, w_frame = frame_img.shape
                
                if is_first_frame:
                    left_pos_frac = processing_params.get("edge_frame_trim_first_left_frac", 0.09)
                    right_pos_frac = processing_params.get("edge_frame_trim_first_right_frac", 0.79)
                    frame_type = "FIRST"
                else:  # is_last_frame
                    left_pos_frac = processing_params.get("edge_frame_trim_last_left_frac", 0.23)
                    right_pos_frac = processing_params.get("edge_frame_trim_last_right_frac", 0.89)
                    frame_type = "LAST"
                
                # Calculate trace region boundaries as absolute pixel positions
                left_boundary_px = int(w_frame * left_pos_frac)
                right_boundary_px = int(w_frame * right_pos_frac)
                
                # Create mask for the region to keep
                if left_boundary_px >= right_boundary_px:
                    print(f"[OVERRIDE {frame_type} FRAME] Frame {frame_idx}: WARNING - invalid boundaries (left {left_boundary_px}px >= right {right_boundary_px}px)")
                    print(f"  Using full signal range as fallback")
                    signal_x_clean = x
                    signal_y_clean = y
                else:
                    mask = (x >= left_boundary_px) & (x <= right_boundary_px)
                    if np.any(mask):
                        signal_x_clean = x[mask]
                        signal_y_clean = y[mask]
                    else:
                        signal_x_clean, signal_y_clean = signal_x, signal_y
            else:
                # Normal trimming for non-edge frames
                signal_x_clean, signal_y_clean = trim_signal_trace(
                    frame_img, signal_x, signal_y, self.config
                )

            if len(signal_x_clean) == 0:
                print(f"No valid signal trace for frame {frame_idx}")
                return None

            # TX pulse and reference line detection
            tx_pulse_col, tx_idx_in_clean = find_tx_pulse(
                signal_x_clean, signal_y_clean, self.config
            )
            ref_row = find_reference_line_blackhat(
                frame_img, base_filename, frame_idx, self.config
            )

            if tx_pulse_col is None or ref_row is None:
                print(f"Critical detection failed for frame {frame_idx}")
                return None

            # Grid detection and calibration
            h_peaks_initial, v_peaks_initial, h_minor_peaks, v_minor_peaks = (
                detect_grid_lines_and_dotted(frame_img, self.config)
            )

            y_range_dB = 8.25 * 10
            y_major, y_minor = interpolate_regular_grid(
                h,
                h_peaks_initial,
                ref_row,
                self.physical_params["y_major_dB"],
                self.physical_params["y_minor_per_major"],
                y_range_dB,
                is_y_axis=True,
                config=self.config,
            )

            x_range_us = self.physical_params.get(
                "x_range_factor"
            ) * self.physical_params.get("x_major_us")
            x_major, x_minor = interpolate_regular_grid(
                w,
                v_peaks_initial,
                tx_pulse_col,
                self.physical_params["x_major_us"],
                self.physical_params["x_minor_per_major"],
                x_range_us,
                is_y_axis=False,
                config=self.config,
            )

            # Calculate calibration factors
            px_per_us_echo, px_per_db_echo = self._calculate_calibration_factors(
                x_major, y_major, w, h
            )

            # Calibrate signal
            power_vals = (
                self.physical_params["y_ref_dB"]
                - (signal_y_clean - ref_row) / px_per_db_echo
            )
            time_vals = (signal_x_clean - tx_pulse_col) / px_per_us_echo

            # Get automatic echo detection results
            from functions.echo_detection import (
                detect_double_transmitter_pulse,
                detect_surface_echo_adaptive,
                detect_bed_echo,
            )

            tx_analysis = detect_double_transmitter_pulse(
                signal_x_clean, signal_y_clean, power_vals, time_vals, self.config
            )
            surface_idx = detect_surface_echo_adaptive(
                power_vals, time_vals, tx_analysis, self.config
            )
            bed_idx = detect_bed_echo(
                power_vals, time_vals, surface_idx, px_per_us_echo, self.config
            )

            # Use enhanced TX detection if available
            if tx_analysis.get("recommended_tx_idx") is not None:
                tx_idx_in_clean = tx_analysis["recommended_tx_idx"]

            return {
                "frame_img": frame_img,
                "signal_x_clean": signal_x_clean,
                "signal_y_clean": signal_y_clean,
                "power_vals": power_vals,
                "time_vals": time_vals,
                "tx_idx": tx_idx_in_clean,
                "surface_idx": surface_idx,
                "bed_idx": bed_idx,
                "calibration_data": {
                    "px_per_us_echo": px_per_us_echo,
                    "px_per_db_echo": px_per_db_echo,
                    "ref_row": ref_row,
                    "tx_pulse_col": tx_pulse_col,
                },
            }

        except Exception as e:
            print(f"ERROR: Frame processing for override failed: {e}")
            import traceback

            traceback.print_exc()
            return None

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

    def _show_montage_and_handle_user_decisions(self):
        """Create montage from deferred per-frame images, show to user, and handle
        approval, manual repicks, or config edits using the new interactive montage viewer.
        Shows all frames processed, regardless of pick status.
        """
        try:
            # Show all frames (no filtering)
            imgs = []
            frame_mapping = []  # Maps montage index to original frame index
            for i, frame_result in enumerate(self.frame_results):
                if i < len(self._deferred_frame_images):
                    imgs.append(self._deferred_frame_images[i])
                    frame_mapping.append(i + 1)  # 1-based frame number
            
            n = len(imgs)
            if n == 0:
                print("No frames available for montage.")
                print(f"Total frames processed: {len(self.frame_results)}")
                return

            total_processed = len(self.frame_results)
            print(f"\n{'='*60}")
            print(f"INTERACTIVE MONTAGE REVIEW: {n} frames with picks detected")
            print(f"{'='*60}")

            # Use the new interactive montage viewer
            if HAVE_QT:
                try:
                    from functions.interactive_montage_viewer import show_interactive_montage

                    montage_dpi = self.output_config.get("montage_dpi", self.output_config.get("plot_dpi", 150))
                    choice, selected_frames = show_interactive_montage(imgs, n, montage_dpi)

                    if choice == "save":
                        print("User approved - saving all frames...")
                        self._save_deferred_images_to_disk()
                        return

                    elif choice == "repick":
                        # User selected specific frames to repick
                        # selected_frames are 1-based frame numbers from the montage dialog, convert to 0-based indices
                        if selected_frames:
                            # Convert 1-based frame numbers to 0-based indices, then map to original frame numbers
                            original_frame_nums = [frame_mapping[idx-1] for idx in selected_frames if 1 <= idx <= len(frame_mapping)]
                            print(f"User selected frames for repicking: {original_frame_nums}")
                            selected_frames = original_frame_nums
                        else:
                            print("No frames selected for repicking")
                            return

                    elif choice == "edit":
                        # User wants to edit config
                        print("User selected 'Edit config and re-run'")
                        choice = '2'

                    else:
                        # Dialog cancelled
                        print("Montage review cancelled")
                        return

                except ImportError as e:
                    print(f"Could not import interactive montage viewer: {e}")
                    print("Falling back to matplotlib...")
                    choice = None
                except Exception as e:
                    print(f"Interactive montage viewer failed: {e}")
                    import traceback
                    traceback.print_exc()
                    choice = None
            else:
                choice = None

            if choice is None:
                # Fallback to matplotlib + console
                try:
                    import matplotlib.pyplot as plt
                    axes_cols = min(6, n)
                    axes_rows = int(np.ceil(n / axes_cols))
                    fig, axes = plt.subplots(axes_rows, axes_cols, figsize=(4 * axes_cols, 3 * axes_rows))
                    axes_flat = axes.flatten() if hasattr(axes, 'flatten') else [axes]
                    for i in range(axes_rows * axes_cols):
                        ax = axes_flat[i]
                        ax.axis('off')
                        if i < n:
                            ax.imshow(imgs[i])
                            ax.set_title(f"Frame {frame_mapping[i]}")
                    plt.suptitle('Per-frame picks montage (only frames with valid picks)')
                    plt.tight_layout()
                    plt.show()
                except Exception:
                    pass

                resp = input("Are you satisfied with the per-frame picks? (Y/N): ").strip().upper()
                if resp == 'Y':
                    self._save_deferred_images_to_disk()
                    return

                print("Options:\n1) Manual repick frames\n2) Edit config and re-run per-frame processing (skip frame detection)")
                choice = input("Choose option 1 or 2: ").strip()
                selected_frames = []

            if choice == '1' or (isinstance(choice, str) and choice == 'repick'):
                # Get frames to repick
                if isinstance(choice, str) and choice == 'repick' and selected_frames:
                    frame_nums = selected_frames
                else:
                    frames_input = input("Enter comma-separated frame numbers to repick (1-based): ")
                    try:
                        frame_nums = [int(x.strip()) for x in frames_input.split(',') if x.strip()]
                    except Exception:
                        print("Invalid input. Aborting repick flow.")
                        return

                for fnum in frame_nums:
                    if fnum < 1 or fnum > len(self.frames):
                        print(f"Frame {fnum} out of range, skipping")
                        continue
                    left, right = self.frames[fnum - 1]
                    frame_data = self._process_frame_for_override(self._last_masked_image, left, right, self.base_filename, fnum)
                    if frame_data is None:
                        print(f"Could not prepare frame {fnum} for override")
                        continue
                    
                    # Retrieve existing noise floor pick if available
                    existing_noise_floor_idx = None
                    if fnum - 1 < len(self.frame_results):
                        frame_result = self.frame_results[fnum - 1]
                        # Try to find the noise floor index by matching time value
                        if not np.isnan(frame_result.get("Noise_Floor_Time_us")):
                            noise_floor_time = frame_result["Noise_Floor_Time_us"]
                            # Find closest index in time_vals
                            distances = np.abs(frame_data["time_vals"] - noise_floor_time)
                            existing_noise_floor_idx = np.argmin(distances)
                            print(f"DEBUG: Retrieved existing noise floor pick: {noise_floor_time:.2f} μs at index {existing_noise_floor_idx}")
                            print(f"DEBUG: Closest match in current frame_data: {frame_data['time_vals'][existing_noise_floor_idx]:.2f} μs")
                    
                    from functions.interactive_override import ManualPickOverride

                    override_session = ManualPickOverride(
                        frame_data["frame_img"],
                        frame_data["signal_x_clean"],
                        frame_data["signal_y_clean"],
                        frame_data["power_vals"],
                        frame_data["time_vals"],
                        frame_data["tx_idx"],
                        frame_data["surface_idx"],
                        frame_data["bed_idx"],
                        self.base_filename,
                        fnum,
                        self.config,
                        noise_floor_idx=existing_noise_floor_idx,
                    )
                    manual_tx, manual_surf, manual_bed, manual_noise_floor, overrides = override_session.start_interactive_session()
                    success = self._save_frame_with_manual_picks(
                        frame_data, manual_tx, manual_surf, manual_bed, manual_noise_floor, overrides, self.base_filename, fnum
                    )
                    if success:
                        # regenerate deferred image for this frame
                        try:
                            self._plot_combined_results(
                                frame_data["frame_img"],
                                frame_data["signal_x_clean"],
                                frame_data["signal_y_clean"],
                                [], [], [], [],
                                frame_data["calibration_data"]["ref_row"],
                                self.base_filename,
                                fnum,
                                frame_data["calibration_data"]["tx_pulse_col"],
                                manual_tx,
                                manual_surf,
                                manual_bed,
                                frame_data["power_vals"],
                                frame_data["time_vals"],
                                frame_data["calibration_data"].get("px_per_us_echo"),
                                frame_data["calibration_data"].get("px_per_db_echo"),
                                replace_index=fnum-1,
                                noise_floor_idx_in_clean=manual_noise_floor,
                            )
                        except Exception as e:
                            print(f"Warning: could not regenerate deferred image for frame {fnum}: {e}")

                # Reload the repicked frames from disk into the deferred images cache
                print("Reloading updated frame images from disk...")
                for fnum in frame_nums:
                    if 1 <= fnum <= len(self.frames):
                        frame_idx = fnum - 1
                        plot_filename = f"{self.output_dir}/{self.base_filename}_frame{fnum:02d}_picked.png"
                        if os.path.exists(plot_filename):
                            try:
                                img = Image.open(plot_filename)
                                if frame_idx < len(self._deferred_frame_images):
                                    self._deferred_frame_images[frame_idx] = img
                                    print(f"  Reloaded frame {fnum} image from disk")
                            except Exception as e:
                                print(f"  Warning: could not reload image for frame {fnum}: {e}")

                # After repicks, show montage again
                self._show_montage_and_handle_user_decisions()

            elif choice == '2':
                print(f"Opening config for editing: {self.config_path_resolved}")
                try:
                    os.startfile(self.config_path_resolved)
                except Exception:
                    print("Could not open config automatically. Please open the file manually:", self.config_path_resolved)
                input("Edit the config as needed, save, then press Enter to continue...")
                try:
                    with open(self.config_path_resolved, 'r') as f:
                        self.config = json.load(f)
                except Exception as e:
                    print(f"Warning: could not reload config: {e}")
                # Update dependent sections
                self.processing_params = self.config.get("processing_params", {})
                self.physical_params = self.config.get("physical_params", {})
                self.output_config = self.config.get("output", {})
                self.output_dir = ensure_output_dirs(self.config)

                # Re-run per-frame processing (skip frame detection)
                print("Re-running per-frame processing with updated config...")
                self.frame_results = []
                self._deferred_frame_images = []
                for idx, (left, right) in enumerate(self.frames):
                    print(f"Re-processing frame {idx+1}/{len(self.frames)}...")
                    self._process_frame(self._last_masked_image, left, right, self.base_filename, idx + 1, total_frames=len(self.frames))

                # Show montage again
                self._show_montage_and_handle_user_decisions()

            else:
                print("Invalid choice. Aborting interactive adjustments.")

        except Exception as e:
            print(f"Error during montage/approval flow: {e}")

    def _save_deferred_images_to_disk(self):
        """Write in-memory deferred images to disk as per-frame picked PNGs.
        Saves all frames with images available (not filtered by pick validity).
        """
        try:
            from PIL import Image
            saved_count = 0
            for i, frame_result in enumerate(self.frame_results):
                if i >= len(self._deferred_frame_images):
                    continue  # Skip if no image available
                
                img = self._deferred_frame_images[i]
                fname = Path(self.output_dir) / f"{self.base_filename}_frame{i+1:02d}_picked.png"
                if img.dtype == np.uint8:
                    out_img = Image.fromarray(img)
                else:
                    out_img = Image.fromarray((img).astype(np.uint8))
                out_img.save(str(fname))
                print(f"Saved deferred plot: {fname}")
                saved_count += 1
            
            print(f"\nSaved {saved_count} frames with images to disk.")
        except Exception as e:
            print(f"Warning: could not save deferred images to disk: {e}")

    # --- PyQt5 helper dialogs ---
    def _qt_init_app(self):
        if not HAVE_QT:
            return None
        if self._qt_app is None:
            self._qt_app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        return self._qt_app

    def _qt_show_image_yesno(self, image_path, window_title, prompt_text):
        """Show an image in a modal PyQt5 dialog with Yes/No buttons. Returns True for Yes."""
        app = self._qt_init_app()
        if app is None:
            raise RuntimeError("PyQt5 not available")

        dialog = QtWidgets.QDialog()
        dialog.setWindowTitle(window_title)
        dialog.setModal(True)
        dialog.setGeometry(100, 100, 1200, 600)  # Set reasonable size
        
        vbox = QtWidgets.QVBoxLayout()

        label = QtWidgets.QLabel(prompt_text)
        vbox.addWidget(label)

        pix = QtGui.QPixmap(image_path)
        img_label = QtWidgets.QLabel()
        # Scale image to fit in dialog
        scaled_pix = pix.scaledToWidth(1100, QtCore.Qt.SmoothTransformation)
        img_label.setPixmap(scaled_pix)
        vbox.addWidget(img_label, 1)  # Give it stretch factor

        hbox = QtWidgets.QHBoxLayout()
        btn_yes = QtWidgets.QPushButton("Yes")
        btn_no = QtWidgets.QPushButton("No")
        hbox.addWidget(btn_yes)
        hbox.addWidget(btn_no)
        vbox.addLayout(hbox)

        dialog.setLayout(vbox)

        result = {'ok': None}

        def on_yes():
            result['ok'] = True
            dialog.accept()

        def on_no():
            result['ok'] = False
            dialog.accept()

        btn_yes.clicked.connect(on_yes)
        btn_no.clicked.connect(on_no)

        # Show dialog and process events
        dialog.show()
        dialog.raise_()
        dialog.activateWindow()
        print(f"DEBUG: Dialog shown, calling exec_()...")
        exit_code = dialog.exec_()
        print(f"DEBUG: Dialog exec_() returned with exit_code={exit_code}, result['ok']={result['ok']}")
        return bool(result['ok'])

    def _qt_show_montage_dialog(self, pil_image, title="Montage"):
        """Show a PIL.Image montage and return action: 'yes', 'repick', 'edit', or None on failure."""
        app = self._qt_init_app()
        if app is None:
            raise RuntimeError("PyQt5 not available")
        # Convert PIL image to QPixmap
        data = pil_image.tobytes("raw", "RGB")
        qimg = QtGui.QImage(data, pil_image.width, pil_image.height, QtGui.QImage.Format_RGB888)
        pix = QtGui.QPixmap.fromImage(qimg)

        # Create a custom dialog class that rescales the displayed pixmap from the
        # original high-resolution pixmap on resize events using SmoothTransformation.
        class MontageDialog(QtWidgets.QDialog):
            def __init__(self, pixmap, title):
                super(MontageDialog, self).__init__()
                self.setWindowTitle(title)
                self._orig_pix = pixmap
                self._user_scale = 1.0

                self.vbox = QtWidgets.QVBoxLayout()

                self.scroll = QtWidgets.QScrollArea()
                self.img_label = QtWidgets.QLabel()
                self.img_label.setAlignment(QtCore.Qt.AlignCenter)
                # Do not allow QLabel to auto-scale contents; we'll set pixmap manually
                self.img_label.setScaledContents(False)
                self.scroll.setWidget(self.img_label)
                self.scroll.setWidgetResizable(True)
                self.vbox.addWidget(self.scroll)

                # Controls
                ctrl_hbox = QtWidgets.QHBoxLayout()
                self.btn_zoom_in = QtWidgets.QPushButton("Zoom +")
                self.btn_zoom_out = QtWidgets.QPushButton("Zoom -")
                self.btn_reset = QtWidgets.QPushButton("Reset")
                ctrl_hbox.addWidget(self.btn_zoom_in)
                ctrl_hbox.addWidget(self.btn_zoom_out)
                ctrl_hbox.addWidget(self.btn_reset)
                self.vbox.addLayout(ctrl_hbox)

                lbl = QtWidgets.QLabel("Are you satisfied with the per-frame picks?")
                self.vbox.addWidget(lbl)

                hbox = QtWidgets.QHBoxLayout()
                self.btn_yes = QtWidgets.QPushButton("Yes - save all")
                self.btn_repick = QtWidgets.QPushButton("Manual repick frames")
                self.btn_edit = QtWidgets.QPushButton("Edit config and re-run")
                hbox.addWidget(self.btn_yes)
                hbox.addWidget(self.btn_repick)
                hbox.addWidget(self.btn_edit)
                self.vbox.addLayout(hbox)

                self.setLayout(self.vbox)

                # Connections
                self.btn_zoom_in.clicked.connect(self.on_zoom_in)
                self.btn_zoom_out.clicked.connect(self.on_zoom_out)
                self.btn_reset.clicked.connect(self.on_reset)

                self.btn_yes.clicked.connect(self.on_yes)
                self.btn_repick.clicked.connect(self.on_repick)
                self.btn_edit.clicked.connect(self.on_edit)

                self._choice = None

                # Initial paint
                self._update_display()

            def _update_display(self):
                # Compute available viewport size
                avail_size = self.scroll.viewport().size()
                if self._orig_pix.isNull():
                    return
                orig_w = self._orig_pix.width()
                orig_h = self._orig_pix.height()

                # Fit-to-window scale
                fit_scale_w = avail_size.width() / orig_w if orig_w > 0 else 1.0
                fit_scale_h = avail_size.height() / orig_h if orig_h > 0 else 1.0
                fit_scale = min(fit_scale_w, fit_scale_h)
                # Use base scale equal to fit_scale (so image fills available area), allow user scaling on top
                target_scale = max(0.01, fit_scale * self._user_scale)
                new_w = max(1, int(orig_w * target_scale))
                new_h = max(1, int(orig_h * target_scale))
                scaled = self._orig_pix.scaled(new_w, new_h, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
                self.img_label.setPixmap(scaled)

            def resizeEvent(self, ev):
                super(MontageDialog, self).resizeEvent(ev)
                self._update_display()

            def on_zoom_in(self):
                self._user_scale *= 1.25
                self._update_display()

            def on_zoom_out(self):
                self._user_scale /= 1.25
                self._update_display()

            def on_reset(self):
                self._user_scale = 1.0
                self._update_display()

            def on_yes(self):
                self._choice = 'yes'
                self.accept()

            def on_repick(self):
                self._choice = 'repick'
                self.accept()

            def on_edit(self):
                self._choice = 'edit'
                self.accept()

        dlg = MontageDialog(pix, title)
        dlg.exec_()
        return dlg._choice

    def _save_frame_with_manual_picks(
        self,
        frame_data,
        manual_tx,
        manual_surface,
        manual_bed,
        manual_noise_floor,
        overrides,
        base_filename,
        frame_idx,
    ):
        """
        Save frame results with manual pick overrides and update main database files.
        Also updates self.frame_results so final export has manual picks.
        """
        try:
            # Get CBD
            if (
                hasattr(self, "cbd_list")
                and self.cbd_list
                and frame_idx <= len(self.cbd_list)
            ):
                correct_cbd = self.cbd_list[frame_idx - 1]
            else:
                # Fallback: extract from filename if cbd_list not available
                import re

                match = re.search(r"C(\d+)_(\d+)", base_filename)
                if match:
                    cbd_start = int(match.group(1))
                    correct_cbd = cbd_start + frame_idx - 1
                else:
                    correct_cbd = np.nan

            # Create frame result with manual picks
            frame_result = {
                "Frame": frame_idx,
                "CBD": correct_cbd,
                "Surface_Time_us": np.nan,
                "Bed_Time_us": np.nan,
                "Ice_Thickness_m": np.nan,
                "Surface_Power_dB": np.nan,
                "Bed_Power_dB": np.nan,
                "Transmitter_Time_us": np.nan,
                "Transmitter_Power_dB": np.nan,
                "Transmitter_X_pixel": np.nan,
                "Noise_Floor_Time_us": np.nan,
                "Noise_Floor_Power_dB": np.nan,
            }

            # Extract data
            power_vals = frame_data["power_vals"]
            time_vals = frame_data["time_vals"]

            # Store results using manual picks
            if manual_surface is not None and manual_surface < len(time_vals):
                frame_result["Surface_Time_us"] = time_vals[manual_surface]
                frame_result["Surface_Power_dB"] = power_vals[manual_surface]

            if manual_bed is not None and manual_bed < len(time_vals):
                frame_result["Bed_Time_us"] = time_vals[manual_bed]
                frame_result["Bed_Power_dB"] = power_vals[manual_bed]

            if manual_tx is not None and manual_tx < len(time_vals):
                frame_result["Transmitter_Time_us"] = time_vals[manual_tx]
                frame_result["Transmitter_Power_dB"] = power_vals[manual_tx]
                frame_result["Transmitter_X_pixel"] = frame_data["calibration_data"][
                    "tx_pulse_col"
                ]

            # Store noise floor if manually picked, otherwise recalculate automatically
            if manual_noise_floor is not None and manual_noise_floor < len(time_vals):
                frame_result["Noise_Floor_Time_us"] = time_vals[manual_noise_floor]
                frame_result["Noise_Floor_Power_dB"] = power_vals[manual_noise_floor]
                print(f"DEBUG: Saving manual noise floor at index {manual_noise_floor}: {time_vals[manual_noise_floor]:.2f} μs, {power_vals[manual_noise_floor]:.1f} dB")
            else:
                # Recalculate noise floor automatically if not manually repicked
                processing_params = self.config.get("processing_params", {})
                noise_floor_start_time = processing_params.get("noise_floor_window_start_us", 5.0)
                noise_floor_end_time = processing_params.get("noise_floor_window_end_us", 6.2)
                
                noise_floor_mask = (time_vals >= noise_floor_start_time) & (time_vals <= noise_floor_end_time)
                if np.any(noise_floor_mask):
                    noise_floor_region = power_vals[noise_floor_mask]
                    noise_floor_power = np.min(noise_floor_region)
                    min_idx_in_window = np.argmin(noise_floor_region)
                    noise_floor_time = time_vals[noise_floor_mask][min_idx_in_window]
                    frame_result["Noise_Floor_Time_us"] = noise_floor_time
                    frame_result["Noise_Floor_Power_dB"] = noise_floor_power

            # Calculate ice thickness
            if not np.isnan(frame_result["Surface_Time_us"]) and not np.isnan(
                frame_result["Bed_Time_us"]
            ):
                ice_thickness = self._calculate_ice_thickness(
                    frame_result["Surface_Time_us"], frame_result["Bed_Time_us"]
                )
                frame_result["Ice_Thickness_m"] = ice_thickness

            print(f"INFO: Updated frame {frame_idx} with manual picks:")
            print(f"  CBD: {frame_result['CBD']}")
            if manual_tx is not None and manual_tx < len(time_vals):
                print(
                    f"  Transmitter: {time_vals[manual_tx]:.2f} μs {'(Manual)' if overrides.get('transmitter') else '(Auto)'}"
                )
            if not np.isnan(frame_result["Surface_Time_us"]):
                print(
                    f"  Surface: {frame_result['Surface_Time_us']:.2f} μs {'(Manual)' if overrides.get('surface') else '(Auto)'}"
                )
            if not np.isnan(frame_result["Bed_Time_us"]):
                print(
                    f"  Bed: {frame_result['Bed_Time_us']:.2f} μs {'(Manual)' if overrides.get('bed') else '(Auto)'}"
                )
            if not np.isnan(frame_result["Noise_Floor_Time_us"]):
                print(
                    f"  Noise Floor: {frame_result['Noise_Floor_Time_us']:.2f} μs {'(Manual)' if overrides.get('noise_floor') else '(Auto)'}"
                )
            if not np.isnan(frame_result["Ice_Thickness_m"]):
                print(f"  Ice thickness: {frame_result['Ice_Thickness_m']:.1f} m")

            # UPDATE self.frame_results with manual picks so final export includes them
            if frame_idx - 1 < len(self.frame_results):
                print(f"DEBUG: Updating self.frame_results[{frame_idx-1}] with manual picks")
                self.frame_results[frame_idx - 1].update(frame_result)
            else:
                print(f"WARNING: Frame {frame_idx} not in self.frame_results, appending new entry")
                self.frame_results.append(frame_result)

            # Update the main CSV file
            main_csv_path = Path(self.output_dir) / f"{base_filename}_pick.csv"
            main_npz_path = Path(self.output_dir) / f"{base_filename}_pick.npz"

            if main_csv_path.exists():
                # Load existing CSV data
                existing_df = pd.read_csv(main_csv_path)

                # Update only the specific columns, preserve CBD
                frame_mask = existing_df["Frame"] == frame_idx
                if frame_mask.any():
                    # Update only the data columns that were manually changed
                    update_columns = [
                        "Surface_Time_us",
                        "Bed_Time_us",
                        "Ice_Thickness_m",
                        "Surface_Power_dB",
                        "Bed_Power_dB",
                        "Transmitter_Time_us",
                        "Transmitter_Power_dB",
                        "Transmitter_X_pixel",
                    ]

                    for col in update_columns:
                        if col in frame_result:
                            existing_df.loc[frame_mask, col] = frame_result[col]

                    # Ensure CBD is preserved/corrected
                    existing_df.loc[frame_mask, "CBD"] = correct_cbd

                    print(f"INFO: Updated frame {frame_idx} in existing CSV database")
                else:
                    # Append new row if frame doesn't exist
                    new_row_df = pd.DataFrame([frame_result])
                    existing_df = pd.concat(
                        [existing_df, new_row_df], ignore_index=True
                    )
                    existing_df = existing_df.sort_values("Frame").reset_index(
                        drop=True
                    )
                    print(f"INFO: Added frame {frame_idx} to CSV database")

                # Save updated CSV
                existing_df.to_csv(
                    main_csv_path, index=False, float_format="%.6f", na_rep="NaN"
                )
                print(f"INFO: Updated main CSV database: {main_csv_path}")

                # Update the main NPZ file
                if main_npz_path.exists():
                    # Load existing NPZ metadata
                    try:
                        npz_data = np.load(main_npz_path, allow_pickle=True)
                        meta_info = (
                            npz_data["meta"].item() if "meta" in npz_data else {}
                        )
                    except:
                        meta_info = {}

                    # Update metadata
                    meta_info["last_manual_override"] = str(datetime.datetime.now())
                    meta_info["manual_override_frame"] = frame_idx
                    meta_info["override_types"] = [k for k, v in overrides.items() if v]
                else:
                    meta_info = {
                        "ascope_file": base_filename,
                        "processing_timestamp": str(datetime.datetime.now()),
                        "ice_velocity_m_per_us": 168.0,
                        "manual_override_frame": frame_idx,
                        "override_types": [k for k, v in overrides.items() if v],
                    }

                # Save updated NPZ with corrected data from CSV
                np.savez(
                    main_npz_path,
                    frame=existing_df["Frame"].values,
                    cbd=existing_df["CBD"].values,
                    lat=existing_df["LAT"].values if "LAT" in existing_df.columns else np.full(len(existing_df), np.nan),
                    lon=existing_df["LON"].values if "LON" in existing_df.columns else np.full(len(existing_df), np.nan),
                    surface_time_us=existing_df["Surface_Time_us"].values,
                    bed_time_us=existing_df["Bed_Time_us"].values,
                    ice_thickness_m=existing_df["Ice_Thickness_m"].values,
                    surface_power_db=existing_df["Surface_Power_dB"].values,
                    bed_power_db=existing_df["Bed_Power_dB"].values,
                    meta=meta_info,
                )
                print(f"INFO: Updated main NPZ database: {main_npz_path}")

            else:
                print("WARNING: Main CSV file not found, creating new database")
                # Create new database with single frame
                df = pd.DataFrame([frame_result])
                df.to_csv(main_csv_path, index=False, float_format="%.6f", na_rep="NaN")

                meta_info = {
                    "ascope_file": base_filename,
                    "processing_timestamp": str(datetime.datetime.now()),
                    "ice_velocity_m_per_us": 168.0,
                    "manual_override_frame": frame_idx,
                    "override_types": [k for k, v in overrides.items() if v],
                }

                np.savez(
                    main_npz_path,
                    frame=df["Frame"].values,
                    cbd=df["CBD"].values,
                    lat=df["LAT"].values if "LAT" in df.columns else np.full(len(df), np.nan),
                    lon=df["LON"].values if "LON" in df.columns else np.full(len(df), np.nan),
                    surface_time_us=df["Surface_Time_us"].values,
                    bed_time_us=df["Bed_Time_us"].values,
                    ice_thickness_m=df["Ice_Thickness_m"].values,
                    surface_power_db=df["Surface_Power_dB"].values,
                    bed_power_db=df["Bed_Power_dB"].values,
                    meta=meta_info,
                )

            # Reconstruct grid data and replace the existing picked.png file
            try:
                from functions.grid_detection import (
                    detect_grid_lines_and_dotted,
                    interpolate_regular_grid,
                )

                frame_img = frame_data["frame_img"]
                h, w = frame_img.shape
                ref_row = frame_data["calibration_data"]["ref_row"]
                tx_pulse_col = frame_data["calibration_data"]["tx_pulse_col"]

                # Re-detect grid lines for proper plotting
                h_peaks_initial, v_peaks_initial, h_minor_peaks, v_minor_peaks = (
                    detect_grid_lines_and_dotted(frame_img, self.config)
                )

                # Reconstruct grid data
                y_range_dB = 8.25 * 10
                y_major, y_minor = interpolate_regular_grid(
                    h,
                    h_peaks_initial,
                    ref_row,
                    self.physical_params["y_major_dB"],
                    self.physical_params["y_minor_per_major"],
                    y_range_dB,
                    is_y_axis=True,
                    config=self.config,
                )

                x_range_us = self.physical_params.get(
                    "x_range_factor"
                ) * self.physical_params.get("x_major_us")
                x_major, x_minor = interpolate_regular_grid(
                    w,
                    v_peaks_initial,
                    tx_pulse_col,
                    self.physical_params["x_major_us"],
                    self.physical_params["x_minor_per_major"],
                    x_range_us,
                    is_y_axis=False,
                    config=self.config,
                )

            except Exception as e:
                print(f"WARNING: Could not reconstruct grid data: {e}")
                # Use empty grid data as fallback
                y_major, y_minor, x_major, x_minor = [], [], [], []

            # Replace the existing picked.png file
            self._plot_combined_results(
                frame_data["frame_img"],
                frame_data["signal_x_clean"],
                frame_data["signal_y_clean"],
                x_major,  # Reconstructed grid data
                y_major,  # Reconstructed grid data
                x_minor,  # Reconstructed grid data
                y_minor,  # Reconstructed grid data
                frame_data["calibration_data"]["ref_row"],
                base_filename,
                frame_idx,
                frame_data["calibration_data"]["tx_pulse_col"],
                manual_tx,
                manual_surface,
                manual_bed,
                power_vals,
                time_vals,
                frame_data["calibration_data"]["px_per_us_echo"],
                frame_data["calibration_data"]["px_per_db_echo"],
            )

            print(
                f"INFO: Replaced frame plot: {base_filename}_frame{frame_idx:02d}_picked.png"
            )
            return True

        except Exception as e:
            print(f"ERROR: Failed to save manual picks: {e}")
            import traceback

            traceback.print_exc()
            return False

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
        replace_index=None,
        noise_floor_idx_in_clean=None):
        """Generate and save a combined plot with debug view and calibrated view."""
        h, w = frame_img.shape

        # RENAMED: Use "picked.png" instead of "combined_annotated.png"
        plot_filename = (
            f"{self.output_dir}/{base_filename}_frame{frame_idx:02d}_picked.png"
        )

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

            # Plot noise floor on debug view
            if time_vals is not None and len(time_vals) > 0:
                # Use manual pick if provided, otherwise calculate automatically
                if noise_floor_idx_in_clean is not None and noise_floor_idx_in_clean < len(signal_x_clean):
                    ax_debug.plot(
                        signal_x_clean[noise_floor_idx_in_clean],
                        signal_y_clean[noise_floor_idx_in_clean],
                        "o",
                        color="orange",
                        ms=6,
                        label="NF",
                    )
                else:
                    # Calculate automatically from time window
                    processing_params = self.config.get("processing_params", {})
                    noise_floor_start_time = processing_params.get("noise_floor_window_start_us", 5.0)
                    noise_floor_end_time = processing_params.get("noise_floor_window_end_us", 6.2)
                    
                    noise_floor_mask = (time_vals >= noise_floor_start_time) & (time_vals <= noise_floor_end_time)
                    if np.any(noise_floor_mask):
                        noise_floor_region = power_vals[noise_floor_mask]
                        min_idx_in_window = np.argmin(noise_floor_region)
                        # Find the actual index in the full time_vals array
                        noise_floor_indices = np.where(noise_floor_mask)[0]
                        noise_floor_idx_in_clean_auto = noise_floor_indices[min_idx_in_window]
                        
                        if noise_floor_idx_in_clean_auto < len(signal_x_clean):
                            ax_debug.plot(
                                signal_x_clean[noise_floor_idx_in_clean_auto],
                                signal_y_clean[noise_floor_idx_in_clean_auto],
                                "o",
                                color="orange",
                                ms=6,
                                label="NF",
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
        plot_max_db = 2.5

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
            # Calculate noise floor mask upfront (used for both manual and automatic)
            processing_params = self.config.get("processing_params", {})
            noise_floor_start_time = processing_params.get("noise_floor_window_start_us", 5.0)
            noise_floor_end_time = processing_params.get("noise_floor_window_end_us", 6.2)
            noise_floor_mask = (time_vals >= noise_floor_start_time) & (time_vals <= noise_floor_end_time)
            
            # Use manual noise floor pick if provided, otherwise calculate automatically
            noise_floor_power = None
            noise_floor_time = None
            
            if noise_floor_idx_in_clean is not None and noise_floor_idx_in_clean < len(time_vals):
                noise_floor_time = time_vals[noise_floor_idx_in_clean]
                noise_floor_power = power_vals[noise_floor_idx_in_clean]
            else:
                # Calculate noise floor automatically from config time window
                if np.any(noise_floor_mask):
                    noise_floor_region = power_vals[noise_floor_mask]
                    noise_floor_power = np.min(noise_floor_region)
                    min_idx_in_window = np.argmin(noise_floor_region)
                    noise_floor_time = time_vals[noise_floor_mask][min_idx_in_window]
            
            # Plot the noise floor marker if we have it
            if noise_floor_time is not None and noise_floor_power is not None:
                ax_calib.plot(
                    noise_floor_time,
                    noise_floor_power,
                    "o",
                    color="orange",
                    label="Noise Floor",
                    markersize=6,
                )
            
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
            # Annotate Tx point
            if tx_idx_in_clean is not None and tx_idx_in_clean < len(time_vals):
                tx_time = time_vals[tx_idx_in_clean]
                tx_power = power_vals[tx_idx_in_clean]
                tx_label = (
                    f"transmitter pulse\n(~{tx_power:.1f} dB at {tx_time:.1f} μs)"
                )

                ax_calib.annotate(
                    tx_label,
                    xy=(tx_time, tx_power),
                    xytext=(-50, -40),
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
                    xy=(surf_time, surf_power),
                    xytext=(-60, -15),
                    textcoords="offset points",
                    ha="right",
                    va="top",
                    fontsize=9,
                    bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.7),
                    arrowprops=dict(
                        arrowstyle="->",
                        connectionstyle="arc3,rad=0.3",
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
                    xy=(bed_time, bed_power),
                    xytext=(60, -15),
                    textcoords="offset points",
                    ha="left",
                    va="top",
                    fontsize=9,
                    bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.7),
                    arrowprops=dict(
                        arrowstyle="->",
                        connectionstyle="arc3,rad=-0.3",
                        color="magenta",
                    ),
                )

            # Annotate Noise Floor point
            if np.any(noise_floor_mask):
                noise_floor_label = f"noise floor\n(~{noise_floor_power:.1f} dB at {noise_floor_time:.1f} μs)"
                ax_calib.annotate(
                    noise_floor_label,
                    xy=(noise_floor_time, noise_floor_power),
                    xytext=(60, 30),
                    textcoords="offset points",
                    ha="left",
                    va="bottom",
                    fontsize=9,
                    bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.7),
                    arrowprops=dict(
                        arrowstyle="->",
                        connectionstyle="arc3,rad=0.3",
                        color="orange",
                    ),
                )
        plt.tight_layout(pad=1.5)

        # If deferring frame plots into memory, capture image instead of saving
        if self.interactive_mode and self.defer_frame_plots:
            try:
                # Render figure to a high-resolution PNG buffer using montage_dpi
                montage_dpi = self.output_config.get(
                    "montage_dpi", self.output_config.get("plot_dpi", 150)
                )
                buf = io.BytesIO()
                try:
                    fig.savefig(buf, format="png", dpi=montage_dpi, bbox_inches="tight")
                    buf.seek(0)
                    from PIL import Image as PILImage

                    pil_img = PILImage.open(buf).convert("RGB")
                    img = np.array(pil_img)
                finally:
                    try:
                        buf.close()
                    except Exception:
                        pass

                if replace_index is None:
                    self._deferred_frame_images.append(img)
                else:
                    # replace existing deferred image
                    if 0 <= replace_index < len(self._deferred_frame_images):
                        self._deferred_frame_images[replace_index] = img
                    else:
                        # pad list if necessary
                        while len(self._deferred_frame_images) < replace_index:
                            self._deferred_frame_images.append(img)
                        self._deferred_frame_images.append(img)

                print(f"Deferred combined plot for frame {frame_idx} (in-memory, dpi={montage_dpi})")
                plt.close(fig)
                return
            except Exception as e:
                print(f"Warning: could not capture deferred image: {e}")

        # Default: save to disk
        plt.savefig(plot_filename, dpi=self.output_config.get("plot_dpi", 200))
        print(f"Saved combined plot: {plot_filename}")
        plt.close(fig)
