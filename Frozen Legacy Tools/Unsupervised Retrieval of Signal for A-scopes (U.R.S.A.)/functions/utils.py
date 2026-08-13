import os
import sys
import numpy as np
import cv2
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt
import json


def load_config(config_path):
    """Load configuration from JSON file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, "r") as f:
        config = json.load(f)

    # Combine all config sections into a flat CONFIG dict for backward compatibility
    flat_config = {}
    for section in config:
        if isinstance(config[section], dict):
            for key, value in config[section].items():
                flat_config[key] = value
        else:
            flat_config[section] = config[section]

    return config, flat_config


def ensure_output_dirs(config):
    """Create output directories if they don't exist."""
    output_dir = config.get("output", {}).get("output_dir", "ascope_processed")
    os.makedirs(output_dir, exist_ok=True)

    processing_params = config.get("processing_params", {})
    if processing_params.get("ref_line_save_intermediate_qa", False):
        qa_dir = os.path.join(output_dir, "ref_line_qa")
        os.makedirs(qa_dir, exist_ok=True)

    return output_dir


def load_and_preprocess_image(file_path=None):
    """Loads and applies CLAHE enhancement to the input image."""
    if file_path is None:
        file_path = input("Enter the full path to the A-scope TIFF image: ").strip()

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    print(f"Loading image: {file_path}")
    img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        raise ValueError("Could not load image.")

    # Get base directory to find config
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = os.path.join(base_dir, "config", "default_config.json")

    # Get config for CLAHE
    _, flat_config = load_config(config_path)

    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(
        clipLimit=flat_config.get("grid_enhance_clip_limit", 2.0), tileGridSize=(8, 8)
    )
    enhanced = clahe.apply(img)
    base_filename = os.path.splitext(os.path.basename(file_path))[0]

    return enhanced, base_filename


def save_plot(fig, filename, dpi=200):
    """Save a matplotlib figure with proper settings."""
    # Get base directory to find config
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = os.path.join(base_dir, "config", "default_config.json")

    config, _ = load_config(config_path)
    output_dir = config.get("output", {}).get("output_dir", "ascope_processed")
    os.makedirs(output_dir, exist_ok=True)

    full_path = os.path.join(output_dir, filename)
    fig.savefig(full_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot: {full_path}")

    return full_path


def get_param(config, section, param_name, default_value):
    """
    Get a parameter from the config with proper fallback.

    Args:
        config (dict): The full configuration dictionary
        section (str): Section name in the config
        param_name (str): Parameter name to retrieve
        default_value: Default value if parameter is not found

    Returns:
        The parameter value or default if not found
    """
    return config.get(section, {}).get(param_name, default_value)


def debug_log(self, message):
    """Print debug messages only when debug mode is enabled."""
    if self.debug_mode:
        print(f"DEBUG: {message}")


def centerize_frames(frames, image, image_width):
    """
    No-op function: return frames as-is without expansion.
    The signal tracing will capture whatever signal exists within the original frame boundaries.
    
    Args:
        frames (list): List of frame tuples (col_start, col_end)
        image (ndarray): The image array
        image_width (int): Width of the image in pixels
    
    Returns:
        list: Original frame boundaries unchanged
    """
    return frames


def show_trace_montage(deferred_frame_images, frame_count, dpi=150):
    """
    Show detected traces in montage style with zoom/zoom-out and satisfaction check.
    Uses the interactive montage viewer from the GUI module.
    For TRACE REVIEW ONLY (Step 4) - simplified buttons (Yes/Edit Config & Re-run)
    
    Args:
        deferred_frame_images (list): List of frame plot images (numpy arrays)
        frame_count (int): Total number of frames
        dpi (int): Display DPI
    
    Returns:
        str: 'save' if user approved (Yes), 'edit' if wants to edit config, None if error
    """
    try:
        from interactive_montage_viewer import show_interactive_montage
        
        print("\n=== Showing Detected Traces Montage ===")
        print(f"Displaying {len(deferred_frame_images)} detected frame traces...")
        
        # For trace review, we show simplified buttons (Yes/Edit Config & Re-run)
        user_choice, _ = show_interactive_montage(
            deferred_frame_images,
            frame_count,
            base_dpi=dpi,
            trace_mode=True  # Simplified buttons for trace review
        )
        
        if user_choice == "save":
            print("✓ User approved traces. Proceeding to full processing...\n")
            return "save"
        elif user_choice == "calib":
            print("ℹ User requested calibrated-only plots (no picks)")
            return "calib"
        elif user_choice == "edit":
            print("⚙ User selected 'Edit Config & Re-run'. Opening config...")
            return "edit"
        else:  # quit or cancelled
            print("⚠ Trace review cancelled or closed.")
            return None

    except Exception as e:
        print(f"ERROR: Failed to show trace montage: {e}")
        import traceback
        traceback.print_exc()
        return None


def show_calibrated_montage(deferred_frame_images, frame_count, dpi=150):
    """
    Show calibrated (power vs time) plots in montage style without picks/annotations.
    Uses the interactive montage viewer in trace_mode so users can inspect plots and optionally
    edit config before proceeding.

    Returns:
        str: 'save' if user approved (proceed), 'edit' if wants to edit config, None if error
    """
    try:
        from interactive_montage_viewer import show_interactive_montage

        print("\n=== Showing Calibrated-only Montage (no picks) ===")
        print(f"Displaying {len(deferred_frame_images)} calibrated frame plots...")

        user_choice, _ = show_interactive_montage(
            deferred_frame_images, frame_count, base_dpi=dpi, trace_mode=True
        )

        if user_choice == "save":
            print("✓ User approved calibrated plots. Proceeding...\n")
            return "save"
        elif user_choice == "edit":
            print("⚙ User selected 'Edit Config & Re-run'. Opening config...")
            return "edit"
        else:
            print("⚠ Calibrated montage cancelled or closed.")
            return None

    except Exception as e:
        print(f"ERROR: Failed to show calibrated montage: {e}")
        import traceback
        traceback.print_exc()
        return None


def show_frame_verification_dialog(verif_image_path):
    """
    Show frame detection verification image with Yes/Edit Config & Re-run buttons.
    
    Args:
        verif_image_path (str): Path to the frame verification plot image
    
    Returns:
        str: 'yes' if user approved, 'edit' if wants to edit config and re-detect, None if error
    """
    try:
        from interactive_montage_viewer import show_interactive_montage
        import matplotlib.image as mpimg
        
        print("\n=== Showing Frame Detection Verification ===")
        print(f"Loading verification image: {verif_image_path}")
        
        # Load the verification image
        verif_img = mpimg.imread(str(verif_image_path))
        
        # Convert to list format expected by show_interactive_montage
        frame_images = [verif_img]
        
        # Show with buttons for Yes/Edit Config
        user_choice, _ = show_interactive_montage(
            frame_images,
            frame_count=1,
            base_dpi=150,
            trace_mode=True  # Uses Yes/Edit Config buttons
        )
        
        if user_choice == "save":
            print("✓ User approved frame detection. Proceeding...\n")
            return "yes"
        elif user_choice == "edit":
            print("⚙ User selected 'Edit Config & Re-detect'. Opening config...")
            return "edit"
        else:  # quit or cancelled
            print("⚠ Frame verification cancelled or closed.")
            return None
    
    except Exception as e:
        print(f"ERROR: Failed to show frame verification dialog: {e}")
        import traceback
        traceback.print_exc()
        return None
