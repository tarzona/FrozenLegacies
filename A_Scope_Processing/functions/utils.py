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
