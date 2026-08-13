"""
ASTRA - Amplitude Signal Tracing & Retrieval Application

README - Installation and Usage Instructions
===========================================

OVERVIEW:
ASTRA is a GUI-based application for processing radar A-scope data from TIFF files.
It allows manual picking of radar features (surface, bed, noise floor, main bang) 
and automatically calculates GPS coordinates and signal parameters.

REQUIRED PACKAGES:
- numpy: Numerical computing
- matplotlib: Plotting and GUI widgets  
- Pillow (PIL): Image processing
- scipy: Scientific computing (Gaussian filtering)
- PyQt5: GUI framework (cross-platform compatibility)
- pandas: Data manipulation and CSV output
- argparse: Command line argument parsing (built-in)
- os, glob, shutil, textwrap, sys: Standard library modules

INSTALLATION:
1. Install pip (if not already installed):
   Download get-pip.py from https://bootstrap.pypa.io/get-pip.py and run: python get-pip.py

2. Install required packages:
   pip install numpy matplotlib Pillow scipy pandas PyQt5

USAGE:

GUI Mode (Default):
1. Place ASTRA.py in the same directory as your TIFF files
2. Ensure navigation CSV file (e.g., "125.csv") is in the same directory
3. Run: python ASTRA.py
4. Follow the GUI workflow:
   - Set up X and Y axis reference lines and scales
   - Select features (Surface, Bed, Noise Floor, Main Bang) and click on A-scope
   - Click "Next A-scope" to move to next position
   - Click "Done" when finished with current TIFF file
   - Automatically loads next TIFF file until all are processed
5. Output files saved to "OUTPUT" folder

CLI Mode (Custom directories):
1. Run: python ASTRA.py tiff "C:/path/to/tiff/directory" nav "C:/path/to/navigation.csv"
2. Opens GUI with specified settings
3. Output files saved to "OUTPUT" folder in ASTRA.py directory

FILE REQUIREMENTS:
- TIFF files: Named as FXXX-CAAAA_CBBBB.tif (e.g., F125-C0999_C1013.tif)
- Navigation CSV: Contains columns CBD, LAT, LON with GPS coordinates

OUTPUT FILES:
- CSV: A-scope data with coordinates and signal parameters
- PNG: Quality control image with marked picks
- All files moved to OUTPUT folder after processing

WORKFLOW STEPS:
1. Set up Y-axis reference line (-60 dB reference)
2. Set up Y-axis scale (pick 4 points for dB scale)
3. Set up X-axis reference line (0 μs reference)  
4. Set up X-axis scale (pick 4 points for time scale)
5. For each A-scope position:
   - Select feature type (Surface/Bed/Noise Floor/Main Bang)
   - Click on the radar trace to pick the feature
   - Repeat for all required features
   - Click "Next A-scope"
6. Click "Done" to finish current TIFF and move to next file

AUTHOR: {Angelo Tarzona, dtarzona@gatech.edu}
DATE: {10/10/2025}
"""

import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PIL import Image
from scipy.ndimage import gaussian_filter
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QWidget, QPushButton, QFrame
from PyQt5.QtCore import QTimer
import pandas as pd
import os
import glob
import shutil
import textwrap
import argparse
import sys

# Disable the decompression bomb check
Image.MAX_IMAGE_PIXELS = None

# Parse command line arguments
def parse_arguments():
    parser = argparse.ArgumentParser(
        description='ASTRA - Amplitude Signal Tracing & Retrieval Application',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  python ASTRA.py --tiff "C:/path/to/tiff/directory" --nav "C:/nav/file.csv"
  python ASTRA.py "C:/path/to/tiff/directory" "C:/nav/file.csv"
  python ASTRA.py --gui  (or just python ASTRA.py for GUI mode)
        '''
    )
    
    parser.add_argument('--gui', action='store_true', 
                       help='Launch GUI mode (default if no arguments provided)')
    parser.add_argument('--tiff', dest='tiff_dir', 
                       help='Path to directory containing TIFF files to process')
    parser.add_argument('--nav', dest='nav_file', 
                       help='Navigation CSV file path')
    parser.add_argument('tiff_pos', nargs='?', 
                       help='Positional argument: Path to directory containing TIFF files')
    parser.add_argument('nav_pos', nargs='?', 
                       help='Positional argument: Navigation CSV file path')
    
    return parser.parse_args()

# Global variables
args = parse_arguments()
CLI_MODE = False
OUTPUT_DIR = None
NAV_FILE = None

# Initialize variables to store data
data = []
current_scope = 1
current_feature = None
crop_start_y = 150
crop_end_y = 1800
processed_files = set()
flash_message = None
flash_counter = 0

# Determine mode and set up file paths
# Check for arguments - support both named (--tiff --nav) and positional arguments
tiff_path = args.tiff_dir or args.tiff_pos
nav_path = args.nav_file or args.nav_pos

if len(sys.argv) == 1 or args.gui:
    # GUI mode - use current directory for TIFF files
    CLI_MODE = False
    tiff_files = []
    for pattern in ["*.tif", "*.tiff", "*.TIF", "*.TIFF"]:
        tiff_files.extend(glob.glob(pattern))
    
    # Remove duplicates and sort
    tiff_files = sorted(list(set(tiff_files)))
    if not tiff_files:
        raise FileNotFoundError("No TIFF files found in the current directory.")
    current_file_index = 0
    image_path = tiff_files[current_file_index]
elif tiff_path and nav_path:
    # CLI mode
    CLI_MODE = True
    if not os.path.exists(tiff_path):
        raise FileNotFoundError(f"TIFF directory not found: {tiff_path}")
    if not os.path.isdir(tiff_path):
        raise ValueError(f"TIFF path must be a directory: {tiff_path}")
    if not os.path.exists(nav_path):
        raise FileNotFoundError(f"Navigation file not found: {nav_path}")
    
    # OUTPUT folder will be created in the same directory as ASTRA.py
    OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "OUTPUT")
    NAV_FILE = nav_path
    
    # Find all TIFF files in the specified directory
    tiff_dir = tiff_path
    tiff_files = []
    for pattern in ["*.tif", "*.tiff", "*.TIF", "*.TIFF"]:
        tiff_files.extend(glob.glob(os.path.join(tiff_dir, pattern)))
    
    # Remove duplicates and sort
    tiff_files = sorted(list(set(tiff_files)))
    
    if not tiff_files:
        raise FileNotFoundError(f"No TIFF files found in directory: {tiff_dir}")
    
    current_file_index = 0
    image_path = tiff_files[current_file_index]
    
    # Create OUTPUT directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print(f"Found {len(tiff_files)} TIFF files to process in {tiff_dir}")
else:
    print("Error: When using CLI mode, both arguments are required: tiff and nav")
    print("Usage:")
    print("  python ASTRA.py --tiff <directory> --nav <file>")
    print("  python ASTRA.py <directory> <file>")
    print("  python ASTRA.py --gui  (for GUI mode)")
    sys.exit(1)

current_filename = os.path.splitext(os.path.basename(image_path))[0]

def load_image_and_reset():
    global img, cropped_img, width, height, zoom_height, zoom_width
    global surface_coords, bed_coords, noisefloor_coords, mainbang_coords
    global reference_line_y, reference_line_y_orig, setting_reference_line
    global y_axis_points, setting_y_axis, y_axis_lines, y_axis_labels
    global x_reference_line, x_reference_line_orig, setting_x_reference_line
    global x_axis_points, setting_x_axis, x_axis_lines, x_axis_labels
    global current_y_axis_pick, current_x_axis_pick
    global data, all_ascope_picks, current_scope, can_click, feature_selected
    global image_path, current_filename
    global current_feature
    global crop_start_y, crop_end_y

    img = Image.open(image_path)
    width, height = img.size
    crop_start_y = 150
    crop_end_y = 1800
    if crop_end_y > height:
        crop_end_y = height
    cropped_img = img.crop((0, crop_start_y, img.width, crop_end_y))
    zoom_width = 2300
    zoom_height = cropped_img.height

    current_feature = None

    # Reset all state variables
    surface_coords = None
    bed_coords = None
    noisefloor_coords = None
    mainbang_coords = None
    reference_line_y = None
    reference_line_y_orig = None
    setting_reference_line = False
    y_axis_points = []
    setting_y_axis = False
    y_axis_lines = []
    y_axis_labels = []
    x_reference_line = None
    x_reference_line_orig = None
    setting_x_reference_line = False
    x_axis_points = []
    setting_x_axis = False
    x_axis_lines = []
    x_axis_labels = []
    current_y_axis_pick = None
    current_x_axis_pick = None
    data = []
    all_ascope_picks = []
    current_scope = 1
    can_click = False
    feature_selected = False
    current_filename = os.path.splitext(os.path.basename(image_path))[0]
    try:
        x_slider_control.valmax = cropped_img.width - zoom_width
        x_slider_control.set_val(0)
        x_slider_control.ax.set_xlim(x_slider_control.valmin, x_slider_control.valmax)
    except Exception:
        pass  # If slider not yet created
    process_and_trace(0, zoom_width)

# Function to position text smartly within the plot area with wrapping
def position_text_smartly(ax, x, y, text, color, fontsize=9, fontweight='bold', max_width=25):
    """
    Position text to stay within the plot boundaries with text wrapping.
    
    Args:
        ax: matplotlib axis object
        x, y: coordinates for the point
        text: text to display
        color: text color
        fontsize: font size
        fontweight: font weight
        max_width: maximum characters per line for wrapping
    """
    # Get the current axis limits (image dimensions)
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    
    # Get image width and height from the current view
    img_width = xlim[1] - xlim[0]
    img_height = ylim[1] - ylim[0]
    
    # Wrap the text to prevent it from being too long
    wrapped_text = '\n'.join(textwrap.wrap(text, width=max_width))
    
    # Default positioning: to the right and slightly below the point
    text_x = x + 15
    text_y = y
    ha = 'left'
    va = 'center'
    
    # If point is in the right half of the image, place text to the left
    if x > img_width * 0.6:
        text_x = x - 15
        ha = 'right'
    
    # If point is near the top, place text below
    if y < img_height * 0.2:
        text_y = y + 20
        va = 'bottom'
    # If point is near the bottom, place text above
    elif y > img_height * 0.8:
        text_y = y - 20
        va = 'top'
    
    ax.text(
        text_x, text_y, wrapped_text,
        color=color, fontsize=fontsize, va=va, ha=ha, 
        fontweight=fontweight, zorder=11,
        bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8, edgecolor=color)
    )

# Function to extract flight info from TIFF filename
def extract_flight_info(filename):
    """
    Extract flight number and CBD range from TIFF filename
    Expected format: FXXX-CAAAA_CBBBB
    Returns: (flight_number, cbd_start, cbd_end)
    """
    try:
        # Remove extension
        basename = os.path.splitext(filename)[0]
        # Split by '-' to get flight and CBD parts
        parts = basename.split('-')
        if len(parts) != 2:
            return None, None, None
        
        flight_part = parts[0]  # FXXX
        cbd_part = parts[1]     # CAAAA_CBBBB
        
        # Extract flight number (remove 'F' prefix)
        if flight_part.startswith('F'):
            flight_number = flight_part[1:]
        else:
            return None, None, None
        
        # Extract CBD range (remove 'C' prefixes and split by '_')
        cbd_range = cbd_part.split('_')
        if len(cbd_range) != 2:
            return None, None, None
        
        cbd_start = cbd_range[0][1:] if cbd_range[0].startswith('C') else cbd_range[0]
        cbd_end = cbd_range[1][1:] if cbd_range[1].startswith('C') else cbd_range[1]
        
        return flight_number, int(cbd_start), int(cbd_end)
    except:
        return None, None, None

# Function to load flight coordinates
def load_flight_coordinates(flight_number):
    """
    Load LAT/LON coordinates from flight CSV file
    Returns: pandas DataFrame with CBD, LAT, LON columns
    """
    try:
        if CLI_MODE and NAV_FILE:
            # In CLI mode, use the specified navigation file
            csv_filename = NAV_FILE
        else:
            # In GUI mode, look for flight number CSV in current directory
            csv_filename = f"{flight_number}.csv"
        
        if os.path.exists(csv_filename):
            df = pd.read_csv(csv_filename)
            return df[['CBD', 'LAT', 'LON']]
        else:
            print(f"Warning: Flight CSV file {csv_filename} not found")
            return None
    except Exception as e:
        print(f"Error loading flight coordinates: {e}")
        return None

# Function to get coordinates for a specific CBD
def get_coordinates_for_cbd(flight_coords_df, cbd_value):
    """
    Get LAT/LON for a specific CBD value
    Returns: (lat, lon) or (None, None) if not found
    """
    if flight_coords_df is None:
        return None, None
    
    # Find the row with the matching CBD
    matching_rows = flight_coords_df[flight_coords_df['CBD'] == cbd_value]
    if len(matching_rows) > 0:
        row = matching_rows.iloc[0]
        return row['LAT'], row['LON']
    else:
        return None, None

# Function to save data to CSV
def save_to_csv():
    # Extract flight information from current filename
    flight_number, cbd_start, cbd_end = extract_flight_info(os.path.basename(image_path))
    
    # Load flight coordinates if available
    flight_coords_df = None
    if flight_number:
        flight_coords_df = load_flight_coordinates(flight_number)
    
    # Calculate CBD values for each A-scope
    # Distribute A-scopes evenly across the CBD range
    cbd_values = []
    lat_values = []
    lon_values = []
    
    if flight_number and cbd_start is not None and cbd_end is not None and len(data) > 0:
        for i, row in enumerate(data):
            ascope_num = row[0]  # A-scope number is the first column
            # Calculate CBD value for this A-scope (linear interpolation)
            if len(data) == 1:
                cbd_value = cbd_start
            else:
                cbd_value = int(cbd_start + (cbd_end - cbd_start) * i / (len(data) - 1))
            
            # Get coordinates for this CBD
            lat, lon = get_coordinates_for_cbd(flight_coords_df, cbd_value)
            
            cbd_values.append(cbd_value)
            lat_values.append(lat if lat is not None else "")
            lon_values.append(lon if lon is not None else "")
    else:
        # If we can't extract flight info, fill with empty values
        for _ in data:
            cbd_values.append("")
            lat_values.append("")
            lon_values.append("")
    
    df = pd.DataFrame(
        data,
        columns=[
            "A-scope Number",
            "FLT",
            "x_surface_px", "y_surface_px",
            "x_bed_px", "y_bed_px",
            "x_noisefloor_px", "y_noisefloor_px",
            "x_mainbang_px", "y_mainbang_px",
            "reference_line_y",
            "reference_line_x",
            "surface_us", "surface_dB",
            "bed_us", "bed_dB",
            "noisefloor_us", "noisefloor_dB",
            "mainbang_us", "mainbang_dB",
            "Filename" 
        ]
    )
    
    # Add the GPS coordinate columns
    df['CBD'] = cbd_values
    df['LAT'] = lat_values
    df['LON'] = lon_values
    
    # Add the meter conversion columns
    df['surface_m'] = df['surface_us'].apply(lambda x: (x/2)*168 if x != "" and x is not None else "")
    df['bed_m'] = df['bed_us'].apply(lambda x: (x/2)*168 if x != "" and x is not None else "")
    df['mainbang_m'] = df['mainbang_us'].apply(lambda x: (x/2)*168 if x != "" and x is not None else "")
    
    # Calculate ice thickness (bed_m - surface_m)
    def calculate_h_ice(row):
        if (row['surface_m'] != "" and row['surface_m'] is not None and 
            row['bed_m'] != "" and row['bed_m'] is not None):
            return row['bed_m'] - row['surface_m']
        else:
            return ""
    
    df['h_ice_m'] = df.apply(calculate_h_ice, axis=1)
    
    # Both CLI and GUI modes save CSV to current directory (will be moved to OUTPUT folder later)
    output_csv_path = f"{current_filename}.csv"
    
    df.to_csv(output_csv_path, index=False)
    print(f"Data saved to {output_csv_path}")
    if flight_number:
        print(f"Flight: {flight_number}, CBD range: {cbd_start}-{cbd_end}")
    
    # Only update GUI if not in CLI mode
    if not CLI_MODE:
        process_and_trace(int(x_slider_control.val), int(x_slider_control.val) + zoom_width)

# Function to apply Gaussian filter and trace signal
def process_and_trace(left, right):
    global surface_coords, bed_coords, noisefloor_coords, mainbang_coords
    global trace_y  # <-- Add this line
    # Zoom into the image based on the left and right bounds
    zoomed_in_img = cropped_img.crop((left, 0, right, zoom_height))
    gray_img = zoomed_in_img  # Convert the zoomed-in image to grayscale
    gray_img_array = np.array(gray_img)

    # Apply Gaussian filter to smooth the zoomed-in image and reduce noise
    smoothed_img = gaussian_filter(gray_img_array, sigma=4)

    # Set the threshold for detecting black areas (signals)
    threshold_value = np.percentile(smoothed_img, 10)  # You can adjust this threshold if needed
    binary_signal = smoothed_img < threshold_value  # Detect black (dark) areas

    # Update the plot with the smoothed, thresholded image
    ax.clear()  # Clear the previous plot
    ax.imshow(smoothed_img, cmap='gray')  # Show the smoothed zoomed-in region

    # Remove default axis ticks and labels
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    # Move the title up to avoid overlap
    title_text = f'A-scope {current_scope}'
    ax.set_title(title_text, pad=30)

    # Set axis labels
    #ax.set_xlabel('Pixels')
    #ax.set_ylabel('Pixels')

    # Trace the signal in red
    trace_y = np.argmax(binary_signal, axis=0)  # Find the Y position of the topmost black pixel for each column
    ax.plot(range(binary_signal.shape[1]), trace_y, color='red', linewidth=2)  # Plot the trace in red

    # Draw the reference line if it has been set
    if reference_line_y is not None:
        ax.axhline(reference_line_y, color='blue', linestyle='--', linewidth=2)
        # Always label reference y-axis as -60 dB (left side)
        ax.text(-40, reference_line_y, "-60 dB", color='blue', va='bottom', ha='right', fontsize=10, fontweight='bold')
        # Always label reference y-axis pixel (right side)
        ax.text(binary_signal.shape[1]+10, reference_line_y, f"{int(reference_line_y)} px", color='blue', va='bottom', ha='left', fontsize=10)

    # Draw the x-axis reference line if set
    if x_reference_line is not None:
        ax.axvline(x_reference_line, color='purple', linestyle='--', linewidth=2)
        # μs label below the plot (outside)
        ax.text(x_reference_line, binary_signal.shape[0] + 55, "0 μs", color='purple', va='bottom', ha='center', fontsize=10, fontweight='bold', clip_on=False)
        # pixel label above the plot (outside, horizontal)
        ax.text(x_reference_line, -45, f"{int(x_reference_line)} px", color='purple', va='top', ha='center', fontsize=10, rotation=0, clip_on=False)

    # Draw y-axis lines and labels if set
    if y_axis_lines:
        for y, db in zip(y_axis_lines, y_axis_labels):
            ax.axhline(y, color='green', linestyle=':', linewidth=1)
            # dB label on the left
            ax.text(-40, y, f"{db}", color='green', va='bottom', ha='right', fontsize=9, fontweight='bold')
            # pixel label on the right
            ax.text(binary_signal.shape[1]+10, y, f"{int(y)}", color='green', va='bottom', ha='left', fontsize=9)

        # Y-axis dB title (left)
        ax.text(-200, binary_signal.shape[0]//2, "Received Power (dB)", color='black', va='center', ha='center', rotation=90, fontsize=11, fontweight='bold')

    # Draw x-axis lines and labels if set
    if x_axis_lines:
        for x, us in zip(x_axis_lines, x_axis_labels):
            ax.axvline(x, color='orange', linestyle=':', linewidth=1)
            # μs label below the plot (outside)
            ax.text(x, binary_signal.shape[0] + 55, f"{us}", color='orange', rotation=0, va='bottom', ha='center', fontsize=9, fontweight='bold', clip_on=False)
            # pixel label above the plot (outside)
            ax.text(x, -50, f"{int(x)}", color='orange', rotation=0, va='top', ha='center', fontsize=9, clip_on=False)

        # X-axis μs title (bottom, outside)
        ax.text(binary_signal.shape[1] // 2, binary_signal.shape[0] + 150, "One Way Travel Time (μs)", color='black', va='bottom', ha='center', fontsize=11, fontweight='bold', clip_on=False)

        # X-axis title above the plot (centered) -- only show after X axis is set up
        ax.text(binary_signal.shape[1] // 2, -55, "X-Axis Pixels", color='black', va='bottom', ha='center', fontsize=11, fontweight='bold', clip_on=False)

        # Y-axis title on the right (centered, vertical) -- only show after X axis is set up
        ax.text(binary_signal.shape[1] + 200, binary_signal.shape[0] // 2, "Y-Axis Pixels", color='black', va='center', ha='center', rotation=90, fontsize=11, fontweight='bold', clip_on=False)

    # Draw guide line for Y axis setup
    if setting_y_axis and current_y_axis_pick is not None:
        ax.axhline(current_y_axis_pick, color='green', linestyle='--', linewidth=2, alpha=0.7)
        # No label

    # Draw guide line for X axis setup
    if setting_x_axis and current_x_axis_pick is not None:
        ax.axvline(current_x_axis_pick, color='orange', linestyle='--', linewidth=2, alpha=0.7)
        # No label

    # Draw guide lines for Y axis setup (all picked points)
    if setting_y_axis and y_axis_points:
        for y in y_axis_points:
            ax.axhline(y, color='green', linestyle='--', linewidth=2, alpha=0.7)

    # Draw guide lines for X axis setup (all picked points)
    if setting_x_axis and x_axis_points:
        for x in x_axis_points:
            ax.axvline(x, color='orange', linestyle='--', linewidth=2, alpha=0.7)

    # Show picked feature points and their labels
    if surface_coords is not None:
        us = pixel_to_us(surface_coords[0])
        db = pixel_to_db(surface_coords[1])
        ax.scatter(surface_coords[0], surface_coords[1], color='green', s=60, marker='o', edgecolors='black', zorder=10)
        position_text_smartly(
            ax, surface_coords[0], surface_coords[1],
            f"Surface (orig: {surface_coords[2]}, {surface_coords[3]} px; {us} μs, {db} dB)",
            color='green', fontsize=8
        )

    if bed_coords is not None:
        us = pixel_to_us(bed_coords[0])
        db = pixel_to_db(bed_coords[1])
        ax.scatter(bed_coords[0], bed_coords[1], color='magenta', s=60, marker='o', edgecolors='black', zorder=10)
        position_text_smartly(
            ax, bed_coords[0], bed_coords[1],
            f"Bed (orig: {bed_coords[2]}, {bed_coords[3]} px; {us} μs, {db} dB)",
            color='magenta', fontsize=8
        )

    if noisefloor_coords is not None:
        us = pixel_to_us(noisefloor_coords[0])
        db = pixel_to_db(noisefloor_coords[1])
        ax.scatter(noisefloor_coords[0], noisefloor_coords[1], color='maroon', s=60, marker='o', edgecolors='black', zorder=10)
        position_text_smartly(
            ax, noisefloor_coords[0], noisefloor_coords[1],
            f"Noise Floor (orig: {noisefloor_coords[2]}, {noisefloor_coords[3]} px; {us} μs, {db} dB)",
            color='maroon', fontsize=8
        )

    if mainbang_coords is not None:
        us = pixel_to_us(mainbang_coords[0])
        db = pixel_to_db(mainbang_coords[1])
        ax.scatter(mainbang_coords[0], mainbang_coords[1], color='blue', s=60, marker='o', edgecolors='black', zorder=10)
        position_text_smartly(
            ax, mainbang_coords[0], mainbang_coords[1],
            f"Main Bang (orig: {mainbang_coords[2]}, {mainbang_coords[3]} px; {us} μs, {db} dB)",
            color='blue', fontsize=8
        )

    # Display flash message if active
    global flash_message, flash_counter
    if flash_message and flash_counter > 0:
        # Display the flash message prominently at the top center
        ax.text(binary_signal.shape[1] // 2, -100, flash_message, 
                color='red', fontsize=16, fontweight='bold', 
                ha='center', va='center', clip_on=False,
                bbox=dict(boxstyle="round,pad=0.5", facecolor='yellow', alpha=0.9, edgecolor='red', linewidth=2))
        flash_counter -= 1
        if flash_counter <= 0:
            flash_message = None

    canvas.draw()

# Function to handle clicks for the different features (Surface, Bed, Noise Floor, Main Bang)
def on_click(event):
    global surface_coords, bed_coords, noisefloor_coords, mainbang_coords
    global current_feature, can_click, feature_selected
    global reference_line_y, reference_line_y_orig, setting_reference_line
    global x_reference_line, x_reference_line_orig, setting_x_reference_line
    global setting_y_axis, y_axis_points, y_axis_lines, y_axis_labels
    global x_reference_line, setting_x_reference_line
    global setting_x_axis, x_axis_points, x_axis_lines, x_axis_labels
    global current_y_axis_pick, current_x_axis_pick

    if event is None:
        return

    # Handle x-axis reference line setup
    if setting_x_reference_line and x_reference_line is None:
        if event.xdata is not None:
            x_reference_line = int(event.xdata)
            x_reference_line_orig = x_reference_line + int(x_slider_control.val)
            process_and_trace(int(x_slider_control.val), int(x_slider_control.val) + zoom_width)
            print(f"X Reference line set at x = {x_reference_line} (orig: {x_reference_line_orig})")
        setting_x_reference_line = False
        return

    # Handle reference line setup (Y axis reference line)
    if setting_reference_line and reference_line_y is None:
        if event.ydata is not None:
            reference_line_y = int(event.ydata)
            reference_line_y_orig = reference_line_y + crop_start_y
            process_and_trace(int(x_slider_control.val), int(x_slider_control.val) + zoom_width)
            print(f"Reference Y-axis line set at y = {reference_line_y} (orig: {reference_line_y_orig})")
        setting_reference_line = False
        return

    # Handle Y axis setup (collect 4 points)
    if setting_y_axis:
        if event.ydata is not None:
            y_axis_points.append(event.ydata)
            current_y_axis_pick = event.ydata  # Store for guide line
            process_and_trace(int(x_slider_control.val), int(x_slider_control.val) + zoom_width)
            if len(y_axis_points) == 4:
                draw_y_axis_lines()
                process_and_trace(int(x_slider_control.val), int(x_slider_control.val) + zoom_width)
                setting_y_axis = False
                current_y_axis_pick = None  # Clear after done
        return

    # Handle X axis setup (collect 4 points)
    if setting_x_axis:
        if event.xdata is not None:
            x_axis_points.append(event.xdata)
            current_x_axis_pick = event.xdata  # Store for guide line
            process_and_trace(int(x_slider_control.val), int(x_slider_control.val) + zoom_width)
            if len(x_axis_points) == 4:
                draw_x_axis_lines()
                process_and_trace(int(x_slider_control.val), int(x_slider_control.val) + zoom_width)
                setting_x_axis = False
                current_x_axis_pick = None  # Clear after done
        return

    if current_feature is None or not can_click or not feature_selected:
        return  # Don't do anything if no feature is selected or if event is None or if feature has not been selected

    x_zoom = int(event.xdata)
    # Snap y_zoom to the red trace if picking a feature
    if current_feature in ["Surface", "Bed", "Noise Floor", "Main Bang"]:
        if 0 <= x_zoom < len(trace_y):
            y_zoom = int(trace_y[x_zoom])
        else:
            y_zoom = int(event.ydata)
    else:
        y_zoom = int(event.ydata)
    x_orig = x_zoom + int(x_slider_control.val)
    y_orig = y_zoom + crop_start_y

    # Record coordinates for the selected feature
    if current_feature == "Surface":
        surface_coords = (x_zoom, y_zoom, x_orig, y_orig)
        print(f"Surface: (zoom: {x_zoom},{y_zoom}) (orig: {x_orig},{y_orig})")
        process_and_trace(int(x_slider_control.val), int(x_slider_control.val) + zoom_width)
    elif current_feature == "No Surface":
        surface_coords = None
        print("Surface set to NaN.")
        process_and_trace(int(x_slider_control.val), int(x_slider_control.val) + zoom_width)
    elif current_feature == "Bed":
        bed_coords = (x_zoom, y_zoom, x_orig, y_orig)
        print(f"Bed: (zoom: {x_zoom},{y_zoom}) (orig: {x_orig},{y_orig})")
        process_and_trace(int(x_slider_control.val), int(x_slider_control.val) + zoom_width)
    elif current_feature == "No Bed":
        bed_coords = None
        print("Bed set to NaN.")
        process_and_trace(int(x_slider_control.val), int(x_slider_control.val) + zoom_width)
    elif current_feature == "Noise Floor":
        noisefloor_coords = (x_zoom, y_zoom, x_orig, y_orig)
        print(f"Noise Floor: (zoom: {x_zoom},{y_zoom}) (orig: {x_orig},{y_orig})")
        process_and_trace(int(x_slider_control.val), int(x_slider_control.val) + zoom_width)
    elif current_feature == "Main Bang":
        mainbang_coords = (x_zoom, y_zoom, x_orig, y_orig)
        print(f"Main Bang: (zoom: {x_zoom},{y_zoom}) (orig: {x_orig},{y_orig})")
        process_and_trace(int(x_slider_control.val), int(x_slider_control.val) + zoom_width)

# Update function for the slider to change the X-axis and zoom in the correct region
def update(val):
    global can_click  # Declare can_click as global to modify the global variable
    left = int(x_slider_control.val)
    right = left + zoom_width
    process_and_trace(left, right)
    
    # Enable feature selection after adjusting the X-axis slider
    if not can_click:
        can_click = True  # Allow feature selection after slider is adjusted

# Function to move to the next A-scope and disable clicking
def next_ascope():
    global current_scope, can_click, feature_selected
    global surface_coords, bed_coords, noisefloor_coords, mainbang_coords
    global all_ascope_picks

    # Only require noisefloor and mainbang to be present
    if noisefloor_coords and mainbang_coords:
        all_ascope_picks.append({
            'surface': surface_coords,
            'bed': bed_coords,
            'noisefloor': noisefloor_coords,
            'mainbang': mainbang_coords
        })

    if current_scope > len(data):
        print(f"Completed A-scope {current_scope}")
        if noisefloor_coords and mainbang_coords:
            # Get flight number for FLT column
            flight_number, _, _ = extract_flight_info(os.path.basename(image_path))
            data.append([
                current_scope,
                flight_number if flight_number else "",  # FLT column
                surface_coords[2] if surface_coords else "", surface_coords[3] if surface_coords else "",  # x_surface_px, y_surface_px (orig)
                bed_coords[2] if bed_coords else "", bed_coords[3] if bed_coords else "",
                noisefloor_coords[2], noisefloor_coords[3],
                mainbang_coords[2], mainbang_coords[3],
                int(reference_line_y_orig) if reference_line_y_orig is not None else "",
                int(x_reference_line_orig) if x_reference_line_orig is not None else "",
                pixel_to_us(surface_coords[0]) if surface_coords else "",
                pixel_to_db(surface_coords[1]) if surface_coords else "",
                pixel_to_us(bed_coords[0]) if bed_coords else "",
                pixel_to_db(bed_coords[1]) if bed_coords else "",
                pixel_to_us(noisefloor_coords[0]),
                pixel_to_db(noisefloor_coords[1]),
                pixel_to_us(mainbang_coords[0]),
                pixel_to_db(mainbang_coords[1]),
                os.path.basename(image_path)
            ])

    current_scope += 1
    can_click = False
    feature_selected = False
    surface_coords = None
    bed_coords = None
    noisefloor_coords = None
    mainbang_coords = None

    print(f"Moving to A-scope {current_scope}")
    
    # Flash message and refresh display
    flash_next_ascope_message()
    save_to_csv()

# Function to flash "MOVE TO NEXT A-SCOPE!" message
def flash_next_ascope_message():
    global flash_message, flash_counter
    flash_message = "MOVE TO NEXT A-SCOPE!"
    flash_counter = 30  # Flash for about 30 refreshes (roughly 3 seconds)
    # Refresh the display to show the cleared picks and flash message
    process_and_trace(int(x_slider_control.val), int(x_slider_control.val) + zoom_width)
    # Start the flash countdown
    update_flash_message()

def update_flash_message():
    global flash_message, flash_counter
    if flash_counter > 0:
        # Refresh the display
        process_and_trace(int(x_slider_control.val), int(x_slider_control.val) + zoom_width)
        # Schedule the next update
        QTimer.singleShot(100, update_flash_message)  # Update every 100ms

# Function to handle setting up the reference line
def setup_reference_line():
    global setting_reference_line
    # Only allow setting if not already set
    if reference_line_y is None:
        setting_reference_line = True
        print("Pick the point for reference line")
    else:
        print("Reference line already set. Click 'Done' to reset.")

def setup_y_axis():
    global setting_y_axis, y_axis_points, y_axis_lines, y_axis_labels
    if not setting_y_axis:
        setting_y_axis = True
        y_axis_points = []
        y_axis_lines = []
        y_axis_labels = []
        print("Pick the points for y-axis (4 points needed)")
    else:
        print("Already setting up Y axis. Finish picking 4 points.")

def draw_y_axis_lines():
    global y_axis_lines, y_axis_labels
    if len(y_axis_points) != 4 or reference_line_y is None:
        return
    # Use the first clicked point as the basepoint
    base_y = y_axis_points[0]
    # Calculate average spacing from the 4 points
    sorted_points = sorted(y_axis_points)
    avg_spacing = np.mean(np.diff(sorted_points))
    # Find how many steps from base to reference line
    # steps_to_ref = round((reference_line_y - base_y) / avg_spacing)
    # The dB value at the base point
    # base_db = -55 - steps_to_ref * 10
    base_db = -55  # <-- Always start at -55 dB

    # Generate lines from basepoint upward, labeling each by +10 dB, until 0 dB or above top
    lines = []
    labels = []
    y = base_y
    db = base_db
    while y > 0 and db <= 0:
        lines.append(y)
        labels.append(db)
        y -= avg_spacing
        db += 10
    y_axis_lines = lines
    y_axis_labels = labels

def save_tiff_with_all_points():
    from PIL import ImageDraw, ImageFont

    # Open the original TIFF
    img = Image.open(image_path)
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", 18)
    except:
        font = ImageFont.load_default()

    for row in data:
        (ascope_num, flt, x_surface, y_surface, x_bed, y_bed, x_noisefloor, y_noisefloor, x_mainbang, y_mainbang, *_ ) = row
        for x, y, color, label in [
            (x_surface, y_surface, "green", "Surface"),
            (x_bed, y_bed, "magenta", "Bed"),
            (x_noisefloor, y_noisefloor, "maroon", "Noise"),
            (x_mainbang, y_mainbang, "blue", "MainBang"),
        ]:
            if x != "" and y != "" and x is not None and y is not None:
                try:
                    x_coord = int(x)
                    y_coord = int(y)  # y is already in original image coordinates!
                    r = 10
                    draw.ellipse((x_coord-r, y_coord-r, x_coord+r, y_coord+r), outline=color, width=3)
                    draw.text((x_coord+r+2, y_coord), f"{label} (A{ascope_num})", fill=color, font=font)
                except (ValueError, TypeError):
                    # Skip if coordinates cannot be converted to integers
                    continue

    # --- OUTPUT PATH LOGIC ---
    # Both CLI and GUI modes now use the same OUTPUT folder approach
    
    # GUI mode - original behavior
    if image_path.lower().endswith('.tiff'):
        out_path = image_path[:-5] + '_QC.png'
    elif image_path.lower().endswith('.tif'):
        out_path = image_path[:-4] + '_QC.png'
    else:
        out_path = image_path + '_QC.png'

    img.save(out_path)
    print(f"Saved QC PNG as {out_path}")

    # --- Move PNG, CSV, and TIFF to OUTPUT folder ---
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "OUTPUT")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Move PNG
    shutil.move(out_path, os.path.join(output_dir, os.path.basename(out_path)))
    # Move CSV
    csv_path = f"{current_filename}.csv"
    if os.path.exists(csv_path):
        shutil.move(csv_path, os.path.join(output_dir, os.path.basename(csv_path)))
    # Move TIFF
    if os.path.exists(image_path):
        shutil.move(image_path, os.path.join(output_dir, os.path.basename(image_path)))
    print(f"Moved PNG, CSV, and TIFF to {output_dir}")

def done_and_clear_reference():
    global reference_line_y, y_axis_lines, y_axis_labels, y_axis_points
    global x_reference_line
    global x_axis_lines, x_axis_labels, x_axis_points, setting_x_axis
    global current_file_index, image_path, current_filename, tiff_files
    global processed_files

    save_to_csv()
    save_tiff_with_all_points()

    # Mark current file as processed
    processed_files.add(image_path)

    # Move to the next unprocessed TIFF file
    while True:
        current_file_index += 1
        if current_file_index >= len(tiff_files):
            print("All TIFF files processed. Exiting.")
            app.quit()
            return
        image_path = tiff_files[current_file_index]
        if image_path not in processed_files:
            current_filename = os.path.splitext(os.path.basename(image_path))[0]
            print(f"Loading next TIFF: {image_path}")
            load_image_and_reset()
            break

def setup_x_reference_line():
    global setting_x_reference_line
    if x_reference_line is None:
        setting_x_reference_line = True
        print("Pick the point for X axis reference line")
    else:
        print("X axis reference line already set. Click 'Done' to reset.")

def setup_x_axis():
    global setting_x_axis, x_axis_points, x_axis_lines, x_axis_labels
    if not setting_x_axis:
        setting_x_axis = True
        x_axis_points = []
        x_axis_lines = []
        x_axis_labels = []
        print("Pick the points for x-axis (4 points needed)")
    else:
        print("Already setting up X axis. Finish picking 4 points.")

def draw_x_axis_lines():
    global x_axis_lines, x_axis_labels
    if len(x_axis_points) != 4:
        return
    base_x = x_axis_points[0]
    sorted_points = sorted(x_axis_points)
    avg_spacing = np.mean(np.diff(sorted_points))
    base_us = 3  # Start at 3 us

    lines = []
    labels = []
    x = base_x
    us = base_us
    while us <= 30:
        lines.append(x)
        labels.append(us)
        x += avg_spacing
        us += 3
    x_axis_lines = lines
    x_axis_labels = labels

def pixel_to_us(x):
    """Convert x pixel to Two Way Travel Time (μs) using x_axis_lines and x_axis_labels."""
    if not x_axis_lines or not x_axis_labels:
        return ""
    # Find the two nearest grid lines
    for i in range(1, len(x_axis_lines)):
        if x < x_axis_lines[i]:
            x0, x1 = x_axis_lines[i-1], x_axis_lines[i]
            us0, us1 = x_axis_labels[i-1], x_axis_labels[i]
            # Linear interpolation
            return round(us0 + (us1 - us0) * (x - x0) / (x1 - x0), 2)
    return x_axis_labels[-1]

def pixel_to_db(y):
    """Convert y pixel to Received Power (dB) using y_axis_lines and y_axis_labels, with extrapolation."""
    if not y_axis_lines or not y_axis_labels:
        return ""
    # If above the topmost line, extrapolate
    if y < y_axis_lines[-1]:
        # Use the last two grid lines for extrapolation
        y0, y1 = y_axis_lines[-2], y_axis_lines[-1]
        db0, db1 = y_axis_labels[-2], y_axis_labels[-1]
        # Linear extrapolation
        return round(db1 + (db1 - db0) * (y - y1) / (y1 - y0), 2)
    # If below the bottommost line, extrapolate
    if y > y_axis_lines[0]:
        y0, y1 = y_axis_lines[0], y_axis_lines[1]
        db0, db1 = y_axis_labels[0], y_axis_labels[1]
        return round(db0 + (db0 - db1) * (y - y0) / (y0 - y1), 2)
    # Otherwise, interpolate as usual
    for i in range(1, len(y_axis_lines)):
        if y > y_axis_lines[i]:
            y0, y1 = y_axis_lines[i-1], y_axis_lines[i]
            db0, db1 = y_axis_labels[i-1], y_axis_labels[i]
            return round(db0 + (db1 - db0) * (y - y0) / (y1 - y0), 2)
    return y_axis_labels[-1]

# Main execution - Both CLI and GUI modes will open the GUI
if CLI_MODE:
    print("ASTRA CLI Mode - Opening GUI with specified settings")
    print(f"TIFF directory: {tiff_path}")
    print(f"OUTPUT folder: {OUTPUT_DIR}")
    print(f"Navigation file: {NAV_FILE}")
    print(f"Found {len(tiff_files)} TIFF files to process:")
    for i, tiff_file in enumerate(tiff_files, 1):
        print(f"  {i}. {os.path.basename(tiff_file)}")
    print("Opening GUI for manual processing...")
    print("All outputs will be saved to OUTPUT folder")
    print("="*50)

# Create the PyQt5 application and main window
app = QApplication(sys.argv)
main_window = QMainWindow()
main_window.setWindowTitle("ASTRA - Amplitude Signal Tracing & Retrieval Application")

# Create central widget and main layout
central_widget = QWidget()
main_window.setCentralWidget(central_widget)
main_layout = QHBoxLayout(central_widget)

# Create a frame to hold the control elements (slider and buttons)
control_frame = QFrame()
control_layout = QVBoxLayout(control_frame)
main_layout.addWidget(control_frame)

# Create a frame to hold the plot
plot_frame = QFrame()
plot_layout = QVBoxLayout(plot_frame)
main_layout.addWidget(plot_frame)

# Create the canvas to embed the plot in the main window
fig, ax = plt.subplots(figsize=(10, 6))
canvas = FigureCanvas(fig)
plot_layout.addWidget(canvas)
canvas.draw()

load_image_and_reset()

# Initialize the plot with the initial zoomed-in region
process_and_trace(0, zoom_width)  # Display the initial zoomed-in image

# Note: Slider controls are handled by matplotlib widgets, no additional frames needed

# Create the X Position slider (Matplotlib widget)
ax_slider_control = fig.add_axes([0.1, 0.01, 0.6, 0.03], facecolor='lightgoldenrodyellow')
global x_slider_control
x_slider_control = Slider(ax_slider_control, 'X Position', 0, cropped_img.width - zoom_width, valinit=0, valstep=1)
x_slider_control.on_changed(update)  # Connect the update function to the slider

# Create Matplotlib "Jump to next A-scope" button above the left/right arrow buttons
ax_button_jump = fig.add_axes([0.80, 0.05, 0.11, 0.04])
button_jump = Button(ax_button_jump, 'Move 1000 px')

def jump_next_ascope(event):
    new_val = min(cropped_img.width - zoom_width, x_slider_control.val + 1000)
    x_slider_control.set_val(new_val)

button_jump.on_clicked(jump_next_ascope)

# Create Matplotlib buttons for left and right arrows, positioned to the right of the slider
ax_button_left = fig.add_axes([0.80, 0.01, 0.04, 0.03])
button_left = Button(ax_button_left, '←')
button_left.on_clicked(lambda event: x_slider_control.set_val(max(0, x_slider_control.val - 10)))

ax_button_right = fig.add_axes([0.87, 0.01, 0.04, 0.03])
button_right = Button(ax_button_right, '→')
button_right.on_clicked(lambda event: x_slider_control.set_val(min(cropped_img.width - zoom_width, x_slider_control.val + 10)))

# Create the buttons for "Surface", "No Surface", "Bed", "No Bed", "Noise Floor", "Main Bang"
def select_surface():
    global current_feature, feature_selected, flash_message, flash_counter
    current_feature = "Surface"
    feature_selected = True  # Mark feature as selected
    flash_message = None  # Clear flash message when starting new picks
    flash_counter = 0
    print("Surface selected")

def select_no_surface():
    global current_feature, feature_selected, flash_message, flash_counter
    current_feature = "No Surface"
    feature_selected = True  # Mark feature as selected
    flash_message = None  # Clear flash message when starting new picks
    flash_counter = 0
    print("No Surface selected")

def select_bed():
    global current_feature, feature_selected, flash_message, flash_counter
    current_feature = "Bed"
    feature_selected = True  # Mark feature as selected
    flash_message = None  # Clear flash message when starting new picks
    flash_counter = 0
    print("Bed selected")

def select_no_bed():
    global current_feature, feature_selected, flash_message, flash_counter
    current_feature = "No Bed"
    feature_selected = True  # Mark feature as selected
    flash_message = None  # Clear flash message when starting new picks
    flash_counter = 0
    print("No Bed selected")

def select_noise_floor():
    global current_feature, feature_selected, flash_message, flash_counter
    current_feature = "Noise Floor"
    feature_selected = True  # Mark feature as selected
    flash_message = None  # Clear flash message when starting new picks
    flash_counter = 0
    print("Noise Floor selected")

def select_main_bang():
    global current_feature, feature_selected, flash_message, flash_counter
    current_feature = "Main Bang"
    feature_selected = True  # Mark feature as selected
    flash_message = None  # Clear flash message when starting new picks
    flash_counter = 0
    print("Main Bang selected")

# Create the Quit button to exit the application
def quit_app():
    app.quit()

# Create PyQt5 buttons and add them to the control layout
button_x_reference_line = QPushButton("Set Up X Axis Reference Line")
button_x_reference_line.clicked.connect(setup_x_reference_line)
control_layout.addWidget(button_x_reference_line)

button_reference_line = QPushButton("Set Up Y Axis Reference Line")
button_reference_line.clicked.connect(setup_reference_line)
control_layout.addWidget(button_reference_line)

button_y_axis = QPushButton("Set Up Y Axis")
button_y_axis.clicked.connect(setup_y_axis)
control_layout.addWidget(button_y_axis)

button_x_axis = QPushButton("Set Up X Axis")
button_x_axis.clicked.connect(setup_x_axis)
control_layout.addWidget(button_x_axis)

button_surface = QPushButton("Surface")
button_surface.clicked.connect(select_surface)
control_layout.addWidget(button_surface)

button_no_surface = QPushButton("No Surface")
button_no_surface.clicked.connect(select_no_surface)
control_layout.addWidget(button_no_surface)

button_bed = QPushButton("Bed")
button_bed.clicked.connect(select_bed)
control_layout.addWidget(button_bed)

button_no_bed = QPushButton("No Bed")
button_no_bed.clicked.connect(select_no_bed)
control_layout.addWidget(button_no_bed)

button_noise_floor = QPushButton("Noise Floor")
button_noise_floor.clicked.connect(select_noise_floor)
control_layout.addWidget(button_noise_floor)

button_main_bang = QPushButton("Main Bang")
button_main_bang.clicked.connect(select_main_bang)
control_layout.addWidget(button_main_bang)

button_next_ascope = QPushButton("Next A-scope")
button_next_ascope.clicked.connect(next_ascope)
control_layout.addWidget(button_next_ascope)

button_done = QPushButton("Done")
button_done.clicked.connect(done_and_clear_reference)
control_layout.addWidget(button_done)

button_quit = QPushButton("Quit")
button_quit.clicked.connect(quit_app)
control_layout.addWidget(button_quit)

# Bind the click event to the on_click function
fig.canvas.mpl_connect('button_press_event', on_click)

# Show the main window and run the PyQt5 event loop
main_window.show()
sys.exit(app.exec_())