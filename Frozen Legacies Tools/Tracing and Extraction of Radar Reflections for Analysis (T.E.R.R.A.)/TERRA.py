"""
TERRA - Tracing & Extraction of Radar Reflections for Analysis

DESCRIPTION:
This tool processes radar TIFF files to extract surface and bed reflections from ice-penetrating 
radar data. It provides an interactive interface for manual digitization of radar reflections 
and exports results to CSV format with optional time-domain calculations using calpip analysis.

REQUIREMENTS:
- Python 3.x
- Required packages: opencv-python, numpy, pandas, matplotlib, tkinter, PIL

IF YOU DO NOT HAVE THE REQUIRED PACKAGES, FOLLOW THESE STEPS:
1. Install/Update pip: python -m ensurepip --upgrade
2. Update pip to latest: python -m pip install --upgrade pip
3. Install required packages: python -m pip install opencv-python numpy pandas matplotlib pillow
4. Verify installation: python -c "import cv2, numpy, pandas, matplotlib, tkinter; print('All packages installed successfully!')"

USAGE:
Run from command line using one of these formats:

1. Fancy labeled format (recommended):
   python TERRA.py tiff "path/to/tiff/directory" output "path/to/output/directory" nav "path/to/navigation.csv"

2. Legacy format:
   python TERRA.py "path/to/tiff/directory" "path/to/output/directory" "path/to/navigation.csv"

3. No arguments (uses default development paths):
   python TERRA.py

ARGUMENTS:
- tiff   : Directory containing TIFF radar files to process
- output : Directory where CSV results and images will be saved
- nav    : Path to navigation CSV file containing CBD coordinates and lat/lon data

EXAMPLE:
python TERRA.py tiff "C:\\Data\\TIFFs" output "C:\\Results" nav "C:\\Navigation\\125.csv"

WORKFLOW:
1. The program loads TIFF files one by one from the specified directory
2. For each file, user performs interactive digitization in this order:
   a) Select CBD (Control Boundary Diamonds) markers - left-click to add points, right-click when done
     b) Pick grouped anchors for surface, bed, and transmitted traces
         (left-click adds points, right-click closes group, Enter finishes feature)
     c) Run column-wise intensity detection guided by each anchor group
     d) Smooth each detected segment independently (no connection across groups)
3. System examines radargram for calpips (calibration pulses) for time-domain analysis
4. Results are automatically saved as CSV files in the output directory
5. High-quality PNG images are generated with picked reflections and axes

OUTPUT FILES:
- [filename].csv: Contains digitized coordinates, lat/lon data, and time-domain calculations
- [filename].png: High-quality visualization of results with dual y-axes (depth/time)

FEATURES:
- Interactive anchor-based digitization with ARIES-style column-wise detection
- Grouped-anchor segmentation (one feature can contain multiple disconnected groups)
- Segment-only local smoothing
- CBD coordinate extraction and lat/lon lookup
- Optional calpip-based time-domain analysis
- High-quality matplotlib figure generation
- Handles missing data with NaN values
- Supports batch processing of multiple files

AUTHOR: [Angelo Tarzona, tarzona@gatech.edu]
VERSION: 2025.10.10
"""

import os
from copy import deepcopy
from functools import partial
import time
import csv
import pandas as pd
import tkinter as tk
from tkinter import messagebox
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from PIL import Image

import cv2  # install OpenCV 
import numpy as np  # install Numpy 

# Key definitions
ESC = 27

# ====================================

class Image:
    """
    Contains an image and helpful methods for alternative displays.
    """
    
    def __init__(self, imgPath: str, cropVals: tuple, DEBUG: bool = False):
        self.DEBUG = DEBUG
        if self.DEBUG:
            print("Creating new image object:", imgPath, cropVals)

        self.imgPath = imgPath
        self.cropVals = cropVals
        self.img = cv2.imread(imgPath)
        self.img = self.img[cropVals[0]:cropVals[1], :]  # vertical crop

    @staticmethod
    def blur(strength: int, image: np.ndarray):
        print("Calculating blur")
        return cv2.blur(image, (strength, strength))

    @staticmethod
    def edgeC(threshold: int, image: np.ndarray):
        print("Calculating edges")
        return cv2.Canny(image=image, threshold1=threshold, threshold2=threshold)

    @staticmethod
    def mask(image1: np.ndarray, image2: np.ndarray):
        print("Calculating mask")
        image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
        return cv2.bitwise_and(image1, image2)

    @staticmethod
    def stitch(listOfImages):
        result = listOfImages[0]
        for image in listOfImages[1:]:
            result = np.concatenate((result, image), axis=1)
        return result

# ====================================

class Window:
    """
    Manages the display window and interactive drawing phases.
    """
    def __init__(self, windowName: str, image: Image = None, detail=10, 
                 drawColourSurface=(255, 0, 0), drawColourBed=(0, 255, 0), drawColourCBD=(150,150,100),
                 drawColourTransmitted=(255, 0, 255), fixedSize=(1200, 800), DEBUG=False, csv_directory=None):
        self.DEBUG = DEBUG
        # Set a fixed window title "TERRA"
        self.windowName = "Tracing & Extraction of Radar Reflections for Analysis (T.E.R.R.A.)"

        self.image = image
        self.sourceImage = image.img if image else None
        self.resultImage = deepcopy(self.sourceImage)
        self.displayedImage = deepcopy(self.sourceImage)
        
        # Store the CSV directory for LAT/LON data
        self.csv_directory = csv_directory
        
        # Fixed display size setup (preserving original TERRA behavior)
        self.fixedSize = fixedSize
        orig_h, orig_w = self.sourceImage.shape[:2]
        self.scale_x = orig_w / fixedSize[0]
        self.scale_y = orig_h / fixedSize[1]
        
        # Phase flags and storage for user selections
        self.CBD_done = False
        self.surface_done = False
        self.bed_done = False
        self.transmitted_done = False
        self.cbd_points = []
        self.surface_points = []
        self.bed_points = []
        self.transmitted_points = []
        self.cbd_boarders = []  # used in CBD selection
        self.detail = detail

        # Anchor-based detection state
        self.surface_anchors = []
        self.bed_anchors = []
        self.transmitted_anchors = []
        self.surface_polarity = 'dark'
        self.bed_polarity = 'dark'
        self.transmitted_polarity = 'bright'
        self.anchor_search_half_window = 80  # pixels; tune to 30-80 as needed
        self.anchor_tracking_max_step = 5  # max y-change per column in pixels
        self.anchor_slope_penalty = 0.75    # higher favors continuity over raw extrema
        self.anchor_guide_penalty = 0.35    # pulls detection toward multi-anchor guide curve
        self.export_dpi = 600
        
        # Calpip processing attributes
        self.calpip_pixel_distance = None
        self.calpip_y_lines = []
        
        # Skip flag for current file
        self.skip_current_file = False

        # Create window at fixed size with title "TERRA" (preserving original behavior)
        cv2.namedWindow(self.windowName, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.windowName, fixedSize[0], fixedSize[1])
 
        # Instantiate drawing helpers (they set their own mouse callbacks)
        self.polyDrawerSurface = PolygonDrawer(self, drawColourSurface, DEBUG=self.DEBUG)
        self.polyDrawerBed = PolygonDrawer(self, drawColourBed, DEBUG=self.DEBUG)
        self.polyDrawerTransmitted = PolygonDrawer(self, drawColourTransmitted, DEBUG=self.DEBUG)
        self.cbdSelector = cbdSelector(self, drawColourCBD, DEBUG=self.DEBUG)

    def refresh(self, image: np.ndarray):
        self.displayedImage = image
        cv2.imshow(self.windowName, image)

    def filterImage(self, image: np.ndarray):
        # For now, simply return a copy; advanced filtering (blur/edge)
        return deepcopy(image)
    
    def resetAllPicks(self):
        """Reset surface, bed, and transmitted picks (CBD picks are kept)."""
        self.polyDrawerSurface.polygons = []
        self.polyDrawerSurface.current_points = []
        self.polyDrawerSurface.preview_lines = []
        self.polyDrawerSurface.done = False
        self.polyDrawerBed.polygons = []
        self.polyDrawerBed.current_points = []
        self.polyDrawerBed.preview_lines = []
        self.polyDrawerBed.done = False
        self.polyDrawerTransmitted.polygons = []
        self.polyDrawerTransmitted.current_points = []
        self.polyDrawerTransmitted.preview_lines = []
        self.polyDrawerTransmitted.done = False
        self.cbd_points = []
        self.surface_points = []
        self.bed_points = []
        self.transmitted_points = []
        self.surface_anchors = []
        self.bed_anchors = []
        self.transmitted_anchors = []
        print("\n[RESET] All picks cleared. Start over.")

    def drawAllPolygons(self):
        # First, resize the source image to the fixed display size
        display = cv2.resize(self.sourceImage, self.fixedSize)
        
        # Draw vertical CBD lines using scaled coordinates
        if self.cbdSelector.points:
            for point in self.cbdSelector.points:
                disp_x = int(point[0] / self.scale_x)
                disp_y = int(point[1] / self.scale_y)
                cv2.line(display, (disp_x, 0), (disp_x, self.fixedSize[1]), (0, 255, 255), 1)
                cv2.circle(display, (disp_x, disp_y), 8, (0, 255, 255), -1)

        # Draw anchor groups (and per-group guide line) for each trace
        anchor_groups = [
            (self.surface_anchors, self.polyDrawerSurface.colour),
            (self.bed_anchors, self.polyDrawerBed.colour),
            (self.transmitted_anchors, self.polyDrawerTransmitted.colour),
        ]
        for grouped_anchors, colour in anchor_groups:
            if not grouped_anchors:
                continue
            for anchors in grouped_anchors:
                if not anchors:
                    continue
                scaled_pts = [(int(p[0] / self.scale_x), int(p[1] / self.scale_y)) for p in anchors]
                if len(scaled_pts) >= 2:
                    sorted_pts = sorted(scaled_pts, key=lambda p: p[0])
                    cv2.polylines(display, [np.array(sorted_pts, dtype=np.int32)], False, colour, 1)
                for pt in scaled_pts:
                    cv2.circle(display, pt, 6, colour, -1)

        # Draw current surface control points being placed (dots + connecting straight line)
        if self.polyDrawerSurface.current_points:
            pts = np.array([(int(p[0] / self.scale_x), int(p[1] / self.scale_y)) for p in self.polyDrawerSurface.current_points])
            if len(pts) >= 2:
                cv2.polylines(display, [pts], False, self.polyDrawerSurface.colour, 1)
            for pt in pts:
                cv2.circle(display, tuple(pt), 5, self.polyDrawerSurface.colour, -1)
        # Draw surface smoothed curves (one per completed segment)
        for preview in self.polyDrawerSurface.preview_lines:
            if len(preview) >= 2:
                scaled = [(int(p[0] / self.scale_x), int(p[1] / self.scale_y)) for p in preview]
                cv2.polylines(display, [np.array(scaled)], False, self.polyDrawerSurface.colour, 3)
        
        # Draw current bed control points being placed (dots + connecting straight line)
        if self.polyDrawerBed.current_points:
            pts = np.array([(int(p[0] / self.scale_x), int(p[1] / self.scale_y)) for p in self.polyDrawerBed.current_points])
            if len(pts) >= 2:
                cv2.polylines(display, [pts], False, self.polyDrawerBed.colour, 1)
            for pt in pts:
                cv2.circle(display, tuple(pt), 5, self.polyDrawerBed.colour, -1)
        # Draw bed smoothed curves (one per completed segment)
        for preview in self.polyDrawerBed.preview_lines:
            if len(preview) >= 2:
                scaled = [(int(p[0] / self.scale_x), int(p[1] / self.scale_y)) for p in preview]
                cv2.polylines(display, [np.array(scaled)], False, self.polyDrawerBed.colour, 3)
        
        # Draw current transmitted control points being placed (dots + connecting straight line)
        if self.polyDrawerTransmitted.current_points:
            pts = np.array([(int(p[0] / self.scale_x), int(p[1] / self.scale_y)) for p in self.polyDrawerTransmitted.current_points])
            if len(pts) >= 2:
                cv2.polylines(display, [pts], False, self.polyDrawerTransmitted.colour, 1)
            for pt in pts:
                cv2.circle(display, tuple(pt), 5, self.polyDrawerTransmitted.colour, -1)
        # Draw transmitted smoothed curves (one per completed segment)
        for preview in self.polyDrawerTransmitted.preview_lines:
            if len(preview) >= 2:
                scaled = [(int(p[0] / self.scale_x), int(p[1] / self.scale_y)) for p in preview]
                cv2.polylines(display, [np.array(scaled)], False, self.polyDrawerTransmitted.colour, 3)
        
        return display

    def drawFinalResult(self):
        """Draw the final result with original image background and only dashed lines (no polygon outlines)."""
        # Start with resized original image (keep original background)
        display = cv2.resize(self.sourceImage, self.fixedSize)
        
        # Get lat/lon data for CBD marks
        lat_lon_data = self.getLatLonForCBDs()
        cbd_numbers = self.getCBDNumbers()
        
        # Draw faint dashed vertical CBD lines (optional, for reference)
        if self.cbdSelector.points:
            for i, point in enumerate(self.cbdSelector.points):
                disp_x = int(point[0] / self.scale_x)
                disp_y = int(point[1] / self.scale_y)
                # Draw dashed line by drawing small segments
                dash_length = 10
                gap_length = 5
                y = 0
                while y < self.fixedSize[1]:
                    y_end = min(y + dash_length, self.fixedSize[1])
                    cv2.line(display, (disp_x, y), (disp_x, y_end), (100, 100, 100), 1)  # Faint gray dashes
                    y += dash_length + gap_length
                # Draw small circle at click point
                cv2.circle(display, (disp_x, disp_y), 3, (100, 100, 100), -1)
                
                # Add CBD number, lat/lon text underneath the actual dashed line
                text_lines = []
                
                # Add CBD number
                if cbd_numbers and i < len(cbd_numbers):
                    text_lines.append(f"CBD: {cbd_numbers[i]}")
                
                # Add lat/lon if available
                if lat_lon_data and i < len(lat_lon_data):
                    lat, lon = lat_lon_data[i]
                    if not (np.isnan(lat) or np.isnan(lon)):
                        text_lines.append(f"Lat: {lat:.2f}")
                        text_lines.append(f"Lon: {lon:.2f}")
                
                # Position text underneath the actual line content (near bottom but within image)
                if text_lines:
                    font_scale = 0.35
                    thickness = 1
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    
                    # Start positioning near the bottom of the image but with margin for text
                    line_height = 15
                    start_y = self.fixedSize[1] - (len(text_lines) * line_height) - 10  # Position from bottom up
                    
                    for j, line in enumerate(text_lines):
                        # Get text size for centering
                        (text_w, text_h), _ = cv2.getTextSize(line, font, font_scale, thickness)
                        text_x = disp_x - text_w // 2  # Center horizontally on the line
                        text_y = start_y + j * line_height
                        
                        # Draw white background rectangle for better visibility
                        cv2.rectangle(display, (text_x-2, text_y-text_h-2), (text_x+text_w+2, text_y+2), (255, 255, 255), -1)
                        # Draw black border around background
                        cv2.rectangle(display, (text_x-2, text_y-text_h-2), (text_x+text_w+2, text_y+2), (0, 0, 0), 1)
                        # Draw text
                        cv2.putText(display, line, (text_x, text_y), font, font_scale, (0, 0, 0), thickness)

        return display

    def getLatLonForCBDs(self):
        """Extract lat/lon coordinates for CBD marks from the flight CSV file."""
        try:
            # Extract flight number and CBD range from filename
            # Example: F125-C0919_0932.tiff -> flight=125, cbd_start=919, cbd_end=932
            filename = os.path.basename(self.image.imgPath)
            filename_no_ext = os.path.splitext(filename)[0]
            
            # Parse filename pattern: F{flight}-C{start}_{end}
            if 'F' in filename_no_ext and '-C' in filename_no_ext:
                parts = filename_no_ext.split('-C')
                flight_num = parts[0].replace('F', '')
                cbd_range = parts[1]
                
                if '_' in cbd_range:
                    cbd_start_str, cbd_end_str = cbd_range.split('_')
                    # Remove 'C' prefix if present (handles C0, C1, etc.)
                    if cbd_start_str.startswith('C'):
                        cbd_start_str = cbd_start_str[1:]  # Remove 'C'
                    if cbd_end_str.startswith('C'):
                        cbd_end_str = cbd_end_str[1:]  # Remove 'C'
                    # Remove leading zeros but keep at least one digit
                    cbd_start_str = cbd_start_str.lstrip('0') or '0'
                    cbd_end_str = cbd_end_str.lstrip('0') or '0'
                    cbd_start, cbd_end = int(cbd_start_str), int(cbd_end_str)
                else:
                    return None
                
                # Load the flight CSV file from specified directory or same directory as TIFF
                if self.csv_directory:
                    # Check if csv_directory is a specific CSV file path (correct usage)
                    if self.csv_directory.lower().endswith('.csv'):
                        # User passed a specific CSV file path - use it directly if it matches flight number
                        csv_filename = os.path.basename(self.csv_directory)
                        csv_flight_num = os.path.splitext(csv_filename)[0]
                        if csv_flight_num == flight_num:
                            csv_path = self.csv_directory
                        else:
                            # Different flight number - construct path in same directory
                            csv_dir = os.path.dirname(self.csv_directory)
                            csv_path = os.path.join(csv_dir, f"{flight_num}.csv")
                    else:
                        # csv_directory is a directory path
                        csv_path = os.path.join(self.csv_directory, f"{flight_num}.csv")
                else:
                    csv_path = os.path.join(os.path.dirname(self.image.imgPath), f"{flight_num}.csv")
                
                if not os.path.exists(csv_path):
                    print(f"Warning: Flight CSV file not found: {csv_path}")
                    return None
                
                # Read CSV and extract lat/lon for the CBD range
                import pandas as pd
                df = pd.read_csv(csv_path)
                
                if self.DEBUG:
                    print(f"CSV file loaded: {csv_path}")
                    print(f"CSV columns: {list(df.columns)}")
                    print(f"CSV shape: {df.shape}")
                
                # Try to find CBD, LAT, LON columns by name (case-insensitive)
                cbd_col = None
                lat_col = None
                lon_col = None
                
                for col in df.columns:
                    col_upper = col.upper()
                    if 'CBD' in col_upper:
                        cbd_col = col
                    elif 'LAT' in col_upper:
                        lat_col = col
                    elif 'LON' in col_upper:
                        lon_col = col
                
                # Fallback to positional if column names not found
                if cbd_col is None: cbd_col = df.columns[0]
                if lat_col is None: lat_col = df.columns[1] if len(df.columns) > 1 else None
                if lon_col is None: lon_col = df.columns[2] if len(df.columns) > 2 else None
                
                if self.DEBUG:
                    print(f"Using columns - CBD: {cbd_col}, LAT: {lat_col}, LON: {lon_col}")
                
                lat_lon_data = []
                
                # Generate CBD numbers for this TIFF file
                num_cbds = len(self.cbdSelector.points) if self.cbdSelector.points else 0
                if num_cbds > 0:
                    # The rightmost CBD (last clicked) is actually the first CBD number
                    # So we need to reverse the assignment
                    cbd_numbers = np.linspace(cbd_end, cbd_start, num_cbds, dtype=int)
                    
                    if self.DEBUG:
                        print(f"Looking for CBD numbers: {cbd_numbers}")
                    
                    for cbd_num in cbd_numbers:
                        # Find matching row in CSV using the identified CBD column
                        matching_rows = df[df[cbd_col] == cbd_num]
                        if not matching_rows.empty:
                            if lat_col and lon_col:
                                lat = matching_rows.iloc[0][lat_col]
                                lon = matching_rows.iloc[0][lon_col]
                                lat_lon_data.append((lat, lon))
                                if self.DEBUG:
                                    print(f"Found CBD {cbd_num}: LAT={lat}, LON={lon}")
                            else:
                                lat_lon_data.append((float('nan'), float('nan')))
                                if self.DEBUG:
                                    print(f"Found CBD {cbd_num} but missing LAT/LON columns")
                        else:
                            lat_lon_data.append((float('nan'), float('nan')))
                            if self.DEBUG:
                                print(f"CBD {cbd_num} not found in CSV")
                
                return lat_lon_data
                
            else:
                print(f"Warning: Could not parse filename format: {filename}")
                return None
                
        except Exception as e:
            print(f"Error reading lat/lon data: {e}")
            return None

    def getCBDNumbers(self):
        """Extract CBD numbers for the current TIFF file."""
        try:
            # Extract CBD range from filename
            filename = os.path.basename(self.image.imgPath)
            filename_no_ext = os.path.splitext(filename)[0]
            
            # Parse filename pattern: F{flight}-C{start}_{end}
            if 'F' in filename_no_ext and '-C' in filename_no_ext:
                parts = filename_no_ext.split('-C')
                cbd_range = parts[1]
                
                if '_' in cbd_range:
                    cbd_start_str, cbd_end_str = cbd_range.split('_')
                    # Remove 'C' prefix if present (handles C0, C1, etc.)
                    if cbd_start_str.startswith('C'):
                        cbd_start_str = cbd_start_str[1:]  # Remove 'C'
                    if cbd_end_str.startswith('C'):
                        cbd_end_str = cbd_end_str[1:]  # Remove 'C'
                    # Remove leading zeros but keep at least one digit
                    cbd_start_str = cbd_start_str.lstrip('0') or '0'
                    cbd_end_str = cbd_end_str.lstrip('0') or '0'
                    cbd_start, cbd_end = int(cbd_start_str), int(cbd_end_str)
                    
                    # Generate CBD numbers for this TIFF file
                    num_cbds = len(self.cbdSelector.points) if self.cbdSelector.points else 0
                    if num_cbds > 0:
                        # The rightmost CBD (last clicked) is actually the first CBD number
                        cbd_numbers = np.linspace(cbd_end, cbd_start, num_cbds, dtype=int)
                        return cbd_numbers.tolist()
                    
            return None
            
        except Exception as e:
            print(f"Error extracting CBD numbers: {e}")
            return None

    def drawAverageLinesAndSave(self):
        start = time.time()
        if self.DEBUG:
            print("Computing average lines...")

        # Initialize with empty lists
        self.surface_points = []
        self.bed_points = []
        self.transmitted_points = []

        # Compute average line for surface if there are completed segments
        if self.polyDrawerSurface.has_polygons():
            self.surface_points = self.polyDrawerSurface.createAvgLine(detail=self.detail, area="Surface")
        else:
            # No surface segments, create NaN points for each CBD border
            if self.cbd_boarders:
                self.surface_points = [(float('nan'), float('nan'))] * len(self.cbd_boarders)
            else:
                self.surface_points = [(float('nan'), float('nan'))] * self.detail
        
        # Compute average line for bed if there are completed segments
        if self.polyDrawerBed.has_polygons():
            self.bed_points = self.polyDrawerBed.createAvgLine(detail=self.detail, area="Bed")
        else:
            # No bed segments, create NaN points for each CBD border
            if self.cbd_boarders:
                self.bed_points = [(float('nan'), float('nan'))] * len(self.cbd_boarders)
            else:
                self.bed_points = [(float('nan'), float('nan'))] * self.detail
        
        # Compute average line for transmitted pulse if there are completed segments
        if self.polyDrawerTransmitted.has_polygons():
            self.transmitted_points = self.polyDrawerTransmitted.createAvgLine(detail=self.detail, area="Transmitted")
        else:
            # No transmitted segments, create NaN points for each CBD border
            if self.cbd_boarders:
                self.transmitted_points = [(float('nan'), float('nan'))] * len(self.cbd_boarders)
            else:
                self.transmitted_points = [(float('nan'), float('nan'))] * self.detail

        self.writeToCSV(self.surface_points, self.bed_points, self.transmitted_points)
        
        # Create the final display image with only lines (no polygons)
        self.resultImage = self.drawFinalResult()
        
        # Draw the average lines on the result image, handling NaN values properly
        self.drawAverageLinesOnResult()
        
        self.refresh(self.resultImage)

        end = time.time()
        if self.DEBUG:
            print("Average lines computed in", int(end - start), "secs.")
    
    def drawAverageLinesOnResult(self):
        """Draw picked traces on result image as independent user-defined segments."""
        for seg in self.polyDrawerSurface.preview_lines:
            if len(seg) >= 2:
                cv2.polylines(self.resultImage, [np.array(seg, dtype=np.int32)], False, self.polyDrawerSurface.colour, 1)
        for seg in self.polyDrawerBed.preview_lines:
            if len(seg) >= 2:
                cv2.polylines(self.resultImage, [np.array(seg, dtype=np.int32)], False, self.polyDrawerBed.colour, 1)
        for seg in self.polyDrawerTransmitted.preview_lines:
            if len(seg) >= 2:
                cv2.polylines(self.resultImage, [np.array(seg, dtype=np.int32)], False, self.polyDrawerTransmitted.colour, 1)
    
    def drawLineSegments(self, points, colour, thickness):
        """Draw dashed horizontal line segments across the width of each individual polygon (not connecting polygons)."""
        if not points:
            return
        
        # Determine which polygon type we're drawing for based on colour
        target_polygons = []
        if colour == self.polyDrawerSurface.colour:
            target_polygons = self.polyDrawerSurface.polygons
        elif colour == self.polyDrawerBed.colour:
            target_polygons = self.polyDrawerBed.polygons
        elif colour == self.polyDrawerTransmitted.colour:
            target_polygons = self.polyDrawerTransmitted.polygons
        
        # Define radargram boundaries (account for extended image with margins)
        margin_left = 100
        radargram_left = margin_left
        radargram_right = margin_left + 1200  # original image width
        
        # For each point, find which specific polygon it belongs to and draw line only within that polygon
        for i, point in enumerate(points):
            if not (np.isnan(point[0]) or np.isnan(point[1])):
                y_level = point[1]
                
                # Check each polygon individually to see if this point belongs to it
                for polygon in target_polygons:
                    # Check if this point is inside this specific polygon
                    point_in_polygon = cv2.pointPolygonTest(np.array(polygon, dtype=np.int32), (point[0], point[1]), False)
                    
                    if point_in_polygon >= 0:  # Point is inside or on the boundary of this polygon
                        # Find the leftmost and rightmost x coordinates at this y level for THIS specific polygon only
                        left_x = float('inf')
                        right_x = float('-inf')
                        
                        for j in range(len(polygon)):
                            p1 = polygon[j]
                            p2 = polygon[(j + 1) % len(polygon)]
                            # Check if this edge crosses the y level
                            if ((p1[1] <= y_level <= p2[1]) or (p2[1] <= y_level <= p1[1])) and p1[1] != p2[1]:
                                # Calculate intersection x coordinate
                                t = (y_level - p1[1]) / (p2[1] - p1[1])
                                x_intersect = p1[0] + t * (p2[0] - p1[0])
                                left_x = min(left_x, x_intersect)
                                right_x = max(right_x, x_intersect)
                        
                        # Draw the dashed line across this specific polygon's width only
                        # But constrain to radargram boundaries in the extended image
                        if left_x != float('inf') and right_x != float('-inf') and left_x < right_x:
                            # Convert to display coordinates but keep within radargram area
                            left_display_x = int(left_x / self.scale_x) + margin_left
                            right_display_x = int(right_x / self.scale_x) + margin_left
                            display_y = int(y_level / self.scale_y)
                            
                            # Constrain to radargram boundaries
                            left_display_x = max(radargram_left, left_display_x)
                            right_display_x = min(radargram_right, right_display_x)
                            
                            if left_display_x < right_display_x:
                                self.drawDashedLine(self.resultImage, (left_display_x, display_y), (right_display_x, display_y), colour, thickness)
                        break  # Found the polygon for this point, no need to check others

    def drawDashedLine(self, img, pt1, pt2, color, thickness, dash_length=10, gap_length=5):
        """Draw a dashed line between two points."""
        # Calculate the distance and direction
        dist = ((pt2[0] - pt1[0]) ** 2 + (pt2[1] - pt1[1]) ** 2) ** 0.5
        dashes = int(dist / (dash_length + gap_length))
        
        for i in range(dashes):
            # Calculate start and end points for each dash
            start_ratio = i * (dash_length + gap_length) / dist
            end_ratio = (i * (dash_length + gap_length) + dash_length) / dist
            
            if end_ratio > 1.0:
                end_ratio = 1.0
            
            start_pt = (
                int(pt1[0] + start_ratio * (pt2[0] - pt1[0])),
                int(pt1[1] + start_ratio * (pt2[1] - pt1[1]))
            )
            end_pt = (
                int(pt1[0] + end_ratio * (pt2[0] - pt1[0])),
                int(pt1[1] + end_ratio * (pt2[1] - pt1[1]))
            )
            
            cv2.line(img, start_pt, end_pt, color, thickness)

    def writeToCSV(self, data_surface, data_bed, data_transmitted):
        # Reference point is now top-left of original TIFF file
        # Add back the crop offset to reference original TIFF coordinates
        CROP_OFFSET_Y = self.image.cropVals[0]  # This is the top crop value (300 in your case)
        
        if self.DEBUG:
            print("Writing to CSV.")

        points_surface = []
        points_bed = []
        points_transmitted = []
        img = cv2.cvtColor(self.sourceImage, cv2.COLOR_BGR2GRAY)
        
        for point in data_surface:
            if not (np.isnan(point[0]) or np.isnan(point[1])):
                # Add crop offset to reference original TIFF top-left corner
                points_surface.append((point[0], point[1]+CROP_OFFSET_Y, img[int(point[1])-1, int(point[0])-1]))
            else:
                points_surface.append((float('nan'), float('nan'), float('nan')))
                
        for point in data_bed:
            if not (np.isnan(point[0]) or np.isnan(point[1])):
                points_bed.append((point[0], point[1]+CROP_OFFSET_Y, img[int(point[1])-1, int(point[0])-1]))
            else:
                points_bed.append((float('nan'), float('nan'), float('nan')))
                
        for point in data_transmitted:
            if not (np.isnan(point[0]) or np.isnan(point[1])):
                points_transmitted.append((point[0], point[1]+CROP_OFFSET_Y, img[int(point[1])-1, int(point[0])-1]))
            else:
                points_transmitted.append((float('nan'), float('nan'), float('nan')))

        # Create arrays for CSV - starting with FLT and CBD data
        flt_data = ["FLT"]
        cbd_number = ["CBD_number"]
        lat_data = ["Latitude"]
        lon_data = ["Longitude"]
        x_cbd_pixel = ["x_CBD_pixel_location"]
        y_cbd_pixel = ["y_CBD_pixel_location"]
        x_surface = ["x_surface"]
        y_surface = ["y_surface"]
        depth_surface = ["pixel_depth_surface"]
        x_bed = ["x_bed"]
        y_bed = ["y_bed"]
        depths_bed = ["pixel_depths_bed"]
        x_transmitted = ["x_transmitted"]
        y_transmitted = ["y_transmitted"]
        depth_transmitted = ["pixel_depth_transmitted"]
        y_surface_corrected = ["y_surface_corrected"]
        y_bed_corrected = ["y_bed_corrected"]
        delta_y = ["delta_y"]

        # Get lat/lon data and CBD numbers
        lat_lon_data = self.getLatLonForCBDs()
        cbd_numbers = self.getCBDNumbers()
        
        # Extract FLT (flight) number from filename
        flt_number = None
        try:
            filename = os.path.basename(self.image.imgPath)
            filename_no_ext = os.path.splitext(filename)[0]
            # Parse filename pattern: F{flight}-C{start}_{end}
            if 'F' in filename_no_ext and '-C' in filename_no_ext:
                parts = filename_no_ext.split('-C')
                flt_number = int(parts[0].replace('F', ''))
        except Exception as e:
            print(f"Warning: Could not extract FLT number from filename: {e}")
            flt_number = float('nan')

        # Now all arrays should have the same length, so we can use a simple loop
        for i in range(len(points_surface)):
            # Add FLT number
            flt_data.append(flt_number)
            
            # Add CBD number
            if cbd_numbers and i < len(cbd_numbers):
                cbd_number.append(cbd_numbers[i])
            else:
                cbd_number.append(float('nan'))
                
            # Add lat/lon data
            if lat_lon_data and i < len(lat_lon_data):
                lat, lon = lat_lon_data[i]
                lat_data.append(lat)
                lon_data.append(lon)
            else:
                lat_data.append(float('nan'))
                lon_data.append(float('nan'))
                
            # Add CBD pixel data - adjust to original TIFF coordinates
            if i < len(self.cbd_boarders):
                x_cbd_pixel.append(self.cbd_boarders[i][0])
                y_cbd_pixel.append(self.cbd_boarders[i][1] + CROP_OFFSET_Y)
            else:
                x_cbd_pixel.append(float('nan'))
                y_cbd_pixel.append(float('nan'))
                
            x_surface.append(points_surface[i][0])
            y_surface.append(points_surface[i][1])
            depth_surface.append(points_surface[i][2])
            x_bed.append(points_bed[i][0])
            y_bed.append(points_bed[i][1])
            depths_bed.append(points_bed[i][2])
            x_transmitted.append(points_transmitted[i][0])
            y_transmitted.append(points_transmitted[i][1])
            depth_transmitted.append(points_transmitted[i][2])
            
            # Calculate corrected y-values and delta_y, handling NaN values
            if not (np.isnan(points_surface[i][1]) or np.isnan(points_transmitted[i][1])):
                surf_corrected = points_surface[i][1] - points_transmitted[i][1]
                y_surface_corrected.append(surf_corrected)
            else:
                y_surface_corrected.append(float('nan'))
                
            if not (np.isnan(points_bed[i][1]) or np.isnan(points_transmitted[i][1])):
                bed_corrected = points_bed[i][1] - points_transmitted[i][1]
                y_bed_corrected.append(bed_corrected)
            else:
                y_bed_corrected.append(float('nan'))
                
            # Calculate delta_y (bed - surface)
            if not (np.isnan(points_bed[i][1]) or np.isnan(points_surface[i][1])):
                if not (np.isnan(points_transmitted[i][1])):
                    # Use corrected values if transmitted data exists
                    delta_y.append(y_bed_corrected[-1] - y_surface_corrected[-1])
                else:
                    # Use original values if no transmitted data
                    delta_y.append(points_bed[i][1] - points_surface[i][1])
            else:
                delta_y.append(float('nan'))
        data = np.transpose([flt_data,cbd_number,lat_data,lon_data,x_cbd_pixel,y_cbd_pixel,x_surface,y_surface,depth_surface,x_bed,y_bed,depths_bed,x_transmitted,y_transmitted,depth_transmitted,y_surface_corrected,y_bed_corrected,delta_y])
        
        # Reverse the rows (excluding header) to order from least to greatest CBD
        if len(data) > 1:  # Make sure there's data beyond the header
            header = data[0]  # Keep the header row
            data_rows = data[1:]  # Get all data rows
            data_rows_reversed = data_rows[::-1]  # Reverse the data rows
            data = np.vstack([header, data_rows_reversed])  # Recombine header with reversed data
        
        # Use image name (without extension) for CSV file name
        imgName = os.path.splitext(os.path.basename(self.image.imgPath))[0]
        with open(imgName + ".csv", 'w', encoding="ISO-8859-1", newline='') as file:
            writer = csv.writer(file)
            writer.writerows(data)
        if self.DEBUG:
            print("CSV saved as", imgName + ".csv")

    def regenerateCSVWithCalpips(self):
        """Regenerate CSV with calpip-based time and depth calculations."""
        if not hasattr(self, 'surface_points') or not self.surface_points:
            print("No surface/bed data available for calpip calculations")
            return
            
        # Recalculate and save CSV with calpip data
        self.writeCSVWithCalpips(self.surface_points, self.bed_points, self.transmitted_points)

    def writeCSVWithCalpips(self, data_surface, data_bed, data_transmitted):
        """Write CSV with time-domain calculations based on calpip data."""
        # Reference point is now top-left of original TIFF file
        # Add back the crop offset to reference original TIFF coordinates
        CROP_OFFSET_Y = self.image.cropVals[0]  # This is the top crop value (300 in your case)
        
        if self.DEBUG:
            print("Writing CSV with calpip calculations.")

        points_surface = []
        points_bed = []
        points_transmitted = []
        img = cv2.cvtColor(self.sourceImage, cv2.COLOR_BGR2GRAY)
        
        for point in data_surface:
            if not (np.isnan(point[0]) or np.isnan(point[1])):
                # Add crop offset to reference original TIFF top-left corner
                points_surface.append((point[0], point[1]+CROP_OFFSET_Y, img[int(point[1])-1, int(point[0])-1]))
            else:
                points_surface.append((float('nan'), float('nan'), float('nan')))
                
        for point in data_bed:
            if not (np.isnan(point[0]) or np.isnan(point[1])):
                points_bed.append((point[0], point[1]+CROP_OFFSET_Y, img[int(point[1])-1, int(point[0])-1]))
            else:
                points_bed.append((float('nan'), float('nan'), float('nan')))
                
        for point in data_transmitted:
            if not (np.isnan(point[0]) or np.isnan(point[1])):
                points_transmitted.append((point[0], point[1]+CROP_OFFSET_Y, img[int(point[1])-1, int(point[0])-1]))
            else:
                points_transmitted.append((float('nan'), float('nan'), float('nan')))

        # Create arrays for CSV - starting with FLT and CBD data
        flt_data = ["FLT"]
        cbd_number = ["CBD_number"]
        lat_data = ["Latitude"]
        lon_data = ["Longitude"]
        x_cbd_pixel = ["x_CBD_pixel_location"]
        y_cbd_pixel = ["y_CBD_pixel_location"]
        x_surface = ["x_surface"]
        y_surface = ["y_surface"]
        depth_surface = ["pixel_depth_surface"]
        x_bed = ["x_bed"]
        y_bed = ["y_bed"]
        depths_bed = ["pixel_depths_bed"]
        x_transmitted = ["x_transmitted"]
        y_transmitted = ["y_transmitted"]
        depth_transmitted = ["pixel_depth_transmitted"]
        y_surface_corrected = ["y_surface_corrected"]
        y_bed_corrected = ["y_bed_corrected"]
        delta_y = ["delta_y"]
        
        # New calpip-based time domain columns
        surf_twt = ["surf_twt_us"]
        bed_twt = ["bed_twt_us"] 
        h_ice_twt = ["h_ice_twt_us"]
        surf_m = ["surf_m"]
        bed_m = ["bed_m"]
        h_ice_m = ["h_ice_m"]
        calpip_pixel_dist = ["calpip_pixel_distance"]

        # Get lat/lon data and CBD numbers
        lat_lon_data = self.getLatLonForCBDs()
        cbd_numbers = self.getCBDNumbers()
        
        # Extract FLT (flight) number from filename
        flt_number = None
        try:
            filename = os.path.basename(self.image.imgPath)
            filename_no_ext = os.path.splitext(filename)[0]
            # Parse filename pattern: F{flight}-C{start}_{end}
            if 'F' in filename_no_ext and '-C' in filename_no_ext:
                parts = filename_no_ext.split('-C')
                flt_number = int(parts[0].replace('F', ''))
        except Exception as e:
            print(f"Warning: Could not extract FLT number from filename: {e}")
            flt_number = float('nan')

        # Constants
        velocity_ice = 168.0  # m/us - aligned with ARIES

        # Now all arrays should have the same length, so we can use a simple loop
        for i in range(len(points_surface)):
            # Add FLT number
            flt_data.append(flt_number)
            
            # Add CBD number
            if cbd_numbers and i < len(cbd_numbers):
                cbd_number.append(cbd_numbers[i])
            else:
                cbd_number.append(float('nan'))
                
            # Add lat/lon data
            if lat_lon_data and i < len(lat_lon_data):
                lat, lon = lat_lon_data[i]
                lat_data.append(lat)
                lon_data.append(lon)
            else:
                lat_data.append(float('nan'))
                lon_data.append(float('nan'))
                
            # Add CBD pixel data - adjust to original TIFF coordinates
            if i < len(self.cbd_boarders):
                x_cbd_pixel.append(self.cbd_boarders[i][0])
                y_cbd_pixel.append(self.cbd_boarders[i][1] + CROP_OFFSET_Y)
            else:
                x_cbd_pixel.append(float('nan'))
                y_cbd_pixel.append(float('nan'))
                
            x_surface.append(points_surface[i][0])
            y_surface.append(points_surface[i][1])
            depth_surface.append(points_surface[i][2])
            x_bed.append(points_bed[i][0])
            y_bed.append(points_bed[i][1])
            depths_bed.append(points_bed[i][2])
            x_transmitted.append(points_transmitted[i][0])
            y_transmitted.append(points_transmitted[i][1])
            depth_transmitted.append(points_transmitted[i][2])
            
            # Calculate corrected y-values relative to transmitted pulse (time zero), handling NaN values
            if not (np.isnan(points_surface[i][1]) or np.isnan(points_transmitted[i][1])):
                # Surface position relative to transmitted pulse (positive = below zero reference)
                surf_corrected = points_surface[i][1] - points_transmitted[i][1]
                y_surface_corrected.append(surf_corrected)
            else:
                y_surface_corrected.append(float('nan'))
                
            if not (np.isnan(points_bed[i][1]) or np.isnan(points_transmitted[i][1])):
                # Bed position relative to transmitted pulse (positive = below zero reference)
                bed_corrected = points_bed[i][1] - points_transmitted[i][1]
                y_bed_corrected.append(bed_corrected)
            else:
                y_bed_corrected.append(float('nan'))
                
            # Calculate delta_y (bed - surface) using corrected values when possible
            if not (np.isnan(points_bed[i][1]) or np.isnan(points_surface[i][1])):
                if not np.isnan(points_transmitted[i][1]):
                    # Use corrected values (thickness relative to zero reference)
                    delta_y.append(y_bed_corrected[-1] - y_surface_corrected[-1])
                else:
                    # Use original values if no transmitted data (fallback)
                    delta_y.append(points_bed[i][1] - points_surface[i][1])
            else:
                delta_y.append(float('nan'))
            
            # Calculate time domain values using calpip data
            if self.calpip_pixel_distance and self.calpip_pixel_distance > 0:
                # Each calpip line = 2 us apart
                us_per_pixel = 2.0 / self.calpip_pixel_distance
                
                # Add calpip pixel distance to each row
                calpip_pixel_dist.append(self.calpip_pixel_distance)
                
                # Time calculations (using corrected y-values)
                if not np.isnan(y_surface_corrected[-1]):
                    surf_twt_val = y_surface_corrected[-1] * us_per_pixel
                    surf_twt.append(surf_twt_val)
                    surf_m.append((surf_twt_val / 2) * velocity_ice)  # SURFACE DEPTH = (SURFACE TWT/2)*V_ICE
                else:
                    surf_twt.append(float('nan'))
                    surf_m.append(float('nan'))
                    
                if not np.isnan(y_bed_corrected[-1]):
                    bed_twt_val = y_bed_corrected[-1] * us_per_pixel
                    bed_twt.append(bed_twt_val)
                    bed_m.append((bed_twt_val / 2) * velocity_ice)  # BED DEPTH = (BED TWT/2)*V_ICE
                else:
                    bed_twt.append(float('nan'))
                    bed_m.append(float('nan'))
                    
                if not np.isnan(delta_y[-1]):
                    h_ice_twt_val = delta_y[-1] * us_per_pixel
                    h_ice_twt.append(h_ice_twt_val)
                    h_ice_m.append((h_ice_twt_val / 2) * velocity_ice)  # H_ICE = ((BED TWT - SURFACE TWT)/2)*V_ICE
                else:
                    h_ice_twt.append(float('nan'))
                    h_ice_m.append(float('nan'))
            else:
                # No calpip data available
                calpip_pixel_dist.append(float('nan'))
                surf_twt.append(float('nan'))
                bed_twt.append(float('nan'))
                h_ice_twt.append(float('nan'))
                surf_m.append(float('nan'))
                bed_m.append(float('nan'))
                h_ice_m.append(float('nan'))

        data = np.transpose([flt_data,cbd_number,lat_data,lon_data,x_cbd_pixel,y_cbd_pixel,x_surface,y_surface,depth_surface,x_bed,y_bed,depths_bed,x_transmitted,y_transmitted,depth_transmitted,y_surface_corrected,y_bed_corrected,delta_y,surf_twt,bed_twt,h_ice_twt,surf_m,bed_m,h_ice_m,calpip_pixel_dist])
        
        # Reverse the rows (excluding header) to order from least to greatest CBD
        if len(data) > 1:  # Make sure there's data beyond the header
            header = data[0]  # Keep the header row
            data_rows = data[1:]  # Get all data rows
            data_rows_reversed = data_rows[::-1]  # Reverse the data rows
            data = np.vstack([header, data_rows_reversed])  # Recombine header with reversed data
        
        # Use image name (without extension) for CSV file name
        imgName = os.path.splitext(os.path.basename(self.image.imgPath))[0]
        with open(imgName + ".csv", 'w', encoding="ISO-8859-1", newline='') as file:
            writer = csv.writer(file)
            writer.writerows(data)
        if self.DEBUG:
            print("CSV with calpip calculations saved as", imgName + ".csv")

    def drawYAxisLines(self):
        """Draw calpip lines with dual-axis: depth (left) and time (right)"""
        if not self.calpip_y_lines:
            return
        
        # Constants
        velocity_ice = 168.0  # m/μs (velocity of ice) - corrected value
        
        # Get transmitted pulse position for 0μs reference
        if self.transmitted_points:
            avg_transmitted_y = np.mean([pt[1] for pt in self.transmitted_points if not np.isnan(pt[1])])
            zero_time_y = avg_transmitted_y
        else:
            # Fallback: estimate from image structure
            zero_time_y = 50
            
        # Convert to display coordinates and draw dashed yellow lines with transparency
        for y_line in self.calpip_y_lines:
            disp_y = int(y_line / self.scale_y)
            self.drawDashedLineWithTransparency(self.resultImage, (0, disp_y), (self.fixedSize[0], disp_y), 
                                              (0, 200, 200), 1, alpha=0.6, dash_length=8, gap_length=4)
        
        # Create extended image with white background for external axes
        if self.calpip_pixel_distance and self.calpip_pixel_distance > 0:
            # Create extended image with white margins for axes
            margin_left = 100  # Increased for better text spacing
            margin_right = 140  # Increased for right axis text
            extended_width = self.fixedSize[0] + margin_left + margin_right
            extended_height = self.fixedSize[1]
            
            # Create white background
            extended_image = np.full((extended_height, extended_width, 3), 255, dtype=np.uint8)
            
            # Copy the original image to the center
            extended_image[0:self.fixedSize[1], margin_left:margin_left + self.fixedSize[0]] = self.resultImage
            
            # Update result image to extended version
            self.resultImage = extended_image
            self.fixedSize = (extended_width, extended_height)
            
            # Axis positions (now in extended image coordinates) - closer to radargram
            left_axis_x = margin_left - 15    # Closer to left edge of radargram
            right_axis_x = margin_left + 1200 + 15  # Closer to right edge of radargram
            
            # Calculate positions and labels
            positions = []
            time_labels = []
            depth_labels = []
            
            # Add 0μs/0m at transmitted pulse (reference point)
            zero_disp_y = int(zero_time_y / self.scale_y)
            positions.append(zero_disp_y)
            time_labels.append("0")
            depth_labels.append("0")
            
            # Add labels for each calpip line based on spacing from transmitted pulse
            for y_line in self.calpip_y_lines:
                disp_y = int(y_line / self.scale_y)
                pixel_offset = y_line - zero_time_y
                
                # Calculate how many 2μs intervals this line is from transmitted pulse
                intervals_from_transmit = pixel_offset / self.calpip_pixel_distance
                time_us = intervals_from_transmit * 2  # Each calpip interval = 2μs
                
                # Only show labels for lines below transmitted pulse (positive time)
                if time_us > 0 and time_us <= 50:  # Reasonable range
                    depth_m = (time_us / 2) * velocity_ice  # DEPTH = (TWO WAY TRAVEL TIME/2)*V_ICE
                    positions.append(disp_y)
                    time_labels.append(f"{time_us:.0f}")
                    depth_labels.append(f"{depth_m:.0f}")
            
            # Draw left axis (depth in meters) - black on white background
            cv2.line(self.resultImage, (left_axis_x, 30), (left_axis_x, extended_height - 30), (0, 0, 0), 2)
            
            # Draw rotated text "Depth (m)" on the left axis - black text, positioned to avoid overlap
            self.drawRotatedText(self.resultImage, "Depth (m)", (left_axis_x - 60, extended_height // 2), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1, angle=90)
            
            # Draw right axis (time in μs) - black on white background
            cv2.line(self.resultImage, (right_axis_x, 30), (right_axis_x, extended_height - 30), (0, 0, 0), 2)
            
            # Draw rotated text "Two-way travel time (μs)" on the right axis - black text
            self.drawRotatedText(self.resultImage, "Two-way travel time (us)", (right_axis_x + 60, extended_height // 2), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1, angle=90)
            
            # Add tick marks and labels - black on white
            for pos, time_label, depth_label in zip(positions, time_labels, depth_labels):
                if 30 < pos < extended_height - 30:  # Within visible area
                    # Left axis (depth) - tick marks and labels, positioned to avoid overlap with axis title
                    cv2.line(self.resultImage, (left_axis_x - 8, pos), (left_axis_x + 8, pos), (0, 0, 0), 2)
                    cv2.putText(self.resultImage, depth_label, (left_axis_x - 35, pos + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1)
                    
                    # Right axis (time) - tick marks and labels
                    cv2.line(self.resultImage, (right_axis_x - 8, pos), (right_axis_x + 8, pos), (0, 0, 0), 2)
                    cv2.putText(self.resultImage, time_label, (right_axis_x + 15, pos + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1)
        
        # Draw the user picks (surface, bed, transmitted) on top of everything
        self.drawAverageLinesOnResult()
        
        # Create final matplotlib figure with high-quality rendering
        self.createMatplotlibFigure()

    def drawDashedLineWithTransparency(self, img, pt1, pt2, color, thickness, alpha=0.6, dash_length=10, gap_length=5):
        """Draw a dashed line with transparency using overlay technique."""
        # Create an overlay image
        overlay = img.copy()
        
        # Calculate the distance and direction
        dist = ((pt2[0] - pt1[0]) ** 2 + (pt2[1] - pt1[1]) ** 2) ** 0.5
        if dist == 0:
            return
            
        dashes = int(dist / (dash_length + gap_length))
        
        for i in range(dashes):
            # Calculate start and end points for each dash
            start_ratio = i * (dash_length + gap_length) / dist
            end_ratio = (i * (dash_length + gap_length) + dash_length) / dist
            
            if end_ratio > 1.0:
                end_ratio = 1.0
            
            start_pt = (
                int(pt1[0] + start_ratio * (pt2[0] - pt1[0])),
                int(pt1[1] + start_ratio * (pt2[1] - pt1[1]))
            )
            end_pt = (
                int(pt1[0] + end_ratio * (pt2[0] - pt1[0])),
                int(pt1[1] + end_ratio * (pt2[1] - pt1[1]))
            )
            
            cv2.line(overlay, start_pt, end_pt, color, thickness)
        
        # Apply transparency
        cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

    def drawRotatedText(self, img, text, position, font, font_scale, color, thickness, angle=0):
        """Draw rotated text using PIL with high-quality font rendering."""
        try:
            from PIL import Image as PILImage, ImageDraw, ImageFont
            import numpy as np
            
            # Calculate high-resolution font size for better quality
            base_font_size = int(font_scale * 40)  # Increased base size for better rendering
            
            # Create a larger PIL image for high-quality text rendering
            canvas_size = (300, 100)
            pil_img = PILImage.new('RGBA', canvas_size, (255, 255, 255, 0))  # Transparent background
            draw = ImageDraw.Draw(pil_img)
            
            # Try to use a better font or default with higher quality
            try:
                # Use default font with high resolution
                pil_font = ImageFont.load_default()
                # For better quality, we'll scale up the rendering
            except:
                pil_font = ImageFont.load_default()
            
            # Draw text on PIL image with high quality
            text_bbox = draw.textbbox((0, 0), text, font=pil_font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]
            
            # Center the text in the canvas
            text_x = (canvas_size[0] - text_width) // 2
            text_y = (canvas_size[1] - text_height) // 2
            
            # Draw with antialiasing
            draw.text((text_x, text_y), text, font=pil_font, fill=(*color[::-1], 255))  # Convert BGR to RGB
            
            # Rotate the PIL image with high quality
            if angle != 0:
                pil_img = pil_img.rotate(angle, expand=True, resample=PILImage.LANCZOS)
            
            # Convert PIL image to OpenCV format
            pil_array = np.array(pil_img)
            
            # Extract the text pixels (non-transparent)
            text_mask = pil_array[:, :, 3] > 0
            if np.any(text_mask):
                # Position the text
                x, y = position
                h, w = pil_array.shape[:2]
                
                # Calculate the position bounds
                x_start = max(0, x - w // 2)
                y_start = max(0, y - h // 2)
                x_end = min(img.shape[1], x_start + w)
                y_end = min(img.shape[0], y_start + h)
                
                # Adjust for actual placement
                w_actual = x_end - x_start
                h_actual = y_end - y_start
                
                if w_actual > 0 and h_actual > 0:
                    # Create the overlay for this region
                    text_region = pil_array[:h_actual, :w_actual]
                    mask_region = text_mask[:h_actual, :w_actual]
                    
                    # Apply text to image with antialiasing
                    img_region = img[y_start:y_end, x_start:x_end]
                    for c in range(3):  # BGR channels
                        img_region[:, :, c][mask_region] = text_region[:, :, 2-c][mask_region]  # Convert RGB to BGR
                        
        except ImportError:
            # Fallback to character-by-character rendering if PIL is not available
            if angle == 90:
                # Draw characters vertically for 90-degree rotation
                char_height = int(font_scale * 25)  # Increased size for better quality
                text_length = len(text) * char_height
                start_y = position[1] - text_length // 2
                for i, char in enumerate(text):
                    cv2.putText(img, char, (position[0], start_y + i * char_height), 
                               font, font_scale * 1.2, color, thickness + 1)  # Increased thickness
            else:
                # Regular horizontal text with better quality
                cv2.putText(img, text, position, font, font_scale * 1.2, color, thickness + 1)

    def createMatplotlibFigure(self):
        """Create high-quality matplotlib figure with proper text rendering."""
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        import os
        
        # Constants
        velocity_ice = 168.0  # m/μs (velocity of ice)
        
        # Global font settings for the exported figure
        plt.rcParams['font.family'] = 'Segoe UI'
        
        # Get transmitted pulse position for 0μs reference
        if self.transmitted_points:
            avg_transmitted_y = np.mean([pt[1] for pt in self.transmitted_points if not np.isnan(pt[1])])
            zero_time_y = avg_transmitted_y
        else:
            zero_time_y = 50
            
        # Create figure with high DPI
        fig_width = 12  # inches
        fig_height = 8   # inches
        fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=self.export_dpi)
        
        # Display the radar image
        # Convert BGR to RGB for matplotlib and mirror horizontally (conventional low-to-high reading direction)
        radar_rgb = cv2.cvtColor(self.sourceImage, cv2.COLOR_BGR2RGB)
        radar_rgb = np.fliplr(radar_rgb)
        ax.imshow(radar_rgb, aspect='auto', extent=[0, self.sourceImage.shape[1], self.sourceImage.shape[0], 0], origin='upper')
        
        # Draw CBD markers and information
        self.drawCBDMarkersMatplotlib(ax)
        ax.set_xlabel('CBD', fontsize=12, fontweight='bold', color='black')
        
        # Draw calpip lines (dashed yellow with transparency)
        if self.calpip_y_lines:
            for y_line in self.calpip_y_lines:
                ax.axhline(y=y_line, color='gold', linestyle='--', alpha=0.6, linewidth=1)
        
        # Draw radar picks (surface, bed, transmitted)
        self.drawRadarPicksMatplotlib(ax)
        
        # Set up dual y-axes for depth and time
        self.setupMatplotlibAxes(ax, zero_time_y, velocity_ice)
        
        # Add legend
        self.addMatplotlibLegend(ax)
        
        # Show only the y-axes, hide top and bottom spines
        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_visible(True)
        ax.spines['left'].set_visible(True)
        ax.spines['right'].set_visible(False)
        
        # Tight layout
        plt.tight_layout()
        
        # Save the matplotlib figure directly for high quality
        # Generate filename based on the original image
        imgName = os.path.splitext(os.path.basename(self.image.imgPath))[0]
        save_path = f"{imgName}.tiff"
        fig.savefig(save_path, dpi=self.export_dpi, bbox_inches='tight', facecolor='white', 
               edgecolor='none', format='tiff')
        print(f"High-quality matplotlib figure saved as: {save_path}")
        
        plt.close(fig)
    
    def drawRadarPicksMatplotlib(self, ax):
        """Draw radar picks using matplotlib for high quality."""
        width = self.sourceImage.shape[1]  # mirrored image, so flip pick x-coordinates to match
        # Draw each completed segment independently so different groups never connect.
        for seg in self.polyDrawerTransmitted.preview_lines:
            if len(seg) >= 2:
                x_coords = [width - pt[0] for pt in seg]
                y_coords = [pt[1] for pt in seg]
                ax.plot(x_coords, y_coords, color='magenta', linewidth=1, linestyle='--', label='_nolegend_')

        for seg in self.polyDrawerSurface.preview_lines:
            if len(seg) >= 2:
                x_coords = [width - pt[0] for pt in seg]
                y_coords = [pt[1] for pt in seg]
                ax.plot(x_coords, y_coords, color='blue', linewidth=1, linestyle='--', label='_nolegend_')

        for seg in self.polyDrawerBed.preview_lines:
            if len(seg) >= 2:
                x_coords = [width - pt[0] for pt in seg]
                y_coords = [pt[1] for pt in seg]
                ax.plot(x_coords, y_coords, color="#15e515", linewidth=1, linestyle='--', label='_nolegend_')
    
    def drawCBDMarkersMatplotlib(self, ax):
        """Draw CBD markers and labels using matplotlib."""
        if not self.cbd_points:
            return
            
        # Get CBD numbers (lat/lon no longer shown on the CBD boxes)
        cbd_numbers = self.getCBDNumbers()
        width = self.sourceImage.shape[1]  # mirrored image, so flip CBD x-coordinates to match
        
        # Draw vertical CBD lines (dashed gray)
        for i, point in enumerate(self.cbd_points):
            x_pos = width - point[0]
            y_pos = point[1]
            
            # Draw vertical dashed line
            ax.axvline(x=x_pos, color='gray', linestyle='--', alpha=0.7, linewidth=1)
            
            # Draw circle at click point
            ax.plot(x_pos, y_pos, 'o', color='gray', markersize=3)
            
        if cbd_numbers:
            tick_positions = [width - point[0] for point in self.cbd_points]
            tick_labels = [str(number) for number in cbd_numbers]
            ax.set_xticks(tick_positions)
            ax.set_xticklabels(tick_labels, fontsize=10, fontweight='semibold', rotation=0, ha='center')
            ax.tick_params(axis='x', labelbottom=True, bottom=True)
    
    def setupMatplotlibAxes(self, ax, zero_time_y, velocity_ice):
        """Set up dual y-axes with proper scaling and labels."""
        if not (self.calpip_pixel_distance and self.calpip_pixel_distance > 0):
            return
            
        # Create secondary y-axis for time
        ax2 = ax.twinx()
        
        # Calculate positions and labels
        positions = []
        time_labels = []
        depth_labels = []
        
        # Add 0μs/0m at transmitted pulse (reference point)
        positions.append(zero_time_y)
        time_labels.append("0")
        depth_labels.append("0")
        
        # Add labels for each calpip line based on spacing from transmitted pulse
        for y_line in self.calpip_y_lines:
            pixel_offset = y_line - zero_time_y
            intervals_from_transmit = pixel_offset / self.calpip_pixel_distance
            time_us = intervals_from_transmit * 2  # Each calpip interval = 2μs
            
            if time_us > 0 and time_us <= 50:  # Reasonable range
                depth_m = (time_us / 2) * velocity_ice  # Two-way travel time; convert to one-way depth
                positions.append(y_line)
                time_labels.append(f"{time_us:.0f}")
                depth_labels.append(f"{depth_m:.0f}")
        
        # Set up left y-axis (depth) - ensure it's visible
        ax.set_yticks(positions)
        ax.set_yticklabels(depth_labels)
        ax.set_ylabel('Depth (m)', fontsize=12, fontweight='bold', color='black', rotation=90, labelpad=15)
        ax.tick_params(axis='y', labelsize=10, colors='black', left=True, labelleft=True)
        plt.setp(ax.get_yticklabels(), fontweight='semibold')
        ax.yaxis.set_label_position('left')
        
        # Set up right y-axis (time)
        ax2.set_yticks(positions)
        ax2.set_yticklabels(time_labels)
        ax2.set_ylabel('Two-way travel time (μs)', fontsize=12, fontweight='bold', color='black', rotation=90, labelpad=15)
        ax2.tick_params(axis='y', labelsize=10, colors='black', right=True, labelright=True)
        plt.setp(ax2.get_yticklabels(), fontweight='semibold')
        ax2.yaxis.set_label_position('right')
        
        # Set y-axis limits to match image coordinates (top=0, bottom=max)
        ax.set_ylim(self.sourceImage.shape[0], 0)  # Inverted so 0 is at top
        ax2.set_ylim(self.sourceImage.shape[0], 0)  # Match the first axis
        
        # Make sure the right spine is visible for the second axis
        ax2.spines['right'].set_visible(True)
        ax2.spines['top'].set_visible(False)
        ax2.spines['bottom'].set_visible(False)
        ax2.spines['left'].set_visible(False)
    
    def addMatplotlibLegend(self, ax):
        """Add legend with high-quality rendering."""
        from matplotlib.lines import Line2D
        
        # Create legend elements with dashed lines to match the plot
        legend_elements = [
            Line2D([0], [0], color='magenta', lw=2, linestyle='--', label='Transmitted pulse'),
            Line2D([0], [0], color='blue', lw=2, linestyle='--', label='Surface'),
            Line2D([0], [0], color='#7fc97f', lw=2, linestyle='--', label='Bed')
        ]
        
        # Add legend in top right
        legend = ax.legend(handles=legend_elements, loc='upper right', fontsize=12, 
                 frameon=True, fancybox=True, shadow=True, framealpha=0.9)
        plt.setp(legend.get_texts(), fontweight='semibold')

    def drawLegend(self):
        """Draw a legend in the top right corner of the image."""
        # Account for the original image area within the extended image
        original_right_edge = 100 + 1200  # margin_left + original_width
        legend_x = original_right_edge - 180  # Position from right edge of original image
        legend_y = 40  # Position from top
        
        # Calculate legend height based on number of items (3 items * 20 spacing, no extra padding)
        legend_height = (3 - 1) * 20 + 15  # 2 gaps between 3 items + minimal padding
        
        # Legend background (semi-transparent)
        overlay = self.resultImage.copy()
        cv2.rectangle(overlay, (legend_x - 10, legend_y - 10), (legend_x + 170, legend_y + legend_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.3, self.resultImage, 0.7, 0, self.resultImage)
        
        # Legend border
        cv2.rectangle(self.resultImage, (legend_x - 10, legend_y - 10), (legend_x + 170, legend_y + legend_height), (255, 255, 255), 1)
        
        # Legend items
        legend_items = [
            ("Transmitted pulse", (255, 0, 255)),  # Magenta
            ("Surface", (255, 0, 0)),              # Red  
            ("Bed", (0, 255, 0))                   # Green
        ]
        
        line_spacing = 20
        for i, (label, color) in enumerate(legend_items):
            y_pos = legend_y + i * line_spacing
            
            # Draw colored line sample
            cv2.line(self.resultImage, (legend_x, y_pos), (legend_x + 20, y_pos), color, 2)
            
            # Draw label text
            cv2.putText(self.resultImage, label, (legend_x + 30, y_pos + 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    def _pick_feature_anchors(self, feature_name: str, attr_name: str, polarity_attr: str):
        groups = []
        current_group = []

        def on_anchor_mouse(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                orig_x = int(x * self.scale_x)
                orig_y = int(y * self.scale_y)
                current_group.append((orig_x, orig_y))
                print(f"{feature_name} group {len(groups) + 1}, point {len(current_group)} added at ({orig_x}, {orig_y}).")
            elif event == cv2.EVENT_RBUTTONDOWN:
                if len(current_group) >= 1:
                    groups.append(current_group.copy())
                    current_group.clear()
                    print(f"{feature_name} group {len(groups)} closed. Start next group or press Enter to finish {feature_name}.")
                else:
                    print(f"Current {feature_name} group is empty. Add points before right-clicking.")

        print(f"Pick {feature_name} anchors (left click).")
        print("  Right-click closes current group, Enter finishes feature, 'u' undo, 'd' dark, 'b' bright, 's' skip, 'r' reset.")
        cv2.setMouseCallback(self.windowName, on_anchor_mouse)

        while True:
            # Show in-progress grouped anchors live
            live_groups = groups.copy()
            if current_group:
                live_groups.append(current_group.copy())
            setattr(self, attr_name, live_groups)
            temp = self.drawAllPolygons()
            self.refresh(temp)
            key = cv2.waitKey(1) & 0xFF
            if key == ESC:
                return False
            elif key == ord('s') or key == ord('S'):
                print("\n[SKIP] Skipping this TIFF file.")
                self.skip_current_file = True
                return False
            elif key == ord('r') or key == ord('R'):
                self.resetAllPicks()
                print("Reset performed. Re-select CBD marks before continuing.")
                return False
            elif key == ord('u') or key == ord('U'):
                if current_group:
                    removed = current_group.pop()
                    print(f"Removed {feature_name} point at {removed} from current group.")
                elif groups:
                    removed_group = groups.pop()
                    print(f"Removed {feature_name} group with {len(removed_group)} point(s).")
            elif key == ord('d') or key == ord('D'):
                setattr(self, polarity_attr, 'dark')
                print(f"{feature_name} polarity set to DARK.")
            elif key == ord('b') or key == ord('B'):
                setattr(self, polarity_attr, 'bright')
                print(f"{feature_name} polarity set to BRIGHT.")
            elif key == 13:  # Enter
                if current_group:
                    groups.append(current_group.copy())
                    current_group.clear()
                    print(f"{feature_name} final group {len(groups)} closed.")
                if groups:
                    setattr(self, attr_name, groups.copy())
                    total_points = sum(len(g) for g in groups)
                    print(f"{feature_name} anchors locked: {len(groups)} group(s), {total_points} point(s), {getattr(self, polarity_attr)} polarity.")
                    return True
                print(f"Add at least one {feature_name} group before pressing Enter.")

    def _detect_anchor_trace(self, anchor_points, half_window: int, polarity: str):
        img_gray = cv2.cvtColor(self.sourceImage, cv2.COLOR_BGR2GRAY)
        h, w = img_gray.shape

        if not anchor_points:
            return [(float('nan'), float('nan'))] * w

        # Sort anchors by x and keep last y when duplicate x exists.
        anchors_sorted = sorted([(int(np.clip(x, 0, w - 1)), int(np.clip(y, 0, h - 1))) for x, y in anchor_points], key=lambda p: p[0])
        anchor_map = {}
        for ax, ay in anchors_sorted:
            anchor_map[ax] = ay
        ax_arr = np.array(sorted(anchor_map.keys()), dtype=float)
        ay_arr = np.array([anchor_map[int(ax)] for ax in ax_arr], dtype=float)

        x_axis = np.arange(w, dtype=float)
        guide_y = np.interp(x_axis, ax_arr, ay_arr)
        x_min = int(np.min(ax_arr))
        x_max = int(np.max(ax_arr))

        # Limit sudden jumps between adjacent columns while still allowing slope.
        max_step = int(max(4, self.anchor_tracking_max_step))
        slope_penalty = float(self.anchor_slope_penalty)
        guide_penalty = float(self.anchor_guide_penalty)

        def detect_at_column(x_col, y_center, y_guide):
            y0 = max(0, int(y_center) - half_window)
            y1 = min(h - 1, int(y_center) + half_window)
            column = img_gray[y0:y1 + 1, x_col]
            if column.size == 0:
                return None

            y_candidates = np.arange(y0, y1 + 1)
            dy = np.abs(y_candidates - int(y_center)).astype(np.float64)
            dg = np.abs(y_candidates - int(y_guide)).astype(np.float64)
            signal = column.astype(np.float64)

            # Continuity-aware objective: choose strong peak while discouraging big y jumps.
            if polarity == 'bright':
                score = (-signal) + slope_penalty * dy + guide_penalty * dg
            else:
                score = signal + slope_penalty * dy + guide_penalty * dg

            idx = int(np.argmin(score))
            return y0 + idx

        trace = [(float('nan'), float('nan'))] * w

        # Track left-to-right using guide curve plus continuity only within this group's x-range.
        prev_y = int(guide_y[x_min])
        for x in range(x_min, x_max + 1):
            if x in anchor_map:
                y_detect = int(anchor_map[x])
            else:
                y_center = int(round(0.6 * prev_y + 0.4 * guide_y[x]))
                y_detect = detect_at_column(x, y_center, guide_y[x])
                if y_detect is None:
                    trace[x] = (float('nan'), float('nan'))
                    continue
                y_detect = int(np.clip(y_detect, prev_y - max_step, prev_y + max_step))
            trace[x] = (x, y_detect)
            prev_y = y_detect

        return trace

    def _preview_trace_confirmation(self, prompt_text):
        print(prompt_text)
        print("  Keys: 'a' approve, '['/'']' window -/+, '-'/'=' step -/+, 's' skip, 'r' reset.")
        while True:
            temp = self.drawAllPolygons()
            self.refresh(temp)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('a') or key == ord('A'):
                return 'approve'
            elif key == ord('s') or key == ord('S'):
                print("\n[SKIP] Skipping this TIFF file.")
                self.skip_current_file = True
                return 'skip'
            elif key == ord('r') or key == ord('R'):
                self.resetAllPicks()
                print("Reset performed. Re-select CBD marks before continuing.")
                return 'reset'
            elif key == ord('['):
                self.anchor_search_half_window = max(15, self.anchor_search_half_window - 5)
                self._build_traces_from_anchors(half_window=self.anchor_search_half_window)
                print(f"Search half-window set to {self.anchor_search_half_window} px")
            elif key == ord(']'):
                self.anchor_search_half_window = min(200, self.anchor_search_half_window + 5)
                self._build_traces_from_anchors(half_window=self.anchor_search_half_window)
                print(f"Search half-window set to {self.anchor_search_half_window} px")
            elif key == ord('-'):
                self.anchor_tracking_max_step = max(4, self.anchor_tracking_max_step - 1)
                self._build_traces_from_anchors(half_window=self.anchor_search_half_window)
                print(f"Tracking max step set to {self.anchor_tracking_max_step} px/column")
            elif key == ord('=') or key == ord('+'):
                self.anchor_tracking_max_step = min(80, self.anchor_tracking_max_step + 1)
                self._build_traces_from_anchors(half_window=self.anchor_search_half_window)
                print(f"Tracking max step set to {self.anchor_tracking_max_step} px/column")

    def _split_valid_segments(self, trace):
        segments = []
        current = []
        for x, y in trace:
            if np.isnan(x) or np.isnan(y):
                if len(current) >= 2:
                    segments.append(current)
                current = []
            else:
                current.append((int(x), int(y)))
        if len(current) >= 2:
            segments.append(current)
        return segments

    def _smooth_segment_local_poly(self, segment, window_size=11):
        if len(segment) < 5:
            return segment

        half = window_size // 2
        xs = np.array([p[0] for p in segment], dtype=float)
        ys = np.array([p[1] for p in segment], dtype=float)
        y_out = ys.copy()

        for i in range(len(segment)):
            left = max(0, i - half)
            right = min(len(segment), i + half + 1)
            xw = xs[left:right]
            yw = ys[left:right]
            if len(xw) < 3:
                continue
            deg = min(2, len(xw) - 1)
            try:
                coeffs = np.polyfit(xw, yw, deg)
                y_out[i] = np.polyval(coeffs, xs[i])
            except Exception:
                pass

        h = self.sourceImage.shape[0]
        return [(int(xs[i]), int(np.clip(y_out[i], 0, h - 1))) for i in range(len(segment))]

    def _build_traces_from_anchors(self, half_window=60):
        configs = [
            (self.polyDrawerSurface, self.surface_anchors, self.surface_polarity, 'Surface'),
            (self.polyDrawerBed, self.bed_anchors, self.bed_polarity, 'Bed'),
            (self.polyDrawerTransmitted, self.transmitted_anchors, self.transmitted_polarity, 'Transmitted')
        ]

        for drawer, anchor_groups, polarity, name in configs:
            if not anchor_groups:
                drawer.preview_lines = []
                drawer.polygons = []
                continue

            all_segments = []
            for group in anchor_groups:
                if not group:
                    continue
                raw = self._detect_anchor_trace(anchor_points=group, half_window=half_window, polarity=polarity)
                segments = self._split_valid_segments(raw)
                smoothed = [self._smooth_segment_local_poly(seg) for seg in segments]
                all_segments.extend(smoothed)

            drawer.preview_lines = all_segments
            drawer.polygons = [seg[:] for seg in all_segments]

            print(f"{name} trace generated with {len(all_segments)} segment(s) across {len(anchor_groups)} anchor group(s).")

    def runInteractiveProcess(self):
        """
        Anchor-based workflow:
          0. CBD selection
                    1. Grouped anchors per feature (surface/bed/transmitted)
                    2. Trace preview and approval
                    3. Save CSV and continue to calpip flow
        """
        while True:  # outer loop allows full reset back to Phase 0
            self.refresh(self.sourceImage)

            print("Phase 0: Select CBD marks. Left-click to add points; right-click when done.")
            print("  Press 's' to skip this file, 'r' to reset all picks.")
            # Reset CBD selector so Phase 0 is re-entered cleanly after a reset
            self.cbdSelector.done = False
            self.cbdSelector.points = []
            cv2.setMouseCallback(self.windowName, self.cbdSelector.on_mouse)
            while not self.cbdSelector.done:
                temp = self.drawAllPolygons()
                self.refresh(temp)
                key = cv2.waitKey(1) & 0xFF
                if key == ESC:
                    return
                elif key == ord('s') or key == ord('S'):
                    print("\n[SKIP] Skipping this TIFF file.")
                    self.skip_current_file = True
                    return
                elif key == ord('r') or key == ord('R'):
                    self.resetAllPicks()

            self.cbd_points = self.cbdSelector.points.copy()
            print("CBD selection complete.")
            print("Phase 1: Anchor-based detection. Pick one or more anchors for each trace (use inflection points).")
            if not self._pick_feature_anchors("Surface", "surface_anchors", "surface_polarity"):
                if self.skip_current_file:
                    return
                continue  # reset — loop back to Phase 0
            if not self._pick_feature_anchors("Bed", "bed_anchors", "bed_polarity"):
                if self.skip_current_file:
                    return
                continue
            if not self._pick_feature_anchors("Transmitted", "transmitted_anchors", "transmitted_polarity"):
                if self.skip_current_file:
                    return
                continue

            print("Phase 2: Trace preview.")
            self._build_traces_from_anchors(half_window=self.anchor_search_half_window)
            final_status = self._preview_trace_confirmation("Review trace preview. Approve to save.")
            if final_status == 'approve':
                break
            elif final_status == 'skip':
                return
            else:  # 'reset' — resetAllPicks already called inside _preview_trace_confirmation
                continue  # loop back to Phase 0

        self.drawAverageLinesAndSave()
        cv2.destroyWindow(self.windowName)

# ====================================

class PolygonDrawer:
    """
    Container for per-feature trace state and helper methods.
    Legacy name kept for compatibility with existing code paths.
    """
    def __init__(self, window: Window, colour: tuple, DEBUG=False):
        self.DEBUG = DEBUG
        if self.DEBUG:
            print("Creating PolygonDrawer for", window.windowName)
        self.window = window
        self.colour = colour
        self.done = False
        self.current_points = []  # Control points for the segment currently being placed
        self.polygons = []  # List of completed control-point sets (one per segment)
        self.preview_lines = []  # Smoothed curve per completed segment

    def on_mouse(self, event, x, y, buttons, user_param):
        if self.done:
            return
        # Convert fixed window coordinates to original image coordinates
        orig_x = int(x * self.window.scale_x)
        orig_y = int(y * self.window.scale_y)
        if event == cv2.EVENT_MOUSEMOVE:
            pass  # Could update a preview position if desired.
        elif event == cv2.EVENT_LBUTTONDOWN:
            if self.DEBUG:
                print(f"PolygonDrawer: Adding point ({orig_x}, {orig_y})")
            self.current_points.append((orig_x, orig_y))
        elif event == cv2.EVENT_RBUTTONDOWN:
            if len(self.current_points) >= 2:
                # Finish current segment and start a new one
                self.polygons.append(self.current_points.copy())
                print(f"Segment {len(self.polygons)} completed. Left-click to start next segment, or press Enter when done.")
                # Compute and store smoothed curve immediately for live preview
                preview = self.computeSmoothedLine(self.current_points.copy())
                self.preview_lines.append(preview)
                self.current_points = []
            else:
                print("Need at least 2 points for a segment.")

    def computeSmoothedLine(self, points):
        """Generate a gentle trace from control points without aggressive over-smoothing."""
        try:
            if len(points) < 2:
                return [(int(p[0]), int(p[1])) for p in points]

            # Sort by x coordinate
            pts_sorted = sorted(points, key=lambda p: p[0])
            xs = np.array([p[0] for p in pts_sorted], dtype=float)
            ys = np.array([p[1] for p in pts_sorted], dtype=float)

            # Remove duplicate x values (keep last occurrence)
            _, unique_idx = np.unique(xs, return_index=True)
            xs = xs[unique_idx]
            ys = ys[unique_idx]

            if len(xs) < 2:
                return [(int(xs[0]), int(ys[0]))]

            # Dense x range: one output point per pixel column
            x_dense = np.arange(int(xs[0]), int(xs[-1]) + 1, dtype=float)

            # Piecewise-linear interpolation follows picks closely and avoids polynomial bowing.
            y_dense = np.interp(x_dense, xs, ys)

            # Apply a very light smooth pass only when enough points are present.
            if len(xs) >= 4 and len(y_dense) >= 5:
                kernel = np.array([1.0, 2.0, 1.0], dtype=float) / 4.0
                y_smooth = y_dense.copy()
                y_smooth[1:-1] = np.convolve(y_dense, kernel, mode='same')[1:-1]

                # Preserve exact endpoints and anchor locations at control-point x's.
                y_smooth[0] = y_dense[0]
                y_smooth[-1] = y_dense[-1]
                for x_anchor, y_anchor in zip(xs, ys):
                    idx = int(round(x_anchor - x_dense[0]))
                    if 0 <= idx < len(y_smooth):
                        y_smooth[idx] = y_anchor
                y_dense = y_smooth

            return [(int(x), int(y)) for x, y in zip(x_dense, y_dense)]
        except Exception as e:
            if self.DEBUG:
                print(f"Smoothing failed: {e}")
            return [(int(p[0]), int(p[1])) for p in points]

    def createAvgLine(self, detail: int, area: str):
        """
        Return trace points from the user-placed and smoothed segment curves.
        If CBDs are present, output one point per CBD x-position by sampling
        the nearest smooth curve. Otherwise return all smoothed points with
        NaN separators between segments.
        """
        if self.DEBUG:
            print("Assembling trace for", area)

        if not self.preview_lines:
            return []

        # Build a combined x -> y lookup from all smoothed segments
        x_to_y = {}
        for seg in self.preview_lines:
            for (x, y) in seg:
                x_to_y[int(x)] = y

        result = []

        if self.window.cbd_boarders:
            sorted_boarders = sorted(self.window.cbd_boarders, key=lambda p: p[0])
            # Compute inter-CBD spacing to set a sensible search tolerance
            if len(sorted_boarders) > 1:
                spacings = [abs(sorted_boarders[i+1][0] - sorted_boarders[i][0])
                            for i in range(len(sorted_boarders) - 1)]
                tolerance = min(spacings) // 2
            else:
                tolerance = 100

            for boarder in sorted_boarders:
                bx = int(boarder[0])
                # Find the closest x in our smoothed curve data
                if x_to_y:
                    closest_x = min(x_to_y.keys(), key=lambda x: abs(x - bx))
                    if abs(closest_x - bx) <= tolerance:
                        result.append((bx, int(x_to_y[closest_x])))
                    else:
                        result.append((float('nan'), float('nan')))
                else:
                    result.append((float('nan'), float('nan')))
        else:
            # No CBDs: return all smoothed points, NaN-separated between segments
            for i, seg in enumerate(self.preview_lines):
                result.extend(seg)
                if i < len(self.preview_lines) - 1:
                    result.append((float('nan'), float('nan')))

        if self.DEBUG:
            print("Trace points:", result)

        return result

    def undo_polygon(self):
        """Remove the last completed segment, or discard in-progress points."""
        if self.current_points:
            self.current_points = []
            print("In-progress points discarded.")
        elif self.polygons:
            self.polygons.pop()
            if self.preview_lines:
                self.preview_lines.pop()
            print(f"Segment removed. {len(self.polygons)} segment(s) remaining.")
        else:
            print("Nothing to undo.")

    def get_all_points(self):
        """Returns all points from all completed polygons."""
        all_points = []
        for polygon in self.polygons:
            all_points.extend(polygon)
        # Include current polygon if it has points
        if self.current_points:
            all_points.extend(self.current_points)
        return all_points

    def has_polygons(self):
        """Returns True if there are completed polygons."""
        return len(self.polygons) > 0

# ====================================

class cbdSelector:
    """
    Selects CBD boarders by recording left clicks; right click marks completion.
    """
    def __init__(self, window: Window, colour: tuple, DEBUG=False):
        self.DEBUG = DEBUG
        if self.DEBUG:
            print("Creating cbdSelector for", window.windowName)
        self.window = window
        self.colour = colour
        self.done = False
        self.points = []  # store the CBD points as vertical boarders

    def on_mouse(self, event, x, y, flags, param):
        if self.done:
            return
        # Convert fixed window coordinates to original image coordinates
        orig_x = int(x * self.window.scale_x)
        orig_y = int(y * self.window.scale_y)
        if event == cv2.EVENT_MOUSEMOVE:
            pass
        elif event == cv2.EVENT_LBUTTONDOWN:
            if self.DEBUG:
                print(f"cbdSelector: Adding point ({orig_x}, {orig_y})")
            self.points.append((orig_x, orig_y))
            self.window.cbd_boarders.append((orig_x, orig_y))
            self.window.cbd_boarders.sort(key=lambda p: p[0])
            # Immediately refresh the window display after a click
            self.window.refresh(self.window.drawAllPolygons())
        elif event == cv2.EVENT_RBUTTONDOWN:
            if len(self.points) >= 1:
                self.done = True
            else:
                print("Select at least one CBD border point.")

# ====================================

class RadarViewer:
    """
    Radar image viewer with horizontal scrolling to examine for calpips
    Based on calpippickerv3.py viewing interface
    """
    def __init__(self, radar_window):
        self.radar_window = radar_window
        self.has_calpips = None
        self.done = False
        
        # Display parameters
        self.WINDOW_W_PX = 800
        self.WINDOW_H_PX = 600
        self.DPI = 100
        self.VIEW_W_PX = 800
        
        # Image data
        self.arr_orig = radar_window.sourceImage
        self.arr = self.enhance_image(self.arr_orig)
        self.view_w = min(self.VIEW_W_PX, self.arr.shape[1])
        self.left = 0
        self.image_artist = None
        
        # Create the viewer interface
        self.create_viewer_window()
    
    def enhance_image(self, arr):
        """Simple enhancement for better calpip visibility"""
        # Convert to float and normalize
        arr_float = arr.astype(np.float32)
        
        # Percentile stretch (1-99%)
        p_lo, p_hi = np.percentile(arr_float, [1, 99])
        if p_hi > p_lo:
            arr_float = (arr_float - p_lo) / (p_hi - p_lo)
            arr_float = np.clip(arr_float, 0, 1)
        
        # Convert to uint8
        return (arr_float * 255).astype(np.uint8)
    
    def create_viewer_window(self):
        """Create the radar viewer interface"""
        matplotlib.use("TkAgg")
        
        self.root = tk.Tk()
        self.root.title("Examine Radargram for Calpips")
        self.root.geometry("1000x700")
        
        # Layout: left controls, right plot
        control_frame = tk.Frame(self.root)
        control_frame.pack(side=tk.LEFT, padx=12, pady=12)
        plot_frame = tk.Frame(self.root)
        plot_frame.pack(side=tk.RIGHT, padx=12, pady=12)
        
        # Instructions
        tk.Label(control_frame, text="Examine for Calpips", font=("Arial", 14, "bold")).pack(pady=10)
        tk.Label(control_frame, text="Use the slider below to scroll\nhorizontally and look for\nregular horizontal lines\n(calpips)", 
                wraplength=200, justify="left").pack(pady=5)
        
        # Buttons
        button_frame = tk.Frame(control_frame)
        button_frame.pack(pady=20)
        
        self.yes_btn = tk.Button(button_frame, text="Has Calpips", command=self.has_calpips_yes, 
                                bg="lightgreen", width=15, height=2)
        self.yes_btn.pack(pady=5)
        
        self.no_btn = tk.Button(button_frame, text="No Calpips", command=self.has_calpips_no, 
                               bg="lightcoral", width=15, height=2)
        self.no_btn.pack(pady=5)
        
        # Figure setup
        figsize = (self.WINDOW_W_PX / self.DPI, self.WINDOW_H_PX / self.DPI)
        self.fig, self.ax = plt.subplots(figsize=figsize, dpi=self.DPI)
        self.ax.set_axis_off()
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
        # Horizontal slider for scrolling
        self.ax_slider = self.fig.add_axes([0.12, 0.05, 0.62, 0.03], facecolor="lightgoldenrodyellow")
        max_left = max(0, self.arr.shape[1] - self.view_w)
        self.slider = Slider(self.ax_slider, "Scroll", 0, max_left, valinit=0, valstep=10)
        self.slider.on_changed(self.on_slider)
        
        # Navigation buttons
        self.ax_left = self.fig.add_axes([0.83, 0.05, 0.05, 0.03])
        self.ax_right = self.fig.add_axes([0.90, 0.05, 0.05, 0.03])
        self.btn_left = Button(self.ax_left, "←")
        self.btn_right = Button(self.ax_right, "→")
        self.btn_left.on_clicked(lambda e: self.slider.set_val(max(0, self.slider.val - 50)))
        self.btn_right.on_clicked(lambda e: self.slider.set_val(min(self.slider.valmax, self.slider.val + 50)))
        
        # Draw initial view
        self.draw_view()
        filename = os.path.basename(self.radar_window.image.imgPath)
        self.ax.set_title(f"Examine: {filename}", fontsize=12, pad=10)
        
        # Make window modal
        self.root.transient()
        self.root.grab_set()
    
    def draw_view(self):
        """Draw the current view slice"""
        L = int(self.left)
        R = int(min(L + self.view_w, self.arr.shape[1]))
        view = self.arr[:, L:R]
        
        if self.image_artist is None:
            self.image_artist = self.ax.imshow(view, cmap="gray", aspect="auto", interpolation="nearest")
        else:
            self.image_artist.set_data(view)
        
        self.ax.set_xlim(0, R - L)
        self.ax.set_ylim(view.shape[0], 0)
        self.canvas.draw_idle()
    
    def on_slider(self, val):
        """Handle slider movement"""
        self.left = int(val)
        self.draw_view()
    
    def has_calpips_yes(self):
        """User confirmed calpips are present"""
        self.has_calpips = True
        self.done = True
        self.root.quit()
    
    def has_calpips_no(self):
        """User confirmed no calpips present"""
        self.has_calpips = False
        self.done = True
        self.root.quit()
    
    def run(self):
        """Run the viewer and return True if calpips present, False otherwise"""
        self.root.mainloop()
        self.root.destroy()
        return self.has_calpips

# ====================================

class CalpipPicker:
    """
    Calpip picker interface based on simplified version of calpippickerv3.py
    """
    def __init__(self, radar_window):
        self.radar_window = radar_window
        self.result_pixel_distance = None
        self.result_y_lines = []
        self.picks_y = []
        self.accept_clicks = False
        self.accept_bound_click = False
        self.end_bound_y = None
        self.done = False
        self.no_calpips = False
        
        # Display parameters
        self.WINDOW_W_PX = 800
        self.WINDOW_H_PX = 600
        self.DPI = 100
        self.VIEW_W_PX = 800
        
        # Image data for matplotlib display
        self.arr_orig = radar_window.sourceImage
        self.arr = self.enhance_image(self.arr_orig)
        self.view_w = min(self.VIEW_W_PX, self.arr.shape[1])
        self.left = 0
        self.image_artist = None
        self.pick_lines = []
        self.interp_lines = []
        self.bound_line = None
        
        # Create the calpip picker window
        self.create_calpip_window()
    
    def enhance_image(self, arr):
        """Simple enhancement for better calpip visibility"""
        # Convert to float and normalize
        arr_float = arr.astype(np.float32)
        
        # Percentile stretch (1-99%)
        p_lo, p_hi = np.percentile(arr_float, [1, 99])
        if p_hi > p_lo:
            arr_float = (arr_float - p_lo) / (p_hi - p_lo)
            arr_float = np.clip(arr_float, 0, 1)
        
        # Convert to uint8
        return (arr_float * 255).astype(np.uint8)
    
    def create_calpip_window(self):
        """Create the calpip picker GUI"""
        matplotlib.use("TkAgg")
        
        self.root = tk.Tk()
        self.root.title("Calpip Picker")
        self.root.geometry("1200x700")
        
        # Layout: left controls, right plot
        control_frame = tk.Frame(self.root)
        control_frame.pack(side=tk.LEFT, padx=12, pady=12)
        plot_frame = tk.Frame(self.root)
        plot_frame.pack(side=tk.RIGHT, padx=12, pady=12)
        
        # Instructions
        tk.Label(control_frame, text="Calpip Processing", font=("Arial", 14, "bold")).pack(pady=10)
        tk.Label(control_frame, text="1. Select end bound\n2. Pick 4 calpips\n3. Click Done", 
                wraplength=200, justify="left").pack(pady=5)
        
        # Buttons
        button_frame = tk.Frame(control_frame)
        button_frame.pack(pady=20)
        
        self.bound_btn = tk.Button(button_frame, text="Select End Bound", command=self.start_bound_selection, 
                                  bg="lightyellow", width=15)
        self.bound_btn.pack(pady=5)
        
        self.pick_btn = tk.Button(button_frame, text="Pick Calpips", command=self.start_picking, 
                                 bg="lightblue", width=15, state="disabled")
        self.pick_btn.pack(pady=5)
        
        self.done_btn = tk.Button(button_frame, text="Done", command=self.finish_picking, 
                                 bg="lightgreen", width=15)
        self.done_btn.pack(pady=5)
        
        self.no_calpips_btn = tk.Button(button_frame, text="No Calpips", command=self.no_calpips_selected, 
                                       bg="lightcoral", width=15)
        self.no_calpips_btn.pack(pady=5)
        
        self.reset_btn = tk.Button(button_frame, text="Reset Picks", command=self.reset_picks,
                                   bg="lightyellow", width=15)
        self.reset_btn.pack(pady=5)
        
        # Status label
        self.status_label = tk.Label(control_frame, text="Click 'Select End Bound' to start", fg="blue",
                                    wraplength=200, justify="left")
        self.status_label.pack(pady=10)
        
        # Figure setup
        figsize = (self.WINDOW_W_PX / self.DPI, self.WINDOW_H_PX / self.DPI)
        self.fig, self.ax = plt.subplots(figsize=figsize, dpi=self.DPI)
        self.ax.set_axis_off()
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
        # Horizontal slider for scrolling
        self.ax_slider = self.fig.add_axes([0.12, 0.05, 0.62, 0.03], facecolor="lightgoldenrodyellow")
        max_left = max(0, self.arr.shape[1] - self.view_w)
        self.slider = Slider(self.ax_slider, "Scroll", 0, max_left, valinit=0, valstep=10)
        self.slider.on_changed(self.on_slider)
        
        # Navigation buttons
        self.ax_left = self.fig.add_axes([0.83, 0.05, 0.05, 0.03])
        self.ax_right = self.fig.add_axes([0.90, 0.05, 0.05, 0.03])
        self.btn_left = Button(self.ax_left, "←")
        self.btn_right = Button(self.ax_right, "→")
        self.btn_left.on_clicked(lambda e: self.slider.set_val(max(0, self.slider.val - 50)))
        self.btn_right.on_clicked(lambda e: self.slider.set_val(min(self.slider.valmax, self.slider.val + 50)))
        
        # Set up matplotlib click handler
        self.fig.canvas.mpl_connect("button_press_event", self.on_matplotlib_click)
        
        # Draw initial view
        self.draw_view()
        filename = os.path.basename(self.radar_window.image.imgPath)
        self.ax.set_title(f"Pick Calpips: {filename}", fontsize=12, pad=10)
        
        # Make window modal
        self.root.transient()
        self.root.grab_set()
    
    def draw_view(self):
        """Draw the current view slice"""
        L = int(self.left)
        R = int(min(L + self.view_w, self.arr.shape[1]))
        view = self.arr[:, L:R]
        
        if self.image_artist is None:
            self.image_artist = self.ax.imshow(view, cmap="gray", aspect="auto", interpolation="nearest")
        else:
            self.image_artist.set_data(view)
        
        self.ax.set_xlim(0, R - L)
        self.ax.set_ylim(view.shape[0], 0)
        self.canvas.draw_idle()
    
    def on_slider(self, val):
        """Handle slider movement"""
        self.left = int(val)
        self.draw_view()
        self.redraw_all_lines()
    
    def redraw_all_lines(self):
        """Redraw all lines after view change"""
        # Clear existing line artists
        for line in self.pick_lines + self.interp_lines:
            if line in self.ax.lines:
                line.remove()
        self.pick_lines = []
        self.interp_lines = []
        if self.bound_line and self.bound_line in self.ax.lines:
            self.bound_line.remove()
            self.bound_line = None
        
        # Redraw bound line
        if self.end_bound_y is not None:
            y_disp = self.end_bound_y
            x0, x1 = self.ax.get_xlim()
            self.bound_line, = self.ax.plot([x0, x1], [y_disp, y_disp], color="red", linewidth=3)
            self.ax.text(x0 + 10, y_disp - 10, "End Bound", color="red", fontweight="bold")
        
        # Redraw pick lines
        for pick_y in self.picks_y:
            y_disp = pick_y
            x0, x1 = self.ax.get_xlim()
            line, = self.ax.plot([x0, x1], [y_disp, y_disp], color="yellow", linewidth=2)
            self.pick_lines.append(line)
        
        # Redraw interpolated lines
        for i, y_line in enumerate(self.result_y_lines):
            y_disp = y_line
            x0, x1 = self.ax.get_xlim()
            line, = self.ax.plot([x0, x1], [y_disp, y_disp], color="cyan", linewidth=1, linestyle="--")
            self.interp_lines.append(line)
            
            # Add time label
            time_us = i * 2
            self.ax.text(x1 - 60, y_disp - 5, f"{time_us}μs", color="cyan", fontsize=8)
        
        self.canvas.draw_idle()
        
    def start_bound_selection(self):
        """Start the end bound selection process"""
        self.accept_bound_click = True
        self.accept_clicks = False
        self.status_label.config(text="Click on the image to select the end bound for y-axis", fg="orange")
        self.bound_btn.config(state="disabled")
        
    def start_picking(self):
        """Start the calpip picking process"""
        self.picks_y = []
        self.accept_clicks = True
        self.accept_bound_click = False
        self.status_label.config(text=f"Click on 4 calpips (picked: {len(self.picks_y)}/4)", fg="green")
        self.pick_btn.config(state="disabled")
        
    def on_matplotlib_click(self, event):
        """Handle matplotlib click events"""
        if event.inaxes != self.ax or event.ydata is None:
            return
            
        # Convert matplotlib coordinates to original image coordinates
        y_click = event.ydata
        
        if self.accept_bound_click:
            # Setting end bound
            self.end_bound_y = y_click
            self.accept_bound_click = False
            
            # Draw bound line
            x0, x1 = self.ax.get_xlim()
            if self.bound_line and self.bound_line in self.ax.lines:
                self.bound_line.remove()
            self.bound_line, = self.ax.plot([x0, x1], [y_click, y_click], color="red", linewidth=3)
            self.ax.text(x0 + 10, y_click - 10, "End Bound", color="red", fontweight="bold")
            
            self.canvas.draw_idle()
            
            self.status_label.config(text="End bound set! Now click 'Pick Calpips'", fg="blue")
            self.bound_btn.config(state="normal")
            self.pick_btn.config(state="normal")
            return
            
        if self.accept_clicks:
            # Picking calpips
            self.picks_y.append(y_click)
            
            # Draw pick line
            x0, x1 = self.ax.get_xlim()
            line, = self.ax.plot([x0, x1], [y_click, y_click], color="yellow", linewidth=2)
            self.pick_lines.append(line)
            
            self.canvas.draw_idle()
            
            self.status_label.config(text=f"Click on 4 calpips (picked: {len(self.picks_y)}/4)")
            
            if len(self.picks_y) >= 4:
                self.accept_clicks = False
                self.calculate_interpolation()
                self.pick_btn.config(state="normal")
                self.status_label.config(text="4 calpips picked! Click Done to continue.")
    
    def calculate_interpolation(self):
        """Calculate calpip spacing and interpolate lines following the yellow picks exactly"""
        if len(self.picks_y) < 2:
            return
            
        # Sort picks and calculate average spacing
        sorted_picks = sorted(self.picks_y)
        spacings = [sorted_picks[i+1] - sorted_picks[i] for i in range(len(sorted_picks)-1)]
        avg_spacing = np.mean(spacings)
        self.result_pixel_distance = avg_spacing
        
        # End at user-selected bound (never go past this)
        if self.end_bound_y is not None:
            end_y = self.end_bound_y
        else:
            # Fallback: use bottom of radar data area
            end_y = self.arr.shape[0] - 50
        
        # Use the first yellow pick as the reference grid point - no shifting
        first_pick_y = sorted_picks[0]
        
        # Generate grid lines using the exact yellow pick spacing pattern
        self.result_y_lines = []
        
        # Start from a position that ensures we cover the entire range
        # Go far enough up to cover any area above the first pick
        start_grid_index = int(np.floor((0 - first_pick_y) / avg_spacing)) - 5
        
        # Generate lines using the yellow pick spacing pattern exactly
        current_index = start_grid_index
        while True:
            current_y = first_pick_y + (current_index * avg_spacing)
            
            # Stop if we've gone past the end bound
            if current_y > end_y:
                break
                
            # Only include lines that are within reasonable bounds
            if current_y >= 0:
                self.result_y_lines.append(current_y)
                
            current_index += 1
        
        # Update status with average spacing
        self.status_label.config(text=f"Average spacing: {avg_spacing:.1f} pixels, {len(self.result_y_lines)} lines")
        
        # Draw interpolated lines
        self.draw_interpolated_lines()
    
    def draw_interpolated_lines(self):
        """Draw clean horizontal calpip lines for picking"""
        # Clear existing interpolated lines
        for line in self.interp_lines:
            if line in self.ax.lines:
                line.remove()
        self.interp_lines = []
        
        # Draw clean horizontal calpip lines in cyan (for picking reference only)
        x0, x1 = self.ax.get_xlim()
        for y_line in self.result_y_lines:
            if 0 <= y_line < self.arr.shape[0]:
                line, = self.ax.plot([x0, x1], [y_line, y_line], color="cyan", linewidth=1, linestyle="--", alpha=0.7)
                self.interp_lines.append(line)
        
        self.canvas.draw_idle()
    
    def finish_picking(self):
        """Finish the calpip picking process"""
        if self.end_bound_y is None:
            messagebox.showwarning("Missing End Bound", "Please select the end bound first!")
        elif len(self.picks_y) < 4:
            messagebox.showwarning("Incomplete", "Please pick 4 calpips!")
        elif not self.result_pixel_distance:
            messagebox.showwarning("No Calculation", "Calpip calculation failed!")
        else:
            self.done = True
            self.root.quit()
    
    def reset_picks(self):
        """Clear all calpip picks and interpolated lines so the user can repick."""
        # Clear pick data
        self.picks_y = []
        self.result_pixel_distance = None
        self.result_y_lines = []
        self.accept_clicks = False

        # Remove yellow pick lines from plot
        for line in self.pick_lines:
            if line in self.ax.lines:
                line.remove()
        self.pick_lines = []

        # Remove cyan interpolated lines from plot
        for line in self.interp_lines:
            if line in self.ax.lines:
                line.remove()
        self.interp_lines = []

        self.canvas.draw_idle()

        # Re-enable Pick Calpips button and update status
        self.pick_btn.config(state="normal")
        self.status_label.config(text="Picks cleared. Click 'Pick Calpips' to repick.", fg="blue")

    def no_calpips_selected(self):
        """User selected no calpips"""
        self.no_calpips = True
        self.done = True
        self.root.quit()
    
    def run(self):
        """Run the calpip picker and return results"""
        self.root.mainloop()
        self.root.destroy()
        
        if self.no_calpips:
            return None
        elif self.done and self.result_pixel_distance:
            return (self.result_pixel_distance, self.result_y_lines)
        else:
            return None

# ====================================

class App:
    """
    Overall manager for processing a folder of TIFF files.
    """
    def __init__(self, path: str, batchSize:int, csv_directory: str = None, output_directory: str = None):
        self.path = path
        self.batchSize = batchSize
        self.csv_directory = csv_directory
        self.output_directory = output_directory
        self.dir_list = [f for f in os.listdir(path) if f.lower().endswith(".tiff") or f.lower().endswith(".tif")]
        if not self.dir_list:
            print("No TIFF files found in", path)
        
        # Global calpip state - persists across files
        self.calpip_pixel_distance = None
        self.calpip_y_lines = []
        # Files processed before first calpip is established are deferred here
        # and backfilled once a calpip reference becomes available.
        self.pending_windows_no_calpip = []

    def process(self, tiff):
        DEBUG = False
        DETAIL = 10  # detail for avg line
        imgPath = os.path.join(self.path, tiff)
        # Adjust crop values as desired.
        image = Image(imgPath, cropVals=(300, 2050), DEBUG=DEBUG)
        win = Window("Reference", image=image, detail=DETAIL, DEBUG=DEBUG, csv_directory=self.csv_directory)
        win.refresh(win.sourceImage)
        # Run interactive phases:
        win.runInteractiveProcess()
        
        # Check if user skipped this file
        if win.skip_current_file:
            print(f"[SKIPPED] {tiff}")
            cv2.destroyAllWindows()
            return None
        
        # After main processing, handle calpip processing
        self.processCalpips(win)
        
        return win.resultImage

    def processCalpips(self, window):
        """Handle calpip processing after main radar analysis."""
        print(f"Examining {os.path.basename(window.image.imgPath)} for calpips...")
        
        # Show radar viewer for user to examine the radargram
        viewer = RadarViewer(window)
        has_calpips = viewer.run()
        found_new_calpip = False
        
        if has_calpips:
            # Open calpip picker
            print("User confirmed calpips present - opening calpip picker...")
            calpip_result = self.openCalpipPicker(window)
            if calpip_result:
                self.calpip_pixel_distance, self.calpip_y_lines = calpip_result
                print(f"New calpip distance: {self.calpip_pixel_distance:.1f} pixels")
                found_new_calpip = True
            else:
                print("Calpip picker was closed or incomplete; keeping previous calpip state.")
        else:
            print("User confirmed no calpips - using existing calpip distance from previous file")
            if not (self.calpip_pixel_distance and self.calpip_y_lines):
                # No calpip reference exists yet. Defer this file and backfill later.
                self.pending_windows_no_calpip.append(window)
                print("No prior calpip available yet; deferring QC generation for this file.")
                return

        # First valid calpip found: backfill all files that were deferred earlier.
        if found_new_calpip and self.pending_windows_no_calpip:
            print(f"Backfilling {len(self.pending_windows_no_calpip)} earlier file(s) with first calpip reference...")
            for pending_window in self.pending_windows_no_calpip:
                self.applyCalpipToWindow(pending_window)
            self.pending_windows_no_calpip.clear()
        
        # Apply calpip data to current file and generate CSV/QC outputs.
        self.applyCalpipToWindow(window)

    def applyCalpipToWindow(self, window):
        """Apply current calpip state to a window and produce calpip-based CSV/QC outputs."""
        # Apply calpip data to current window and regenerate CSV with time calculations
        window.calpip_pixel_distance = self.calpip_pixel_distance
        window.calpip_y_lines = self.calpip_y_lines.copy() if self.calpip_y_lines else []
        window.regenerateCSVWithCalpips()

        # Update result image with y-axis lines and save high-quality QC PNG
        window.drawYAxisLines()

    def openCalpipPicker(self, window):
        """Open the calpip picker interface and return (pixel_distance, y_lines) or None."""
        print("Opening calpip picker interface...")
        
        # Create and run the calpip picker
        picker = CalpipPicker(window)
        result = picker.run()
        
        return result

    def save_high_dpi_image(self, image, save_path, dpi=500):
        """Save image with high DPI using PIL."""
        try:
            from PIL import Image as PILImage
            # Convert BGR to RGB
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # Create PIL image
            pil_image = PILImage.fromarray(rgb_image)
            # Save with high DPI
            pil_image.save(save_path, dpi=(dpi, dpi), optimize=True)
        except ImportError:
            # Fallback to OpenCV if PIL is not available
            cv2.imwrite(save_path, image)

    def find_last_processed_index(self):
        """Find the index of the last processed TIFF file based on existing CSVs in output directory."""
        if not self.output_directory or not os.path.exists(self.output_directory):
            return 0
        
        # Get all CSV files in output directory
        csv_files = [f for f in os.listdir(self.output_directory) if f.lower().endswith(".csv")]
        if not csv_files:
            return 0
        
        # Extract TIFF filenames from CSV names (remove .csv extension)
        processed_tiff_names = set(f[:-4] for f in csv_files)  # Remove .csv
        
        # Find the index of the last processed TIFF file
        last_index = -1
        for i, tiff_file in enumerate(self.dir_list):
            tiff_base = os.path.splitext(tiff_file)[0]
            if tiff_base in processed_tiff_names:
                last_index = i
        
        # Return the next index to process (or 0 if none found)
        return last_index + 1
    
    def launch(self, saveDirectory=""):
        anchorCounter = self.find_last_processed_index()
        print(f"TARGET: {len(self.dir_list)} files | Starting from index {anchorCounter}")
        if anchorCounter > 0:
            print(f"Resuming from file {anchorCounter}/{len(self.dir_list)} - {self.dir_list[anchorCounter] if anchorCounter < len(self.dir_list) else 'complete'}")
        while anchorCounter < len(self.dir_list):
            resImages = []
            for count in range(self.batchSize):
                if anchorCounter+count >= len(self.dir_list):
                    break
                currImage = self.dir_list[anchorCounter+count]
                result = self.process(currImage)
                if result is not None:  # Only add to resImages if not skipped
                    resImages.append(result)
                print("Processed image index:", anchorCounter+count)
            anchorCounter += self.batchSize

        if self.pending_windows_no_calpip:
            print(
                f"Warning: {len(self.pending_windows_no_calpip)} file(s) were processed without any calpip reference in this run; "
                "QC PNG generation was deferred but could not be completed because no calpip was ever selected."
            )

            # DEBUG: Stitching functionality commented out for now
            # if resImages:
            #     res = Image.stitch(resImages)
            #     imageIndex = anchorCounter - self.batchSize + 1
            #     save_path = os.path.join(saveDirectory, f"{imageIndex}.png")
            #     self.save_high_dpi_image(res, save_path, dpi=500)
            #     print("Saved stitched image to", save_path)

# ====================================

if __name__ == "__main__":
    import sys
    
    # Check if command-line arguments are provided
    if len(sys.argv) == 7:
        # Fancy labeled mode: python TERRA.py tiff "path" output "path" nav "path"
        try:
            # Parse labeled arguments
            args = {}
            for i in range(1, len(sys.argv), 2):
                label = sys.argv[i].lower()
                path = sys.argv[i + 1]
                args[label] = path
            
            # Validate we have all required labels
            if 'tiff' not in args or 'output' not in args or 'nav' not in args:
                raise ValueError("Missing required arguments")
                
            tiff_directory = args['tiff']
            output_directory = args['output']
            navigation_directory = args['nav']
            
        except (IndexError, ValueError):
            print("Error: Invalid argument format.")
            print("Usage: python TERRA.py tiff \"path/to/tiffs\" output \"path/to/output\" nav \"path/to/nav.csv\"")
            sys.exit(1)
        
        # Validate directories exist
        if not os.path.exists(tiff_directory):
            print(f"Error: TIFF directory does not exist: {tiff_directory}")
            sys.exit(1)
        
        if not os.path.exists(navigation_directory):
            print(f"Error: Navigation directory does not exist: {navigation_directory}")
            sys.exit(1)
        
        # Create output directory if it doesn't exist
        if not os.path.exists(output_directory):
            try:
                os.makedirs(output_directory)
                print(f"Created output directory: {output_directory}")
            except Exception as e:
                print(f"Error creating output directory: {e}")
                sys.exit(1)
        
        print(f"TERRA - Tracing & Extraction of Radar Reflections for Analysis")
        print(f"TIFF Directory: {tiff_directory}")
        print(f"Output Directory: {output_directory}")
        print(f"Navigation Directory: {navigation_directory}")
        print("=" * 60)
        
        # Change to output directory so CSV files are saved there
        original_cwd = os.getcwd()
        os.chdir(output_directory)
        
        try:
            app = App(tiff_directory, batchSize=1, csv_directory=navigation_directory, output_directory=output_directory)
            app.launch(output_directory)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        finally:
            # Restore original working directory
            os.chdir(original_cwd)
            
    elif len(sys.argv) == 4:
        # Legacy mode: python TERRA.py <tiff_directory> <output_directory> <navigation_directory>
        tiff_directory = sys.argv[1]
        output_directory = sys.argv[2]
        navigation_directory = sys.argv[3]
        
        # Validate directories exist
        if not os.path.exists(tiff_directory):
            print(f"Error: TIFF directory does not exist: {tiff_directory}")
            sys.exit(1)
        
        if not os.path.exists(navigation_directory):
            print(f"Error: Navigation directory does not exist: {navigation_directory}")
            sys.exit(1)
        
        # Create output directory if it doesn't exist
        if not os.path.exists(output_directory):
            try:
                os.makedirs(output_directory)
                print(f"Created output directory: {output_directory}")
            except Exception as e:
                print(f"Error creating output directory: {e}")
                sys.exit(1)
        
        print(f"TERRA - Tracing & Extraction of Radar Reflections for Analysis")
        print(f"TIFF Directory: {tiff_directory}")
        print(f"Output Directory: {output_directory}")
        print(f"Navigation Directory: {navigation_directory}")
        print("=" * 60)
        
        # Change to output directory so CSV files are saved there
        original_cwd = os.getcwd()
        os.chdir(output_directory)
        
        try:
            app = App(tiff_directory, batchSize=1, csv_directory=navigation_directory, output_directory=output_directory)
            app.launch(output_directory)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        finally:
            # Restore original working directory
            os.chdir(original_cwd)
            
    else:
        # No arguments or wrong number - use default paths and show usage
        print("TERRA - Tracing & Extraction of Radar Reflections for Analysis")
        print("")
        print("Usage (Recommended):")
        print('  python TERRA.py tiff "path/to/tiffs" output "path/to/output" nav "path/to/nav.csv"')
        print("")
        print("Usage (Legacy):")
        print('  python TERRA.py "path/to/tiffs" "path/to/output" "path/to/nav.csv"')
        print("")
        print("Arguments:")
        print("  tiff   : Directory containing TIFF files to process")
        print("  output : Directory where CSV files and results will be saved")
        print("  nav    : Path to navigation CSV file (e.g., 125.csv)")
        print("")
        print("Example:")
        print('  python TERRA.py tiff "C:\\TIFFs\\F125" output "C:\\Results" nav "C:\\Nav\\125.csv"')
        print("")
        print("Running with default development paths...")
        
        # Use default development paths
        source = r"ZScope_TERRA\125"
        destination = r"ZScope_TERRA"
        csv_source = r"ZScope_TERRA\125.csv"

        original_cwd = os.getcwd()
        os.chdir(destination)
        
        try:
            app = App(source, batchSize=1, csv_directory=csv_source, output_directory=destination)
            app.launch(destination)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        finally:
            os.chdir(original_cwd)