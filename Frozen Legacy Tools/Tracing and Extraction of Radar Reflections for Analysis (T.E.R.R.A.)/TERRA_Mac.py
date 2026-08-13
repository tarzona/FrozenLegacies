"""
TERRA - Tracing & Extraction of Radar Reflections for Analysis

DESCRIPTION:
This tool processes radar TIFF files to extract surface and bed reflections from ice-penetrating 
radar data. It provides an interactive interface for manual digitization of radar reflections 
and exports results to CSV format with optional time-domain calculations using calpip analysis.

REQUIREMENTS:
- Python 3.x
- Required packages: opencv-python, numpy, pandas, matplotlib, tkinter, PIL, PyQt5

IF YOU DO NOT HAVE THE REQUIRED PACKAGES, FOLLOW THESE STEPS:
1. Install/Update pip: python -m ensurepip --upgrade
2. Update pip to latest: python -m pip install --upgrade pip
3. Install required packages: python -m pip install opencv-python numpy pandas matplotlib pillow PyQt5
4. Verify installation: python -c "import cv2, numpy, pandas, matplotlib, tkinter; print('All packages installed successfully!')"

NOTE FOR MAC USERS: If you encounter issues with tkinter, try using PyQt5 backend by ensuring PyQt5 is installed.

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
   b) Draw surface polygons - left-click to draw, right-click to finish polygon, Enter when all done
   c) Draw bed polygons - left-click to draw, right-click to finish polygon, Enter when all done  
   d) Draw transmitted pulse polygons - left-click to draw, right-click to finish polygon, Enter when all done
3. System examines radargram for calpips (calibration pulses) for time-domain analysis
4. Results are automatically saved as CSV files in the output directory
5. High-quality PNG images are generated with picked reflections and axes

OUTPUT FILES:
- [filename].csv: Contains digitized coordinates, lat/lon data, and time-domain calculations
- [filename].png: High-quality visualization of results with dual y-axes (depth/time)

FEATURES:
- Interactive polygon-based digitization
- Automatic averaging of reflections within polygons
- CBD coordinate extraction and lat/lon lookup
- Optional calpip-based time-domain analysis
- High-quality matplotlib figure generation
- Handles missing data with NaN values
- Supports batch processing of multiple files

AUTHOR: [Angelo Tarzona, dtarzona@gatech.edu]
VERSION: 2025.10.10
"""

import os
import sys
from copy import deepcopy
from functools import partial
import time
import csv
import pandas as pd
import matplotlib
# Force matplotlib to use Qt5Agg backend on macOS
if sys.platform == 'darwin':
    os.environ['MPLBACKEND'] = 'Qt5Agg'
    matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button

# PyQt5 imports instead of tkinter
try:
    from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QMessageBox, QFrame
    from PyQt5.QtCore import Qt, QTimer
    from PyQt5.QtGui import QFont
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
    QT_AVAILABLE = True
    print("Using PyQt5 backend")
except ImportError:
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg as FigureCanvas
    QT_AVAILABLE = False
    print("PyQt5 not available, falling back to matplotlib backends")
    
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
        
        # Fixed display size setup
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
        
        # Calpip processing attributes
        self.calpip_pixel_distance = None
        self.calpip_y_lines = []
        
        # Create window at fixed size with title "TERRA"
        cv2.namedWindow(self.windowName, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.windowName, fixedSize[0], fixedSize[1])
        # Move window to a consistent position on screen
        cv2.moveWindow(self.windowName, 100, 50)
 
        # Instantiate drawing helpers (they set their own mouse callbacks)
        self.polyDrawerSurface = PolygonDrawer(self, drawColourSurface, DEBUG=self.DEBUG)
        self.polyDrawerBed = PolygonDrawer(self, drawColourBed, DEBUG=self.DEBUG)
        self.polyDrawerTransmitted = PolygonDrawer(self, drawColourTransmitted, DEBUG=self.DEBUG)
        self.cbdSelector = cbdSelector(self, drawColourCBD, DEBUG=self.DEBUG)

    def refresh(self, image: np.ndarray):
        self.displayedImage = image
        # Resize image to fit the fixed window size for display
        display_image = cv2.resize(image, self.fixedSize)
        cv2.imshow(self.windowName, display_image)

    def filterImage(self, image: np.ndarray):
        # For now, simply return a copy; advanced filtering (blur/edge)
        return deepcopy(image)

    def drawAllPolygons(self):
        # First, resize the source image to the fixed display size
        display = cv2.resize(self.sourceImage, self.fixedSize)
        
        # Draw vertical CBD lines using scaled coordinates
        if self.cbdSelector.points:
            for point in self.cbdSelector.points:
                disp_x = int(point[0] / self.scale_x)
                disp_y = int(point[1] / self.scale_y)
                cv2.line(display, (disp_x, 0), (disp_x, self.fixedSize[1]), (0, 255, 255), 4)
                cv2.circle(display, (disp_x, disp_y), 8, (0, 255, 255), -1)
        
        # Draw the surface polygons (if any) using scaled coordinates
        for polygon in self.polyDrawerSurface.polygons:
            pts = np.array([(int(p[0] / self.scale_x), int(p[1] / self.scale_y)) for p in polygon])
            cv2.polylines(display, [pts], False, self.polyDrawerSurface.colour, 2)
        # Draw current surface polygon being drawn
        if self.polyDrawerSurface.current_points:
            pts = np.array([(int(p[0] / self.scale_x), int(p[1] / self.scale_y)) for p in self.polyDrawerSurface.current_points])
            cv2.polylines(display, [pts], False, self.polyDrawerSurface.colour, 1)
        
        # Draw the bed polygons (if any) using scaled coordinates
        for polygon in self.polyDrawerBed.polygons:
            pts = np.array([(int(p[0] / self.scale_x), int(p[1] / self.scale_y)) for p in polygon])
            cv2.polylines(display, [pts], False, self.polyDrawerBed.colour, 2)
        # Draw current bed polygon being drawn
        if self.polyDrawerBed.current_points:
            pts = np.array([(int(p[0] / self.scale_x), int(p[1] / self.scale_y)) for p in self.polyDrawerBed.current_points])
            cv2.polylines(display, [pts], False, self.polyDrawerBed.colour, 1)
        
        # Draw the transmitted pulse polygons (if any) using scaled coordinates
        for polygon in self.polyDrawerTransmitted.polygons:
            pts = np.array([(int(p[0] / self.scale_x), int(p[1] / self.scale_y)) for p in polygon])
            cv2.polylines(display, [pts], False, self.polyDrawerTransmitted.colour, 2)
        # Draw current transmitted polygon being drawn
        if self.polyDrawerTransmitted.current_points:
            pts = np.array([(int(p[0] / self.scale_x), int(p[1] / self.scale_y)) for p in self.polyDrawerTransmitted.current_points])
            cv2.polylines(display, [pts], False, self.polyDrawerTransmitted.colour, 1)
        
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

        # Compute average line for surface if there are completed polygons
        if self.polyDrawerSurface.has_polygons():
            # Create combined stencil from all surface polygons
            for polygon in self.polyDrawerSurface.polygons:
                cv2.fillPoly(self.polyDrawerSurface.stencil, np.array([polygon]), (255,255,255))
            self.surface_points = self.polyDrawerSurface.createAvgLine(detail=self.detail, area="Surface")
        else:
            # No surface polygons, create NaN points for each CBD border
            if self.cbd_boarders:
                self.surface_points = [(float('nan'), float('nan'))] * len(self.cbd_boarders)
            else:
                self.surface_points = [(float('nan'), float('nan'))] * self.detail
        
        # Compute average line for bed if there are completed polygons
        if self.polyDrawerBed.has_polygons():
            # Create combined stencil from all bed polygons
            for polygon in self.polyDrawerBed.polygons:
                cv2.fillPoly(self.polyDrawerBed.stencil, np.array([polygon]), (255,255,255))
            self.bed_points = self.polyDrawerBed.createAvgLine(detail=self.detail, area="Bed")
        else:
            # No bed polygons, create NaN points for each CBD border
            if self.cbd_boarders:
                self.bed_points = [(float('nan'), float('nan'))] * len(self.cbd_boarders)
            else:
                self.bed_points = [(float('nan'), float('nan'))] * self.detail
        
        # Compute average line for transmitted pulse if there are completed polygons
        if self.polyDrawerTransmitted.has_polygons():
            # Create combined stencil from all transmitted polygons
            for polygon in self.polyDrawerTransmitted.polygons:
                cv2.fillPoly(self.polyDrawerTransmitted.stencil, np.array([polygon]), (255,255,255))
            self.transmitted_points = self.polyDrawerTransmitted.createAvgLine(detail=self.detail, area="Transmitted")
        else:
            # No transmitted polygons, create NaN points for each CBD border
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
        """Draw average lines on result image, properly handling NaN values by drawing segments."""
        # Draw surface line segments
        self.drawLineSegments(self.surface_points, self.polyDrawerSurface.colour, 2)
        # Draw bed line segments  
        self.drawLineSegments(self.bed_points, self.polyDrawerBed.colour, 2)
        # Draw transmitted line segments
        self.drawLineSegments(self.transmitted_points, self.polyDrawerTransmitted.colour, 2)
    
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
        velocity_ice = 169.0  # m/us

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
                    surf_m.append(surf_twt_val / 2 * velocity_ice)
                else:
                    surf_twt.append(float('nan'))
                    surf_m.append(float('nan'))
                    
                if not np.isnan(y_bed_corrected[-1]):
                    bed_twt_val = y_bed_corrected[-1] * us_per_pixel
                    bed_twt.append(bed_twt_val)
                    bed_m.append(bed_twt_val / 2 * velocity_ice)
                else:
                    bed_twt.append(float('nan'))
                    bed_m.append(float('nan'))
                    
                if not np.isnan(delta_y[-1]):
                    h_ice_twt_val = delta_y[-1] * us_per_pixel
                    h_ice_twt.append(h_ice_twt_val)
                    h_ice_m.append(h_ice_twt_val / 2 * velocity_ice)
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
                    depth_m = time_us * velocity_ice  # Don't divide by 2 as requested
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
        
        # Get transmitted pulse position for 0μs reference
        if self.transmitted_points:
            avg_transmitted_y = np.mean([pt[1] for pt in self.transmitted_points if not np.isnan(pt[1])])
            zero_time_y = avg_transmitted_y
        else:
            zero_time_y = 50
            
        # Create figure with high DPI
        fig_width = 12  # inches
        fig_height = 8   # inches
        fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=500)
        
        # Display the radar image
        # Convert BGR to RGB for matplotlib
        radar_rgb = cv2.cvtColor(self.sourceImage, cv2.COLOR_BGR2RGB)
        ax.imshow(radar_rgb, aspect='auto', extent=[0, self.sourceImage.shape[1], self.sourceImage.shape[0], 0], origin='upper')
        
        # Draw CBD markers and information
        self.drawCBDMarkersMatplotlib(ax)
        
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
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(True)
        ax.spines['right'].set_visible(False)
        
        # Remove x-axis ticks and labels (we only want the xlabel)
        ax.set_xticks([])
        
        # Tight layout
        plt.tight_layout()
        
        # Save the matplotlib figure directly for high quality
        # Generate filename based on the original image
        imgName = os.path.splitext(os.path.basename(self.image.imgPath))[0]
        save_path = f"{imgName}.png"
        fig.savefig(save_path, dpi=500, bbox_inches='tight', facecolor='white', 
                   edgecolor='none', format='png')
        print(f"High-quality matplotlib figure saved as: {save_path}")
        
        # Also convert to numpy array for backward compatibility
        fig.canvas.draw()
        buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        
        # Convert RGBA to RGB (remove alpha channel), then RGB to BGR for OpenCV
        buf_rgb = buf[:, :, :3]  # Remove alpha channel
        self.resultImage = cv2.cvtColor(buf_rgb, cv2.COLOR_RGB2BGR)
        self.fixedSize = (buf_rgb.shape[1], buf_rgb.shape[0])
        
        plt.close(fig)
    
    def drawRadarPicksMatplotlib(self, ax):
        """Draw radar picks using matplotlib for high quality."""
        # Draw transmitted pulse (magenta, dashed)
        if self.transmitted_points:
            valid_points = [(pt[0], pt[1]) for pt in self.transmitted_points if not (np.isnan(pt[0]) or np.isnan(pt[1]))]
            if valid_points:
                x_coords, y_coords = zip(*valid_points)
                ax.plot(x_coords, y_coords, color='magenta', linewidth=2, linestyle='--', label='Transmitted pulse')
        
        # Draw surface (blue, dashed) - matching the picking color
        if self.surface_points:
            valid_points = [(pt[0], pt[1]) for pt in self.surface_points if not (np.isnan(pt[0]) or np.isnan(pt[1]))]
            if valid_points:
                x_coords, y_coords = zip(*valid_points)
                ax.plot(x_coords, y_coords, color='blue', linewidth=2, linestyle='--', label='Surface')
        
        # Draw bed (green, dashed) - matching the original BGR color (0, 255, 0) which is green
        if self.bed_points:
            valid_points = [(pt[0], pt[1]) for pt in self.bed_points if not (np.isnan(pt[0]) or np.isnan(pt[1]))]
            if valid_points:
                x_coords, y_coords = zip(*valid_points)
                ax.plot(x_coords, y_coords, color='green', linewidth=2, linestyle='--', label='Bed')
    
    def drawCBDMarkersMatplotlib(self, ax):
        """Draw CBD markers and labels using matplotlib."""
        if not self.cbd_points:
            return
            
        # Get lat/lon data and CBD numbers
        lat_lon_data = self.getLatLonForCBDs()
        cbd_numbers = self.getCBDNumbers()
        
        # Draw vertical CBD lines (dashed gray)
        for i, point in enumerate(self.cbd_points):
            x_pos = point[0]
            y_pos = point[1]
            
            # Draw vertical dashed line
            ax.axvline(x=x_pos, color='gray', linestyle='--', alpha=0.7, linewidth=1)
            
            # Draw circle at click point
            ax.plot(x_pos, y_pos, 'o', color='gray', markersize=3)
            
            # Create text label with correct format
            text_lines = []
            
            # Add CBD number
            if cbd_numbers and i < len(cbd_numbers):
                text_lines.append(f"CBD: {cbd_numbers[i]}")
            
            # Add lat/lon if available
            if lat_lon_data and i < len(lat_lon_data):
                lat, lon = lat_lon_data[i]
                if not (np.isnan(lat) or np.isnan(lon)):
                    text_lines.append(f"LAT: {lat:.2f}")
                    text_lines.append(f"LON: {lon:.2f}")
            
            # Position text at bottom of image, but keep within frame boundaries
            if text_lines:
                # Combine all lines with newlines for proper format
                full_text = '\n'.join(text_lines)
                text_y = self.sourceImage.shape[0] - 80  # A bit higher from bottom to stay in frame
                
                # Adjust x position to keep text boxes within image boundaries
                text_box_width = 60  # Approximate width of text box
                x_min = text_box_width // 2  # Left boundary
                x_max = self.sourceImage.shape[1] - text_box_width // 2  # Right boundary
                
                # Constrain x position within boundaries
                adjusted_x_pos = max(x_min, min(x_max, x_pos))
                
                ax.text(adjusted_x_pos, text_y, full_text, fontsize=6, ha='center', va='bottom',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
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
                depth_m = time_us * velocity_ice
                positions.append(y_line)
                time_labels.append(f"{time_us:.0f}")
                depth_labels.append(f"{depth_m:.0f}")
        
        # Set up left y-axis (depth) - ensure it's visible
        ax.set_yticks(positions)
        ax.set_yticklabels(depth_labels)
        ax.set_ylabel('Depth (m)', fontsize=10, color='black', rotation=90, labelpad=15)
        ax.tick_params(axis='y', labelsize=8, colors='black', left=True, labelleft=True)
        ax.yaxis.set_label_position('left')
        
        # Set up right y-axis (time)
        ax2.set_yticks(positions)
        ax2.set_yticklabels(time_labels)
        ax2.set_ylabel('Two-way travel time (μs)', fontsize=10, color='black', rotation=90, labelpad=15)
        ax2.tick_params(axis='y', labelsize=8, colors='black', right=True, labelright=True)
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
            Line2D([0], [0], color='green', lw=2, linestyle='--', label='Bed')
        ]
        
        # Add legend in top right
        ax.legend(handles=legend_elements, loc='upper right', fontsize=8, 
                 frameon=True, fancybox=True, shadow=True, framealpha=0.9)

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

    def runInteractiveProcess(self):
        """
        Runs the four-step process:
          1. CBD selection
          2. Surface polygon(s) drawing
          3. Bed polygon(s) drawing
          4. Transmitted pulse polygon(s) drawing
        After the final Enter key press, the average lines are computed and CSV is saved.
        """
        # Initialize the window with proper display
        initial_display = self.drawAllPolygons()
        self.refresh(initial_display)
        
        # Show filename being processed
        filename = os.path.basename(self.image.imgPath)
        print(f"\n{'='*60}")
        print(f"🔍 PROCESSING FILE: {filename}")
        print(f"{'='*60}")
        print("\n📍 Phase 1: Select the CBD. Left-click to add points; right-click when done.")        # Activate CBD selector and wait until done
        cv2.setMouseCallback(self.windowName, self.cbdSelector.on_mouse)
        while not self.cbdSelector.done:
            temp = self.drawAllPolygons()
            self.refresh(temp)
            key = cv2.waitKey(1) & 0xFF
            if key == ESC:
                return
        self.cbd_points = self.cbdSelector.points.copy()
        print("✅ CBD selection complete.")

        time.sleep(0.5)  # slight pause

        print("\n🏔️  Phase 2: Select the Surface area(s). Left-click to add points; right-click to finish polygon; press Enter when all polygons done.")
        cv2.setMouseCallback(self.windowName, self.polyDrawerSurface.on_mouse)
        while not self.polyDrawerSurface.done:
            temp = self.drawAllPolygons()
            self.refresh(temp)
            key = cv2.waitKey(1) & 0xFF
            if key == ESC:
                                return
            elif key == 13:  # Enter key
                if self.polyDrawerSurface.has_polygons():
                    self.polyDrawerSurface.done = True
                    print(f"✅ Surface selection complete with {len(self.polyDrawerSurface.polygons)} polygon(s).")
                else:
                    print("Please draw at least one polygon before finishing.")
        
        time.sleep(0.5)

        print("\n🏔️  Phase 3: Select the Bed area(s). Left-click to add points; right-click to finish polygon; press Enter when all polygons done.")
        cv2.setMouseCallback(self.windowName, self.polyDrawerBed.on_mouse)
        while not self.polyDrawerBed.done:
            temp = self.drawAllPolygons()
            self.refresh(temp)
            key = cv2.waitKey(1) & 0xFF
            if key == ESC:
                return
            elif key == 13:  # Enter key
                if self.polyDrawerBed.has_polygons():
                    self.polyDrawerBed.done = True
                    print(f"✅ Bed selection complete with {len(self.polyDrawerBed.polygons)} polygon(s).")
                else:
                    print("Please draw at least one polygon before finishing.")

        time.sleep(0.5)

        print("\n📡 Phase 4: Select the Transmitted Pulse area(s). Left-click to add points; right-click to finish polygon; press Enter when all polygons done.")
        cv2.setMouseCallback(self.windowName, self.polyDrawerTransmitted.on_mouse)
        while not self.polyDrawerTransmitted.done:
            temp = self.drawAllPolygons()
            self.refresh(temp)
            key = cv2.waitKey(1) & 0xFF
            if key == ESC:
                return
            elif key == 13:  # Enter key
                if self.polyDrawerTransmitted.has_polygons():
                    self.polyDrawerTransmitted.done = True
                    print(f"✅ Transmitted pulse selection complete with {len(self.polyDrawerTransmitted.polygons)} polygon(s).")
                else:
                    print("Please draw at least one polygon before finishing.")
        
        print("\n🔄 Computing average lines and saving results...")
        # Compute average lines and save CSV
        self.drawAverageLinesAndSave()
        print("\n🎉 === PROCESSING COMPLETED ===")
        print("✅ CSV file saved and high-quality image generated.")
        print("👆 Press any key to continue to next file...")

        # Make sure the window is visible and focused for key input
        cv2.imshow(self.windowName, self.resultImage)
        cv2.setWindowProperty(self.windowName, cv2.WND_PROP_TOPMOST, 1)  # Bring to front
        cv2.setWindowProperty(self.windowName, cv2.WND_PROP_TOPMOST, 0)  # Remove topmost flag
        
        # Wait so user can see result before closing
        while True:
            key = cv2.waitKey(1) & 0xFF
            if key != 255:  # Any key pressed (255 means no key)
                break
        cv2.destroyWindow(self.windowName)

# ====================================

class PolygonDrawer:
    """
    Draws multiple polygons by recording left clicks; right click finishes current polygon.
    Press 'Enter' key to finish all polygons for this feature.
    """
    def __init__(self, window: Window, colour: tuple, DEBUG=False):
        self.DEBUG = DEBUG
        if self.DEBUG:
            print("Creating PolygonDrawer for", window.windowName)
        self.window = window
        self.colour = colour
        self.done = False
        self.current_points = []  # Current polygon being drawn
        self.polygons = []  # List of completed polygons
        self.stencil = np.zeros(self.window.sourceImage.shape, dtype=self.window.sourceImage.dtype)

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
            if len(self.current_points) >= 3:
                # Finish current polygon and start a new one
                self.polygons.append(self.current_points.copy())
                print(f"Polygon {len(self.polygons)} completed. Left-click to start next polygon, or press Enter to finish all polygons.")
                self.current_points = []
            else:
                print("Need at least 3 points for a polygon.")

    def createAvgLine(self, detail: int, area: str):
        """
        Divides the polygon-extracted area into vertical segments and computes the average point in each.
        Now works with multiple polygons combined in the stencil.
        """
        if self.DEBUG:
            print("Creating average line for", area)
        
        # Use the stencil that has been filled with all polygons
        mask = cv2.cvtColor(self.stencil, cv2.COLOR_BGR2GRAY)
        
        # Get all points from all polygons to determine bounding box
        all_points = self.get_all_points()
        if not all_points:
            return []
            
        # Determine bounding box of all drawn polygons
        ys = [pt[1] for pt in all_points]
        xs = [pt[0] for pt in all_points]
        y0, y1 = int(min(ys)), int(max(ys))
        x0, x1 = int(min(xs)), int(max(xs))
        
        # Crop the mask to the polygon bounding box
        cropped = mask[y0:y1, x0:x1]
        if self.DEBUG:
            print("Cropped region dims:", cropped.shape)
        # Get coordinates where mask is nonzero
        Y, X = np.where(cropped == 255)
        whitePoints = np.column_stack((X, Y))
        if len(whitePoints) == 0:
            return []
        # Sort by x and segment based on detail
        whitePoints = whitePoints[np.argsort(whitePoints[:, 0])]
        averagedPoints = []
        # Check if window has CBD boarders for segmentation; otherwise divide evenly.
        if self.window.cbd_boarders:
            # Ensure we have one point per CBD border, using NaN for missing data
            for boarder in self.window.cbd_boarders:
                bx = boarder[0] - x0  # adjust relative to cropped region
                segment = whitePoints[(whitePoints[:, 0] > bx - 20) & (whitePoints[:, 0] <= bx + 20)]
                if segment.size:
                    avg = segment.mean(axis=0)
                    averagedPoints.append((int(avg[0] + x0), int(avg[1] + y0)))
                else:
                    # No data in this CBD segment, add NaN point
                    averagedPoints.append((float('nan'), float('nan')))
        else:
            segSize = (x1 - x0) / detail
            for n in range(detail):
                seg_start = x0 + n * segSize
                seg_end = seg_start + segSize
                segment = whitePoints[(whitePoints[:, 0] > (seg_start - x0)) & 
                                      (whitePoints[:, 0] <= (seg_end - x0))]
                if segment.size:
                    avg = segment.mean(axis=0)
                    averagedPoints.append((int(avg[0] + x0), int(avg[1] + y0)))
                else:
                    # No data in this segment, add NaN point
                    averagedPoints.append((float('nan'), float('nan')))
        if self.DEBUG:
            print("Averaged points:", averagedPoints)
        # Draw computed average line for feedback (only valid points, skip NaN)
        valid_points = [pt for pt in averagedPoints if not (np.isnan(pt[0]) or np.isnan(pt[1]))]
        if valid_points:
            cv2.polylines(self.window.resultImage, np.array([valid_points]), False, self.colour, 1)
        return averagedPoints

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
        if not QT_AVAILABLE:
            print("Error: PyQt5 not available. Cannot create calpip viewer.")
            self.has_calpips = False
            self.done = True
            return
            
        self.app = QApplication.instance()
        if self.app is None:
            self.app = QApplication([])
            
        self.root = QMainWindow()
        self.root.setWindowTitle("Examine Radargram for Calpips")
        self.root.resize(1000, 700)
        
        # Central widget
        central_widget = QWidget()
        self.root.setCentralWidget(central_widget)
        
        # Layout: left controls, right plot
        main_layout = QHBoxLayout(central_widget)
        
        # Control frame
        control_frame = QWidget()
        control_layout = QVBoxLayout(control_frame)
        
        # Instructions
        title_label = QLabel("Examine for Calpips")
        title_label.setFont(QFont("Arial", 14, QFont.Bold))
        control_layout.addWidget(title_label)
        
        instruction_label = QLabel("Use the slider below to scroll\nhorizontally and look for\nregular horizontal lines\n(calpips)")
        instruction_label.setWordWrap(True)
        control_layout.addWidget(instruction_label)
        
        # Buttons
        self.yes_btn = QPushButton("Has Calpips")
        self.yes_btn.setStyleSheet("background-color: lightgreen")
        self.yes_btn.setFixedSize(150, 50)
        self.yes_btn.clicked.connect(self.has_calpips_yes)
        control_layout.addWidget(self.yes_btn)
        
        self.no_btn = QPushButton("No Calpips")
        self.no_btn.setStyleSheet("background-color: lightcoral")
        self.no_btn.setFixedSize(150, 50)
        self.no_btn.clicked.connect(self.has_calpips_no)
        control_layout.addWidget(self.no_btn)
        
        control_layout.addStretch()
        main_layout.addWidget(control_frame)
        
        # Figure setup
        figsize = (self.WINDOW_W_PX / self.DPI, self.WINDOW_H_PX / self.DPI)
        self.fig, self.ax = plt.subplots(figsize=figsize, dpi=self.DPI)
        self.ax.set_axis_off()
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setParent(central_widget)
        main_layout.addWidget(self.canvas)
        
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
        
        # Show window
        self.root.show()
    
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
        self.root.close()
    
    def has_calpips_no(self):
        """User confirmed no calpips present"""
        self.has_calpips = False
        self.done = True
        self.root.close()
    
    def run(self):
        """Run the viewer and return True if calpips present, False otherwise"""
        if not QT_AVAILABLE:
            return False
        self.app.exec_()
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
        if not QT_AVAILABLE:
            print("Error: PyQt5 not available. Cannot create calpip picker.")
            self.no_calpips = True
            self.done = True
            return
            
        self.app = QApplication.instance()
        if self.app is None:
            self.app = QApplication([])
            
        self.root = QMainWindow()
        self.root.setWindowTitle("Calpip Picker")
        self.root.resize(1200, 700)
        
        # Central widget
        central_widget = QWidget()
        self.root.setCentralWidget(central_widget)
        
        # Layout: left controls, right plot
        main_layout = QHBoxLayout(central_widget)
        
        # Control frame
        control_frame = QWidget()
        control_layout = QVBoxLayout(control_frame)
        
        # Instructions
        title_label = QLabel("Calpip Processing")
        title_label.setFont(QFont("Arial", 14, QFont.Bold))
        control_layout.addWidget(title_label)
        
        instruction_label = QLabel("1. Select end bound\n2. Pick 4 calpips\n3. Click Done")
        instruction_label.setWordWrap(True)
        control_layout.addWidget(instruction_label)
        
        # Buttons
        self.bound_btn = QPushButton("Select End Bound")
        self.bound_btn.setStyleSheet("background-color: lightyellow")
        self.bound_btn.setFixedWidth(150)
        self.bound_btn.clicked.connect(self.start_bound_selection)
        control_layout.addWidget(self.bound_btn)
        
        self.pick_btn = QPushButton("Pick Calpips")
        self.pick_btn.setStyleSheet("background-color: lightblue")
        self.pick_btn.setFixedWidth(150)
        self.pick_btn.setEnabled(False)
        self.pick_btn.clicked.connect(self.start_picking)
        control_layout.addWidget(self.pick_btn)
        
        self.done_btn = QPushButton("Done")
        self.done_btn.setStyleSheet("background-color: lightgreen")
        self.done_btn.setFixedWidth(150)
        self.done_btn.clicked.connect(self.finish_picking)
        control_layout.addWidget(self.done_btn)
        
        self.no_calpips_btn = QPushButton("No Calpips")
        self.no_calpips_btn.setStyleSheet("background-color: lightcoral")
        self.no_calpips_btn.setFixedWidth(150)
        self.no_calpips_btn.clicked.connect(self.no_calpips_selected)
        control_layout.addWidget(self.no_calpips_btn)
        
        # Status label
        self.status_label = QLabel("Click 'Select End Bound' to start")
        self.status_label.setStyleSheet("color: blue")
        self.status_label.setWordWrap(True)
        control_layout.addWidget(self.status_label)
        
        control_layout.addStretch()
        main_layout.addWidget(control_frame)
        
        # Figure setup
        figsize = (self.WINDOW_W_PX / self.DPI, self.WINDOW_H_PX / self.DPI)
        self.fig, self.ax = plt.subplots(figsize=figsize, dpi=self.DPI)
        self.ax.set_axis_off()
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setParent(central_widget)
        main_layout.addWidget(self.canvas)
        
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
        
        # Show window
        self.root.show()
    
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
        self.status_label.setText("Click on the image to select the end bound for y-axis")
        self.status_label.setStyleSheet("color: orange")
        self.bound_btn.setEnabled(False)
        
    def start_picking(self):
        """Start the calpip picking process"""
        self.picks_y = []
        self.accept_clicks = True
        self.accept_bound_click = False
        self.status_label.setText(f"Click on 4 calpips (picked: {len(self.picks_y)}/4)")
        self.status_label.setStyleSheet("color: green")
        self.pick_btn.setEnabled(False)
        
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
            self.status_label.setText("End bound set! Now click 'Pick Calpips'")
            self.status_label.setStyleSheet("color: blue")
            self.bound_btn.setEnabled(True)
            self.pick_btn.setEnabled(True)
            return
            
        if self.accept_clicks:
            # Picking calpips
            self.picks_y.append(y_click)
            
            # Draw pick line
            x0, x1 = self.ax.get_xlim()
            line, = self.ax.plot([x0, x1], [y_click, y_click], color="yellow", linewidth=2)
            self.pick_lines.append(line)
            
            self.canvas.draw_idle()
            
            self.status_label.setText(f"Click on 4 calpips (picked: {len(self.picks_y)}/4)")
            
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
        self.status_label.setText(f"Average spacing: {avg_spacing:.1f} pixels, {len(self.result_y_lines)} lines")
        
        # Draw interpolated lines
        self.draw_interpolated_lines()
    
    def draw_interpolated_lines(self):
        """Draw clean horizontal calpip lines for picking"""        # Clear existing interpolated lines
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
            QMessageBox.warning(self.root, "Missing End Bound", "Please select the end bound first!")
        elif len(self.picks_y) < 4:
            QMessageBox.warning(self.root, "Incomplete", "Please pick 4 calpips!")
        elif not self.result_pixel_distance:
            QMessageBox.warning(self.root, "No Calculation", "Calpip calculation failed!")
        else:
            self.done = True
            self.root.close()
    
    def no_calpips_selected(self):
        """User selected no calpips"""
        self.no_calpips = True
        self.done = True
        self.root.close()
    
    def run(self):
        """Run the calpip picker and return results"""
        self.app.exec_()
        
        # Clean up after dialog closes
        
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
    def __init__(self, path: str, batchSize:int, csv_directory: str = None):
        self.path = path
        self.batchSize = batchSize
        self.csv_directory = csv_directory
        self.dir_list = [f for f in os.listdir(path) if f.lower().endswith(".tiff") or f.lower().endswith(".tif")]
        if not self.dir_list:
            print("No TIFF files found in", path)
        
        # Global calpip state - persists across files
        self.calpip_pixel_distance = None
        self.calpip_y_lines = []

    def process(self, tiff):
        DEBUG = True
        DETAIL = 10  # detail for avg line
        imgPath = os.path.join(self.path, tiff)
        # Adjust crop values as desired.
        image = Image(imgPath, cropVals=(300, 2050), DEBUG=DEBUG)
        win = Window("Reference", image=image, detail=DETAIL, DEBUG=DEBUG, csv_directory=self.csv_directory)
        win.refresh(win.sourceImage)
        # Run interactive phases:
        win.runInteractiveProcess()
        
        # After main processing, handle calpip processing
        self.processCalpips(win)
        
        return win.resultImage

    def processCalpips(self, window):
        """Handle calpip processing after main radar analysis."""
        print(f"Examining {os.path.basename(window.image.imgPath)} for calpips...")
        
        # Show radar viewer for user to examine the radargram
        viewer = RadarViewer(window)
        has_calpips = viewer.run()
        
        if has_calpips:
            # Open calpip picker
            print("User confirmed calpips present - opening calpip picker...")
            calpip_result = self.openCalpipPicker(window)
            if calpip_result:
                self.calpip_pixel_distance, self.calpip_y_lines = calpip_result
                print(f"New calpip distance: {self.calpip_pixel_distance:.1f} pixels")
        else:
            print("User confirmed no calpips - using existing calpip distance from previous file")
        
        # Apply calpip data to current window and regenerate CSV with time calculations
        window.calpip_pixel_distance = self.calpip_pixel_distance
        window.calpip_y_lines = self.calpip_y_lines.copy() if self.calpip_y_lines else []
        window.regenerateCSVWithCalpips()
        
        # Update result image with y-axis lines
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

    def launch(self, saveDirectory=""):
        anchorCounter = 0
        print("TARGET:", len(self.dir_list))
        while anchorCounter < len(self.dir_list):
            resImages = []
            for count in range(self.batchSize):
                if anchorCounter+count >= len(self.dir_list):
                    break
                currImage = self.dir_list[anchorCounter+count]
                resImages.append(self.process(currImage))
                print("Processed image index:", anchorCounter+count)
            anchorCounter += self.batchSize
        
        # Show completion message after processing all files
        print("\n" + "="*60)
        print("🎉 ALL FILES PROCESSED SUCCESSFULLY! 🎉")
        print(f"Total files processed: {len(self.dir_list)}")
        print("All CSV files and images have been saved to the output directory.")
        print("="*60)
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
            app = App(tiff_directory, batchSize=1, csv_directory=navigation_directory)
            app.launch(output_directory)
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
            app = App(source, batchSize=1, csv_directory=csv_source)
            app.launch(destination)
        finally:
            os.chdir(original_cwd)