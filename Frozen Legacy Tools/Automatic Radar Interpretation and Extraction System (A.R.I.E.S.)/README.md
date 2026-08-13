<p align="left">
  <img src="docs/ARIES.png" alt="ARIES Logo" height="120">
  <span style="font-size:2em; vertical-align: middle;">
</p>

# ARIES — Automatic Radar Interpretation and Extraction System

**ARIES** is an advanced Python system for automatically processing and interpreting historical Z-scope radar sounding images (echograms) collected in the 1970s SPRI/NSF/TUD airborne radar surveys across Antarctica. The system uses computer vision and signal processing techniques to automatically detect ice surface and bed interfaces, providing quantitative ice thickness measurements.

---

## Table of Contents

- [Features](#-features)
- [Requirements](#-requirements)
- [Installation](#️-installation)
- [Usage](#-usage)
  - [Single Image Processing](#single-image-processing)
  - [Batch Processing](#batch-processing)
- [Configuration](#configuration)
  - [Configuration tuning: move tx pulse / surface / bed detection](#configuration-tuning-move-tx-pulse--surface--bed-detection)
- [Configuration Reference Sheet](#-configuration-reference-sheet)
- [Core Processing Pipeline](#-core-processing-pipeline)
- [Output Files](#-output-files)
  
---

## 🚀 Features

- **Automated Image Processing**: Load and preprocess historical Z-scope radar echograms with adaptive enhancement
- **Intelligent Feature Detection**: Automatically detect film artifact boundaries, transmitter pulse, and calibration pips
- **Advanced Echo Tracing**: Multi-scale adaptive algorithms for automatic surface and bed echo detection
- **Dual Calibration Methods**: Choose between ARIES automatic pip detection or TERRA-style manual 4-point selection
- **Time Domain Calibration**: Precise time-to-depth conversion using 2μs interval calibration (TERRA methodology)
- **Coordinate Integration**: Full GPS coordinate interpolation using flight navigation data
- **Interactive Optimization**: Real-time parameter adjustment with config editing and re-run detection capability
- **Quality Validation**: Automatic validation and parameter optimization with user approval workflow
- **Comprehensive Output**: Enhanced CSV export with pixel coordinates, travel times, and ice thickness measurements
- **Smart Batch Processing**: Calibration reuse functionality for efficient processing of similar images
- **Interactive Refinement**: Manual CBD tick selection with filename-based count detection
- **Modular Architecture**: Separate modules for artifact detection, calibration, feature detection, and visualization


---

## 📋 Requirements

### System Requirements
- **Python 3.8+** (Python 3.11 or 3.12 recommended for optimal performance)
- **Operating System**: Windows 10/11, macOS 10.14+, or Linux
- **RAM**: 8GB minimum, 16GB recommended for batch processing
- **Storage**: ~1GB per processed flight for output data

### Python Dependencies
- **Core Libraries**:
  - `numpy` - Numerical computing and array operations
  - `scipy` - Scientific computing and signal processing
  - `opencv-python` - Computer vision and image processing
  - `matplotlib` - Plotting and visualization
  - `pandas` - Data manipulation and CSV handling
  - `pathlib` - Modern file path handling

### Development Tools
- [VSCode](https://code.visualstudio.com/) (recommended for development and debugging)
- Git (for version control and updates) 

---

## 🛠️ Installation

### Quick Setup

1. **Clone the repository**:
```bash
git clone https://github.com/GT-PGSL/FrozenLegacies.git
cd "FrozenLegacies/Frozen Legacy Tools/Automatic Radar Interpretation and Extraction System (A.R.I.E.S.)"
```

2. **Create and activate virtual environment** (recommended):
```bash
# Windows
python -m venv aries_env
aries_env\Scripts\activate

# macOS/Linux  
python3 -m venv aries_env
source aries_env/bin/activate
```

3. **Install dependencies**:
```bash
pip install --upgrade pip
pip install numpy scipy opencv-python matplotlib pandas pathlib2
```

4. **Verify installation**:
```bash
python -c "import numpy, cv2, matplotlib, pandas, scipy; print('All packages installed successfully!')"
```

### Common interpreter name problems (python vs python3)

Some systems use the `python` command, others use `python3` (Linux/macOS), and Windows also supports the Python Launcher `py` which can select versions. If you get "command not found" or packages seem installed for a different interpreter, use these recommendations:

- Check which interpreter is available and its version:
  - Windows (PowerShell): `py -3 --version` or `python --version`
  - macOS / Linux: `python3 --version` or `python --version`

- Prefer invoking pip with the same interpreter to avoid installing into a different Python than you run:
  - `python -m pip install ...` or `python3 -m pip install ...` (use whichever matches your `python` command)
  - Windows (explicit): `py -3 -m pip install ...`

- Creating and activating venvs safely (examples):
  - Windows PowerShell (recommended):
    ```powershell
    py -3 -m venv .venv
    .\.venv\Scripts\Activate.ps1
    py -3 -m pip install --upgrade pip
    py -3 -m pip install numpy scipy opencv-python matplotlib pandas
    ```
  - macOS / Linux:
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    python3 -m pip install --upgrade pip
    python3 -m pip install numpy scipy opencv-python matplotlib pandas
    ```

- If activation fails in PowerShell because of execution policy, run as admin once:
  - `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser` (see PowerShell docs and use with caution)

Using these patterns ensures the packages are installed into the interpreter you're actually running. If in doubt, try `python -c "import sys; print(sys.executable)"` (or `py -3 -c "import sys; print(sys.executable)"`) to see which binary will run the code.

### Directory Structure
```
ARIES/
├── main.py                    # Main processing script
├── zscope_processor.py        # Core processor class
├── config/
│   ├── default_config.json    # Processing parameters
│   └── physical_constants.json # Physical constants
├── functions/                 # Modular processing functions
├── F125_Zscope/              # Sample data directory
└── output/                   # Processed results
```

---

## 📖 Usage

### Single Image Processing

Process a single Z-scope radar image with automatic feature detection:

```powershell
# interactive mode (will prompt for calpip selection unless you pass --non_interactive_pip_x)
python main.py --output <output_dir> <path\to\zscope_image.tiff>

# non-interactive (supply approximate calpip X coordinate)
python main.py --output <output_dir> --non_interactive_pip_x 19000 <path\to\zscope_image.tiff>
```

**Arguments**:
- `--batch_dir`: Path to directory containing radar image file(s)
- `--output`: Directory for output files and results
- `--nav`: Flight navigation CSV file with GPS coordinates

**Example**:
```bash
python main.py --batch_dir F125_Zscope --output output --nav F125_Zscope/125.csv
```

### Batch Processing

Process multiple images in a directory with adaptive parameter learning:

```powershell
# Batch mode example (Windows PowerShell):
python main.py --batch_dir F125_Zscope --output output\F125_batch --nav F125_Zscope\125.csv
```

**Arguments**:
- `--batch_dir`: Directory containing multiple `.tiff` files
- `--output`: Directory for batch processing results
- `--nav`: Flight navigation CSV file

**Example**:
```bash
python main.py --batch_dir F125_Zscope --output output/F125_batch --nav F125_Zscope/125.csv
```

### Interactive Workflow

Note: **Calibration pip selection (calpip) and method choice are prompted up-front** in both single-image interactive mode and batch mode. In single-image mode the program will open a click selector to pick an approximate X-coordinate for calpip when `--non_interactive_pip_x` is not provided. In batch mode the user is first asked to choose a method (ARIES or TERRA), and if ARIES is chosen an initial click-selection is requested on the first image.

1. **Calibration (ARIES or TERRA)** — The workflow begins by asking the user to choose the calibration pip (calpip) method and/or pick the approximate pip location:
  - Single-image interactive: ClickSelector opens to pick the approximate X-coordinate (unless `--non_interactive_pip_x` is supplied).
  - Batch mode: The script first asks whether to use ARIES (automatic) or TERRA (manual). ARIES then requires an initial ClickSelector on the first image; TERRA proceeds to 4-point manual selection without an initial area click.
2. **Image preprocessing** with automatic feature detection and boundary identification (the pipeline reads the whole image file).
3. **Echo detection & optimization**:
  - Automatic surface and bed echo detection
  - Results review with approval dialog
  - **Choice 1**: Re-run detection with updated configuration (real-time parameter adjustment)
  - **Choice 2**: Proceed with current results or run guided calibration
4. **CBD tick selection**: Manually select all CBD positions (count auto-detected from filename range)
5. **Time-domain calibration**: 2μs interval calibration using the chosen method (ARIES or TERRA)
6. **Results visualization**: Time-calibrated echogram with detected features
7. **Enhanced output**: CSV export with coordinates, travel times, and ice thickness

### Batch Processing Workflow

1. **Startup / First image**: At the start of batch mode the user is asked to choose a calpip method (ARIES or TERRA). If a non-interactive X coordinate was supplied (`--non_interactive_pip_x`) the calpip selection step is skipped and that X value is used. If ARIES is chosen and no non-interactive X was supplied, the program will open the ClickSelector on the first image to get an approximate X location. TERRA uses manual 4-point selection and does not require the initial area-click.
2. **Read & process whole files**: The pipeline then reads the whole image files in the batch directory and applies preprocessing, echo detection and calibration as configured.
3. **Per-image calpip decision**:
  - **"y" (new calibration)**: Prompt to select a new calpip for the current image using the chosen method
  - **"n" (reuse calibration)**: Use the previous calpip values (approx X or stored calibration) and skip new selection
  - **"q" (quit)**: Exit batch processing
4. **Adaptive processing**: The chosen calibration method is preserved for the remainder of the batch unless explicitly changed
5. **Efficiency optimization**: Reusing calibration data speeds up processing for similar images

### Command Line Options

- `--batch_dir`: Directory containing radar image files (required)
- `--output`: Output directory for processed results (required)
- `--nav`: Navigation CSV file path (required for coordinate interpolation)
- `--config`: Custom configuration file path (optional)


---

## Configuration

ARIES uses JSON configuration files to control processing parameters:

- **Processing parameters**: `config/default_config.json` - Main algorithm settings
- **Physical constants**: `config/physical_constants.json` - Radar wave velocity, ice properties

You can customize detection thresholds, search windows, and output settings by editing these files.

### Configuration tuning: move transmitter pulse / surface / bed detection

If you need to shift where the algorithms search (for the transmitter pulse, the surface echo, or the bed echo), change the following keys in `config/default_config.json`.

- Transmitter pulse (where to look for the TX pulse near the top of the image):
  - `transmitter_pulse_params.search_height_ratio` — controls how far down (vertical fraction or multiplier depending on image crop) the processor will search for the pulse. Decrease to search higher, increase to search deeper.
  - `transmitter_pulse_params.peak_prominence` — detection sensitivity; reduce to make detection easier for weak pulses, raise to ignore noisy peaks.
  - `transmitter_pulse_params.fallback_depth_ratio` — fallback vertical position when no peak is confidently detected.

  Example: to search closer to the top of the file try lowering `search_height_ratio` (e.g., reduce from 1.2 to 0.5–0.8 in heavily-cropped images).

- Surface echo (how far below the transmitter pulse to start looking for the ice surface):
  - `echo_tracing_params.surface_detection.search_start_offset_px` — pixels below the transmitter pulse where surface search begins. Lower this value if the surface echo appears closer to the transmit pulse, raise it if the surface is deeper.
  - `echo_tracing_params.surface_detection.search_depth_px` — vertical size (px) of the search window for the surface echo — increase to allow deeper search.
  - `echo_tracing_params.surface_detection.peak_prominence` — strength threshold for surface echo detection.

  Example: if the detected surface is consistently too low, decrease `search_start_offset_px` (e.g. from 300 to 250 or 200). If the surface is missed, lower `peak_prominence` to 5–15.

- Bed echo (how far below the surface the algorithm looks for the bed):
  - `echo_tracing_params.bed_detection.search_start_offset_from_surface_px` — start searching this many pixels below the detected surface.
  - `echo_tracing_params.bed_detection.search_end_offset_from_z_boundary_px` — how close to image bottom to stop searching.
  - `echo_tracing_params.bed_detection.peak_prominence` — bed echo detection threshold; lower for weak returns.

  Example: for thicker ice, increase `search_start_offset_from_surface_px` (e.g., from 260 to 400–800); for noisy images, raise `peak_prominence` to avoid false positives.

Notes, diagnostics and iteration:
- Use the visualization flags in `default_config.json` to enable debug plots (e.g. `visualize_tx_pulse_detection`, `visualize_pip_detection`, `save_intermediate_plots`) to inspect the search windows and detections.
- Make single-image changes and re-run interactive mode until you find settings that work — then save a config copy and use `--config` to process a batch consistently.
- For GUI-based calpip picking (TERRA method) use the interactive picker; ARIES method requires initial approximate X position and will try automatic pip detection.

---

## 🔧 Configuration Reference Sheet

### Key Parameters in `config/default_config.json`

#### Adjusting Transmitter Pulse Detection
```json
"transmitter_pulse_params": {
    "search_height_ratio": 0.35,        // Higher = search deeper (0.2-0.5)
    "peak_prominence": 0.7,             // Higher = more selective (0.3-1.0) 
    "fallback_depth_ratio": 0.3,       // Fallback position if no peaks found
    "visualize_tx_pulse_detection": true // Enable debug visualization
}
```

**Quick Adjustments**:
- **Transmit pulse too high**: Decrease `search_height_ratio` to 0.25
- **Transmit pulse too low**: Increase `search_height_ratio` to 0.45  
- **No pulse detected**: Lower `peak_prominence` to 0.3-0.5

#### Surface Echo Detection Parameters
```json
"echo_tracing_params": {
  "surface_detection": {
    "search_start_offset_px": 355,      // Offset from transmit pulse (pixels)
    "search_depth_px": 150,             // Search window depth
    "peak_prominence": 30,              // Echo strength threshold
    "enhancement_clahe_clip": 2.8       // Image contrast enhancement
  }
}
```

**Quick Adjustments**:
- **Surface too close to transmit pulse**: Increase `search_start_offset_px` to 400-500
- **Surface too far from transmit pulse**: Decrease `search_start_offset_px` to 250-300
- **Missing weak surface echoes**: Lower `peak_prominence` to 15-25
- **Too many false surface detections**: Increase `peak_prominence` to 35-50
- **Adjust search window size**: Modify `search_depth_px` (100-250 typical range)

#### Bed Echo Detection Parameters  
```json
"echo_tracing_params": {
  "bed_detection": {
    "search_start_offset_from_surface_px": 495, // Offset from surface echo
    "search_end_offset_from_z_boundary_px": 22, // Distance from image bottom
    "peak_prominence": 95,                       // Echo strength threshold
    "enhancement_clahe_clip": 5.0               // Image contrast enhancement
  }
}
```

**Quick Adjustments**:
- **Bed too close to surface**: Increase `search_start_offset_from_surface_px` to 600-800
- **Bed too far from surface**: Decrease `search_start_offset_from_surface_px` to 300-400  
- **Missing weak bed echoes**: Lower `peak_prominence` to 50-70
- **Too many false bed detections**: Increase `peak_prominence` to 120-150
- **Adjust bottom boundary**: Modify `search_end_offset_from_z_boundary_px` (10-50 typical range)

#### Image Enhancement Settings
```json
"preprocessing_params": {
    "percentile_low": 2,                // Lower contrast percentile (1-5)
    "percentile_high": 98               // Upper contrast percentile (95-99)
}
```

**Quick Adjustments**:
- **Image too dark**: Increase `percentile_high` to 99
- **Image too bright/washed out**: Decrease `percentile_high` to 95-97
- **Need more contrast**: Adjust `percentile_low` to 1 and `percentile_high` to 99

### Quick Reference: Key Parameters by Section

| Section | Parameter | Purpose | Typical Range |
|---------|-----------|---------|---------------|
| **Transmitter Pulse** | `search_height_ratio` | Search depth from top | 0.2 - 0.5 |
| | `peak_prominence` | Detection sensitivity | 0.3 - 1.0 |
| **Surface Detection** | `search_start_offset_px` | Offset from TX pulse | 250 - 500 |
| | `search_depth_px` | Search window size | 100 - 250 |
| | `peak_prominence` | Echo strength threshold | 15 - 50 |
| | `enhancement_clahe_clip` | Contrast enhancement | 2.0 - 4.0 |
| **Bed Detection** | `search_start_offset_from_surface_px` | Offset from surface | 300 - 800 |
| | `search_end_offset_from_z_boundary_px` | Distance from bottom | 10 - 50 |
| | `peak_prominence` | Echo strength threshold | 50 - 150 |
| | `enhancement_clahe_clip` | Contrast enhancement | 3.0 - 6.0 |

### Parameter Adjustment Workflow

1. **Start with transmitter pulse**: Ensure it's correctly detected first
2. **Adjust surface detection**: Tune `search_start_offset_px` and `peak_prominence` based on ice type
3. **Refine bed detection**: Adjust `search_start_offset_from_surface_px` and prominence for bedrock visibility
4. **Fine-tune enhancement**: Modify `enhancement_clahe_clip` parameters for optimal echo visibility

### Common Parameter Combinations

| Ice Type | `search_start_offset_px` | `peak_prominence` | `search_start_offset_from_surface_px` | `peak_prominence` (bed) |
|----------|--------------------------|-------------------|---------------------------------------|-------------------------|
| **Thick Ice** | 355 | 30 | 495 | 95 |
| **Thin Ice** | 300 | 25 | 350 | 70 |
| **Noisy Data** | 400 | 40 | 550 | 120 |
| **Clear Data** | 320 | 20 | 450 | 60 |

---
## 🔧 Core Processing Pipeline

### 1. Image Preprocessing & Enhancement
- **Format Conversion**: 16-bit to 8-bit TIFF conversion with percentile normalization
- **Contrast Enhancement**: CLAHE (Contrast Limited Adaptive Histogram Equalization) 
- **Artifact Detection**: Automatic film artifact and sprocket hole boundary detection
- **Quality Assessment**: Image quality metrics and preprocessing optimization

### 2. Transmitter Pulse Detection
- **Search Window Definition**: Configurable search area in upper portion of image
- **Peak Detection**: Multi-scale peak prominence analysis with position weighting
- **Validation**: Automatic validation using expected pulse characteristics
- **Fallback Mechanisms**: Robust handling of difficult detection cases

### 3. CBD Tick Mark Selection
- **Filename-Based Count**: Automatically determines number of CBDs from filename (e.g., C0218_C0231 = 14 CBDs)
- **Manual Positioning**: User manually clicks each CBD tick mark position for maximum accuracy
- **Dynamic Color Coding**: Each selected CBD gets a unique color marker and sequential numbering
- **No Interpolation**: Uses exact clicked positions, eliminating spacing calculation errors

### 4. Dual Calibration Pip Methods

#### ARIES Automatic Method:
- **Click Selection**: User clicks approximate calibration pip location
- **Automatic Detection**: Computer vision algorithms detect individual tick marks (~33px spacing)
- **Local Refinement**: Advanced algorithms for precise position detection
- **Multi-Approach**: Primary and aggressive detection methods with ranking

#### TERRA Manual Method:
- **4-Point Selection**: Manual selection of exactly 4 calibration points
- **Windowed Interface**: PyQt5-based picker with image navigation and zoom
- **Major Calpip Lines**: Focuses on major calibration lines (~137px spacing)
- **2μs Intervals**: Each major line represents 2μs travel time (TERRA methodology)
- **Slider Navigation**: Easy navigation through large radar images

#### Method Selection:
- **User Choice**: Select method at start of processing session
- **Batch Consistency**: Chosen method preserved throughout batch processing
- **Calibration Reuse**: Option to reuse previous calibration data for similar images

### 5. Automated Echo Tracing & Optimization
- **Initial Detection**: 
  - Multi-scale adaptive filtering and enhancement
  - Configurable search parameters with default settings
  - Peak prominence analysis for surface and bed interfaces
- **Interactive Review**:
  - Visual results display with coverage statistics
  - User approval dialog: satisfied vs. optimize parameters
  - **Real-time parameter editing**: Direct config file modification
  - **Re-run capability**: Test new settings immediately
- **Optimization Options**:
  - **Option 1**: Interactive point selection (guided calibration)
  - **Option 2**: Direct configuration file editing with system editor
- **Quality Metrics**: Coverage statistics and confidence scoring for validation

### 6. Data Integration & Export
- **Time Domain Calibration**: Precise calibration using 2μs interval methodology (TERRA-compatible)
- **Two-Way Travel Time**: Ice thickness calculated using round-trip electromagnetic wave propagation
- **Coordinate Interpolation**: High-resolution lat/lon interpolation from flight navigation data
- **Enhanced Output**: Complete pixel coordinates, travel times, and ice thickness measurements
- **Batch Efficiency**: Calibration reuse and method consistency for streamlined processing

--- 
## 📊 Output Files

For each processed image, ARIES generates comprehensive output files:

### 1. Enhanced Ice Thickness CSV: `{filename}_thickness.csv`
Contains full-resolution measurements with the following columns:
- **X (pixel)**: Horizontal pixel coordinate
- **Latitude**: High-precision interpolated coordinates (decimal degrees)
- **Longitude**: High-precision interpolated coordinates (decimal degrees)
- **CBD**: Control Block Distance navigation reference (where available)
- **Surface Depth (μs)**: One-way travel time to ice surface interface
- **Bed Depth (μs)**: One-way travel time to ice bed interface  
- **Ice Thickness (m)**: Calculated ice thickness using two-way travel time methodology (like TERRA)

### 2. Calibrated Visualization Plots
- **`{filename}_picked.png`**: Main time-calibrated echogram with:
  - Detected transmitter pulse location
  - Calibration pip tick marks and time grid
  - CBD position labels with coordinates
  - Time and depth axis calibration
  
- **`{filename}_time_calibrated_auto_echoes.png`**: Echogram with automatic detections:
  - Automatically traced surface echo (green dashed line)
  - Automatically traced bed echo (magenta dashed line)
  - Detection confidence indicators
  - Processing parameter annotations

### 3. Quality Assurance & Debug Output
- **`{filename}_enhanced_local_refinement_validation.png`**: CBD detection validation
- **`debug_output/`** directory containing:
  - Transmitter pulse detection diagnostics
  - Echo tracing intermediate results  
  - Parameter optimization logs
  - Processing statistics and quality metrics

### 4. Batch Processing Summaries
For batch operations, additional files include:
- **Processing logs**: Success/failure statistics across image sequence
- **Parameter optimization**: Adaptive parameter updates for flight consistency
- **Quality reports**: Automated validation results and recommendations

### Example Output Structure
```
output/
├── F125-C0468_0481_thickness.csv                    # Main data export
├── F125-C0468_0481_picked.png                       # Calibrated visualization  
├── F125-C0468_0481_time_calibrated_auto_echoes.png  # Auto-detection results
├── F125-C0468_0481_enhanced_local_refinement_validation.png
└── debug_output/
    ├── F125-C0468_0481_transmitter_pulse_detection.png
    ├── F125-C0468_0481_surface_detection_debug.png
    └── F125-C0468_0481_bed_detection_debug.png
```

---

## 🔧 Troubleshooting

### Common Issues and Solutions

#### Detection Problems
- **Transmitter pulse not found**: 
  - Check `visualize_tx_pulse_detection: true` in config
  - Adjust `search_height_ratio` (try 0.25-0.45)
  - Lower `peak_prominence` to 0.3-0.5

- **Surface echo missing or incorrect**:
  - Verify transmitter pulse detection first
  - Adjust `search_start_offset_px` based on ice type
  - Lower `peak_prominence` for weak echoes
  - Check image quality and contrast settings

- **Bed echo not detected**:
  - Ensure surface echo is correctly detected
  - Increase `search_start_offset_from_surface_px` for thick ice
  - Lower `peak_prominence` for weak bed returns
  - Check `search_end_offset_from_z_boundary_px` setting

#### Image Quality Issues
- **Poor contrast**: Adjust `percentile_low` and `percentile_high` in preprocessing
- **Too much noise**: Increase `enhancement_clahe_clip` values
- **Artifacts interfering**: Check film artifact boundary detection parameters

#### Processing Errors
- **Import errors**: Verify all dependencies installed: `pip install numpy scipy opencv-python matplotlib pandas`
- **Memory issues**: Process images individually rather than batch mode
- **File path errors**: Use absolute paths and ensure directories exist

### Debug Mode
Enable detailed debugging by setting visualization flags to `true` in config files:
```json
"visualize_tx_pulse_detection": true,
"visualize_pip_detection": true,
"save_intermediate_plots": true
```

### Getting Help
1. Check the `debug_output/` directory for diagnostic plots
2. Review processing logs for specific error messages
3. Test with provided sample data first
4. Verify configuration parameters match your data characteristics

---

## 📈 Performance Notes

- **Processing Speed**: ~30-60 seconds per image depending on size and complexity
- **Memory Usage**: ~2-4GB RAM per image during processing
- **Batch Efficiency**: Adaptive parameter learning improves accuracy in batch mode
- **Output Size**: ~1-5MB per processed image (CSV + visualizations)

---

*Part of the FrozenLegacies project — preserving and analyzing historical Antarctic radar data for climate science research through advanced automated interpretation techniques.*

