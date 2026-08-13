<p align="left">
  <img src="docs/URSA.png" alt="URSA Logo" height="120">
  <span style="font-size:2em; vertical-align: middle;">
</p>

# URSA — Unsupervised Retrieval of Signal for A-scopes

**Advanced automated detection and analysis of A-scope radar data from historical TIFF images.**

URSA is a sophisticated system for processing A-scope radar data, featuring automated detection of individual A-scope frames, signal trace extraction, grid line calibration, and intelligent echo detection. The system automatically detects transmitter pulses, surface echoes, and bed echoes while providing precise time-domain calibration and ice thickness calculations.

---

## Table of Contents

- [Quick Start](#-quick-start)
- [Features](#-features)  
- [Installation](#️-installation)
- [Usage](#-usage)
- [Configuration](#️-configuration)
- [Configuration Reference Sheet](#-configuration-reference-sheet)
- [Core Processing Pipeline](#-core-processing-pipeline)
- [Output Files](#-output-files)
- [Troubleshooting](#-troubleshooting)

---

## 🚀 Quick Start

### Windows
```cmd
pip install numpy opencv-python matplotlib scipy pandas
python main.py --input path/to/your/image.tiff
```

### Mac/Linux  
```bash
pip install numpy opencv-python matplotlib scipy pandas
python3 main.py --input path/to/your/image.tiff
```

---

## 🎯 Features

- **Automated Frame Detection**: Intelligent detection of individual A-scope frames within TIFF images
- **Signal Trace Extraction**: Advanced algorithms for extracting radar signal traces from noisy backgrounds
- **Grid Line Calibration**: Automatic detection and interpolation of time/power grid lines for accurate calibration
- **Multi-Modal Echo Detection**: 
  - Double transmitter pulse detection and analysis
  - Adaptive surface echo detection with noise floor estimation
  - Bed echo detection with geometric loss compensation
- **Time Domain Processing**: Precise conversion from pixels to microseconds and dB power levels
- **Ice Thickness Calculation**: Automatic calculation using electromagnetic wave propagation in ice
- **Navigation Integration**: CBD-based coordinate interpolation from flight navigation data
- **Quality Assurance**: Comprehensive validation and debug output for each processing stage
- **Batch Processing**: Process entire directories with consistent parameter optimization

---

## 🛠️ Installation

### System Requirements
- **Python 3.8+** (Python 3.9+ recommended)
- **RAM**: 4GB minimum, 8GB recommended for batch processing
- **Storage**: ~500MB per processed image directory

### Dependencies Installation

**Windows**:
```cmd
# Create virtual environment (recommended)
python -m venv ursa_env
ursa_env\Scripts\activate

# Install dependencies
pip install --upgrade pip
pip install numpy opencv-python matplotlib scipy pandas pathlib2
```

**Mac/Linux**:
```bash
# Create virtual environment (recommended)  
python3 -m venv ursa_env
source ursa_env/bin/activate

# Install dependencies
pip install --upgrade pip
pip install numpy opencv-python matplotlib scipy pandas
```

### Verify Installation
```bash
python -c "import numpy, cv2, matplotlib, scipy, pandas; print('All packages installed successfully!')"
```

---

## 📖 Usage

### Single Image Processing

Process a single A-scope TIFF image:
```bash
python main.py --input path/to/your/image.tiff
```

### Batch Processing  

Process all TIFF files in a directory:
```bash
python main.py --batch path/to/tiff/directory --nav path/to/navigation.csv
```

### Interactive Frame Reprocessing

Reprocess a specific frame with manual adjustments:
```bash
python main.py --input path/to/image.tiff --interactive frame_number
```

**Example**:
```bash
python main.py --input data/103/F103-C0467_0479.tiff --interactive 4
```

### Calibrated-only Montage (new)

When using Step 4 (trace review) interactively there's now a "Calibrated Plots (no picks)" button. This opens a second montage showing calibrated power vs time plots without surface/bed picks — useful when you want to visually inspect the calibrated A-scope frames and then edit `config/default_config.json` to set absolute time windows for surface/bed selection. After editing the config, return to the trace review and re-run Step 4 to evaluate your new config.

### Command Line Options

- `--input`: Single TIFF image file path
- `--batch`: Directory containing multiple TIFF files
- `--nav`: Navigation CSV file with CBD, LAT, LON columns
- `--config`: Custom configuration file path
- `--interactive`: Reprocess specific frame number with manual picker
- `--output`: Custom output directory

---

## ⚙️ Configuration

URSA uses a comprehensive JSON configuration system:

- **Main configuration**: `config/default_config.json` - Processing parameters and thresholds
- **Override capability**: Use `--config custom_config.json` to specify custom settings

Key configuration sections:
- `processing_params`: Frame detection, signal processing, echo detection parameters
- `physical_params`: Time and power axis calibration constants  
- `output`: Output directory and visualization settings

---

## 🔧 Configuration Reference Sheet

### Key Parameters in `config/default_config.json`

#### A-Scope Frame Detection
```json
"processing_params": {
    "min_frame_width_px": 1200,           // Minimum frame width to detect
    "frame_detect_min_frame_width_px": 1200, // Minimum width for valid frames
    "min_frame_gap_px": 30,               // Minimum gap between frames
    "frame_detect_gap_threshold": 0.85    // Brightness threshold for gaps
}
```

**Quick Adjustments**:
- **Missing A-scope frames**: Decrease `min_frame_width_px` to 800-1000
- **False frame detection**: Increase `min_frame_width_px` to 1400-1600
- **Frames too close together**: Increase `min_frame_gap_px` to 50-100
- **Not detecting frame gaps**: Lower `frame_detect_gap_threshold` to 0.7-0.8

#### Transmitter Pulse Detection
```json
"processing_params": {
    "tx_search_window_us": 5.0,           // Search window for TX pulse (μs)
    "tx_min_separation_us": 0.1,          // Min separation for double pulse
    "tx_max_separation_us": 3.5,          // Max separation for double pulse
    "tx_power_diff_threshold_db": 8.0     // Power difference threshold
    "tx_search_start_us": null,           // Optional absolute search start (μs) - primary if set
    "tx_search_end_us": null,             // Optional absolute search end (μs) - primary if set
}
```

**Quick Adjustments**:
- **TX pulse not found**: Increase `tx_search_window_us` to 6.0-8.0
- **False TX detections**: Decrease `tx_search_window_us` to 3.0-4.0
- **Double pulse not detected**: Adjust `tx_min_separation_us` (0.05-0.2) and `tx_max_separation_us` (2.0-4.0)

#### Surface Echo Detection
```json
"processing_params": {
    "surface_search_start_offset_us": 8.0,    // Start search offset from TX (μs)
    "surface_search_window_us": 6.0,          // Search window duration
    "surface_min_snr_db": 10.0,               // Minimum signal-to-noise ratio
    "surface_search_start_offset_px": 20,     // Pixel offset from TX pulse
    "surface_peak_distance_px": 15            // Minimum distance between peaks
    "surface_search_start_us": null,         // Optional absolute surface search start (μs)
    "surface_search_end_us": null,           // Optional absolute surface search end (μs)
}
```

**Quick Adjustments**:
- **Surface too close to TX**: Increase `surface_search_start_offset_us` to 10-12
- **Surface too far from TX**: Decrease `surface_search_start_offset_us` to 5-7
- **Missing weak surface**: Lower `surface_min_snr_db` to 5-8
- **Too many surface candidates**: Increase `surface_min_snr_db` to 12-15
- **Refine pixel-level detection**: Adjust `surface_peak_distance_px` (10-25)

#### Bed Echo Detection  
```json
"processing_params": {
    "bed_decay_start_offset_us": 1.0,         // Start of geometric loss region
    "bed_search_start_offset_us": 4.0,        // Search start from surface
    "bed_min_time_after_surface_us": 2.0,     // Minimum time gap from surface
    "bed_geometric_loss_margin_db": 20,       // Margin for geometric loss
    "bed_min_power_db": -45,                  // Minimum detectable power
    "bed_peak_prominence_db": 0.1,            // Peak prominence threshold
    "bed_relative_fallback_db_drop": 25       // Fallback threshold
    "bed_search_start_us": null,             // Optional absolute bed search start (μs)
    "bed_search_end_us": null,               // Optional absolute bed search end (μs)
    "bed_select_mode": "highest_peak",    // How to choose a bed candidate inside absolute window: "highest_peak", "highest_power" (raw max), or "most_prominent"
}
```

Note: If any of the *_search_start_us / *_search_end_us keys are set, URSA will use the specified calibrated time window (μs) as the primary method to locate the highest-power peak for the corresponding feature (TX, Surface, Bed). If no valid candidate is found inside that absolute window (or the keys are unset), URSA falls back to the original relative/heuristic search logic (offsets, decay detection, and pixel-based searches).

Important: When an absolute window is set for Surface or Bed, URSA will choose the highest local peak inside that window as the primary pick and will ignore configured minimum-power checks (e.g. `surface_min_snr_db` or `bed_min_power_db`) for that absolute-window selection. This makes absolute windows deterministic and robust to thresholding — but also means the window must be chosen carefully to avoid false positives.

### 🔎 How URSA chooses the "primary" peak (absolute window vs automatic)

You can control how URSA selects the primary peak for the transmitter (TX), surface and bed echoes in two ways:

- 1) By specifying an absolute time window in the configuration file (recommended when you know roughly where the echo should be). When any of these *_search_start_us / *_search_end_us keys are set, URSA will use that absolute window as the PRIMARY selection mechanism and will pick the single highest local peak inside the window.
- 2) If an absolute window is not set, or if URSA finds no valid peaks inside a user-specified window, URSA falls back to the automatic (heuristic) search algorithm. The automatic algorithm uses relative offsets from the detected TX pulse, SNR thresholds, geometric-loss compensation and peak prominence checks.

When an absolute window is used (absolute-window selection):
- URSA will pick the highest local peak inside the window, ignoring configurable minimum-power checks and some relative-time safeguards (this behavior makes an absolute window deterministic, but it can also select false positives if the window is too wide or misplaced).

When the absolute-window selection fails (no valid peak inside the window):
- URSA will fall back to the automatic heuristic logic to search based on offsets, SNR, pixel-based trimming and geometric loss handling. This is the original behavior when absolute windows are not provided.

### ⚙️ Config keys you can change

Here are the specific keys in `config/default_config.json` that control absolute-window behavior and related thresholds:

- Transmitter (TX) selection:
  - `tx_search_start_us` — optional absolute start time (μs) for TX search
  - `tx_search_end_us` — optional absolute end time (μs) for TX search
  - `tx_min_separation_us`, `tx_max_separation_us` — used by automatic double-pulse detection
  - `tx_power_diff_threshold_db` — threshold for double-pulse power difference

- Surface selection:
  - `surface_search_start_us` — optional absolute start time (μs) for surface search (primary if set)
  - `surface_search_end_us` — optional absolute end time (μs) for surface search
  - `surface_min_snr_db` — when using automatic detection, minimum SNR threshold for accepting a surface candidate
  - `surface_search_start_offset_us`, `surface_search_window_us` — offset-based window used by automatic detection when no absolute window is set

- Bed selection:
  - `bed_search_start_us` — optional absolute start time (μs) for bed search (primary if set)
  - `bed_search_end_us` — optional absolute end time (μs) for bed search
  - `bed_select_mode` — controls which candidate wins inside an absolute window. Valid values:
      - `highest_peak` (default): selects the peak with the highest prominence within the window
      - `highest_power`: selects the highest raw-power sample inside the window
      - `most_prominent`: selects by peak prominence (shape-based)
  - `bed_min_time_after_surface_us` — minimum time separation required by automatic detection when NOT using absolute windows
  - `bed_min_power_db` — minimum power threshold used by automatic detection

### Practical guidance

- If you already know roughly where surface/bed echoes should lie, set `surface_search_start_us`/`surface_search_end_us` and/or `bed_search_start_us`/`bed_search_end_us` to narrow windows. URSA will choose the single highest peak inside those windows, which can help make results deterministic across datasets.
- Use `bed_select_mode` to influence how an absolute-window bed pick is chosen — `highest_power` tends to bias towards strong, narrow peaks while `most_prominent` prefers wider, well-shaped returns.
- If your window contains noisy or spurious peaks, make it narrower rather than relying on thresholds; absolute-window selection ignores some threshold checks.

#### Fallback behavior

If a user-specified window fails (no peak found inside), URSA will automatically attempt the regular heuristic (offset + SNR + prominence) search — so you can safely supply windows as hints without risking total failure.

**Quick Adjustments**:
- **Bed too close to surface**: Increase `bed_min_time_after_surface_us` to 3-4
- **Bed too far from surface**: Decrease `bed_search_start_offset_us` to 2-3
- **Missing weak bed echoes**: Lower `bed_min_power_db` to -50 or -55
- **Too many bed candidates**: Increase `bed_min_power_db` to -40 or -35
- **Bed in wrong location**: Adjust `bed_decay_start_offset_us` (0.5-2.0)

### Quick Reference: Critical Parameters

| Parameter | Purpose | Typical Range | Impact |
|-----------|---------|---------------|---------|
| **Frame Detection** | | | |
| `min_frame_width_px` | A-scope frame width | 800-1600 | Frame detection sensitivity |
| `frame_detect_gap_threshold` | Gap detection | 0.7-0.9 | Inter-frame separation |
| **TX Pulse** | | | |  
| `tx_search_window_us` | TX search duration | 3.0-8.0 | TX detection window |
| `tx_power_diff_threshold_db` | Double pulse threshold | 5-12 | Double TX sensitivity |
| **Surface Echo** | | | |
| `surface_search_start_offset_us` | Surface search start | 5-12 | Surface detection timing |
| `surface_min_snr_db` | Surface SNR threshold | 5-15 | Surface detection sensitivity |
| **Bed Echo** | | | |
| `bed_min_power_db` | Minimum bed power | -55 to -35 | Bed detection threshold |
| `bed_min_time_after_surface_us` | Surface-bed separation | 1-5 | Minimum ice thickness |

### Parameter Adjustment Workflow

1. **Frame Detection First**: Ensure all A-scope frames are detected correctly
2. **TX Pulse Tuning**: Verify transmitter pulse detection in each frame  
3. **Surface Echo Optimization**: Adjust timing and sensitivity for ice surface
4. **Bed Echo Refinement**: Fine-tune power thresholds for bedrock detection
5. **Validation**: Review debug output and QA plots for each adjustment

---

## � Core Processing Pipeline

### 1. Image Preprocessing & Frame Detection
- **Sprocket Hole Masking**: Automatic masking of film sprocket holes and artifacts
- **Frame Boundary Detection**: Intelligent detection of individual A-scope frame boundaries
- **Frame Validation**: Quality assessment and frame count verification against expected CBD sequence

### 2. Signal Trace Extraction
- **Signal Detection**: Advanced algorithms to detect radar signal traces within each frame
- **Trace Cleaning**: Noise reduction and signal conditioning using adaptive smoothing
- **Quality Assessment**: Trace quality scoring to validate signal extraction

### 3. Grid Line Detection & Calibration
- **Reference Line Detection**: Automatic detection of power reference lines (e.g., -60 dB)
- **Grid Line Interpolation**: Detection and interpolation of time and power grid lines
- **Calibration Factor Calculation**: Conversion factors from pixels to microseconds and dB

### 4. Echo Detection & Analysis
- **Transmitter Pulse Detection**: 
  - Single and double transmitter pulse detection
  - Enhanced algorithm with confidence scoring
  - Time-zero reference establishment
- **Surface Echo Detection**:
  - Adaptive search window based on TX pulse location
  - Signal-to-noise ratio analysis
  - Multi-scale peak detection with validation
- **Bed Echo Detection**:
  - Geometric loss compensation
  - Search window optimization relative to surface
  - Power threshold adaptation for weak returns

### 5. Data Integration & Export
- **Time Domain Calibration**: Precise conversion to one-way travel times
- **Ice Thickness Calculation**: Physical thickness using electromagnetic wave velocity in ice
- **Navigation Integration**: CBD-based coordinate interpolation from flight data
- **Quality Assurance**: Comprehensive validation and debug output

---

## 📊 Output Files

For each processed TIFF image, URSA generates:

### 1. Primary Data Export: `{filename}_pick.csv`
Contains per-frame measurements with columns:
- **Frame**: A-scope frame number  
- **CBD**: Control Block Distance navigation reference
- **LAT, LON**: Interpolated GPS coordinates (decimal degrees)
- **Surface_Time_us**: One-way travel time to ice surface (microseconds)
- **Bed_Time_us**: One-way travel time to ice bed (microseconds)
- **Ice_Thickness_m**: Calculated ice thickness (meters)
- **Surface_Power_dB, Bed_Power_dB**: Echo power levels (dB)
- **Transmitter_Time_us, Transmitter_Power_dB**: TX pulse characteristics
- **Transmitter_X_pixel**: TX pulse pixel location

### 2. Binary Data Archive: `{filename}_pick.npz`
NumPy archive containing:
- All CSV data in structured arrays
- Processing metadata and configuration
- Quality metrics and confidence scores

### 3. Quality Assurance Visualizations
- **`{filename}_frame_verification.png`**: Frame detection validation
- **`{filename}_frame{XX}_grid_QA.png`**: Grid detection validation for each frame
- **`{filename}_frame{XX}_picked.png`**: Final results with detected echoes and calibration

### 4. Debug Output (when enabled)
- Detailed processing logs for each frame
- Intermediate processing results
- Parameter optimization traces
- Signal quality assessments

### Example Output Structure
```
output/
├── F103-C0467_0479_pick.csv                    # Main data export
├── F103-C0467_0479_pick.npz                    # Binary archive  
├── F103-C0467_0479_frame_verification.png      # Frame detection QA
├── F103-C0467_0479_frame01_grid_QA.png         # Grid detection QA
├── F103-C0467_0479_frame01_picked.png          # Final results
├── ... (additional frames)
└── debug/ (if enabled)
    ├── processing_log.txt
    └── intermediate_results/
```

---

## 🔧 Troubleshooting

### Common Issues and Solutions

#### Frame Detection Problems
- **Too few frames detected**: 
  - Lower `min_frame_width_px` (try 800-1000)
  - Adjust `frame_detect_gap_threshold` (try 0.7-0.8)
- **Too many frames detected**:
  - Increase `min_frame_width_px` (try 1400-1600)
  - Increase `min_frame_gap_px` (try 50-100)

#### Echo Detection Issues
- **Transmitter pulse not found**:
  - Increase `tx_search_window_us` to 6-8 μs
  - Check for double pulse with `double_tx_detection_enabled: true`
- **Surface echo missing**:
  - Adjust `surface_search_start_offset_us` (try 5-12 μs)
  - Lower `surface_min_snr_db` for weak echoes (try 5-8 dB)
- **Bed echo not detected**:
  - Lower `bed_min_power_db` (try -50 to -55 dB)
  - Adjust `bed_search_start_offset_us` (try 2-6 μs)

#### Processing Errors
- **Import errors**: Verify dependencies: `pip install numpy opencv-python matplotlib scipy pandas`
- **Memory issues**: Process files individually, check available RAM
- **Path errors**: Use absolute paths, ensure directories exist
- **Config errors**: Validate JSON syntax in configuration files

### Debug Mode
Enable detailed debugging in config:
```json
"output": {
    "debug_mode": true,
    "plot_dpi": 150
}
```

### Performance Optimization
- **Processing Speed**: ~2-5 minutes per image depending on complexity
- **Memory Usage**: ~1-3GB RAM per image during processing  
- **Batch Efficiency**: Use batch mode for consistent parameter optimization
- **Storage**: ~10-50MB output per processed image

---

## 📚 Additional Resources

- **Sample Data**: Contact project maintainers for access to sample TIFF files
- **Parameter Tuning**: Use interactive mode to fine-tune detection parameters
- **Batch Processing**: Process entire flight directories for consistent results
- **Navigation Data**: CBD-based coordinate files available from project archive

---

## ❓ Getting Help

1. **Check Configuration**: Verify parameter settings using the cheat sheet above
2. **Review Debug Output**: Enable debug mode and examine QA plots
3. **Test with Sample Data**: Validate installation with known-good data
4. **Contact Support**: Open an issue with configuration details and error logs

---

*Part of the FrozenLegacies project — preserving and analyzing historical Antarctic radar data through advanced automated signal processing and echo detection techniques.*




