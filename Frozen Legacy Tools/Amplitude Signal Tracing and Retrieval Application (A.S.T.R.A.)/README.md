<p align="left">
  <img src="docs/ASTRA.png" alt="ASTRA Logo" height="120">
  <span style="font-size:2em; vertical-align: middle;">
</p>

# ASTRA — Amplitude Signal Tracing & Retrieval Application

**ASTRA** is a GUI tool for interactively picking A‑scope features (surface, bed, noise floor, and main bang) from historical radar **TIFF** images, then exporting per‑trace measurements and geolocation to **CSV**, with a QC **PNG** saved for each processed image.

## 🚀 Quick Start

### Windows
```cmd
pip install numpy matplotlib Pillow scipy pandas
python ASTRA.py
```

### Mac  
```bash
pip install numpy matplotlib Pillow scipy pandas PyQt5
python3 ASTRA_Mac.py
```

---

## Table of Contents

- [Features](#-features)
- [Requirements](#-requirements)
- [Installation](#-installation)
- [Usage](#-usage)
  - [GUI mode (default)](#gui-mode-default)
  - [CLI mode (custom directories)](#cli-mode-custom-directories)
- [Core Workflow](#-core-workflow)
- [Output Files](#-output-files)
- [File Naming & Navigation CSV](#-file-naming--navigation-csv)
- [Notes & Tips](#-notes--tips)

---

## 🚀 Features

- **Pick four A‑scope features:** Surface, Bed, Noise Floor, Main Bang
- **Axis setup tools:**
  - Y‑axis reference line (sets the −60 dB line) and 4‑point dB scale
  - X‑axis reference line (0 µs) and 4‑point time scale
- **Snap‑to‑trace semi‑automatic picking:** y‑coordinates snap to a smoothed, thresholded red trace
- **Automatic travel‑time & power readouts** for each picked feature (in µs and dB)
- **Per‑trace geolocation:** Interpolated CBD values with **LAT/LON** lookup from a flight navigation CSV
- **QC image generation:** Saves a PNG with all picks and labels overlaid
- **Batch over folders:** Walks through all TIFFs in a directory
- **OUTPUT management:** Moves the original **TIFF**, the **CSV**, and the **QC PNG** into an `OUTPUT/` folder

---

## 📋 Requirements

### System Requirements
- **Python 3.7+** (recommended: Python 3.8 or newer)
- **Operating System**: Windows 10/11 or macOS 10.14+

### Python Packages

**For Windows (`ASTRA.py`)**:
- `numpy` - Numerical computing
- `matplotlib` - Plotting and GUI widgets  
- `Pillow` (PIL) - Image processing
- `scipy` - Scientific computing (Gaussian filtering)
- `pandas` - Data manipulation and CSV output
- `tkinter` - GUI framework (usually included with Python)

**For Mac (`ASTRA_Mac.py`)**:
- `numpy`, `matplotlib`, `Pillow`, `scipy`, `pandas` (same as Windows)
- `PyQt5` - Cross-platform GUI framework (better Mac compatibility)

**Standard Library** (included with Python):
- `argparse`, `os`, `glob`, `shutil`, `textwrap`, `sys`

### Quick Install Commands

**Windows**:
```cmd
pip install numpy matplotlib Pillow scipy pandas
```

**Mac**:
```bash
pip install numpy matplotlib Pillow scipy pandas PyQt5
```

---

## 🛠️ Installation

ASTRA runs as a single script; no package install required.

### Windows Installation

1. **Install Python 3.x** (if not already installed):
   - Download from [python.org](https://python.org/downloads/)
   - During installation, check "Add Python to PATH"

2. **Open Command Prompt or PowerShell** and run:
```cmd
# Create virtual environment (recommended)
python -m venv astra_env
astra_env\Scripts\activate

# Install required packages
python -m pip install --upgrade pip
pip install numpy matplotlib Pillow scipy pandas
```

3. **Place files**:
   - Put `ASTRA.py` (or `ASTRA_Mac.py`) in your working directory
   - Place TIFF files in the same directory, OR prepare a separate TIFF directory
   - Place navigation CSV file (e.g., `125.csv`) in same directory or prepare path

### Mac Installation

1. **Install Python 3.x** (if not already installed):
   - Using Homebrew: `brew install python`
   - Or download from [python.org](https://python.org/downloads/)

2. **Open Terminal** and run:
```bash
# Create virtual environment (recommended)
python3 -m venv astra_env
source astra_env/bin/activate

# Install required packages
python3 -m pip install --upgrade pip
pip install numpy matplotlib Pillow scipy pandas PyQt5

# For Mac compatibility, use ASTRA_Mac.py which includes PyQt5 support
```

3. **Place files**:
   - Put `ASTRA_Mac.py` in your working directory
   - Place TIFF files in the same directory, OR prepare a separate TIFF directory  
   - Place navigation CSV file (e.g., `125.csv`) in same directory or prepare path

### File Structure
```
your_project_folder/
├── ASTRA.py (Windows) or ASTRA_Mac.py (Mac)
├── *.tif or *.tiff files (radar images)
├── 125.csv (navigation file, optional)
├── NAV/ (optional navigation folder)
└── OUTPUT/ (created automatically)
```

---

## 🖱️ Usage

### Windows Usage

#### GUI mode (default)
1) Open **Command Prompt** or **PowerShell** in your project folder
2) Activate environment: `astra_env\Scripts\activate`
3) Run:
```cmd
python ASTRA.py
```
4) Follow the on‑screen buttons to set axes and make picks
5) Click **Next A‑scope** after each trace; **Done** when finishing a file

#### CLI mode (custom directories)
```cmd
# Activate environment first
astra_env\Scripts\activate

# Named arguments
python ASTRA.py --tiff "C:\path\to\tiff_dir" --nav "C:\path\to\125.csv"

# Or positional arguments
python ASTRA.py "C:\path\to\tiff_dir" "C:\path\to\125.csv"
```

### Mac Usage

#### GUI mode (default)
1) Open **Terminal** in your project folder
2) Activate environment: `source astra_env/bin/activate`
3) Run:
```bash
python3 ASTRA_Mac.py
```
4) Follow the on‑screen buttons to set axes and make picks
5) Click **Next A‑scope** after each trace; **Done** when finishing a file

#### CLI mode (custom directories)
```bash
# Activate environment first
source astra_env/bin/activate

# Named arguments
python3 ASTRA_Mac.py --tiff "/path/to/tiff_dir" --nav "/path/to/125.csv"

# Or positional arguments
python3 ASTRA_Mac.py "/path/to/tiff_dir" "/path/to/125.csv"
```

**Note**: All outputs go to `OUTPUT/` folder next to the ASTRA script.

---

## 🔧 Core Workflow

1. **Open image & set crop** (internally crops rows ~150–1800 px to avoid margins).
2. **Set Y‑axis reference (−60 dB):** click the reference line location once.
3. **Build Y dB scale:** click **4** y‑points; ASTRA draws green gridlines and labels (`-55, -45, ... 0 dB`).
4. **Set X‑axis reference (0 µs):** click once to mark time zero.
5. **Build X µs scale:** click **4** x‑points; ASTRA draws orange gridlines every **3 µs** up to **30 µs**.
6. **Pick features per A‑scope:** choose **Surface**, **Bed**, **Noise Floor**, or **Main Bang**, then click on the trace.
   - y‑picks **snap** to the red auto‑trace (smoothed & thresholded) for consistency.
   - “**No Surface**” / “**No Bed**” record missing values when appropriate.
7. **Move along the image:** use the **X Position** slider (or **Move 1000 px**) to pan.
8. **Next A‑scope:** stores the current picks and advances the A‑scope counter.
9. **Done:** writes the CSV, saves a QC PNG, and moves **TIFF+CSV+PNG** into `OUTPUT/`.

---

## 📊 Output Files

For each input TIFF:

1) **CSV** — `<filename>.csv` with per‑A‑scope measurements and geolocation. Columns include:
   - `A-scope Number`, `FLT`
   - `x_surface_px`, `y_surface_px`, `x_bed_px`, `y_bed_px`
   - `x_noisefloor_px`, `y_noisefloor_px`, `x_mainbang_px`, `y_mainbang_px`
   - `reference_line_y`, `reference_line_x`
   - `surface_us`, `surface_dB`, `bed_us`, `bed_dB`
   - `noisefloor_us`, `noisefloor_dB`, `mainbang_us`, `mainbang_dB`
   - `Filename`
   - `CBD`, `LAT`, `LON` *(if nav CSV present)*
   - `surface_m`, `bed_m`, `mainbang_m`, `h_ice_m`
     - Distance conversion uses: **meters = (time_us / 2) × 168**

2) **QC PNG** — `<filename>_QC.png` with all picked points and labels drawn on the original image.

3) **OUTPUT folder** — ASTRA moves the **TIFF**, **CSV**, and **QC PNG** into `OUTPUT/` (created next to `ASTRA.py`).

---

## 🗂️ File Naming & Navigation CSV

- **TIFF filenames** should be `FXXX-CAAAA_CBBBB.tif` (e.g., `F125-C0999_C1013.tif`).  
  ASTRA parses the **flight number** and **CBD range** from this pattern.
- **Navigation CSV** (either `--nav` path or `FXXX.csv` in the working folder) must include:  
  `CBD, LAT, LON`  
  ASTRA linearly distributes A‑scopes across the file’s CBD range and looks up **LAT/LON** for each CBD.

---

## 💡 Notes & Tips

- **Axis lines:** dB grid starts at **−55 dB** by design; time grid steps by **3 µs**.
- **Reference capture:** pixel locations for the −60 dB line and 0 µs are stored in the CSV (`reference_line_y`, `reference_line_x`).
- **Auto‑trace:** computed via Gaussian smoothing and dark‑pixel thresholding; picks snap to the trace along x.
- **Required to advance:** **Noise Floor** and **Main Bang** must be present for an A‑scope to be saved.
- **Large TIFFs:** `Image.MAX_IMAGE_PIXELS = None` avoids PIL's DecompressionBomb warning.
- **GUI everywhere:** even in CLI mode, ASTRA opens the same GUI for consistent interaction.
- **Platform differences:** Use `ASTRA.py` for Windows (tkinter), `ASTRA_Mac.py` for Mac (PyQt5).

### Platform-Specific Features

**Windows Version (`ASTRA.py`)**:
- Uses `tkinter` for GUI (built into most Python installations)
- Native Windows look and feel
- Optimized for Windows file path handling

**Mac Version (`ASTRA_Mac.py`)**:
- Uses `PyQt5` for better Mac compatibility
- Native Mac look and feel  
- Enhanced cross-platform matplotlib backend (`Qt5Agg`)
- Better handling of Mac-specific display issues

---

## 🔧 Troubleshooting

### Common Issues

**Windows**:
- **"tkinter not found"**: Reinstall Python with "tcl/tk and IDLE" option checked
- **"pip not recognized"**: Add Python to PATH during installation or reinstall Python
- **Permission errors**: Run Command Prompt as Administrator, or use `--user` flag: `pip install --user package_name`

**Mac**:
- **GUI not appearing**: Install PyQt5: `pip install PyQt5`
- **"python3 not found"**: Install Python via Homebrew: `brew install python`
- **Package installation fails**: Try using `python3 -m pip install` instead of `pip`

**Both Platforms**:
- **"No TIFF files found"**: Ensure TIFF files are in correct directory and have extensions: `.tif`, `.tiff`, `.TIF`, `.TIFF`
- **Navigation CSV issues**: Check that CSV has columns `CBD`, `LAT`, `LON` and filename matches flight number
- **Memory issues with large TIFFs**: Close other applications; ASTRA automatically handles large image processing

### Getting Help
If you encounter issues:
1. Check that all required packages are installed
2. Verify Python version (3.7+ required)
3. Try running with Python directly: `python -c "import numpy, matplotlib, pandas; print('All packages OK')"`
4. Check file permissions in your working directory

---

*Part of the FrozenLegacies project — preserving and analyzing historical Antarctic radar data for climate science research.*
