<p align="center">
  <img src="frozenlegacytools_img.png" alt="FrozenLegacy Tools" width="600">
</p>

# FrozenLegacy Tools

**A comprehensive suite of automated and interactive tools for processing and analyzing historical Antarctic radar sounding data collected in the 1970s SPRI/NSF/TUD airborne surveys.**

The FrozenLegacy Tools project preserves and digitizes decades of ice-penetrating radar data, enabling modern climate science research through advanced computer vision and signal processing techniques. Our tools automatically detect ice surface and bed interfaces, calculate ice thickness, and provide precise geolocation from historical radar echograms and A-scope data.

---

## 📊 Download Sample Data

**⚠️ IMPORTANT: Download sample Z-scope and A-scope files to test the tools:**

### 📥 [Download Sample Data Files](https://gatech.box.com/s/125xh7e7vera3gawucu1gybczwg002kw)

This link contains sample TIFF files and navigation CSV data needed to run all the tools in this suite. Download and extract the files before using any of the applications.

---

## 🛠️ Available Tools

### 🎯 [ASTRA - Amplitude Signal Tracing & Retrieval Application](./Amplitude%20Signal%20Tracing%20and%20Retrieval%20Application%20%28A.S.T.R.A.%29/)
**Interactive GUI tool for manual A-scope feature picking**

- **Purpose**: Interactive digitization of A-scope radar features (surface, bed, noise floor, main bang)
- **Input**: Historical radar TIFF images from A-scope displays
- **Method**: Manual feature picking with guided workflow and axis calibration
- **Output**: Georeferenced CSV data with travel-time and power measurements + QC visualization
- **Best for**: Detailed analysis of specific radar traces, training datasets, quality control

**Key Features:**
- Snap-to-trace semi-automatic picking for consistency
- Real-time travel-time and power readouts (µs and dB)
- Automatic geolocation using flight navigation CSV files
- QC image generation with overlaid picks and labels
- Batch processing across entire directories

---

### 🚀 [ARIES - Automatic Radar Interpretation and Extraction System](./Automatic%20Radar%20Interpretation%20and%20Extraction%20System%20%28A.R.I.E.S.%29/)
**Fully automated Z-scope echogram processing**

- **Purpose**: Automated processing and interpretation of Z-scope radar echograms
- **Input**: Historical Z-scope radar sounding images (echograms)
- **Method**: Computer vision and signal processing for automatic feature detection
- **Output**: Quantitative ice thickness measurements with coordinate data
- **Best for**: Large-scale batch processing, consistent automated analysis

**Key Features:**
- Intelligent transmitter pulse detection and time-domain calibration
- Multi-scale adaptive surface and bed echo detection algorithms  
- Automatic film artifact boundary detection and removal
- GPS coordinate interpolation using flight navigation data
- Quality validation with automatic parameter optimization
- Comprehensive debug output and visualization

---

### 🎨 [TERRA - Tracing & Extraction of Radar Reflections for Analysis](./Tracing%20and%20Extraction%20of%20Radar%20Reflections%20for%20Analysis%20%28T.E.R.R.A.%29/)
**Interactive polygon-based radar reflection digitization**

- **Purpose**: Interactive digitization of surface and bed reflections from radar images
- **Input**: Radar TIFF images (works with both A-scope and Z-scope data)
- **Method**: Polygon-based manual tracing with calpip time-domain conversion
- **Output**: Geolocated picks with optional time-domain travel-time calculations
- **Best for**: Complex radar signatures requiring detailed manual interpretation

**Key Features:**
- Polygon-based digitization for complex reflection patterns
- CBD (Control Block Distance) labeling with lat/lon interpolation
- Calpip detection workflow for precise time-domain calculations
- High-DPI PNG visualization with dual y-axes (depth/time)
- NaN-tolerant handling of missing or undefined picks
- Automatic averaging of pixel intensities within drawn polygons

---

### 🔍 [URSA - Unsupervised Retrieval of Signal for A-scopes](./Unsupervised%20Retrieval%20of%20Signal%20for%20A-scopes%20%28U.R.S.A.%29/)
**Advanced automated A-scope frame detection and analysis**

- **Purpose**: Sophisticated automated detection and analysis of A-scope radar data
- **Input**: Historical A-scope TIFF images containing multiple radar frames
- **Method**: Automated frame detection, signal extraction, and intelligent echo detection
- **Output**: Comprehensive analysis with time-domain calibration and ice thickness
- **Best for**: Processing images with multiple A-scope frames, automated workflow

**Key Features:**
- Automated detection of individual A-scope frames within TIFF images
- Advanced signal trace extraction from noisy backgrounds
- Grid line calibration with automatic time/power axis detection
- Multi-modal echo detection (transmitter pulse, surface, bed echoes)
- Quality assurance with comprehensive validation and debug output
- Interactive override capability for manual refinement

---

## 🚀 Quick Start Guide

### Prerequisites
All tools require **Python 3.7+** with the following packages:
```bash
pip install numpy matplotlib opencv-python scipy pandas pillow
```

**Additional requirements:**
- **Windows**: `tkinter` (usually included with Python)
- **Mac**: `PyQt5` for better compatibility: `pip install PyQt5`

### Basic Usage

#### 1. Download Sample Data
First, download the sample files from the link above and extract them to a working directory.

#### 2. Choose Your Tool Based on Data Type

**For A-scope data (single radar traces):**
```bash
# Interactive manual picking
python ASTRA/ASTRA.py

# Automated batch processing  
python URSA/main.py --input path/to/ascope.tiff
```

**For Z-scope data (radar echograms):**
```bash
# Automated processing
python ARIES/main.py --input path/to/zscope.tiff --nav path/to/nav.csv

# Interactive polygon tracing
python TERRA/TERRA.py tiff "path/to/tiffs" output "path/to/results" nav "path/to/nav.csv"
```

#### 3. Expected File Structure
```
your_project_folder/
├── TIFF_files/
│   ├── F125-C0919_C0932.tiff    # Radar images
│   └── ...
├── NAV/
│   └── 125.csv                  # Navigation data (CBD, LAT, LON)
└── OUTPUT/                      # Results (created automatically)
```

---

## 📁 File Requirements

### TIFF Files
- **Naming Convention**: `FXXX-CAAAA_CBBBB.tiff` (e.g., `F125-C0919_C0932.tiff`)
- **Format**: Standard TIFF format with radar image data
- **Content**: Historical radar sounding data (A-scope or Z-scope)

### Navigation CSV
- **Required Columns**: `CBD`, `LAT`, `LON`
- **Purpose**: GPS coordinate interpolation for georeferencing
- **Example**: `125.csv` for flight F125

---

## 📊 Output Files

All tools generate comprehensive output including:

### CSV Data Files
- **Coordinates**: Pixel positions of detected/picked features
- **Measurements**: Travel-time (µs), power (dB), ice thickness (m)
- **Geolocation**: CBD, latitude, longitude for each measurement
- **Quality Metrics**: Processing parameters and confidence scores

### Visualization
- **QC Images**: High-quality PNG files with overlaid picks and annotations
- **Debug Plots**: Detailed processing validation (when enabled)
- **Time-Calibrated Views**: Dual-axis plots showing both pixels and physical units

### Data Products
- **Binary Archives**: NumPy `.npz` files with structured processing results
- **Processing Logs**: Detailed workflow information for reproducibility

---

## 🔧 Installation Guide

### Windows
```cmd
# Create virtual environment
python -m venv frozenlegacy_env
frozenlegacy_env\Scripts\activate

# Install dependencies
pip install --upgrade pip
pip install numpy matplotlib opencv-python scipy pandas pillow

# Verify installation
python -c "import numpy, cv2, matplotlib, pandas, scipy; print('Installation successful!')"
```

### Mac/Linux
```bash
# Create virtual environment  
python3 -m venv frozenlegacy_env
source frozenlegacy_env/bin/activate

# Install dependencies
pip install --upgrade pip
pip install numpy matplotlib opencv-python scipy pandas pillow PyQt5

# Verify installation
python3 -c "import numpy, cv2, matplotlib, pandas, scipy; print('Installation successful!')"
```

---

## 📚 Documentation

Each tool includes comprehensive documentation:

- **[ASTRA Documentation](./Amplitude%20Signal%20Tracing%20and%20Retrieval%20Application%20%28A.S.T.R.A.%29/README.md)** - Interactive A-scope feature picking
- **[ARIES Documentation](./Automatic%20Radar%20Interpretation%20and%20Extraction%20System%20%28A.R.I.E.S.%29/README.md)** - Automated Z-scope processing  
- **[TERRA Documentation](./Tracing%20and%20Extraction%20of%20Radar%20Reflections%20for%20Analysis%20%28T.E.R.R.A.%29/README.md)** - Interactive polygon digitization
- **[URSA Documentation](./Unsupervised%20Retrieval%20of%20Signal%20for%20A-scopes%20%28U.R.S.A.%29/README.md)** - Advanced A-scope automation

---

## 🔍 Tool Selection Guide

| **Data Type** | **Processing Method** | **Recommended Tool** | **Use Case** |
|---------------|----------------------|---------------------|--------------|
| A-scope | Manual/Interactive | **ASTRA** | Detailed feature analysis, training data |
| A-scope | Automated | **URSA** | Batch processing, consistent analysis |
| Z-scope | Automated | **ARIES** | Large-scale processing, research datasets |
| Z-scope | Manual/Interactive | **TERRA** | Complex signatures, custom analysis |

### Processing Workflow Recommendations

#### For Research Projects:
1. **Start with automated tools** (ARIES/URSA) for initial processing
2. **Use interactive tools** (ASTRA/TERRA) for quality control and validation
3. **Combine outputs** for comprehensive analysis

#### For Data Preservation:
1. **Use ARIES** for systematic Z-scope processing
2. **Use URSA** for comprehensive A-scope digitization
3. **Archive both raw and processed data**

---

## 🚨 Troubleshooting

### Common Issues

#### Installation Problems
```bash
# Python not found
# Windows: Add Python to PATH during installation
# Mac: Install via Homebrew: brew install python

# Package installation fails
pip install --user package_name  # Install for current user only
python -m pip install --upgrade pip  # Update pip first
```

#### Processing Issues
```bash
# No TIFF files found
# Ensure files have extensions: .tif, .tiff, .TIF, .TIFF

# Navigation CSV issues  
# Check columns: CBD, LAT, LON
# Verify filename matches flight number (e.g., 125.csv for F125)

# Memory issues
# Close other applications
# Process files individually instead of batch mode
```

#### GUI Problems
```bash
# Windows: tkinter not found
# Reinstall Python with "tcl/tk and IDLE" option

# Mac: GUI not appearing
# Install PyQt5: pip install PyQt5
# Use Mac-specific versions when available
```

---

## 💡 Tips for Success

### Data Preparation
- **Organize files** by flight number and chronological order
- **Verify navigation data** matches TIFF filename conventions
- **Check image quality** - tools work best with clear, contrast-rich radar data

### Processing Strategy
- **Start small** - process individual files before batch operations
- **Use debug mode** to understand processing steps
- **Save intermediate results** for quality control
- **Document processing parameters** for reproducibility

### Quality Control
- **Review QC images** for detection accuracy
- **Check coordinate interpolation** against known locations
- **Validate ice thickness** calculations against field measurements
- **Use multiple tools** for cross-validation of results

---

## 🎯 Research Applications

### Climate Science
- **Ice sheet evolution**: Long-term thickness change analysis
- **Basal conditions**: Bed topography and subglacial processes  
- **Mass balance studies**: Historical ice thickness for modern comparisons

### Glaciology
- **Ice dynamics**: Flow patterns and velocity calculations
- **Structural analysis**: Internal layer detection and dating
- **Thermal regime**: Surface and basal echo characteristics

### Geophysics
- **Bedrock mapping**: Subglacial topography reconstruction
- **Sediment analysis**: Basal reflection characterization
- **Electromagnetic properties**: Ice permittivity and attenuation

---

## 📞 Support & Community

### Getting Help
1. **Check tool-specific documentation** for detailed usage instructions
2. **Review sample data** to understand expected input formats  
3. **Use debug modes** to diagnose processing issues
4. **Contact project maintainers** for technical support

### Contributing
- **Report bugs** and suggest improvements
- **Share processing parameters** for different ice types
- **Contribute sample datasets** for tool validation
- **Develop new analysis modules** using the existing framework

---

## 🏛️ Project Background

The FrozenLegacy Tools project digitizes and preserves historical Antarctic radar sounding data collected during pioneering 1970s airborne surveys conducted by the Scott Polar Research Institute (SPRI), National Science Foundation (NSF), and Technical University of Denmark (TUD). 

These surveys represent some of the earliest systematic ice-penetrating radar measurements across Antarctica, providing crucial baseline data for understanding long-term ice sheet changes in the context of modern climate research.

### Historical Context
- **Data Collection**: 1970s airborne radar sounding campaigns
- **Geographic Coverage**: Major Antarctic ice sheets and outlet glaciers
- **Technical Specifications**: VHF ice-penetrating radar systems
- **Data Products**: Film-based A-scope traces and Z-scope echograms
- **Scientific Value**: Baseline measurements for climate change studies

### Modern Digitization
- **Preservation**: Converting analog film records to digital formats
- **Analysis**: Extracting quantitative measurements from historical images
- **Integration**: Combining with modern datasets for temporal analysis
- **Accessibility**: Making historical data available to the research community

---

## 📄 Citation & Acknowledgments

If you use FrozenLegacy Tools in your research, please cite:

```
FrozenLegacy Tools: Automated and Interactive Processing Suite for Historical Antarctic Radar Data
Georgia Institute of Technology - Polar Glaciology and Satellite Laser Altimetry Laboratory
```

### Acknowledgments
- **Historical Data**: Scott Polar Research Institute, National Science Foundation, Technical University of Denmark
- **Development**: Georgia Institute of Technology
- **Funding**: [Grant/funding information]
- **Community**: Antarctic research community for validation and feedback

---

## 📋 License

[License information - specify appropriate license for the project]

---

*Preserving the past to understand the future - FrozenLegacy Tools enables modern analysis of historical Antarctic radar data for climate science research.*