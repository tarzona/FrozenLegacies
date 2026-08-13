# SPRI-NSF-TUD Campaign README file

This repository serves as a hub of scientific journals, processing codes, and REU and Mentees guide for analyzing the historical SPRI-NSF-TUD Campaign in Antarctica. 

## Repository Structure

### Frozen Legacy Tools
This folder contains the latest, production-ready processing tools for analyzing SPRI-NSF-TUD radar data:

- **ASTRA** (Amplitude Signal Tracing & Retrieval Application): Interactive GUI tool for picking A-scope features (surface, bed, noise floor, and main bang) from historical radar TIFF images
- **ARIES** (Automatic Radar Interpretation and Extraction System): Advanced automated system for processing Z-scope radar sounding images (echograms) using computer vision and signal processing
- **TERRA** (Tracing & Extraction of Radar Reflections for Analysis): Interactive tool for digitizing surface and bed reflections from historical radar TIFF images with CBD labeling and time-domain conversion
- **URSA** (Unsupervised Retrieval of Signal for A-scopes): Sophisticated automated system for A-scope radar data processing with intelligent echo detection and calibration

### OLD_Versions
Contains legacy versions of processing tools for reference and backward compatibility:
- **A_Scope_Processing**: Earlier version of A-scope analysis tools
- **Z_Scope_Processing**: Earlier version of Z-scope analysis tools

### Papers
Scientific literature and documentation related to the SPRI-NSF-TUD Campaign, including 5 key papers covering positioning, processing methods, and data collection techniques. Includes guidance materials for literature review.

### REU_and_Mentees
Resources and guidance for Research Experience for Undergraduates (REU) students and mentees, including project descriptions, tentative schedules, and semester-specific materials.

## For REU Students and Mentees

Please go to the **REU_and_Mentees** folder to get started with literature review about the SPRI-NSF-TUD Campaign. There is also a mini-syllabus in the folder for guidance with what to read first. Once familiar with the SPRI-NSF-TUD Campaign, explore the **Frozen Legacy Tools** for the latest processing capabilities, or refer to **OLD_Versions** for legacy tools. For processing examples, consult with Angelo T. for guidance on calibration and processing of this historical dataset. 

## For Visitors

Please read the 'Data Processing'subsection of this README file to know more about A- and Z-scope calibration and processing.

## Data Availability

You can explore the SPRI-NSF-TUD Campaign dataset at https://www.radarfilm.studio/

If you want to download and analyze this dataset (downloading A- and Z-scopes), please request an invite link to Angelo T. (dtarzona@gatech.edu) or Brian A. (bamaro@stanford.edu).

LAT/LON/CBD for different flight numbers are located in the GitHub Repository of Radar Film Studio: https://github.com/radioglaciology/radarfilmstudio/tree/2beee065a5b9bbdda5369d91507a6dca5a48cd5f/antarctica_original_positioning
  - If you need different thematic and flight maps please talk to Angelo T.

## Data Processing

### Current Tools (Frozen Legacy Tools)
For the most up-to-date and feature-rich processing capabilities, use the tools in the **Frozen Legacy Tools** folder:

- **ASTRA**: Interactive A-scope feature picking with GUI interface and CSV export
- **ARIES**: Fully automated Z-scope processing with computer vision techniques
- **TERRA**: Interactive radar reflection digitization with geolocated outputs
- **URSA**: Advanced automated A-scope processing with intelligent detection algorithms

### Legacy Tools (OLD_Versions)
The **OLD_Versions** folder contains earlier versions of processing tools:
  - **A_Scope_Processing**: Contains manual picker (MATLAB-based) and automatic picker (Python-based) codes for semi-automatically picking the Transmitter Pulse, Surface Echo, and Bed Echo of each A-scope in a film
  - **Z_Scope_Processing**: Contains semi-manual and fully-automatic picker (Python-based) code to trace Z-scopes' transmitter pulse, surface and bed echo returns

### Scientific Background
To learn about the 5 main scientific papers that discuss positioning, processing methods, and data collection of the SPRI-NSF-TUD Campaign, explore the **Papers** folder. This includes key publications from Schroeder et al. (2019), Millar et al. (1981), Drewry et al. (1982), Bingham and Siegert (2009), and Karlsson et al. (2024). 

