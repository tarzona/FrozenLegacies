# SPRI-NSF-TUD Campaign README file

This repository serves as a hub of scientific journals, processing codes, and REU and Mentees guide for analyzing the historical SPRI-NSF-TUD Campaign in Antarctica. 

## For REU Students and Mentees

Please go to the REU_and_Mentees folder to get started with literature review about the SPRI-NSF-TUD Campaign. There is also a mini-syllabus in the folder for guidance with what to read first. Once familiar with the SPRI-NSF-TUD Campaign, please see A_Scope_Processing and Z_Scope_Processing. Once you have picked the examples for A- and Z-scope, consult with Angelo T. for Z_Calibration, and A_Calibration to learn how to process and calibrate this historical dataset. 

## For Visitors

Please read the 'Data Processing'subsection of this README file to know more about A- and Z-scope calibration and processing.

## Data Availability

You can explore the SPRI-NSF-TUD Campaign dataset at https://www.radarfilm.studio/

If you want to download and analyze this dataset (downloading A- and Z-scopes), please request an invite link to Angelo T. (dtarzona@gatech.edu) or Brian A. (bamaro@stanford.edu).

LAT/LON/CBD for different flight numbers are located in the GitHub Repository of Radar Film Studio: https://github.com/radioglaciology/radarfilmstudio/tree/2beee065a5b9bbdda5369d91507a6dca5a48cd5f/antarctica_original_positioning
  - If you need different thematic and flight maps please talk to Angelo T.

## Data Processing

To know more about the 5 main scientific papers that discuss positioning, ways to process, and data collection of the SPRI-NSF-TUD Campaign, go to 'Papers' section.

Each 'Scope_Processing' folder has an example A- or Z-scope.
  - For A_Scope_Processing:
    -   This folder contains a MATLab code for semi-automatically picking the Main Bang, Surface Echo, and Bed Echo of each A-scope in a film.
  - For A_Scope_Calibration:
    -   This folder contains a MATLab code for calibrating the semi-automatically picks from pixels to dB relative to the scale provided by Neal 1977, Rose 1978, and Millar 1981
  - For Z_Scope_Processing_SemiAutomatic:
    -   This folder contains a Python code cowritten with Abdullah A. to semi-automatically pick Z-scopes' surface and bed feature in order to get the ice thickness.
  - For Z_Scope_Calibration:
    -   This folder contains series of MATLab code for calibratiing the semi-automatic picks from pixels to meters relative to the 'calibration pips'. See Schroeder et al. 2019

## Future Works
- Need to create a folder for 'Z_Scope_Processing_Automatic' to share with potential users as to how use picked data via semi-automatic picker as boundaries for automatic picking.
  - Need to fix film cropping between automatic and semi-automatic picker. 
