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

To know more about the 5 main scientific papers that discuss positioning, ways to process, and data collection of the SPRI-NSF-TUD Campaign, go to __Papers__ section.

Each 'Scope_Processing' folder has an example A- or Z-scope.
  - For A_Scope_Processing:
    -   This folder contains two sets of manual picker (MATLab-based) codes and automatic  picker (Python-based) for semi-automatically picking the Transmitter Pulse, Surface Echo, and Bed Echo of each A-scope in a film.
  - For Z_Scope_Processing
    -   This folder contains two sets of semi-manual and fully-automatic picker (Python-based) Python code to trace Z-scopes' transmitter pulse, surface and bed echo returns. 

