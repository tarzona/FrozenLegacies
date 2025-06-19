## Quick Start

This package provides tools for processing A-scope radar data from TIFF images. It detects signal traces, reference lines, transmitter pulses, surface echoes, and bed echoes.

# Installation

Clone this repository: git clone cd ascope

Install the required dependencies: pip install -r docs/requirements.txt

# Usage

Command Line Interface

Process a specific image python runme/main.py –input path/to/your/image.tiff
Process with a custom configuration python runme/main.py –input path/to/your/image.tiff –config path/to/config.json
Process with a custom output directory python runme/main.py –input path/to/your/image.tiff –output-dir path/to/output
Enable debug mode python runme/main.py –input path/to/your/image.tiff –debug


# Configuration

The default configuration is stored in config/default_config.json. Physical parameters are stored in config/physical_params.json.
You can override these by providing a custom configuration file with the --config option.

# Output

Processed results are saved to the output directory specified in the configuration (default: ascope_processed).
