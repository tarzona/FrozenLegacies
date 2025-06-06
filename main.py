#!/usr/bin/env python3
import os
import sys
import argparse

# Get the directory of the current script (main.py)
current_dir = os.path.dirname(os.path.abspath(__file__))
# Get the directory containing the 'ascope' package (parent of 'runme')
parent_dir = os.path.dirname(current_dir)
# Add to Python path
sys.path.insert(0, parent_dir)

from ascope.ascope_processor import AScope


def main():
    """Main function to process A-scope radar data."""
    parser = argparse.ArgumentParser(description="Process A-scope radar data.")
    parser.add_argument("--input", required=True, help="Path to input image file")

    # Set default config path relative to main.py
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_config = os.path.join(script_dir, "config", "default_config.json")

    parser.add_argument(
        "--config", default=default_config, help="Path to configuration file"
    )
    parser.add_argument("--output", default=None, help="Path to output directory")
    args = parser.parse_args()

    # Create processor instance
    processor = AScope(args.config)

    # Process the image
    processor.process_image(args.input, args.output)

    print("Processing complete!")


if __name__ == "__main__":
    main()
