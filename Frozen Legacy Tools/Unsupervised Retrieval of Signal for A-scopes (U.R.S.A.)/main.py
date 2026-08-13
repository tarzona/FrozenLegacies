#!/usr/bin/env python3

import os
import sys
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from ascope_processor import AScope

# Get the directory of the current script (main.py)
current_dir = os.path.dirname(os.path.abspath(__file__))

# Add the current directory to Python path for imports
sys.path.insert(0, current_dir)

def load_navigation_file(nav_path):
    """
    Load navigation CSV file with CBD, LAT, LON columns.
    
    Args:
        nav_path (str): Path to navigation CSV file
        
    Returns:
        dict: Dictionary mapping CBD to (LAT, LON) tuples
    """
    if not nav_path or not os.path.exists(nav_path):
        return {}
        
    try:
        nav_df = pd.read_csv(nav_path)
        
        # Validate required columns
        required_cols = ['CBD', 'LAT', 'LON']
        missing_cols = [col for col in required_cols if col not in nav_df.columns]
        if missing_cols:
            print(f"ERROR: Navigation file missing columns: {missing_cols}")
            return {}
            
        # Create CBD -> (LAT, LON) mapping
        nav_dict = {}
        for _, row in nav_df.iterrows():
            cbd = int(row['CBD'])
            lat = float(row['LAT'])
            lon = float(row['LON'])
            nav_dict[cbd] = (lat, lon)
            
        print(f"INFO: Loaded navigation data for {len(nav_dict)} CBD points")
        return nav_dict
        
    except Exception as e:
        print(f"ERROR: Failed to load navigation file {nav_path}: {e}")
        return {}


def process_batch_folder(folder_path, nav_path=None, config_path=None, output_dir=None, interactive_approval=True):
    """
    Process all TIFF files in a folder with optional navigation data.
    
    Args:
        folder_path (str): Path to folder containing TIFF files
        nav_path (str): Path to navigation CSV file
        config_path (str): Path to config file
        output_dir (str): Output directory
    """
    folder_path = Path(folder_path)
    if not folder_path.exists() or not folder_path.is_dir():
        print(f"ERROR: Folder not found: {folder_path}")
        return False
        
    # Find all TIFF files
    tiff_files = list(folder_path.glob("*.tiff")) + list(folder_path.glob("*.tif"))
    if not tiff_files:
        print(f"ERROR: No TIFF files found in {folder_path}")
        return False
        
    print(f"INFO: Found {len(tiff_files)} TIFF files for batch processing")
    
    # Load navigation data if provided
    nav_dict = load_navigation_file(nav_path) if nav_path else {}
    
    # Process each file
    success_count = 0
    for i, tiff_file in enumerate(tiff_files, 1):
        print(f"\n{'='*60}")
        print(f"PROCESSING FILE {i}/{len(tiff_files)}: {tiff_file.name}")
        print(f"{'='*60}")
        
        try:
            # Initialize processor for each file
            processor = AScope(config_path)
            if output_dir:
                processor.set_output_directory(output_dir)
                
            # Set navigation data if available
            if nav_dict:
                processor.nav_dict = nav_dict
            # Enable interactive approval flow for this processor if requested
            processor.set_interactive_mode(enabled=bool(interactive_approval), defer_frame_plots=bool(interactive_approval))
            
            # Process the image
            processor.process_image(str(tiff_file), output_dir)
            success_count += 1
            
        except Exception as e:
            print(f"ERROR: Failed to process {tiff_file.name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\n{'='*60}")
    print(f"BATCH PROCESSING COMPLETE: {success_count}/{len(tiff_files)} files processed successfully")
    print(f"{'='*60}")
    
    return success_count == len(tiff_files)


def process_single_frame_interactive(
    input_file, frame_number, config_path=None, output_dir=None
):
    """
    Process a single frame with interactive manual override capability.

    Args:
        input_file (str): Path to input TIFF file
        frame_number (int): Frame number to process (1-indexed)
        config_path (str): Path to config file
        output_dir (str): Output directory
    """
    from functions.interactive_override import ManualPickOverride
    from functions.utils import load_and_preprocess_image
    from functions.preprocessing import mask_sprocket_holes, detect_ascope_frames

    print(f"\n{'=' * 60}")
    print("INTERACTIVE MANUAL PEAK OVERRIDE SESSION")
    print(f"{'=' * 60}")
    print(f"Input file: {input_file}")
    print(f"Frame: {frame_number}")

    # Initialize processor
    processor = AScope(config_path)
    if output_dir:
        processor.set_output_directory(output_dir)

    # Load and preprocess image
    image, base_filename = load_and_preprocess_image(input_file)

    # Extract CBD sequence
    cbd_list = processor._extract_cbd_sequence_from_filename(base_filename)

    # Mask sprocket holes and detect frames
    masked_image, _ = mask_sprocket_holes(image, processor.config)
    frames = detect_ascope_frames(masked_image, processor.config)

    if frame_number < 1 or frame_number > len(frames):
        print(f"ERROR: Frame number {frame_number} out of range (1-{len(frames)})")
        return False

    # Extract frame bounds
    left, right = frames[frame_number - 1]

    print(f"INFO: Processing frame {frame_number} (columns {left}-{right})...")

    # Process frame to the point where we have automatic picks
    frame_data = processor._process_frame_for_override(
        masked_image, left, right, base_filename, frame_number
    )

    if frame_data is None:
        print("ERROR: Frame processing failed")
        return False

    # Launch interactive manual override session
    override_session = ManualPickOverride(
        frame_data["frame_img"],
        frame_data["signal_x_clean"],
        frame_data["signal_y_clean"],
        frame_data["power_vals"],
        frame_data["time_vals"],
        frame_data["tx_idx"],
        frame_data["surface_idx"],
        frame_data["bed_idx"],
        base_filename,
        frame_number,
        processor.config,
    )

    # Get manual picks from interactive session
    manual_tx, manual_surface, manual_bed, overrides = (
        override_session.start_interactive_session()
    )

    # Save frame with manual picks
    success = processor._save_frame_with_manual_picks(
        frame_data,
        manual_tx,
        manual_surface,
        manual_bed,
        overrides,
        base_filename,
        frame_number,
    )

    if success:
        print(
            f"SUCCESS: Results saved for frame {frame_number} in {processor.output_dir}"
        )
    else:
        print(f"ERROR: Failed to save results for frame {frame_number}")

    return success


def main():
    """
    Main function to process A-scope radar data with enhanced functionality.
    Features:
    - Ice thickness calculation using standard ice velocity (168 m/μs)
    - Automatic CSV and NPZ export with frame-by-frame data
    - CBD assignment from filename parsing
    - Interactive manual peak override for individual frames
    - Enhanced error handling and validation
    """
    parser = argparse.ArgumentParser(
        description="Process A-scope radar data with enhanced ice thickness calculation and interactive override.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Single file processing:
    python main.py --input F103-C0467_C0479.tiff
    python main.py --input F103-C0467_C0479.tiff --output ./results
    python main.py --input F103-C0467_C0479.tiff --interactive 5
    
  Batch processing:
    python main.py /path/to/tiff_folder --nav /path/to/navigation.csv
    python main.py /path/to/tiff_folder --nav /path/to/navigation.csv --output ./results

Navigation CSV format:
  Required columns: CBD, LAT, LON
  CBD values will be matched with TIFF filenames (FXXX-CAAAA_CBBBB.tiff)

Output files:
  - Frame verification: {filename}_frame_verification.png
  - Individual frame plots: {filename}_frame{XX}_picked.png  
  - Grid QA plots: {filename}_frame{XX}_grid_QA.png
  - CSV database: {filename}_pick.csv (includes LAT/LON if nav provided)
  - NPZ database: {filename}_pick.npz (includes LAT/LON if nav provided)

CSV columns:
  Frame, CBD, LAT, LON, Surface_Time_us, Bed_Time_us, Ice_Thickness_m, Surface_Power_dB, Bed_Power_dB

Interactive Override (single file only):
  Use --interactive FRAME_NUMBER to manually correct automatic peak detection
  Press 't' to redefine transmitter pulse, 's' for surface, 'b' for bed
  Click on calibrated plot to select new peak positions
        """,
    )

    # Input arguments - either single file or batch folder
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--input",
        help="Path to input A-scope TIFF image file (e.g., F103-C0467_0479.tiff)",
    )
    input_group.add_argument(
        "folder_path", 
        nargs="?",
        help="Path to folder containing TIFF files for batch processing",
    )

    # Optional arguments
    parser.add_argument(
        "--output", default=None, help="Path to output directory (default: ./output/)"
    )
    parser.add_argument(
        "--nav", 
        help="Path to navigation CSV file with columns: CBD, LAT, LON (required for batch processing)"
    )

    # Set default config path relative to main.py
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_config = os.path.join(script_dir, "config", "default_config.json")

    parser.add_argument(
        "--config",
        default=default_config,
        help=f"Path to configuration file (default: {default_config})",
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode for additional outputs and verbose logging",
    )

    # Interactive override functionality
    parser.add_argument(
        "--interactive",
        type=int,
        metavar="FRAME_NUMBER",
        help="Run interactive manual peak override on specified frame (1-based index). "
        "Allows manual correction of automatic transmitter, surface, and bed picks.",
    )

    parser.add_argument(
        "--interactive-approval",
        action="store_true",
        help="Enable interactive montage approval for batch runs (prompts per-file).",
    )

    parser.add_argument(
        "--no-approve",
        action="store_true",
        help="Disable interactive approval for single-file runs (non-interactive).",
    )

    args = parser.parse_args()

    # Determine processing mode
    is_batch_mode = bool(args.folder_path)
    is_single_mode = bool(args.input)
    
    if is_batch_mode:
        # Batch processing mode
        folder_path = Path(args.folder_path)
        if not folder_path.exists() or not folder_path.is_dir():
            print(f"ERROR: Folder not found: {folder_path}")
            sys.exit(1)
            
        # Navigation file is recommended for batch processing
        if not args.nav:
            print("WARNING: No navigation file provided. CBD coordinates will not be included.")
        elif not Path(args.nav).exists():
            print(f"ERROR: Navigation file not found: {args.nav}")
            sys.exit(1)
            
    else:
        # Single file processing mode
        input_path = Path(args.input)
        if not input_path.exists():
            print(f"ERROR: Input file not found: {input_path}")
            sys.exit(1)

        if not input_path.suffix.lower() in [".tiff", ".tif"]:
            print(f"WARNING: Input file doesn't appear to be a TIFF: {input_path}")
            print("Proceeding anyway...")

    # Validate config file
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"ERROR: Configuration file not found: {config_path}")
        sys.exit(1)

    # Set up output directory
    if args.output:
        output_path = Path(args.output)
        output_path.mkdir(parents=True, exist_ok=True)
        print(f"INFO: Using output directory: {output_path.resolve()}")
    else:
        output_path = None
        print("INFO: Using default output directory from config")

    # Display processing information
    print("=" * 60)
    print("A-SCOPE RADAR DATA PROCESSOR")
    print("Enhanced with Ice Thickness Calculation & Interactive Override")
    print("=" * 60)
    
    if is_batch_mode:
        print(f"Mode: BATCH PROCESSING")
        print(f"Input folder: {folder_path.resolve()}")
        print(f"Navigation file: {args.nav if args.nav else 'None'}")
        effective_interactive = not args.no_approve
        print(f"Interactive approval: {'Enabled' if effective_interactive else 'Disabled'} (use --no-approve to disable)")
    else:
        print(f"Mode: SINGLE FILE PROCESSING")
        print(f"Input file: {input_path.resolve()}")
        
        # Extract filename info for validation (single file mode only)
        filename = input_path.stem
        import re

        cbd_match = re.search(r"C(\d+)_C(\d+)", filename)
        if cbd_match:
            cbd_start = int(cbd_match.group(1))
            cbd_end = int(cbd_match.group(2))
            expected_frames = abs(cbd_end - cbd_start) + 1
            print(
                f"Expected CBD range: {cbd_start} to {cbd_end} ({expected_frames} frames)"
            )

            # Validate frame number for interactive mode
            if args.interactive:
                if args.interactive < 1 or args.interactive > expected_frames:
                    print(
                        f"ERROR: Frame number {args.interactive} is out of range (1-{expected_frames})"
                    )
                    sys.exit(1)
        else:
            print("WARNING: Could not extract CBD range from filename")
            print("Expected filename format: FXX-CXXXX_CXXXX.tiff")

        # Show interactive mode
        if args.interactive:
            print(f"Interactive override: Frame {args.interactive}")
    
    print(f"Config file: {config_path.resolve()}")
    if output_path:
        print(f"Output directory: {output_path.resolve()}")
    print(f"Debug mode: {'Enabled' if args.debug else 'Disabled'}")
    print("=" * 60)

    try:
        if is_batch_mode:
            # Batch processing mode
            print("INFO: Starting batch processing...")
            success = process_batch_folder(
                str(folder_path.resolve()),
                args.nav,
                str(config_path.resolve()),
                str(output_path.resolve()) if output_path else None,
                interactive_approval=(not args.no_approve),
            )
            
            if success:
                print("\n" + "=" * 60)
                print("BATCH PROCESSING COMPLETED SUCCESSFULLY!")
                print("=" * 60)
                sys.exit(0)
            else:
                print("\n" + "=" * 60)
                print("SOME FILES FAILED DURING BATCH PROCESSING!")
                print("=" * 60)
                sys.exit(1)
                
        else:
            # Single file processing mode
            # Handle interactive mode FIRST
            if args.interactive:
                print(
                    f"INFO: Starting interactive override for frame {args.interactive}..."
                )
                success = process_single_frame_interactive(
                    str(input_path.resolve()),
                    args.interactive,
                    str(config_path.resolve()),
                    str(output_path.resolve()) if output_path else None,
                )

                if success:
                    print("\n" + "=" * 60)
                    print("INTERACTIVE OVERRIDE COMPLETED SUCCESSFULLY!")
                    print("=" * 60)
                    sys.exit(0)
                else:
                    print("\n" + "=" * 60)
                    print("INTERACTIVE OVERRIDE FAILED!")
                    print("=" * 60)
                    sys.exit(1)

            # Normal single file processing mode
            print("INFO: Initializing A-scope processor...")
            processor = AScope(str(config_path.resolve()))

            # Set debug mode if requested
            if args.debug:
                processor.set_debug_mode(True)
                print("INFO: Debug mode enabled - additional outputs will be generated")

            # Set output directory if specified
            if output_path:
                processor.set_output_directory(str(output_path.resolve()))

            # Enable or disable interactive approval flow for single-file processing
            processor.set_interactive_mode(enabled=not args.no_approve, defer_frame_plots=not args.no_approve)

            # Process the image
            print(f"INFO: Starting processing of {input_path.name}...")
            processor.process_image(
                str(input_path.resolve()),
                str(output_path.resolve()) if output_path else None,
            )

        # Display completion summary
        print("\n" + "=" * 60)
        print("PROCESSING COMPLETED SUCCESSFULLY!")
        print("=" * 60)

        # Show output files
        output_dir = processor.output_dir
        base_name = filename  # Keep original filename (with dashes)

        expected_outputs = [
            f"{base_name}_frame_verification.png",
            f"{base_name}_pick.csv",
            f"{base_name}_pick.npz",
        ]

        print("Generated output files:")
        for output_file in expected_outputs:
            output_file_path = Path(output_dir) / output_file
            if output_file_path.exists():
                print(f"  ✅ {output_file}")
            else:
                print(f"  ❌ {output_file} (not found)")

        # Show frame-specific outputs
        if hasattr(processor, "frame_results") and processor.frame_results:
            frame_count = len(processor.frame_results)
            print(f"\nFrame-specific outputs ({frame_count} frames):")
            for i in range(1, frame_count + 1):
                frame_picked = f"{base_name}_frame{i:02d}_picked.png"
                frame_grid = f"{base_name}_frame{i:02d}_grid_QA.png"

                frame_picked_path = Path(output_dir) / frame_picked
                frame_grid_path = Path(output_dir) / frame_grid

                if frame_picked_path.exists():
                    print(f"  ✅ {frame_picked}")
                if frame_grid_path.exists():
                    print(f"  ✅ {frame_grid}")

        print(f"\nAll outputs saved to: {output_dir}")

        # Processing statistics with proper pandas usage
        if hasattr(processor, "frame_results") and processor.frame_results:
            results = processor.frame_results
            total_frames = len(results)

            # Use numpy operations to avoid scope issues
            valid_surface = sum(
                1 for r in results if not pd.isna(r.get("Surface_Time_us", np.nan))
            )
            valid_bed = sum(
                1 for r in results if not pd.isna(r.get("Bed_Time_us", np.nan))
            )
            valid_thickness = sum(
                1 for r in results if not pd.isna(r.get("Ice_Thickness_m", np.nan))
            )

            print(f"\nProcessing Statistics:")
            print(f"  Total frames processed: {total_frames}")
            print(f"  Frames with surface picks: {valid_surface}")
            print(f"  Frames with bed picks: {valid_bed}")
            print(f"  Frames with ice thickness: {valid_thickness}")

            if valid_thickness > 0:
                try:
                    df = pd.DataFrame(results)
                    thickness_values = df["Ice_Thickness_m"].dropna()
                    if len(thickness_values) > 0:
                        print(
                            f"  Ice thickness range: {thickness_values.min():.1f} - {thickness_values.max():.1f} m"
                        )
                        print(
                            f"  Average ice thickness: {thickness_values.mean():.1f} m"
                        )
                except Exception as e:
                    print(f"  Note: Could not calculate thickness statistics: {e}")
                    # Fallback calculation without pandas
                    thickness_list = [
                        r.get("Ice_Thickness_m", np.nan)
                        for r in results
                        if not pd.isna(r.get("Ice_Thickness_m", np.nan))
                    ]
                    if thickness_list:
                        print(
                            f"  Ice thickness range: {min(thickness_list):.1f} - {max(thickness_list):.1f} m"
                        )
                        print(
                            f"  Average ice thickness: {np.mean(thickness_list):.1f} m"
                        )

        print("=" * 60)

    except KeyboardInterrupt:
        print("\n\nINFO: Processing interrupted by user")
        sys.exit(1)

    except FileNotFoundError as e:
        print(f"ERROR: File not found - {e}")
        sys.exit(1)

    except ValueError as e:
        print(f"ERROR: Invalid input - {e}")
        sys.exit(1)

    except Exception as e:
        print(f"ERROR: Processing failed - {e}")
        if args.debug:
            import traceback

            print("\nFull error traceback:")
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
