from PIL import Image, ImageFile
import numpy as np
import cv2
from pathlib import Path
import os  # For ensuring output directory exists if we add saving utils later


def load_and_preprocess_image(image_path_str, preprocessing_params=None):
    """
    Loads an image, converts it to a NumPy array, and performs basic preprocessing.

    Args:
        image_path_str (str): The path to the image file.
        preprocessing_params (dict, optional): A dictionary containing parameters for preprocessing.
            Expected keys:
            - "percentile_low" (float): Lower percentile for contrast stretching (default: 2).
            - "percentile_high" (float): Upper percentile for contrast stretching (default: 98).

    Returns:
        numpy.ndarray: The preprocessed grayscale image as a NumPy array, or None if loading fails.
    """
    if preprocessing_params is None:
        preprocessing_params = {}

    image_path = Path(image_path_str)
    if not image_path.is_file():
        print(f"Error: Image file not found at {image_path}")
        return None

    try:
        # Allow PIL to load truncated images to prevent common errors with large/corrupted files [2]
        ImageFile.LOAD_TRUNCATED_IMAGES = True

        pil_img = Image.open(image_path)
        img_np = np.array(pil_img)

        print(f"Successfully loaded image: {image_path.name}")
        print(f"Original image dtype: {img_np.dtype}, shape: {img_np.shape}")

        # Convert to uint8 with contrast stretching if image is uint16
        if img_np.dtype == np.uint16:
            percentile_low = preprocessing_params.get("percentile_low", 2)
            percentile_high = preprocessing_params.get("percentile_high", 98)

            print(
                f"Converting image from uint16 to uint8 using percentiles ({percentile_low}%, {percentile_high}%)..."
            )
            p_low_val, p_high_val = np.percentile(
                img_np, (percentile_low, percentile_high)
            )

            # Avoid division by zero if p_low_val and p_high_val are the same
            if p_high_val == p_low_val:
                if p_low_val > 0:  # If image is not all black
                    img_np = np.where(img_np > 0, 255, 0).astype(np.uint8)
                else:  # Image is all black or uniform
                    img_np = np.zeros_like(img_np, dtype=np.uint8)
            else:
                img_np = np.clip(
                    (img_np - p_low_val) * 255.0 / (p_high_val - p_low_val), 0, 255
                ).astype(np.uint8)

            print(f"Converted image dtype: {img_np.dtype}")

        # Convert to grayscale if the image has multiple channels (e.g., RGB, RGBA)
        if len(img_np.shape) == 3:
            if img_np.shape[2] == 3:  # RGB
                print("Converting RGB image to grayscale...")
                img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
            elif img_np.shape[2] == 4:  # RGBA
                print("Converting RGBA image to grayscale...")
                img_np = cv2.cvtColor(img_np, cv2.COLOR_RGBA2GRAY)
            else:
                print(
                    f"Warning: Image has {img_np.shape[2]} channels. Taking the first channel as grayscale."
                )
                img_np = img_np[:, :, 0]
            print(f"Grayscale image shape: {img_np.shape}")

        return img_np

    except FileNotFoundError:
        print(f"Error: Image file not found at {image_path}")
        return None
    except Exception as e:
        print(f"Error loading or preprocessing image {image_path}: {e}")
        return None


# Example of a simple save utility that could be added here if needed:
# def save_debug_image(image_array, filename_suffix, base_filename, output_dir="debug_output"):
#     """
#     Saves a NumPy array as an image to the debug output directory.
#     """
#     if not os.path.exists(output_dir):
#         os.makedirs(output_dir, exist_ok=True)

#     output_path = Path(output_dir) / f"{base_filename}_{filename_suffix}.png"
#     try:
#         cv2.imwrite(str(output_path), image_array)
#         print(f"Saved debug image: {output_path}")
#     except Exception as e:
#         print(f"Error saving debug image {output_path}: {e}")
