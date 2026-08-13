import numpy as np


def refine_peaks(profile, peak_indices, window_size=5):
    """
    Refines the positions of detected peaks in a 1D profile using a center-of-mass calculation.

    For each initially detected peak, this function considers a small window around it.
    It then calculates the center of mass within this window, using the profile values
    as weights. This can provide a sub-pixel estimate of the peak's true location,
    which is often more accurate than the initial integer pixel index.

    This method is useful for improving the precision of feature localization,
    such as the exact position of calibration tick marks.

    Args:
        profile (np.ndarray): A 1D NumPy array representing the signal or intensity profile
                              where peaks were detected (e.g., a vertical intensity sum).
        peak_indices (np.ndarray or list): A list or NumPy array of integer indices
                                           representing the initially detected peak positions
                                           within the `profile`.
        window_size (int, optional): The ODD integer size of the window (number of points)
                                     to consider around each peak for the center-of-mass
                                     calculation. Defaults to 5. A larger window considers
                                     more surrounding data but might be influenced by nearby
                                     features. A smaller window is more local.

    Returns:
        np.ndarray: A NumPy array of the refined peak positions (float values).
                    The length of this array is the same as `peak_indices`.
                    If a window sum is zero (e.g., all-zero profile in window),
                    the original peak index is returned for that peak.

    Example:
        >>> profile = np.array([0, 0, 1, 2, 5, 2, 1, 0, 0])
        >>> initial_peaks = np.array([4]) # Peak at index 4 (value 5)
        >>> refined = refine_peaks(profile, initial_peaks, window_size=3)
        >>> print(refined) # Output will be close to 4.0, possibly slightly shifted
                           # depending on symmetry. For [0,5,2], CoM = (3*0+4*5+5*2)/(0+5+2) = 30/7 = 4.28
                           # For this example, window around index 4 (value 5) with size 3 is [2, 5, 2]
                           # indices are [3, 4, 5]. CoM = (3*2 + 4*5 + 5*2) / (2+5+2) = (6+20+10)/9 = 36/9 = 4.0
                           # If window_size=5, data: [1,2,5,2,1], indices: [2,3,4,5,6]
                           # CoM = (2*1+3*2+4*5+5*2+6*1)/(1+2+5+2+1) = (2+6+20+10+6)/11 = 44/11 = 4.0
    """
    if not isinstance(peak_indices, np.ndarray):
        peak_indices = np.array(peak_indices)

    if peak_indices.size == 0:
        return np.array([])  # Return empty if no peaks to refine

    if window_size % 2 == 0:
        # print(f"Warning: refine_peaks window_size ({window_size}) should be odd. Using {window_size + 1}.")
        window_size += 1  # Ensure window size is odd for symmetry around the peak

    half_window = window_size // 2
    refined_positions = []

    for peak_idx_int in peak_indices.astype(int):  # Ensure integer indices for slicing
        # Define the start and end of the window for the current peak
        start_index = max(0, peak_idx_int - half_window)
        end_index = min(
            len(profile), peak_idx_int + half_window + 1
        )  # Slice end is exclusive

        # Extract the data and corresponding indices for the window
        window_data = profile[start_index:end_index]
        indices_in_window = np.arange(start_index, end_index)

        # Calculate the sum of data in the window (denominator for center of mass)
        sum_of_window_data = np.sum(window_data)

        if sum_of_window_data > 0:
            # Calculate center of mass: sum(index * data_value) / sum(data_value)
            center_of_mass = (
                np.sum(indices_in_window * window_data) / sum_of_window_data
            )
            refined_positions.append(center_of_mass)
        else:
            # If the sum of window data is zero (e.g., all zeros in the window),
            # it's not possible to calculate CoM. Fallback to the original peak index.
            refined_positions.append(float(peak_idx_int))

    return np.array(refined_positions)


#
#
# import os
# from pathlib import Path
#
# def ensure_directory_exists(dir_path_str):
#     """
#     Checks if a directory exists, and if not, creates it.
#
#     Args:
#         dir_path_str (str): The path to the directory.
#     """
#     path_obj = Path(dir_path_str)
#     if not path_obj.exists():
#         print(f"INFO: Directory '{path_obj}' does not exist. Creating it.")
#         os.makedirs(path_obj, exist_ok=True) # exist_ok=True prevents error if dir was created by another process
#     elif not path_obj.is_dir():
#         print(f"ERROR: '{path_obj}' exists but is not a directory.")
#         raise NotADirectoryError(f"'{path_obj}' is not a directory.")
