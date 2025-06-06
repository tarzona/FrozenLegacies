import matplotlib.pyplot as plt
import numpy as np  # Not strictly needed for ClickSelector but often useful for interactive tools


class ClickSelector:
    """
    An interactive tool to select a point (specifically an X-coordinate) on an image.

    When instantiated with an image, it displays the image in a Matplotlib window.
    The user can click on the image. The class captures the X and Y coordinates of the click.
    The window closes automatically after the first click.

    Attributes:
        image (np.ndarray): The image to be displayed.
        selected_x (int or None): The X-coordinate of the point clicked by the user.
                                  None if no click has occurred or window closed.
        selected_y (int or None): The Y-coordinate of the point clicked by the user.
                                  None if no click has occurred or window closed.
        fig (matplotlib.figure.Figure): The Matplotlib figure object.
        ax (matplotlib.axes.Axes): The Matplotlib axes object displaying the image.
    """

    def __init__(self, image_to_display, title="Click on the target location"):
        """
        Initializes the ClickSelector and displays the image for selection.

        Args:
            image_to_display (np.ndarray): The image (as a NumPy array) on which the user will click.
            title (str, optional): The title for the Matplotlib window.
                                   Defaults to "Click on the target location".
        """
        self.image = image_to_display
        self.selected_x = None
        self.selected_y = None

        # Determine figure size. For very wide images, a wide figure is helpful.
        img_height, img_width = self.image.shape[:2]
        # Aim for a figure height of ~6 inches, adjust width proportionally, max width ~24 inches.
        fig_height_inches = 6
        aspect_ratio = img_width / img_height
        fig_width_inches = min(24, fig_height_inches * aspect_ratio)

        # If image is very tall and narrow, this might result in too narrow a figure,
        # so ensure a minimum width too, e.g., 8 inches.
        fig_width_inches = max(8, fig_width_inches)

        self.fig, self.ax = plt.subplots(figsize=(fig_width_inches, fig_height_inches))
        self.ax.imshow(self.image, cmap="gray", aspect="auto")
        self.ax.set_title(title, fontsize=12)
        self.ax.set_xlabel("X-pixel coordinate")
        self.ax.set_ylabel("Y-pixel coordinate")

        # Connect the click event to the onclick method
        self.cid = self.fig.canvas.mpl_connect("button_press_event", self._onclick)

        print(
            "INFO: Displaying image for selection. Click the desired location in the pop-up window."
        )
        print("      The window will close automatically after your click.")
        plt.show()  # This will block until the window is closed

    def _onclick(self, event):
        """
        Handles the mouse click event on the Matplotlib figure.

        Stores the click coordinates and closes the figure.

        Args:
            event (matplotlib.backend_bases.MouseEvent): The Matplotlib mouse event.
        """
        # Check if the click was within the axes
        if event.inaxes == self.ax:
            if event.xdata is not None and event.ydata is not None:
                self.selected_x = int(round(event.xdata))
                self.selected_y = int(round(event.ydata))
                print(f"INFO: User selected X={self.selected_x}, Y={self.selected_y}")
            else:
                print(
                    "INFO: Click was outside image data area. No coordinates captured."
                )
        else:
            print("INFO: Click was outside the main axes. No coordinates captured.")

        # Disconnect the event handler and close the figure
        # This ensures the selector is used only once per instance.
        if hasattr(self, "cid") and self.cid is not None:
            self.fig.canvas.mpl_disconnect(self.cid)
            self.cid = None  # Prevent multiple disconnects if somehow called again

        # Close the figure to unblock plt.show() and return control to the script
        plt.close(self.fig)


def get_manual_feature_annotations(
    default_features,
    pixels_per_microsecond,
    transmitter_pulse_y_abs,
    prompt_message="Do you want to manually annotate radar features? (yes/no): ",
):
    """
    Prompts the user to manually input or confirm pixel coordinates for radar features.

    It iterates through a dictionary of default features, allowing the user to
    update the 'pixel_abs' (absolute Y-coordinate) for each. If updated, the
    corresponding 'time_us' is recalculated.

    Args:
        default_features (dict): A dictionary where keys are feature identifiers (e.g., 'i')
                                 and values are dictionaries containing:
                                     'name' (str): Display name (e.g., "Ice Surface").
                                     'pixel_abs' (int): Default absolute Y-pixel coordinate.
                                     'color' (str): Color for visualization.
                                     (Optionally 'time_us' can be pre-filled or will be calculated).
        pixels_per_microsecond (float): Calibration factor (pixels / µs) used to calculate time.
        transmitter_pulse_y_abs (int): Absolute Y-coordinate of the transmitter pulse (0 µs reference).
        prompt_message (str, optional): The message to display when asking if the user wants to annotate.

    Returns:
        tuple: (updated_features_dict, bool)
               - updated_features_dict (dict): The dictionary of features, potentially updated by the user.
               - user_did_annotate (bool): True if the user chose to annotate, False otherwise.
    """
    updated_features = default_features.copy()  # Work on a copy
    user_did_annotate = False

    while True:
        annotate_choice = input(prompt_message).strip().lower()
        if annotate_choice in ["yes", "y", "no", "n"]:
            break
        print("Invalid input. Please enter 'yes' (or 'y') or 'no' (or 'n').")

    if annotate_choice in ["yes", "y"]:
        user_did_annotate = True
        print("\n--- Manual Feature Annotation ---")
        print(
            "For each feature, enter the absolute Y-pixel coordinate from the original image."
        )
        print("Press Enter to keep the current default value.")

        for key, feature_details in updated_features.items():
            current_pixel = feature_details.get("pixel_abs", "Not set")
            prompt_text = (
                f"Enter Y-pixel for '{feature_details['name']}' "
                f"(current: {current_pixel}): "
            )

            # We usually don't ask to re-input the transmitter pulse if it's auto-detected
            if (
                key == "t" and "pixel_abs" in feature_details
            ):  # Assuming 't' is key for Tx pulse
                print(
                    f"INFO: Transmitter Pulse ('{feature_details['name']}') is set to {feature_details['pixel_abs']}."
                )
                # Ensure time is 0 for Tx pulse if not already set
                updated_features[key]["time_us"] = 0.0
                continue

            while True:
                try:
                    user_input = input(prompt_text).strip()
                    if not user_input:  # User pressed Enter, keep default
                        print(f"Keeping current value for '{feature_details['name']}'.")
                        # Ensure time is calculated if pixel_abs exists
                        if (
                            "pixel_abs" in feature_details
                            and pixels_per_microsecond > 0
                        ):
                            updated_features[key]["time_us"] = (
                                feature_details["pixel_abs"] - transmitter_pulse_y_abs
                            ) / pixels_per_microsecond
                        break

                    pixel_abs_val = int(user_input)
                    updated_features[key]["pixel_abs"] = pixel_abs_val
                    if (
                        pixels_per_microsecond > 0
                    ):  # Avoid division by zero if not calibrated
                        updated_features[key]["time_us"] = (
                            pixel_abs_val - transmitter_pulse_y_abs
                        ) / pixels_per_microsecond
                    else:
                        updated_features[key]["time_us"] = float(
                            "nan"
                        )  # Indicate time cannot be calculated

                    print(
                        f"Set '{feature_details['name']}' to Y-pixel {pixel_abs_val} (Time: {updated_features[key]['time_us']:.1f} µs)."
                    )
                    break
                except ValueError:
                    print(
                        "Invalid input. Please enter a whole number for the pixel coordinate."
                    )
                except Exception as e:
                    print(f"An error occurred: {e}. Please try again.")
        print("--- End of Manual Feature Annotation ---\n")
    else:
        print("INFO: Skipping manual feature annotation.")
        # Ensure times are calculated for default features if not already present
        for key, feature_details in updated_features.items():
            if (
                "pixel_abs" in feature_details
                and "time_us" not in feature_details
                and pixels_per_microsecond > 0
            ):
                updated_features[key]["time_us"] = (
                    feature_details["pixel_abs"] - transmitter_pulse_y_abs
                ) / pixels_per_microsecond
            elif "pixel_abs" in feature_details and pixels_per_microsecond <= 0:
                updated_features[key]["time_us"] = float("nan")

    return updated_features, user_did_annotate
