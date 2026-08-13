import matplotlib.pyplot as plt
import numpy as np
import cv2


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


class EchoPointSelector:
    """Interactive tool for selecting surface and bed echo points with crosshair precision."""

    def __init__(self, image, title="Select Echo Points"):
        self.image = image
        self.title = title
        self.surface_points = []
        self.bed_points = []
        self.current_mode = "surface"  # "surface" or "bed"
        self.fig = None
        self.ax = None
        self.expected_points = 5  # Increased from 3 for better coverage

    def start_selection(self):
        """Start interactive point selection with crosshair cursor."""
        height, width = self.image.shape

        self.fig, self.ax = plt.subplots(figsize=(24, 10))

        # Enhanced image display
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(self.image)

        self.ax.imshow(enhanced, cmap="gray", aspect="auto")
        self.ax.set_title(
            f"{self.title}\nMode: {self.current_mode.upper()} echo selection - Click on clear echo peaks",
            fontsize=16,
            fontweight="bold",
        )

        # Add crosshair cursor
        self.crosshair_v = self.ax.axvline(
            x=0, color="yellow", linestyle="-", linewidth=1, alpha=0.8, visible=False
        )
        self.crosshair_h = self.ax.axhline(
            y=0, color="yellow", linestyle="-", linewidth=1, alpha=0.8, visible=False
        )

        # Connect events
        self.motion_cid = self.fig.canvas.mpl_connect(
            "motion_notify_event", self._on_mouse_move
        )
        self.click_cid = self.fig.canvas.mpl_connect(
            "button_press_event", self._on_click
        )
        self.key_cid = self.fig.canvas.mpl_connect(
            "key_press_event", self._on_key_press
        )

        # Instructions
        self._update_instructions()

        plt.tight_layout()
        plt.show()

        return self.surface_points, self.bed_points

    def _on_click(self, event):
        """Handle mouse click events for point selection."""
        if event.inaxes != self.ax:
            return

        if event.xdata is None or event.ydata is None:
            return

        x, y = int(event.xdata), int(event.ydata)

        if self.current_mode == "surface":
            if len(self.surface_points) < self.expected_points:
                self.surface_points.append((x, y))
                color = "cyan"
                label = f"Surface Point {len(self.surface_points)}"
                print(f"Added surface point {len(self.surface_points)}: X={x}, Y={y}")
        else:  # bed mode
            if len(self.bed_points) < self.expected_points:
                self.bed_points.append((x, y))
                color = "orange"
                label = f"Bed Point {len(self.bed_points)}"
                print(f"Added bed point {len(self.bed_points)}: X={x}, Y={y}")

        # Draw the selected point
        self.ax.plot(
            x,
            y,
            "o",
            color=color,
            markersize=8,
            markeredgewidth=2,
            markeredgecolor="white",
            markerfacecolor=color,
            label=label,
            alpha=0.9,
        )

        # Add text annotation
        self.ax.annotate(
            f"{self.current_mode.upper()}\n{len(self.surface_points if self.current_mode == 'surface' else self.bed_points)}",
            xy=(x, y),
            xytext=(10, 10),
            textcoords="offset points",
            bbox=dict(boxstyle="round,pad=0.3", fc=color, alpha=0.8),
            fontsize=10,
            color="white",
            fontweight="bold",
        )

        self.ax.legend()
        self.fig.canvas.draw()

        # Update instructions and check completion
        self._update_instructions()
        self._check_completion()

    def _on_mouse_move(self, event):
        """Handle mouse movement to update crosshair position."""
        if event.inaxes == self.ax:
            self.crosshair_v.set_xdata([event.xdata])
            self.crosshair_h.set_ydata([event.ydata])
            self.crosshair_v.set_visible(True)
            self.crosshair_h.set_visible(True)
            self.fig.canvas.draw_idle()
        else:
            self.crosshair_v.set_visible(False)
            self.crosshair_h.set_visible(False)
            self.fig.canvas.draw_idle()

    def _on_key_press(self, event):
        """Handle keyboard events for mode switching."""
        if event.key == "s":
            self.current_mode = "surface"
            self.ax.set_title(
                f"{self.title}\nMode: {self.current_mode.upper()} echo selection - Click on clear echo peaks",
                fontsize=16,
                fontweight="bold",
            )
            print("Switched to SURFACE echo selection mode")
            self.fig.canvas.draw()
        elif event.key == "b":
            self.current_mode = "bed"
            self.ax.set_title(
                f"{self.title}\nMode: {self.current_mode.upper()} echo selection - Click on clear echo peaks",
                fontsize=16,
                fontweight="bold",
            )
            print("Switched to BED echo selection mode")
            self.fig.canvas.draw()
        elif event.key == "q" or event.key == "escape":
            print("Selection completed by user")
            plt.close(self.fig)

    def _update_instructions(self):
        """Update instruction text on the plot."""
        surface_count = len(self.surface_points)
        bed_count = len(self.bed_points)

        instructions = (
            "ECHO POINT SELECTION FOR SEARCH OPTIMIZATION:\n"
            f"Current Mode: {self.current_mode.upper()}\n"
            "• Press 'S' for Surface mode, 'B' for Bed mode\n"
            f"• Surface points: {surface_count}/{self.expected_points}\n"
            f"• Bed points: {bed_count}/{self.expected_points}\n"
            "• Press 'Q' or 'Escape' when done\n"
            "• Close window to finish selection"
        )

        # Clear previous instructions
        for txt in self.ax.texts:
            if "ECHO POINT SELECTION" in txt.get_text():
                txt.remove()

        self.ax.text(
            0.98,
            0.02,
            instructions,
            transform=self.ax.transAxes,
            fontsize=12,
            verticalalignment="bottom",
            horizontalalignment="right",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.9),
        )

    def _check_completion(self):
        """Check if enough points have been selected."""
        surface_count = len(self.surface_points)
        bed_count = len(self.bed_points)

        if (
            surface_count >= self.expected_points
            and bed_count >= self.expected_points
        ):
            print(
                f"\nSUCCESS: Selected {surface_count} surface and {bed_count} bed points"
            )
            print("You can close the window or press 'Q' to finish selection")

            self.ax.set_title(
                f"{self.title}\nCOMPLETE: {surface_count} surface, {bed_count} bed points selected",
                fontsize=16,
                fontweight="bold",
                color="green",
            )
            self.fig.canvas.draw()

    def cleanup(self):
        """Clean up event connections."""
        if hasattr(self, "motion_cid") and self.motion_cid is not None:
            self.fig.canvas.mpl_disconnect(self.motion_cid)
        if hasattr(self, "click_cid") and self.click_cid is not None:
            self.fig.canvas.mpl_disconnect(self.click_cid)
        if hasattr(self, "key_cid") and self.key_cid is not None:
            self.fig.canvas.mpl_disconnect(self.key_cid)
