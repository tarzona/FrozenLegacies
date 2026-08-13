"""
Interactive Montage Viewer for A-scope Frame Results

Provides a PyQt5-based GUI for viewing detected and picked A-scope frames
with dynamic quality adjustment based on window size and zoom level.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import io

try:
    from PyQt5 import QtWidgets, QtCore, QtGui
    HAVE_QT = True
except ImportError:
    HAVE_QT = False


class InteractiveMontageViewer:
    """
    Interactive montage viewer that displays A-scope frame results with:
    - Dynamic quality adjustment based on window size
    - Zoom with quality adaptation
    - User approval workflow (save/repick/edit config)
    """

    def __init__(self, deferred_frame_images, frame_count, dpi=150, trace_mode=False):
        """
        Initialize the montage viewer.

        Args:
            deferred_frame_images: List of numpy arrays (frame plots)
            frame_count: Total number of frames
            dpi: Base DPI for rendering
            trace_mode: If True, show simplified buttons (Yes/Quit) for trace review.
                       If False, show full buttons (Save All/Repick/Edit Config) for picks review.
        """
        if not HAVE_QT:
            raise RuntimeError("PyQt5 is required for interactive montage viewer")

        self.deferred_frame_images = deferred_frame_images
        self.frame_count = frame_count
        self.base_dpi = dpi
        self.trace_mode = trace_mode
        self.user_choice = None
        self.selected_frames = []
        self._create_app()
        self._create_dialog()

    def _create_app(self):
        """Create or get QApplication instance."""
        self.app = QtWidgets.QApplication.instance()
        if self.app is None:
            self.app = QtWidgets.QApplication([])

    def _create_dialog(self):
        """Create the main montage dialog."""
        self.dialog = QtWidgets.QDialog()
        self.dialog.setModal(True)  # Ensure modal behavior
        
        # Set window title based on mode
        if self.trace_mode:
            window_title = f"Step 4: Trace Detection - {self.frame_count} frames"
        else:
            window_title = f"Step 4.5: Picks Review - {self.frame_count} frames with valid picks"
        self.dialog.setWindowTitle(window_title)
        self.dialog.setGeometry(100, 100, 1200, 800)

        layout = QtWidgets.QVBoxLayout()

        # Create canvas area with scroll
        self.scroll = QtWidgets.QScrollArea()
        self.scroll.setWidgetResizable(True)

        self.canvas_widget = QtWidgets.QLabel()
        self.canvas_widget.setAlignment(QtCore.Qt.AlignCenter)
        self.canvas_widget.setScaledContents(False)
        self.scroll.setWidget(self.canvas_widget)

        layout.addWidget(self.scroll)

        # Control buttons
        ctrl_layout = QtWidgets.QHBoxLayout()

        self.zoom_in_btn = QtWidgets.QPushButton("Zoom In (+)")
        self.zoom_out_btn = QtWidgets.QPushButton("Zoom Out (-)")
        self.reset_zoom_btn = QtWidgets.QPushButton("Reset Zoom")

        self.zoom_in_btn.clicked.connect(self._on_zoom_in)
        self.zoom_out_btn.clicked.connect(self._on_zoom_out)
        self.reset_zoom_btn.clicked.connect(self._on_reset_zoom)

        ctrl_layout.addWidget(self.zoom_in_btn)
        ctrl_layout.addWidget(self.zoom_out_btn)
        ctrl_layout.addWidget(self.reset_zoom_btn)

        layout.addLayout(ctrl_layout)

        # Status label
        self.status_label = QtWidgets.QLabel(
            f"Showing {self.frame_count} frames - Resize window to adjust quality"
        )
        layout.addWidget(self.status_label)

        # Decision buttons
        decision_layout = QtWidgets.QHBoxLayout()

        if self.trace_mode:
            # Trace review: simplified buttons (Yes/Edit Config)
            self.save_btn = QtWidgets.QPushButton("✓ Yes, Proceed")
            self.quit_btn = QtWidgets.QPushButton("⚙ Edit Config & Re-run")
            # New: show calibrated-only plots (no picks)
            self.calib_btn = QtWidgets.QPushButton("📐 Calibrated Plots (no picks)")
            self.repick_btn = None
            self.edit_btn = None
            
            self.save_btn.clicked.connect(self._on_save)
            self.quit_btn.clicked.connect(self._on_edit)
            self.calib_btn.clicked.connect(self._on_calib)
            
            decision_layout.addWidget(self.save_btn)
            decision_layout.addWidget(self.quit_btn)
            decision_layout.addWidget(self.calib_btn)
        else:
            # Picks review: full buttons (Save All/Repick/Edit Config)
            self.save_btn = QtWidgets.QPushButton("✓ Save All")
            self.repick_btn = QtWidgets.QPushButton("⟲ Manual Repick Frames")
            self.edit_btn = QtWidgets.QPushButton("⚙ Edit Config & Re-run")
            self.quit_btn = None
            
            self.save_btn.clicked.connect(self._on_save)
            self.repick_btn.clicked.connect(self._on_repick)
            self.edit_btn.clicked.connect(self._on_edit)
            
            decision_layout.addWidget(self.save_btn)
            decision_layout.addWidget(self.repick_btn)
            decision_layout.addWidget(self.edit_btn)

        layout.addLayout(decision_layout)

        self.dialog.setLayout(layout)

        # Montage display state
        self.current_zoom = 1.0
        self.montage_pixmap = None
        self.orig_pixmap = None

        # Build and display initial montage
        self._build_montage()
        self._update_display()

        # Connect resize event
        self.dialog.resizeEvent = self._on_dialog_resize
        
        print(f"[Montage Dialog] Dialog created and configured. Ready to show.")

    def _build_montage(self):
        """Build montage from deferred frame images."""
        if not self.deferred_frame_images:
            print("No deferred frame images to display")
            return

        try:
            from PIL import Image as PILImage

            # Convert PIL Images to numpy arrays if needed
            images_array = []
            for img in self.deferred_frame_images:
                if isinstance(img, PILImage.Image):
                    images_array.append(np.array(img))
                else:
                    images_array.append(img)

            # Parameters for montage layout
            frames_per_row = min(4, self.frame_count)  # 4 frames per row max
            rows = int(np.ceil(self.frame_count / frames_per_row))

            # Get max dimensions from all images
            max_w = max(img.shape[1] for img in images_array)
            max_h = max(img.shape[0] for img in images_array)

            # Create montage canvas
            gap = 20  # Gap between frames
            total_w = frames_per_row * (max_w + gap) + gap
            total_h = rows * (max_h + gap) + gap

            montage = PILImage.new("RGB", (int(total_w), int(total_h)), color=(240, 240, 240))

            # Place each frame
            for idx, img_array in enumerate(images_array):
                row = idx // frames_per_row
                col = idx % frames_per_row

                # Convert numpy array to PIL Image
                if img_array.dtype != np.uint8:
                    img_array = (np.clip(img_array, 0, 255)).astype(np.uint8)

                pil_img = PILImage.fromarray(img_array)

                # Resize to max dimensions if needed
                if pil_img.width != max_w or pil_img.height != max_h:
                    pil_img = pil_img.resize(
                        (int(max_w), int(max_h)), PILImage.LANCZOS
                    )

                # Calculate position
                x = int(col * (max_w + gap) + gap)
                y = int(row * (max_h + gap) + gap)

                # Paste image
                montage.paste(pil_img, (x, y))

                # Add frame number label
                from PIL import ImageDraw, ImageFont

                draw = ImageDraw.Draw(montage)
                frame_num_text = f"Frame {idx + 1}"
                try:
                    # Try to use a built-in font
                    font = ImageFont.truetype("arial.ttf", 16)
                except:
                    font = ImageFont.load_default()

                text_bbox = draw.textbbox((0, 0), frame_num_text, font=font)
                text_w = text_bbox[2] - text_bbox[0]
                text_h = text_bbox[3] - text_bbox[1]

                text_x = int(x + (max_w - text_w) / 2)
                text_y = int(y + max_h - text_h - 5)

                draw.text(
                    (text_x, text_y),
                    frame_num_text,
                    fill=(0, 0, 0),
                    font=font,
                )

            # Convert to QPixmap
            pil_image_data = montage.tobytes("raw", "RGB")
            qimage = QtGui.QImage(
                pil_image_data,
                montage.width,
                montage.height,
                QtGui.QImage.Format_RGB888,
            )
            self.orig_pixmap = QtGui.QPixmap.fromImage(qimage)
            print(f"Built montage: {montage.width}x{montage.height} pixels")

        except Exception as e:
            print(f"Error building montage: {e}")
            import traceback
            traceback.print_exc()

    def _update_display(self):
        """Update the displayed montage based on current zoom and window size."""
        if self.orig_pixmap is None:
            return

        try:
            # Get available viewport size
            viewport = self.scroll.viewport()
            avail_w = viewport.width()
            avail_h = viewport.height()

            if avail_w <= 0 or avail_h <= 0:
                return

            # Get original size
            orig_w = self.orig_pixmap.width()
            orig_h = self.orig_pixmap.height()

            # Calculate fit-to-window scale
            fit_scale_w = avail_w / orig_w if orig_w > 0 else 1.0
            fit_scale_h = avail_h / orig_h if orig_h > 0 else 1.0
            fit_scale = min(fit_scale_w, fit_scale_h)

            # Apply zoom on top of fit scale
            target_scale = max(0.1, fit_scale * self.current_zoom)

            # Calculate new dimensions
            new_w = max(1, int(orig_w * target_scale))
            new_h = max(1, int(orig_h * target_scale))

            # Scale with smooth transformation
            scaled_pixmap = self.orig_pixmap.scaled(
                new_w, new_h, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation
            )

            self.montage_pixmap = scaled_pixmap
            self.canvas_widget.setPixmap(scaled_pixmap)

            # Update DPI label
            current_dpi = int(self.base_dpi * target_scale)
            self.status_label.setText(
                f"Frames: {self.frame_count} | Zoom: {self.current_zoom:.1f}x | Effective DPI: {current_dpi} | Window: {avail_w}x{avail_h}"
            )

        except Exception as e:
            print(f"Error updating display: {e}")

    def _on_dialog_resize(self, event):
        """Handle window resize events - update quality dynamically."""
        QtWidgets.QDialog.resizeEvent(self.dialog, event)
        self._update_display()

    def _on_zoom_in(self):
        """Zoom in and increase quality."""
        self.current_zoom *= 1.25
        self._update_display()

    def _on_zoom_out(self):
        """Zoom out and decrease quality."""
        self.current_zoom = max(0.1, self.current_zoom / 1.25)
        self._update_display()

    def _on_reset_zoom(self):
        """Reset zoom to fit window."""
        self.current_zoom = 1.0
        self._update_display()

    def _on_save(self):
        """User selected 'save all' or 'yes, proceed'."""
        print(f"[Button Click] Save/Proceed button clicked")
        self.user_choice = "save"
        self.dialog.accept()

    def _on_quit(self):
        """User selected 'quit' (for trace mode)."""
        print(f"[Button Click] Quit button clicked")
        self.user_choice = "quit"
        self.dialog.reject()

    def _on_repick(self):
        """User selected 'manual repick'."""
        # Open dialog to select frames
        frames_input, ok = QtWidgets.QInputDialog.getText(
            self.dialog,
            "Select Frames to Repick",
            "Enter comma-separated frame numbers (1-based):\nExample: 1, 3, 5-7, 10",
        )

        if ok and frames_input:
            try:
                # Parse frame numbers
                selected = set()
                for part in frames_input.split(","):
                    part = part.strip()
                    if "-" in part:
                        # Range: e.g., "5-7"
                        start, end = map(int, part.split("-"))
                        selected.update(range(start, end + 1))
                    else:
                        # Single frame
                        selected.add(int(part))

                # Validate ranges
                valid_selected = [
                    f for f in selected if 1 <= f <= self.frame_count
                ]
                if valid_selected:
                    self.selected_frames = sorted(valid_selected)
                    self.user_choice = "repick"
                    print(f"Selected frames for repicking: {self.selected_frames}")
                    self.dialog.accept()
                else:
                    QtWidgets.QMessageBox.warning(
                        self.dialog,
                        "Invalid Selection",
                        f"No valid frame numbers in range 1-{self.frame_count}",
                    )
            except Exception as e:
                QtWidgets.QMessageBox.warning(
                    self.dialog,
                    "Parse Error",
                    f"Could not parse frame numbers: {e}",
                )

    def _on_edit(self):
        """User selected 'edit config'."""
        self.user_choice = "edit"
        self.dialog.accept()

    def _on_calib(self):
        """User selected 'show calibrated-only plots'."""
        print(f"[Button Click] Calibrated-only montage requested")
        self.user_choice = "calib"
        self.dialog.accept()

    def show_and_wait(self):
        """
        Show the montage dialog and wait for user decision.

        Returns:
            tuple: (choice, selected_frames)
                choice: 'save', 'repick', 'quit', or 'edit'
                selected_frames: list of frame numbers to repick (or empty if not 'repick')
        """
        print(f"\n{'='*70}")
        print(f"[Montage Dialog] Showing montage (trace_mode={self.trace_mode})")
        print(f"{'='*70}")
        
        # Ensure window is shown and in foreground
        self.dialog.show()
        self.dialog.raise_()
        self.dialog.activateWindow()
        
        print(f"[Montage Dialog] Window shown. Waiting for user input...")
        result = self.dialog.exec_()  # Blocking modal call - wait for user
        print(f"[Montage Dialog] Dialog closed. User choice: {self.user_choice}")
        
        return self.user_choice, self.selected_frames


def show_interactive_montage(
    deferred_frame_images, frame_count, base_dpi=150, trace_mode=False
):
    """
    Convenience function to show interactive montage and get user decision.

    Args:
        deferred_frame_images: List of frame plot numpy arrays
        frame_count: Total number of frames
        base_dpi: Base DPI for rendering
        trace_mode: If True, show simplified buttons (Yes/Quit) for trace review.
                   If False, show full buttons (Save All/Repick/Edit Config) for picks review.

    Returns:
        tuple: (choice, selected_frames) where choice is 'save', 'quit', 'repick', or 'edit'
    """
    viewer = InteractiveMontageViewer(deferred_frame_images, frame_count, base_dpi, trace_mode=trace_mode)
    return viewer.show_and_wait()
