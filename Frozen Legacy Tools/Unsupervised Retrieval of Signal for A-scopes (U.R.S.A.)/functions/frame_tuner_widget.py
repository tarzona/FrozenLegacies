"""
Interactive Frame Detection Tuner Widget
Integrated into URSA Step 3 verification dialog
"""

import json
import numpy as np
from pathlib import Path
from PIL import Image as PILImage

try:
    from PyQt5.QtWidgets import (
        QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QSlider, QLabel,
        QPushButton, QScrollArea, QApplication
    )
    from PyQt5.QtGui import QPixmap, QImage
    from PyQt5.QtCore import Qt
    HAS_QT = True
except ImportError:
    HAS_QT = False


def detect_frames_with_params(image, config, min_gap, gap_threshold, min_width):
    """Detect frames with specified parameters"""
    from preprocessing import detect_ascope_frames
    
    # Create temporary config with current parameters
    temp_config = json.loads(json.dumps(config))
    params = temp_config.get("processing_params", {})
    params["min_frame_gap_px"] = min_gap
    params["frame_detect_gap_threshold"] = gap_threshold
    params["min_frame_width_px"] = min_width
    params["frame_detect_min_frame_width_px"] = min_width
    temp_config["processing_params"] = params
    
    # Detect frames - returns just frames list, not a tuple
    frames = detect_ascope_frames(image, temp_config)
    return frames


class FrameTunerWidget(QMainWindow):
    """Interactive frame tuning window"""
    
    def __init__(self, image, config_path):
        super().__init__()
        self.image = image
        self.config_path = config_path
        self.frames = []
        
        print(f"DEBUG: FrameTunerWidget init - image type: {type(image)}, shape: {getattr(image, 'shape', 'N/A')}")
        
        # Load config
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        self.result = False
        self.init_ui()
        
    def init_ui(self):
        """Initialize UI"""
        self.setWindowTitle("Frame Detection Parameter Tuner")
        self.setGeometry(100, 100, 1600, 700)
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QHBoxLayout()
        
        # Left side: Controls
        left_layout = QVBoxLayout()
        left_layout.addWidget(QLabel("=== Interactive Frame Tuning ===\n"))
        
        # Parameters
        params = self.config.get("processing_params", {})
        
        # min_frame_gap_px
        left_layout.addWidget(QLabel("min_frame_gap_px:"))
        self.gap_px_label = QLabel(str(params.get("min_frame_gap_px", 2200)))
        self.gap_px_slider = QSlider(Qt.Horizontal)
        self.gap_px_slider.setMinimum(1000)
        self.gap_px_slider.setMaximum(3500)
        self.gap_px_slider.setValue(int(params.get("min_frame_gap_px", 2200)))
        self.gap_px_slider.setTickPosition(QSlider.TicksBelow)
        self.gap_px_slider.setTickInterval(500)
        self.gap_px_slider.valueChanged.connect(self.on_param_changed)
        left_layout.addWidget(self.gap_px_label)
        left_layout.addWidget(self.gap_px_slider)
        
        # frame_detect_gap_threshold
        left_layout.addWidget(QLabel("\nframe_detect_gap_threshold:"))
        gap_thresh = params.get("frame_detect_gap_threshold", 0.88)
        self.gap_thresh_label = QLabel(f"{gap_thresh:.2f}")
        self.gap_thresh_slider = QSlider(Qt.Horizontal)
        self.gap_thresh_slider.setMinimum(70)
        self.gap_thresh_slider.setMaximum(99)
        self.gap_thresh_slider.setValue(int(gap_thresh * 100))
        self.gap_thresh_slider.setTickPosition(QSlider.TicksBelow)
        self.gap_thresh_slider.setTickInterval(5)
        self.gap_thresh_slider.valueChanged.connect(self.on_param_changed)
        left_layout.addWidget(self.gap_thresh_label)
        left_layout.addWidget(self.gap_thresh_slider)
        
        # min_frame_width_px
        left_layout.addWidget(QLabel("\nmin_frame_width_px:"))
        self.width_px_label = QLabel(str(params.get("min_frame_width_px", 2200)))
        self.width_px_slider = QSlider(Qt.Horizontal)
        self.width_px_slider.setMinimum(1000)
        self.width_px_slider.setMaximum(3500)
        self.width_px_slider.setValue(int(params.get("min_frame_width_px", 2200)))
        self.width_px_slider.setTickPosition(QSlider.TicksBelow)
        self.width_px_slider.setTickInterval(500)
        self.width_px_slider.valueChanged.connect(self.on_param_changed)
        left_layout.addWidget(self.width_px_label)
        left_layout.addWidget(self.width_px_slider)
        
        # Status
        left_layout.addWidget(QLabel("\n--- Results ---"))
        self.status_label = QLabel("Frames: 0\nStatus: Ready")
        left_layout.addWidget(self.status_label)
        
        # Buttons
        left_layout.addStretch()
        
        button_layout = QVBoxLayout()
        save_btn = QPushButton("✓ SAVE & CLOSE")
        save_btn.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold; padding: 10px;")
        save_btn.clicked.connect(self.save_and_close)
        
        cancel_btn = QPushButton("✗ CANCEL (No Changes)")
        cancel_btn.setStyleSheet("background-color: #f44336; color: white; padding: 10px;")
        cancel_btn.clicked.connect(self.cancel_close)
        
        button_layout.addWidget(save_btn)
        button_layout.addWidget(cancel_btn)
        left_layout.addLayout(button_layout)
        
        # Right side: Image display
        self.image_label = QLabel()
        self.image_label.setMinimumWidth(1000)
        self.image_label.setAlignment(Qt.AlignCenter)
        
        scroll = QScrollArea()
        scroll.setWidget(self.image_label)
        scroll.setWidgetResizable(True)
        
        main_layout.addLayout(left_layout, 1)
        main_layout.addWidget(scroll, 2)
        
        central_widget.setLayout(main_layout)
        
        # Initial detection
        self.detect_and_display()
        
    def on_param_changed(self):
        """Called when any slider is moved"""
        self.detect_and_display()
        
    def detect_and_display(self):
        """Detect frames and display with current parameters"""
        min_gap = self.gap_px_slider.value()
        gap_threshold = self.gap_thresh_slider.value() / 100.0
        min_width = self.width_px_slider.value()
        
        # Update labels
        self.gap_px_label.setText(str(min_gap))
        self.gap_thresh_label.setText(f"{gap_threshold:.2f}")
        self.width_px_label.setText(str(min_width))
        
        # Detect frames
        try:
            print(f"DEBUG: Starting frame detection with min_gap={min_gap}, gap_threshold={gap_threshold}, min_width={min_width}")
            self.frames = detect_frames_with_params(
                self.image, self.config, min_gap, gap_threshold, min_width
            )
            print(f"DEBUG: Detected {len(self.frames)} frames")
            self.status_label.setText(f"Frames: {len(self.frames)}\nStatus: ✓ Detected")
        except Exception as e:
            print(f"ERROR in detect_and_display: {e}")
            import traceback
            traceback.print_exc()
            self.frames = []
            self.status_label.setText(f"Frames: 0\nStatus: ✗ Error: {str(e)[:30]}")
            return
        
        # Display
        self.display_frames()
        
    def display_frames(self):
        """Display image with frame boundaries"""
        try:
            # Validate image
            if self.image is None or self.image.size == 0:
                print("ERROR: Image is None or empty")
                self.status_label.setText("Status: ✗ No image data")
                return
            
            print(f"DEBUG: Image shape: {self.image.shape}, dtype: {self.image.dtype}")
            
            # Convert to RGB
            if len(self.image.shape) == 2:
                display_image = np.stack([self.image] * 3, axis=2)
            else:
                display_image = self.image[:, :, :3].copy() if self.image.shape[2] >= 3 else self.image.copy()
            
            height, width = self.image.shape[:2]
            print(f"DEBUG: Display image shape: {display_image.shape}")
            
            # Draw frame boundaries
            for i, (left, right) in enumerate(self.frames):
                display_image[:, max(0, left-3):min(width, left+4)] = [255, 0, 0]
                display_image[:, max(0, right-3):min(width, right+4)] = [255, 0, 0]
            
            # Convert to QPixmap
            display_image = display_image.astype(np.uint8)
            h, w = display_image.shape[:2]
            
            # Make sure data is contiguous for QImage
            display_image = np.ascontiguousarray(display_image)
            
            if len(display_image.shape) == 3 and display_image.shape[2] == 3:
                bytes_per_line = 3 * w
                q_img = QImage(display_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
            else:
                print(f"ERROR: Unexpected image shape for display: {display_image.shape}")
                self.status_label.setText("Status: ✗ Image format error")
                return
            
            pixmap = QPixmap.fromImage(q_img)
            
            if pixmap.isNull():
                print("ERROR: QPixmap is null")
                self.status_label.setText("Status: ✗ Pixmap creation failed")
                return
            
            print(f"DEBUG: Pixmap created successfully: {pixmap.width()}x{pixmap.height()}")
            
            # Scale
            scaled = pixmap.scaledToWidth(1000, Qt.SmoothTransformation)
            self.image_label.setPixmap(scaled)
            print("DEBUG: Image displayed successfully")
        except Exception as e:
            print(f"ERROR in display_frames: {e}")
            import traceback
            traceback.print_exc()
            self.status_label.setText(f"Status: ✗ Display error: {str(e)[:30]}")
        
    def save_and_close(self):
        """Save parameters and close"""
        params = self.config.get("processing_params", {})
        params["min_frame_gap_px"] = self.gap_px_slider.value()
        params["frame_detect_gap_threshold"] = self.gap_thresh_slider.value() / 100.0
        params["min_frame_width_px"] = self.width_px_slider.value()
        params["frame_detect_min_frame_width_px"] = self.width_px_slider.value()
        
        self.config["processing_params"] = params
        
        # Save
        with open(self.config_path, 'w') as f:
            json.dump(self.config, f, indent=2)
        
        print(f"✓ Saved {len(self.frames)} frame parameters")
        self.result = True
        self.close()
        
    def cancel_close(self):
        """Close without saving"""
        print("✗ Tuning cancelled")
        self.result = False
        self.close()


def launch_frame_tuner(image, config_path):
    """Launch the frame tuner window"""
    if not HAS_QT:
        print("ERROR: PyQt5 not available")
        return False
    
    app = QApplication.instance() or QApplication([])
    tuner = FrameTunerWidget(image, config_path)
    tuner.show()
    app.exec_()
    
    return tuner.result
