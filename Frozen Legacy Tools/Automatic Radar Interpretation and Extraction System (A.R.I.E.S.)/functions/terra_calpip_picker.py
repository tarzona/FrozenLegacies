"""
TERRA-style manual calpip picker for ARIES
Implements the exact same calpip selection interface as TERRA with windowed view and slider
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QMessageBox
from PyQt5.QtCore import Qt
import os


class TerraCalpipPicker:
    """
    Full TERRA-style calpip picker with windowed view and horizontal slider navigation
    Exact replica of TERRA's CalpipPicker class
    """
    
    def __init__(self, image, base_filename):
        # Image and radar data
        self.arr = image  # Full radar image array
        self.base_filename = base_filename
        
        # Window and view parameters (optimized for proper scaling)
        self.WINDOW_W_PX = 1000
        self.WINDOW_H_PX = 600
        self.DPI = 80
        self.view_w = min(1000, self.arr.shape[1])  # Width of viewable area
        self.left = 0  # Current left position in full image
        
        # Picker state
        self.picks_y = []
        self.result_pixel_distance = None
        self.result_y_lines = []
        self.calpip_grid_lines = []
        self.end_bound_y = None
        
        # UI state
        self.accept_clicks = False
        self.accept_bound_click = False
        self.done = False
        self.no_calpips = False
        
        # Visual elements
        self.image_artist = None
        self.pick_lines = []
        self.interp_lines = []
        self.bound_line = None
        self.grid_lines = []
        
    def run(self):
        """Run the TERRA-style calpip picker and return pixel distance or None"""
        print(f"Opening TERRA-style calpip picker for {self.base_filename}...")
        
        # Create the TERRA-style windowed interface
        self.create_calpip_window()
        
        if self.no_calpips:
            return None
        elif self.done and self.result_pixel_distance:
            print(f"TERRA method: Selected {len(self.picks_y)} calpip lines")
            print(f"TERRA method: Average spacing = {self.result_pixel_distance:.2f} pixels")
            print(f"TERRA method: Each line = 2.0 μs")
            print(f"TERRA method: Calibration = {self.result_pixel_distance:.2f} px ÷ 2.0 μs = {self.result_pixel_distance/2.0:.3f} px/μs")
            return self.result_pixel_distance
        else:
            return None
    
    def create_calpip_window(self):
        """Create the complete TERRA-style calpip picker GUI with slider navigation using PyQt5"""
        # Create PyQt5 application if it doesn't exist
        app = QApplication.instance()
        if app is None:
            app = QApplication([])
        
        # Create main window
        self.window = QMainWindow()
        self.window.setWindowTitle("TERRA Method - Calpip Picker")
        self.window.setGeometry(100, 100, 1400, 800)
        
        # Central widget with horizontal layout
        central_widget = QWidget()
        self.window.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        
        # Left control panel
        control_widget = QWidget()
        control_widget.setFixedWidth(250)
        control_layout = QVBoxLayout(control_widget)
        main_layout.addWidget(control_widget)
        
        # Instructions
        title_label = QLabel("TERRA Method - Calpip Processing")
        title_label.setStyleSheet("font: bold 14px;")
        control_layout.addWidget(title_label)
        
        instructions = QLabel("1. Select end bound\n2. Pick 4 calpips\n3. Click Done\n\nUse slider to navigate")
        instructions.setWordWrap(True)
        control_layout.addWidget(instructions)
        
        # Buttons
        self.bound_btn = QPushButton("Select End Bound")
        self.bound_btn.setStyleSheet("background-color: lightyellow;")
        self.bound_btn.clicked.connect(self.start_bound_selection)
        control_layout.addWidget(self.bound_btn)
        
        self.pick_btn = QPushButton("Pick Calpips")
        self.pick_btn.setStyleSheet("background-color: lightblue;")
        self.pick_btn.setEnabled(False)
        self.pick_btn.clicked.connect(self.start_picking)
        control_layout.addWidget(self.pick_btn)
        
        self.done_btn = QPushButton("Done")
        self.done_btn.setStyleSheet("background-color: lightgreen;")
        self.done_btn.clicked.connect(self.finish_picking)
        control_layout.addWidget(self.done_btn)
        
        self.no_calpips_btn = QPushButton("No Calpips")
        self.no_calpips_btn.setStyleSheet("background-color: lightcoral;")
        self.no_calpips_btn.clicked.connect(self.no_calpips_selected)
        control_layout.addWidget(self.no_calpips_btn)
        
        # Status label
        self.status_label = QLabel("Click 'Select End Bound' to start")
        self.status_label.setStyleSheet("color: blue;")
        self.status_label.setWordWrap(True)
        control_layout.addWidget(self.status_label)
        
        control_layout.addStretch()  # Push everything to top
        
        # Right plot area
        plot_widget = QWidget()
        plot_layout = QVBoxLayout(plot_widget)
        main_layout.addWidget(plot_widget)
        
        # Figure setup with proper sizing
        figsize = (self.WINDOW_W_PX / self.DPI, self.WINDOW_H_PX / self.DPI)
        self.fig, self.ax = plt.subplots(figsize=figsize, dpi=self.DPI)
        self.fig.tight_layout(pad=2.0)
        
        # Configure axis for proper image display
        self.ax.set_xlabel('X Position (pixels)')
        self.ax.set_ylabel('Y Position (pixels)')
        
        self.canvas = FigureCanvas(self.fig)
        plot_layout.addWidget(self.canvas)
        
        # Horizontal slider for scrolling
        self.ax_slider = self.fig.add_axes([0.12, 0.05, 0.62, 0.03], facecolor="lightgoldenrodyellow")
        max_left = max(0, self.arr.shape[1] - self.view_w)
        self.slider = Slider(self.ax_slider, "Scroll", 0, max_left, valinit=0, valstep=10)
        self.slider.on_changed(self.on_slider)
        
        # Navigation buttons
        self.ax_left = self.fig.add_axes([0.83, 0.05, 0.05, 0.03])
        self.ax_right = self.fig.add_axes([0.90, 0.05, 0.05, 0.03])
        self.btn_left = Button(self.ax_left, "←")
        self.btn_right = Button(self.ax_right, "→")
        self.btn_left.on_clicked(lambda e: self.slider.set_val(max(0, self.slider.val - 50)))
        self.btn_right.on_clicked(lambda e: self.slider.set_val(min(self.slider.valmax, self.slider.val + 50)))
        
        # Set up matplotlib click handler
        self.fig.canvas.mpl_connect("button_press_event", self.on_matplotlib_click)
        
        # Draw initial view
        self.draw_view()
        self.ax.set_title(f"TERRA Method: Pick Calpips - {self.base_filename}", fontsize=12, pad=10)
        
        # Show window and run
        self.window.show()
        app.exec_()
    
    def draw_view(self):
        """Draw the current view slice"""
        L = int(self.left)
        R = int(min(L + self.view_w, self.arr.shape[1]))
        view = self.arr[:, L:R]
        
        if self.image_artist is None:
            # Use proper aspect ratio to prevent distortion
            self.image_artist = self.ax.imshow(view, cmap="gray", aspect="equal", 
                                             interpolation="nearest", origin="upper")
        else:
            self.image_artist.set_data(view)
            self.image_artist.set_extent([0, R - L, view.shape[0], 0])
        
        self.ax.set_xlim(0, R - L)
        self.ax.set_ylim(view.shape[0], 0)
        
        # Ensure proper scaling
        self.ax.set_aspect('equal', adjustable='box')
        
        self.canvas.draw()
    
    def on_slider(self, val):
        """Handle slider movement"""
        self.left = int(val)
        self.draw_view()
        self.redraw_all_lines()
    
    def redraw_all_lines(self):
        """Redraw all lines after view change"""
        # Clear existing line artists
        for line in self.pick_lines + self.interp_lines:
            if line in self.ax.lines:
                line.remove()
        self.pick_lines = []
        self.interp_lines = []
        if self.bound_line and self.bound_line in self.ax.lines:
            self.bound_line.remove()
            self.bound_line = None
        
        # Redraw bound line
        if self.end_bound_y is not None:
            y_disp = self.end_bound_y
            x0, x1 = self.ax.get_xlim()
            self.bound_line, = self.ax.plot([x0, x1], [y_disp, y_disp], color="red", linewidth=3)
            self.ax.text(x0 + 10, y_disp - 10, "End Bound", color="red", fontweight="bold")
        
        # Redraw pick lines
        for pick_y in self.picks_y:
            y_disp = pick_y
            x0, x1 = self.ax.get_xlim()
            line, = self.ax.plot([x0, x1], [y_disp, y_disp], color="yellow", linewidth=2)
            self.pick_lines.append(line)
        
        # Redraw interpolated lines
        for i, y_line in enumerate(self.result_y_lines):
            y_disp = y_line
            x0, x1 = self.ax.get_xlim()
            line, = self.ax.plot([x0, x1], [y_disp, y_disp], color="cyan", linewidth=1, linestyle="--")
            self.interp_lines.append(line)
            
            # Add time label
            time_us = i * 2
            self.ax.text(x1 - 60, y_disp - 5, f"{time_us}μs", color="cyan", fontsize=8)
        
        self.canvas.draw()
        
    def start_bound_selection(self):
        """Start the end bound selection process"""
        self.accept_bound_click = True
        self.accept_clicks = False
        self.status_label.setText("Click on the image to select the end bound for y-axis")
        self.status_label.setStyleSheet("color: orange;")
        self.bound_btn.setEnabled(False)
        
    def start_picking(self, event=None):
        """Start the calpip picking process"""
        self.picks_y = []
        self.accept_clicks = True
        self.accept_bound_click = False
        self.status_label.setText(f"Click on 4 calpips (picked: {len(self.picks_y)}/4)")
        self.status_label.setStyleSheet("color: green;")
        self.pick_btn.setEnabled(False)
        
    def on_matplotlib_click(self, event):
        """Handle matplotlib click events"""
        if event.inaxes != self.ax or event.ydata is None:
            return
            
        # Convert matplotlib coordinates to original image coordinates
        y_click = event.ydata
        
        if self.accept_bound_click:
            # Setting end bound
            self.end_bound_y = y_click
            self.accept_bound_click = False
            
            # Draw bound line
            x0, x1 = self.ax.get_xlim()
            if self.bound_line and self.bound_line in self.ax.lines:
                self.bound_line.remove()
            self.bound_line, = self.ax.plot([x0, x1], [y_click, y_click], color="red", linewidth=3)
            self.ax.text(x0 + 10, y_click - 10, "End Bound", color="red", fontweight="bold")
            
            self.canvas.draw()
            
            self.status_label.setText("End bound set! Now click 'Pick Calpips'")
            self.status_label.setStyleSheet("color: blue;")
            self.bound_btn.setEnabled(True)
            self.pick_btn.setEnabled(True)
            return
            
        if self.accept_clicks:
            # Picking calpips
            self.picks_y.append(y_click)
            
            # Draw pick line
            x0, x1 = self.ax.get_xlim()
            line, = self.ax.plot([x0, x1], [y_click, y_click], color="yellow", linewidth=2)
            self.pick_lines.append(line)
            
            self.canvas.draw()
            
            self.status_label.setText(f"Click on 4 calpips (picked: {len(self.picks_y)}/4)")
            
            if len(self.picks_y) >= 4:
                self.accept_clicks = False
                self.calculate_interpolation()
                self.pick_btn.setEnabled(True)
                self.status_label.setText("4 calpips picked! Click Done to continue.")
    
    def calculate_spacing(self):
        """Calculate average spacing between picked calpip lines and generate full grid"""
        if len(self.picks_y) < 2:
            return
        
        # Sort the y-coordinates
        sorted_picks = sorted(self.picks_y)
        
        # Calculate spacings between adjacent lines
        spacings = []
        for i in range(len(sorted_picks) - 1):
            spacing = sorted_picks[i + 1] - sorted_picks[i]
            spacings.append(spacing)
        
        # Calculate average spacing
        self.result_pixel_distance = np.mean(spacings)
        
        # Generate full calpip grid like TERRA does
        self.generate_calpip_grid(sorted_picks[0], self.result_pixel_distance)
        
        # Update status
        self.update_status(f"Average spacing: {self.result_pixel_distance:.2f} pixels. Full grid generated. Click 'Done' to confirm.")
        
        print(f"TERRA method: Individual spacings: {spacings}")
        print(f"TERRA method: Average spacing: {self.result_pixel_distance:.2f} pixels")
        print(f"TERRA method: Calibration factor: {self.result_pixel_distance/2.0:.3f} px/μs")
        print(f"TERRA method: Generated full calpip grid for entire radargram")

    def generate_calpip_grid(self, first_pick_y, avg_spacing):
        """Generate full calpip grid lines like TERRA does"""
        # Clear any existing grid lines
        if hasattr(self, 'grid_lines'):
            for line in self.grid_lines:
                if line in self.ax.lines:
                    line.remove()
        
        self.grid_lines = []
        
        # Generate grid lines covering the entire image
        # Start from a position that ensures we cover the entire range
        image_height = self.image.shape[0]
        
        # Calculate grid starting point (go far enough up to cover any area above first pick)
        start_grid_index = int(np.floor((0 - first_pick_y) / avg_spacing)) - 5
        
        # Generate lines using the picked spacing pattern
        current_index = start_grid_index
        grid_y_positions = []
        
        while True:
            current_y = first_pick_y + (current_index * avg_spacing)
            
            # Stop if we've gone past the bottom of the image
            if current_y > image_height:
                break
                
            # Only include lines that are within the image bounds
            if 0 <= current_y < image_height:
                grid_y_positions.append(current_y)
                
                # Draw the grid line
                x_limits = self.ax.get_xlim()
                line = self.ax.axhline(y=current_y, color='cyan', linewidth=1, 
                                     linestyle='--', alpha=0.6)
                self.grid_lines.append(line)
                
                # Add time label every few lines (like TERRA does)
                if current_index % 2 == 0:  # Label every other line
                    time_us = current_index * 2  # Each line = 2μs in TERRA method
                    self.ax.text(x_limits[1] * 0.02, current_y - 3, f'{time_us}μs', 
                               color='cyan', fontsize=8, alpha=0.8)
                
            current_index += 1
        
        # Store grid information
        self.calpip_grid_lines = grid_y_positions
        
        # Redraw the figure
        self.fig.canvas.draw()
        
        print(f"TERRA method: Generated {len(grid_y_positions)} calpip grid lines")
        print("TERRA method: Grid covers entire radargram with 2μs intervals")
    
    def finish_picking(self, event=None):
        """Finish the picking process"""
        if len(self.picks_y) < 2:
            messagebox.showwarning("Incomplete", "Please pick at least 2 calpip lines!")
            return
        
        if self.result_pixel_distance is None:
            self.calculate_spacing()
        
        if self.result_pixel_distance:
            self.done = True
            plt.close(self.fig)
            print(f"TERRA method: Finished with {len(self.picks_y)} lines, spacing={self.result_pixel_distance:.2f}px")
    
    def cancel_picking(self, event=None):
        """Cancel the picking process"""
        self.cancelled = True
        plt.close(self.fig)
        print("TERRA method: Cancelled by user")
    
    def update_status(self, message):
        """Update the status text"""
        self.status_text.set_text(message)
        self.canvas.draw_idle()

    def calculate_interpolation(self):
        """Calculate calpip spacing and interpolate lines following the yellow picks exactly (TERRA method)"""
        if len(self.picks_y) < 2:
            return
            
        # Sort picks and calculate average spacing
        sorted_picks = sorted(self.picks_y)
        spacings = [sorted_picks[i+1] - sorted_picks[i] for i in range(len(sorted_picks)-1)]
        avg_spacing = np.mean(spacings)
        self.result_pixel_distance = avg_spacing
        
        # End at user-selected bound (never go past this)
        if self.end_bound_y is not None:
            end_y = self.end_bound_y
        else:
            # Fallback: use bottom of radar data area
            end_y = self.arr.shape[0] - 50
        
        # Use the first yellow pick as the reference grid point - no shifting
        first_pick_y = sorted_picks[0]
        
        # Generate grid lines using the exact yellow pick spacing pattern
        self.result_y_lines = []
        
        # Start from a position that ensures we cover the entire range
        # Go far enough up to cover any area above the first pick
        start_grid_index = int(np.floor((0 - first_pick_y) / avg_spacing)) - 5
        
        # Generate lines using the yellow pick spacing pattern exactly
        current_index = start_grid_index
        while True:
            current_y = first_pick_y + (current_index * avg_spacing)
            
            # Stop if we've gone past the end bound
            if current_y > end_y:
                break
                
            # Only include lines that are within reasonable bounds
            if current_y >= 0:
                self.result_y_lines.append(current_y)
                
            current_index += 1
        
        # Store as calpip grid lines for return
        self.calpip_grid_lines = self.result_y_lines.copy()
        
        # Update status with average spacing
        self.status_label.setText(f"Average spacing: {avg_spacing:.1f} pixels, {len(self.result_y_lines)} lines")
        
        # Draw interpolated lines
        self.draw_interpolated_lines()
        
        print(f"TERRA method: Individual spacings: {spacings}")
        print(f"TERRA method: Average spacing: {self.result_pixel_distance:.2f} pixels")
        print(f"TERRA method: Generated {len(self.result_y_lines)} total grid lines")
        print(f"TERRA method: Calibration factor: {self.result_pixel_distance/2.0:.3f} px/μs")
    
    def draw_interpolated_lines(self):
        """Draw clean horizontal calpip lines for picking"""
        # Clear existing interpolated lines
        for line in self.interp_lines:
            if line in self.ax.lines:
                line.remove()
        self.interp_lines = []
        
        # Draw clean horizontal calpip lines in cyan (for picking reference only)
        x0, x1 = self.ax.get_xlim()
        for i, y_line in enumerate(self.result_y_lines):
            if 0 <= y_line < self.arr.shape[0]:
                line, = self.ax.plot([x0, x1], [y_line, y_line], color="cyan", linewidth=1, linestyle="--", alpha=0.7)
                self.interp_lines.append(line)
                
                # Add time labels every few lines
                if i % 2 == 0:  # Label every other line
                    time_us = i * 2  # Each line represents 2μs
                    self.ax.text(x1 - 60, y_line - 5, f"{time_us}μs", color="cyan", fontsize=8)
        
        self.canvas.draw()
    
    def finish_picking(self):
        """Finish the calpip picking process"""
        if self.end_bound_y is None:
            QMessageBox.warning(self.window, "Missing End Bound", "Please select the end bound first!")
        elif len(self.picks_y) < 4:
            QMessageBox.warning(self.window, "Incomplete", "Please pick 4 calpips!")
        elif not self.result_pixel_distance:
            QMessageBox.warning(self.window, "No Calculation", "Calpip calculation failed!")
        else:
            self.done = True
            self.window.close()
    
    def no_calpips_selected(self):
        """User selected no calpips"""
        self.no_calpips = True
        self.done = True
        self.window.close()


def terra_calpip_method(image, base_filename):
    """
    Run TERRA-style manual calpip picking
    
    Args:
        image: 2D numpy array of the radar image
        base_filename: Base filename for display
        
    Returns:
        dict or None: Calpip detection results in ARIES format, or None if cancelled
    """
    picker = TerraCalpipPicker(image, base_filename)
    pixel_distance = picker.run()
    
    if pixel_distance is None:
        return None
    
    # Get full grid information if available
    full_grid_lines = getattr(picker, 'calpip_grid_lines', picker.picks_y)
    
    # Return calibration details, y-line positions, and pixel distance (for ARIES compatibility)
    calibration_details = {
        'x_position': image.shape[1] // 2,  # Center of image
        'y_start': 0,
        'y_end': image.shape[0],
        'tick_count': len(full_grid_lines),  # Total grid lines, not just picked ones
        'mean_spacing': pixel_distance,
        'tick_positions': full_grid_lines,  # Full grid positions
        'manual_picks': picker.picks_y,  # Original manual picks
        'z_boundary': image.shape[0],
        'match_score': 1.0,  # Perfect score for manual selection
        'method': 'TERRA Manual',
        'calibration_notes': f'Manual selection of {len(picker.picks_y)} calpip lines, generated {len(full_grid_lines)} total grid lines, 2.0 μs per line'
    }
    
    print(f"TERRA method calibration completed: {len(picker.picks_y)} lines, {pixel_distance:.2f}px spacing")
    print(f"TERRA method: Generated {len(full_grid_lines)} y-axis grid lines")
    return calibration_details, full_grid_lines, pixel_distance