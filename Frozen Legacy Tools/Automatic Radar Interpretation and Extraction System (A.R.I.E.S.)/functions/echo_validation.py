import matplotlib.pyplot as plt
import numpy as np
import cv2


class EchoValidationInterface:
    """Interface for validating and iteratively refining automatic echo detection results."""

    def __init__(self, image, surface_trace, bed_trace, optimized_params):
        self.image = image
        self.surface_trace = surface_trace.copy()
        self.bed_trace = bed_trace.copy()
        self.optimized_params = optimized_params
        self.problem_regions = []
        self.refinement_iteration = 0
        self.max_iterations = 3

    def validate_results(self):
        """Show results and allow iterative refinement until user is satisfied."""

        while self.refinement_iteration < self.max_iterations:
            print(f"\n=== VALIDATION ITERATION {self.refinement_iteration + 1} ===")

            # Show current results
            user_satisfied = self._show_current_results()

            if user_satisfied:
                print("User satisfied with results - proceeding to CBD selection")
                break

            # Get problem regions for refinement
            problem_regions = self._get_problem_regions()

            if not problem_regions:
                print("No problem regions selected - proceeding with current results")
                break

            # Perform refinement
            self._refine_problem_regions(problem_regions)
            self.refinement_iteration += 1

        if self.refinement_iteration >= self.max_iterations:
            print(f"Maximum refinement iterations ({self.max_iterations}) reached")

        return self.problem_regions

    def _show_current_results(self):
        """Display current detection results and get user feedback."""

        fig, ax = plt.subplots(figsize=(24, 12))

        # Display image with detected traces
        enhanced = cv2.createCLAHE(clipLimit=3.0).apply(self.image)
        ax.imshow(enhanced, cmap="gray", aspect="auto")

        # Plot detected traces
        x_coords = np.arange(len(self.surface_trace))

        # Surface trace
        valid_surface = np.isfinite(self.surface_trace)
        if np.any(valid_surface):
            ax.plot(
                x_coords[valid_surface],
                self.surface_trace[valid_surface],
                "cyan",
                linewidth=2,
                label="Detected Surface",
                alpha=0.8,
            )

        # Bed trace
        valid_bed = np.isfinite(self.bed_trace)
        if np.any(valid_bed):
            ax.plot(
                x_coords[valid_bed],
                self.bed_trace[valid_bed],
                "orange",
                linewidth=2,
                label="Detected Bed",
                alpha=0.8,
            )

        # Add quality metrics
        surface_coverage = np.sum(valid_surface) / len(self.surface_trace) * 100
        bed_coverage = np.sum(valid_bed) / len(self.bed_trace) * 100

        ax.set_title(
            f"Echo Detection Results - Iteration {self.refinement_iteration + 1}\n"
            f"Surface Coverage: {surface_coverage:.1f}% | Bed Coverage: {bed_coverage:.1f}%\n"
            "Review the results and decide if refinement is needed",
            fontsize=16,
            fontweight="bold",
        )
        ax.legend()

        # Add instruction text
        ax.text(
            0.02,
            0.02,
            "RESULT REVIEW:\n"
            "• Cyan = Surface echoes\n"
            "• Orange = Bed echoes\n"
            "• Close window to continue",
            transform=ax.transAxes,
            fontsize=12,
            verticalalignment="bottom",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgreen", alpha=0.9),
        )

        plt.tight_layout()
        plt.show()

        # Get user satisfaction feedback
        while True:
            user_input = (
                input("\nAre you satisfied with these echo detection results? (y/n): ")
                .strip()
                .lower()
            )

            if user_input in ["y", "yes"]:
                return True
            elif user_input in ["n", "no"]:
                return False
            else:
                print("Please enter 'y' for yes or 'n' for no")

    def _get_problem_regions(self):
        """Allow user to select problem regions for refinement."""

        fig, ax = plt.subplots(figsize=(24, 12))

        # Display image with current traces
        enhanced = cv2.createCLAHE(clipLimit=3.0).apply(self.image)
        ax.imshow(enhanced, cmap="gray", aspect="auto")

        # Plot current traces
        x_coords = np.arange(len(self.surface_trace))

        valid_surface = np.isfinite(self.surface_trace)
        if np.any(valid_surface):
            ax.plot(
                x_coords[valid_surface],
                self.surface_trace[valid_surface],
                "cyan",
                linewidth=2,
                label="Current Surface",
                alpha=0.7,
            )

        valid_bed = np.isfinite(self.bed_trace)
        if np.any(valid_bed):
            ax.plot(
                x_coords[valid_bed],
                self.bed_trace[valid_bed],
                "orange",
                linewidth=2,
                label="Current Bed",
                alpha=0.7,
            )

        ax.set_title(
            "Select Problem Regions for Refinement\n"
            "Click and drag to select regions that need improvement",
            fontsize=16,
            fontweight="bold",
        )
        ax.legend()

        # Add region selector for problem areas
        from matplotlib.widgets import RectangleSelector

        self.problem_regions = []

        def on_problem_region_select(eclick, erelease):
            x1, x2 = sorted([eclick.xdata, erelease.xdata])
            y1, y2 = sorted([eclick.ydata, erelease.ydata])

            if x1 is not None and x2 is not None:
                self.problem_regions.append(
                    {
                        "x_range": (int(x1), int(x2)),
                        "y_range": (int(y1), int(y2)),
                        "needs_retuning": True,
                    }
                )

                print(f"Marked problem region: X={int(x1)}-{int(x2)}")

                # Draw the selected region
                from matplotlib.patches import Rectangle

                rect = Rectangle(
                    (x1, y1),
                    x2 - x1,
                    y2 - y1,
                    linewidth=2,
                    edgecolor="red",
                    facecolor="red",
                    alpha=0.3,
                    label=f"Problem Region {len(self.problem_regions)}",
                )
                ax.add_patch(rect)
                ax.legend()
                fig.canvas.draw()

        problem_selector = RectangleSelector(
            ax,
            on_problem_region_select,
            useblit=True,
            button=[1],
            minspanx=10,
            minspany=10,
        )

        # Add instruction text
        ax.text(
            0.02,
            0.98,
            "PROBLEM REGION SELECTION:\n"
            "• Click and drag to select problematic areas\n"
            "• Red boxes show selected regions\n"
            "• Close window when done selecting",
            transform=ax.transAxes,
            fontsize=12,
            verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="yellow", alpha=0.9),
        )

        plt.tight_layout()
        plt.show()

        return self.problem_regions

    def _refine_problem_regions(self, problem_regions):
        """Refine detection in selected problem regions."""
        from functions.echo_tracing import detect_surface_echo, detect_bed_echo

        print(f"Refining {len(problem_regions)} problem regions...")

        for i, region in enumerate(problem_regions):
            x_start, x_end = region["x_range"]
            print(f"Refining region {i + 1}: X={x_start}-{x_end}")

            # Extract problem region
            region_data = self.image[:, x_start:x_end]

            if region_data.shape[1] > 0:
                # Apply more aggressive parameters for problem regions
                adjusted_surface_params = self.optimized_params.get(
                    "surface_detection", {}
                ).copy()
                adjusted_bed_params = self.optimized_params.get(
                    "bed_detection", {}
                ).copy()

                # Reduce prominence threshold for difficult regions
                adjusted_surface_params["peak_prominence"] = (
                    adjusted_surface_params.get("peak_prominence", 20) * 0.7
                )
                adjusted_bed_params["peak_prominence"] = (
                    adjusted_bed_params.get("peak_prominence", 30) * 0.7
                )

                # Increase CLAHE enhancement
                adjusted_surface_params["enhancement_clahe_clip"] = min(
                    8.0,
                    adjusted_surface_params.get("enhancement_clahe_clip", 3.0) * 1.5,
                )
                adjusted_bed_params["enhancement_clahe_clip"] = min(
                    8.0, adjusted_bed_params.get("enhancement_clahe_clip", 3.0) * 1.5
                )

                # Re-run detection on problem region
                try:
                    # Assuming we have access to transmitter pulse position
                    tx_pulse_y_rel = 50  # This should be passed from the main processor

                    region_surface = detect_surface_echo(
                        region_data, tx_pulse_y_rel, adjusted_surface_params
                    )
                    region_bed = detect_bed_echo(
                        region_data,
                        region_surface,
                        region_data.shape[0] - 50,
                        adjusted_bed_params,
                    )

                    # Update the main traces with refined results
                    self.surface_trace[x_start:x_end] = region_surface
                    self.bed_trace[x_start:x_end] = region_bed

                    print(f"Region {i + 1} refinement completed")

                except Exception as e:
                    print(f"Error refining region {i + 1}: {e}")
