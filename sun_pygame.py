import tkinter as tk
import math


class SunDisplay:
    def __init__(self, width=400, height=400, title="Sun Display"):
        self.width = width
        self.height = height
        self.center_x = width // 2
        self.center_y = height // 2
        self.max_arrow_length = min(width, height) // 3

        # Initialize the GUI
        self.root = tk.Tk()
        self.root.title(title)
        self.root.geometry(f"{width}x{height}")

        self.canvas = tk.Canvas(self.root, width=width, height=height, bg='black')
        self.canvas.pack()

        # Store arrow values and visual elements
        self.values = [0] * 6
        self.theta = None  # Store interpolated theta angle
        self.arrows = []
        self.theta_indicator = None  # Visual indicator for theta
        self.theta_label = None  # Text label for theta
        self.angle_step = 60  # 360 degrees / 6 arrows

        # Store custom angles and values
        self.custom_angles = None
        self.custom_angle_values = None

        # Create initial arrows
        self.create_arrows()

        # Process any pending events so window appears
        self.root.update_idletasks()

        # Make window non-blocking and set it to stay on top initially
        self.root.lift()
        self.root.attributes('-topmost', True)
        self.root.after(100, lambda: self.root.attributes('-topmost', False))

    def create_arrows(self):
        """Create the initial 6 arrows pointing outward"""
        self.arrows = []
        for i in range(6):
            angle = i * self.angle_step
            arrow_id = self.canvas.create_line(
                self.center_x, self.center_y,
                self.center_x, self.center_y,
                fill='yellow', width=3, arrow=tk.LAST, arrowshape=(10, 12, 3)
            )
            self.arrows.append(arrow_id)

        # Draw center circle (sun core)
        self.sun_core = self.canvas.create_oval(
            self.center_x - 10, self.center_y - 10,
            self.center_x + 10, self.center_y + 10,
            fill='orange', outline='yellow', width=2
        )

    def update(self, values, theta=None, angles=None, angle_values=None):
        """Update the sun with new values and optional interpolated theta angle

        Args:
            values: List of 6 values for default arrows (used when angles/angle_values not provided)
            theta: Optional interpolated theta angle
            angles: Optional array of custom angles (in radians)
            angle_values: Optional array of values corresponding to custom angles
        """
        # Store the parameters
        self.values = values.copy()
        self.theta = theta

        # Check if we should use custom angles
        if (angles is not None and angle_values is not None and
                len(angles) == len(angle_values)):
            self.theta = None
            self.custom_angles = angles.copy()
            self.custom_angle_values = angle_values.copy()
        else:
            self.custom_angles = None
            self.custom_angle_values = None
            if len(values) != 6:
                raise ValueError("Exactly 6 values required when not using custom angles")

        self.redraw_arrows()

        # Process GUI events so the display updates
        # Use update_idletasks() instead of update() to avoid conflicts with matplotlib
        try:
            self.root.update_idletasks()
        except tk.TclError:
            # Window was closed
            pass

    def redraw_arrows(self):
        """Redraw all arrows based on current values"""
        # Clear existing arrows
        for arrow_id in self.arrows:
            self.canvas.delete(arrow_id)
        self.arrows = []

        # Determine which values and angles to use
        if self.custom_angles is not None and self.custom_angle_values is not None:
            # Use custom angles and values
            angles_to_use = self.custom_angles
            values_to_use = self.custom_angle_values
        else:
            # Use default 6 arrows
            angles_to_use = [math.radians(i * self.angle_step) for i in range(6)]
            values_to_use = self.values

        # Calculate max value for normalization
        max_val = max([abs(v) for v in values_to_use]) if any(v != 0 for v in values_to_use) else 1

        # Create arrows for the specified angles
        for i, (angle, value) in enumerate(zip(angles_to_use, values_to_use)):
            # Calculate arrow length based on absolute value (normalized)
            normalized_length = abs(value * 40)

            # Calculate end point
            end_x = self.center_x + normalized_length * math.cos(angle)
            end_y = self.center_y + normalized_length * math.sin(angle)

            # Create arrow
            arrow_id = self.canvas.create_line(
                self.center_x, self.center_y,
                end_x, end_y,
                width=3, arrow=tk.LAST, arrowshape=(10, 12, 3)
            )
            self.arrows.append(arrow_id)

            # Color based on intensity and sign
            if value >= 0:
                # Positive values: Yellow to red gradient
                intensity = int((value / max_val) * 255)
                color = f"#{intensity:02x}{intensity // 2:02x}00"
            else:
                # Negative values: Blue to green gradient
                # Use absolute value for intensity calculation
                intensity = int((abs(value) / max_val) * 255)
                color = f"#00{intensity // 2:02x}{intensity:02x}"
            self.canvas.itemconfig(arrow_id, fill=color)

        # Draw theta indicator if theta is provided
        self.draw_theta_indicator()

    def draw_theta_indicator(self):
        """Draw a visual indicator for the interpolated theta angle"""
        # Remove previous theta indicator
        if self.theta_indicator:
            self.canvas.delete(self.theta_indicator)
            self.theta_indicator = None

        if self.theta is not None:
            # Convert theta to radians (assuming theta is in degrees)
            theta_rad = self.theta  # math.radians(self.theta - 90)  # -90 to start from top

            # Create a longer line to show the interpolated direction
            indicator_length = self.max_arrow_length * 1.2
            end_x = self.center_x + indicator_length * math.cos(theta_rad)
            end_y = self.center_y + indicator_length * math.sin(theta_rad)

            # Draw the theta indicator as a thick white line with different arrow
            self.theta_indicator = self.canvas.create_line(
                self.center_x, self.center_y,
                end_x, end_y,
                fill='white', width=5, arrow=tk.LAST,
                arrowshape=(15, 18, 5)
            )

            # Add text label showing the theta value
            label_x = self.center_x + (indicator_length + 30) * math.cos(theta_rad)
            label_y = self.center_y + (indicator_length + 30) * math.sin(theta_rad)

            if hasattr(self, 'theta_label') and self.theta_label:
                self.canvas.delete(self.theta_label)

            self.theta_label = self.canvas.create_text(
                label_x, label_y,
                text=f"θ: {(self.theta * 180 / math.pi):.1f}°",
                fill='white', font=('Arial', 12, 'bold')
            )

    def is_open(self):
        """Check if the window is still open"""
        try:
            return self.root.winfo_exists()
        except tk.TclError:
            return False

    def close(self):
        """Close the display"""
        try:
            self.root.quit()
            self.root.destroy()
        except tk.TclError:
            pass

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
import matplotlib.cm as cm


class SunDisplay3D:
    def __init__(self, size=8, title="3D Sun Display"):
        """
        Initialize 3D Sun Display

        Args:
            size: Size of the display window
            title: Window title
        """
        # Create figure and 3D axis
        self.fig = plt.figure(figsize=(size, size))
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.fig.suptitle(title)

        # Display parameters
        self.max_arrow_length = 2.0
        # self.arrow_scale = 1.5 # This was defined but not clearly used, consider if needed

        # Data storage
        self.theta = None
        self.phi = None
        self.r = None
        self.theta_array = None
        self.phi_array = None
        self.r_array = None

        # Visual elements storage
        self.arrows = []  # Stores individual Line2D artists from array updates
        self.sun_core = None
        self.interpolated_arrow_parts = [] # Stores [shaft, head] Line2D artists for single updates
        self.text_annotations = [] # Stores text artists
        self.colorbar = None # Store colorbar instance

        # Setup the display
        self.setup_display()
        self.create_sun_core()

        # Show the plot
        plt.ion()  # Interactive mode
        plt.show()

    def setup_display(self):
        """Setup the 3D display properties"""
        # Set equal aspect ratio and limits (will be adjusted dynamically)
        limit = self.max_arrow_length * 1.5
        self.ax.set_xlim([-limit, limit])
        self.ax.set_ylim([-limit, limit])
        self.ax.set_zlim([-limit, limit])

        # Set labels
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')

        # Set background color
        self.ax.xaxis.pane.fill = False
        self.ax.yaxis.pane.fill = False
        self.ax.zaxis.pane.fill = False

        # Make panes transparent
        self.ax.xaxis.pane.set_alpha(0.1)
        self.ax.yaxis.pane.set_alpha(0.1)
        self.ax.zaxis.pane.set_alpha(0.1)

        # Set grid
        self.ax.grid(True, alpha=0.3)

        # Set viewing angle
        self.ax.view_init(elev=20, azim=45)

        # Ensure equal aspect ratio for true scaling
        self.ax.set_box_aspect([1,1,1])

    def create_sun_core(self):
        """Create the central sun core"""
        # Create a sphere for the sun core
        u = np.linspace(0, 2 * np.pi, 20)
        v = np.linspace(0, np.pi, 20)
        x_core = 0.2 * np.outer(np.cos(u), np.sin(v))
        y_core = 0.2 * np.outer(np.sin(u), np.sin(v))
        z_core = 0.2 * np.outer(np.ones(np.size(u)), np.cos(v))

        self.sun_core = self.ax.plot_surface(x_core, y_core, z_core,
                                             color='orange', alpha=0.8, zorder=0) # zorder for sun core

    def spherical_to_cartesian(self, theta, phi, r=1.0):
        """
        Convert spherical coordinates to cartesian

        Args:
            theta: Azimuthal angle (radians)
            phi: Polar angle (radians)
            r: Radius

        Returns:
            x, y, z coordinates
        """
        x = r * np.sin(phi) * np.cos(theta)
        y = r * np.sin(phi) * np.sin(theta)
        z = r * np.cos(phi)
        return x, y, z

    def adjust_display_limits(self, max_magnitude):
        """Adjust display limits to accommodate the largest arrow"""
        limit = max(abs(max_magnitude) * 1.3, 1.0)  # At least 1.0 for visibility, use abs for max_magnitude
        self.ax.set_xlim([-limit, limit])
        self.ax.set_ylim([-limit, limit])
        self.ax.set_zlim([-limit, limit])
        self.ax.set_box_aspect([1,1,1])  # Maintain equal aspect ratio

    def clear_arrows(self):
        """Clear all existing arrows and associated text from the display."""
        # Remove multiple arrows from array update
        for artist in self.arrows:
            if artist:
                try:
                    artist.remove()
                except (ValueError, AttributeError): pass # May already be gone or not a valid artist
        self.arrows.clear()

        # Remove single interpolated arrow (shaft and head parts)
        for artist in self.interpolated_arrow_parts:
            if artist:
                try:
                    artist.remove()
                except (ValueError, AttributeError): pass
        self.interpolated_arrow_parts.clear()

        # Remove text annotations
        for text_artist in self.text_annotations:
            if text_artist:
                try:
                    text_artist.remove()
                except (ValueError, AttributeError): pass
        self.text_annotations.clear()

        # Remove colorbar if it exists (handled in _update_array, but good practice if generalized)
        if hasattr(self, 'colorbar') and self.colorbar:
            try:
                self.colorbar.remove()
            except: # Catch any error during removal
                pass
            self.colorbar = None


    def create_3d_arrow(self, start, end, color='red', alpha=0.8, width=0.05):
        """
        Create a 3D arrow from start to end point

        Args:
            start: Starting point (x, y, z)
            end: Ending point (x, y, z)
            color: Arrow color
            alpha: Transparency
            width: Arrow base width factor

        Returns:
            A list of Matplotlib Line2D artists [shaft_artist, head_artist], or empty list if length is zero.
        """
        # Calculate direction vector
        direction = np.array(end) - np.array(start)
        length = np.linalg.norm(direction)

        if length < 1e-9: # Use a small epsilon for zero length check
            return []

        # Normalize direction
        direction_normalized = direction / length

        # Create arrow shaft (e.g., 80% of total length)
        shaft_length_ratio = 0.85
        shaft_end = np.array(start) + direction_normalized * length * shaft_length_ratio

        shaft_artist = self.ax.plot([start[0], shaft_end[0]],
                                    [start[1], shaft_end[1]],
                                    [start[2], shaft_end[2]],
                                    color=color, linewidth=width*6, alpha=alpha, solid_capstyle='round')[0] # Adjusted width scaling

        # Create arrowhead (simplified: a thicker line from shaft_end to end)
        # For a more proper arrowhead, one might use quiver3d or draw a cone/polygon.
        # This implementation uses a thicker line for the head.
        head_artist = self.ax.plot([shaft_end[0], end[0]],
                                   [shaft_end[1], end[1]],
                                   [shaft_end[2], end[2]],
                                   color=color, linewidth=width*12, alpha=alpha, solid_capstyle='round')[0] # Adjusted width scaling

        return [shaft_artist, head_artist]

    def update(self, *args):
        """
        Update method with three possible signatures:
        1. update(theta, phi, r) - Single direction arrow with specified magnitude
        2. update(theta, phi) - Single direction arrow with default magnitude
        3. update(theta_array, phi_array, r_array) - Multiple arrows with directions and magnitudes
        """
        if len(args) == 2:
            # Single theta, phi update with default magnitude
            theta, phi = args
            self._update_single(theta, phi, self.max_arrow_length) # Use a default length like max_arrow_length
        elif len(args) == 3:
            theta, phi, r = args
            if np.isscalar(theta) and np.isscalar(phi) and np.isscalar(r):
                # Single arrow with specified magnitude
                self._update_single(theta, phi, r)
            else:
                # Array update
                self._update_array(theta, phi, r)
        else:
            raise ValueError("update() takes 2 arguments (theta, phi) or 3 arguments (theta, phi, r) or (theta_array, phi_array, r_array)")

    def _update_single(self, theta, phi, r):
        """
        Update display with a single arrow

        Args:
            theta: Azimuthal angle (radians)
            phi: Polar angle (radians)
            r: Arrow magnitude/length
        """
        self.theta = theta
        self.phi = phi
        self.r = r
        self.theta_array = None
        self.phi_array = None
        self.r_array = None

        # Clear existing arrows and annotations
        self.clear_arrows()

        # Adjust display limits to fit the arrow
        self.adjust_display_limits(abs(r))

        # Create single arrow with specified magnitude
        start = (0, 0, 0)
        end = self.spherical_to_cartesian(theta, phi, r)

        # Color based on magnitude
        max_ref = max(abs(r), 1.0) # Reference for color intensity
        color_intensity = min(1.0, abs(r) / max_ref if max_ref != 0 else 1.0)

        if r >= 0:
            color = (1.0, 1.0 - color_intensity * 0.7, 0)  # Yellow (low mag) to Red (high mag)
        else:
            color = (0, 0.3 + color_intensity * 0.7, 1.0)  # Light blue (low mag) to Blue (high mag)

        # Arrow width proportional to magnitude for visibility
        width = 0.02 + 0.05 * min(color_intensity, 1.0) # Base width factor for create_3d_arrow

        arrow_artists = self.create_3d_arrow(
            start, end, color=color, alpha=0.9, width=width
        )
        if arrow_artists:
            self.interpolated_arrow_parts.extend(arrow_artists)

        # Add text annotation
        text_offset_factor = abs(r) * 0.2 + 0.3 # Dynamic offset based on arrow length
        text_pos_r = abs(r) + text_offset_factor
        text_pos = self.spherical_to_cartesian(theta, phi, text_pos_r)

        text_content = f'θ: {np.degrees(theta):.1f}°\nφ: {np.degrees(phi):.1f}°\nr: {r:.2f}'
        text_artist = self.ax.text(text_pos[0], text_pos[1], text_pos[2],
                                   text_content, fontsize=9, color='black', # Changed to black for visibility
                                   ha='center', va='center', bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", lw=0.5, alpha=0.7))
        self.text_annotations.append(text_artist)

        # Refresh display
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

    def _update_array(self, theta_array, phi_array, r_array):
        """
        Update display with arrays of directions and magnitudes

        Args:
            theta_array: Array of azimuthal angles (radians)
            phi_array: Array of polar angles (radians)
            r_array: Array of magnitudes/lengths corresponding to each direction
        """
        self.theta = None
        self.phi = None
        self.r = None
        self.theta_array = np.array(theta_array)
        self.phi_array = np.array(phi_array)
        self.r_array = np.array(r_array)

        # Clear existing arrows and colorbar
        self.clear_arrows() # This will also clear the colorbar via its new logic if it was managed by clear_arrows.
                            # However, explicit removal before recreation is safer.
        if hasattr(self, 'colorbar') and self.colorbar:
            try:
                self.colorbar.remove()
            except: pass
            self.colorbar = None


        # Ensure all arrays have the same shape
        if self.theta_array.shape != self.phi_array.shape or self.theta_array.shape != self.r_array.shape:
            raise ValueError("theta_array, phi_array, and r_array must have the same shape")

        # Flatten arrays for easier processing
        theta_flat = self.theta_array.flatten()
        phi_flat = self.phi_array.flatten()
        r_flat = self.r_array.flatten()

        if r_flat.size == 0: # No data to plot
            self.fig.canvas.draw_idle()
            self.fig.canvas.flush_events()
            return

        # Find max absolute value for color normalization and display limits
        max_abs_r = np.max(np.abs(r_flat)) if np.any(r_flat) else 1.0
        self.adjust_display_limits(max_abs_r)


        # Create colormap for magnitudes
        norm = Normalize(vmin=-max_abs_r, vmax=max_abs_r) # Symmetrical normalization around 0
        colormap = cm.RdYlBu_r  # Red-Yellow-Blue colormap (Red positive, Blue negative)

        # Create arrows for each theta/phi/r combination
        for i, (theta, phi, r_val) in enumerate(zip(theta_flat, phi_flat, r_flat)):
            if abs(r_val) < 1e-9:  # Skip near-zero magnitudes
                continue

            # Calculate end position using the specified magnitude
            start = (0, 0, 0)
            end = self.spherical_to_cartesian(theta, phi, r_val)

            # Get color based on magnitude
            color = colormap(norm(r_val))

            # Create arrow with width proportional to magnitude
            width_factor = 0.015 + 0.04 * (abs(r_val) / max_abs_r if max_abs_r != 0 else 1.0)
            arrow_artists = self.create_3d_arrow(
                start, end, color=color, alpha=0.8, width=width_factor
            )

            if arrow_artists:
                self.arrows.extend(arrow_artists)

        # Add colorbar only if arrows were created
        if self.arrows:
            scalar_map = cm.ScalarMappable(norm=norm, cmap=colormap)
            scalar_map.set_array([]) # Important for the mappable to work without data array
            self.colorbar = self.fig.colorbar(scalar_map, ax=self.ax, shrink=0.6, aspect=15, pad=0.1)
            self.colorbar.set_label('Magnitude (r)', rotation=270, labelpad=20)

        # Refresh display
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

    def is_open(self):
        """Check if the display window is still open"""
        return plt.fignum_exists(self.fig.number)

    def close(self):
        """Close the display"""
        plt.ioff() # Turn off interactive mode before closing
        plt.close(self.fig)