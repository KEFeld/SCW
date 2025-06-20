import numpy as np

# Geometric Parameters
BASE_RADIUS = 5              # Base radius
HEIGHT = 10                  # Height of the spiral
MAX_ROTATION_ANGLE = 8 * np.pi  # Total angle range (number of turns)
NUM_POINTS = 2000            # Resolution
CYLINDER_RADIUS = BASE_RADIUS * 0.2  # Radius of inner cylinder
NUM_LINES = 8                # Number of guiding lines

# Visual Parameters
TAPER_FACTOR = 0.7           # Controls the tapering of the cone
CIRCLE_POINTS = 100          # Points for cap circles
CYLINDER_Z_POINTS = 100      # Z-axis resolution for cylinder
CYLINDER_THETA_POINTS = 50   # Angular resolution for cylinder
CAP_RADIUS_POINTS = 20       # Radial resolution for caps
VERTICAL_POINTS = 50         # Points for vertical guiding lines
RADIAL_LINES_COUNT = 16      # Number of radial lines

# Colors and Styling
SPIRAL_COLOR = 'orange'
CAP_COLOR = 'yellow'
CYLINDER_COLOR = 'orange'
GUIDING_LINE_COLOR = 'red'
RADIAL_GUIDE_COLOR = 'cyan'
BACKGROUND_COLOR = 'black'

# Alpha Values
SPIRAL_ALPHA = 1.0
CAP_ALPHA = 0.3
CYLINDER_ALPHA = 0.2
GUIDING_LINE_ALPHA = 0.6
RADIAL_GUIDE_ALPHA = 0.4

# Line Widths
SPIRAL_LINE_WIDTH = 2
GUIDING_LINE_WIDTH = 2
RADIAL_GUIDE_LINE_WIDTH = 1 