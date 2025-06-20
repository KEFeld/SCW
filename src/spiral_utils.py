import numpy as np
import matplotlib.pyplot as plt
from spiral_constants import *

def generate_spiral_points():
    """Generate points for the main spiral."""
    t = np.linspace(0, MAX_ROTATION_ANGLE, NUM_POINTS)
    r = BASE_RADIUS * (1 - TAPER_FACTOR * np.sin(np.pi * t / (2 * MAX_ROTATION_ANGLE)))
    x = r * np.cos(t)
    y = r * np.sin(t)
    z = HEIGHT * (t / MAX_ROTATION_ANGLE)
    return x, y, z

def generate_cap_points():
    """Generate points for top and bottom caps."""
    theta = np.linspace(0, 2*np.pi, CIRCLE_POINTS)
    r_grid, theta_grid = np.meshgrid(np.linspace(0, BASE_RADIUS, CAP_RADIUS_POINTS), theta)
    
    # Bottom cap
    bottom_x = r_grid * np.cos(theta_grid)
    bottom_y = r_grid * np.sin(theta_grid)
    bottom_z = np.zeros_like(bottom_x)
    
    # Top cap
    top_r = BASE_RADIUS * (1 - TAPER_FACTOR)
    r_grid_top, theta_grid_top = np.meshgrid(np.linspace(0, top_r, CAP_RADIUS_POINTS), theta)
    top_x = r_grid_top * np.cos(theta_grid_top)
    top_y = r_grid_top * np.sin(theta_grid_top)
    top_z = np.full_like(top_x, HEIGHT)
    
    return bottom_x, bottom_y, bottom_z, top_x, top_y, top_z

def generate_cylinder_points():
    """Generate points for the inner cylinder surface."""
    cylinder_z = np.linspace(0, HEIGHT, CYLINDER_Z_POINTS)
    cylinder_theta = np.linspace(0, 2*np.pi, CYLINDER_THETA_POINTS)
    cylinder_z_grid, cylinder_theta_grid = np.meshgrid(cylinder_z, cylinder_theta)
    cylinder_x = CYLINDER_RADIUS * np.cos(cylinder_theta_grid)
    cylinder_y = CYLINDER_RADIUS * np.sin(cylinder_theta_grid)
    return cylinder_x, cylinder_y, cylinder_z_grid

def draw_radial_lines(ax):
    """Draw radial lines from cylinder to spiral."""
    for i in range(RADIAL_LINES_COUNT):
        t_point = i * (MAX_ROTATION_ANGLE / RADIAL_LINES_COUNT)
        
        # Get point on spiral
        spiral_r = BASE_RADIUS * (1 - TAPER_FACTOR * np.sin(np.pi * t_point / (2 * MAX_ROTATION_ANGLE)))
        spiral_x = spiral_r * np.cos(t_point)
        spiral_y = spiral_r * np.sin(t_point)
        spiral_z = HEIGHT * (t_point / MAX_ROTATION_ANGLE)
        
        # Get point on cylinder
        cylinder_x_point = CYLINDER_RADIUS * np.cos(t_point)
        cylinder_y_point = CYLINDER_RADIUS * np.sin(t_point)
        
        # Draw horizontal line (commented out in original)
        # ax.plot3D([cylinder_x_point, spiral_x], 
        #           [cylinder_y_point, spiral_y], 
        #           [spiral_z, spiral_z], 
        #           color='orange', alpha=0.5, linewidth=1)  # type: ignore

def draw_guiding_lines(ax):
    """Draw vertical guiding lines along the cone periphery."""
    guiding_angles = [i * (2*np.pi/NUM_LINES) for i in range(NUM_LINES)]
    
    for angle in guiding_angles:
        # Create points along the height of the cone
        z_vertical = np.linspace(0, HEIGHT, VERTICAL_POINTS)
        
        # Calculate radius at each height using the same curve as the spiral
        r_vertical = BASE_RADIUS * (1 - TAPER_FACTOR * np.sin(np.pi * z_vertical / (2 * HEIGHT)))
        
        # Calculate x, y coordinates at each height
        x_vertical = r_vertical * np.cos(angle)
        y_vertical = r_vertical * np.sin(angle)
        
        # Draw the guiding line
        ax.plot3D(x_vertical, y_vertical, z_vertical,
                  color=GUIDING_LINE_COLOR, alpha=GUIDING_LINE_ALPHA, 
                  linewidth=GUIDING_LINE_WIDTH)  # type: ignore

def draw_spiral_radial_guidelines(ax):
    """Draw radial guidelines connecting cylinder spiral to outer spiral."""
    # Calculate angles for full spiral rotation
    angles = [i * (MAX_ROTATION_ANGLE/NUM_LINES) for i in range(NUM_LINES)]
    
    # For each angle, get points on cylinder and outer spiral
    for angle in angles:
        # Point on cylinder
        cylinder_x = CYLINDER_RADIUS * np.cos(angle)
        cylinder_y = CYLINDER_RADIUS * np.sin(angle)
        
        # Calculate height based on angle relative to MAX_ROTATION_ANGLE
        z_height = HEIGHT * (angle / MAX_ROTATION_ANGLE)
        
        # Get corresponding point on outer spiral at same angle
        spiral_r = BASE_RADIUS * (1 - TAPER_FACTOR * np.sin(np.pi * angle / (2 * MAX_ROTATION_ANGLE)))
        spiral_x = spiral_r * np.cos(angle)
        spiral_y = spiral_r * np.sin(angle)
        
        # Draw radial line connecting cylinder to spiral
        ax.plot3D([cylinder_x, spiral_x],
                  [cylinder_y, spiral_y], 
                  [z_height, z_height],
                  color=RADIAL_GUIDE_COLOR, alpha=RADIAL_GUIDE_ALPHA, 
                  linewidth=RADIAL_GUIDE_LINE_WIDTH)  # type: ignore

def style_plot(ax, fig):
    """Apply consistent styling to the plot."""
    ax.set_facecolor(BACKGROUND_COLOR)
    fig.patch.set_facecolor(BACKGROUND_COLOR)
    ax.grid(False)
    ax.axis('off')
    ax.set_xlim((-BASE_RADIUS, BASE_RADIUS))
    ax.set_ylim((-BASE_RADIUS, BASE_RADIUS))
    ax.set_zlim((0, HEIGHT))  # type: ignore
    plt.tight_layout()

def draw_complete_spiral(ax):
    """Draw all spiral elements on the given axes."""
    # Generate and plot spiral
    x, y, z = generate_spiral_points()
    ax.plot3D(x, y, z, color=SPIRAL_COLOR, linewidth=SPIRAL_LINE_WIDTH)  # type: ignore
    
    # Generate and plot caps
    bottom_x, bottom_y, bottom_z, top_x, top_y, top_z = generate_cap_points()
    ax.plot_surface(bottom_x, bottom_y, bottom_z, color=CAP_COLOR, alpha=CAP_ALPHA)  # type: ignore
    ax.plot_surface(top_x, top_y, top_z, color=CAP_COLOR, alpha=CAP_ALPHA)  # type: ignore
    
    # Generate and plot cylinder
    cylinder_x, cylinder_y, cylinder_z_grid = generate_cylinder_points()
    ax.plot_surface(cylinder_x, cylinder_y, cylinder_z_grid, color=CYLINDER_COLOR, alpha=CYLINDER_ALPHA)  # type: ignore
    
    # Draw all connecting lines
    draw_radial_lines(ax)
    draw_guiding_lines(ax)
    draw_spiral_radial_guidelines(ax) 