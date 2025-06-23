import numpy as np
import matplotlib.pyplot as plt
from spiral_constants import *

def generate_spiral_points(base_radius, height, max_rotation_angle, taper_factor, num_points=NUM_POINTS):
    """Generate points for the main spiral."""
    t = np.linspace(0, max_rotation_angle, num_points)
    r = base_radius * (1 - taper_factor * np.sin(np.pi * t / (2 * max_rotation_angle)))
    x = r * np.cos(t)
    y = r * np.sin(t)
    z = height * (t / max_rotation_angle)
    return x, y, z

def generate_cap_points(base_radius, height, taper_factor, circle_points=CIRCLE_POINTS, cap_radius_points=CAP_RADIUS_POINTS):
    """Generate points for top and bottom caps."""
    theta = np.linspace(0, 2*np.pi, circle_points)
    r_grid, theta_grid = np.meshgrid(np.linspace(0, base_radius, cap_radius_points), theta)
    
    # Bottom cap
    bottom_x = r_grid * np.cos(theta_grid)
    bottom_y = r_grid * np.sin(theta_grid)
    bottom_z = np.zeros_like(bottom_x)
    
    # Top cap
    top_r = base_radius * (1 - taper_factor)
    r_grid_top, theta_grid_top = np.meshgrid(np.linspace(0, top_r, cap_radius_points), theta)
    top_x = r_grid_top * np.cos(theta_grid_top)
    top_y = r_grid_top * np.sin(theta_grid_top)
    top_z = np.full_like(top_x, height)
    
    return bottom_x, bottom_y, bottom_z, top_x, top_y, top_z

def generate_cylinder_points(cylinder_radius, height, cylinder_z_points=CYLINDER_Z_POINTS, cylinder_theta_points=CYLINDER_THETA_POINTS):
    """Generate points for the inner cylinder surface."""
    cylinder_z = np.linspace(0, height, cylinder_z_points)
    cylinder_theta = np.linspace(0, 2*np.pi, cylinder_theta_points)
    cylinder_z_grid, cylinder_theta_grid = np.meshgrid(cylinder_z, cylinder_theta)
    cylinder_x = cylinder_radius * np.cos(cylinder_theta_grid)
    cylinder_y = cylinder_radius * np.sin(cylinder_theta_grid)
    return cylinder_x, cylinder_y, cylinder_z_grid

def draw_guiding_lines(ax, base_radius, height, taper_factor, num_lines, vertical_points=VERTICAL_POINTS):
    """Draw vertical guiding lines along the cone periphery."""
    guiding_angles = [i * (2*np.pi/num_lines) for i in range(num_lines)]
    
    for angle in guiding_angles:
        # Create points along the height of the cone
        z_vertical = np.linspace(0, height, vertical_points)
        
        # Calculate radius at each height using the same curve as the spiral
        r_vertical = base_radius * (1 - taper_factor * np.sin(np.pi * z_vertical / (2 * height)))
        
        # Calculate x, y coordinates at each height
        x_vertical = r_vertical * np.cos(angle)
        y_vertical = r_vertical * np.sin(angle)
        
        # Draw the guiding line
        ax.plot3D(x_vertical, y_vertical, z_vertical,
                  color=GUIDING_LINE_COLOR, alpha=GUIDING_LINE_ALPHA, 
                  linewidth=GUIDING_LINE_WIDTH)

def draw_spiral_radial_guidelines(ax, base_radius, height, cylinder_radius, max_rotation_angle, taper_factor, num_lines):
    """Draw radial guidelines connecting cylinder spiral to outer spiral."""
    # Calculate angles for full spiral rotation
    num_spiral_guidelines = 4* num_lines;
    angles = [i * (max_rotation_angle/num_spiral_guidelines) for i in range(num_spiral_guidelines)]
    
    # For each angle, get points on cylinder and outer spiral
    for angle in angles:
        # Point on cylinder
        cylinder_x = cylinder_radius * np.cos(angle)
        cylinder_y = cylinder_radius * np.sin(angle)
        
        # Calculate height based on angle relative to max_rotation_angle
        z_height = height * (angle / max_rotation_angle)
        
        # Get corresponding point on outer spiral at same angle
        spiral_r = base_radius * (1 - taper_factor * np.sin(np.pi * angle / (2 * max_rotation_angle)))
        spiral_x = spiral_r * np.cos(angle)
        spiral_y = spiral_r * np.sin(angle)
        
        # Draw radial line connecting cylinder to spiral
        ax.plot3D([cylinder_x, spiral_x],
                  [cylinder_y, spiral_y], 
                  [z_height, z_height],
                  color=RADIAL_GUIDE_COLOR, alpha=RADIAL_GUIDE_ALPHA, 
                  linewidth=RADIAL_GUIDE_LINE_WIDTH)

def style_plot(ax, fig, base_radius, height):
    """Apply consistent styling to the plot."""
    ax.set_facecolor(BACKGROUND_COLOR)
    fig.patch.set_facecolor(BACKGROUND_COLOR)
    ax.grid(False)
    ax.axis('off')
    
    # Use fixed limits instead of scaling with base_radius
    # This will allow you to see the actual size changes
    max_radius = 10  # Fixed maximum radius for view
    ax.set_xlim((-max_radius, max_radius))
    ax.set_ylim((-max_radius, max_radius))
    ax.set_zlim((0, height))
    plt.tight_layout()

def draw_complete_spiral(ax, base_radius, height, max_rotation_angle, cylinder_radius, taper_factor, num_lines):
    """Draw all spiral elements on the given axes with the specified parameters."""
    # Generate and plot spiral
    x, y, z = generate_spiral_points(base_radius, height, max_rotation_angle, taper_factor)
    ax.plot3D(x, y, z, color=SPIRAL_COLOR, linewidth=SPIRAL_LINE_WIDTH)
    
    # Generate and plot caps
    bottom_x, bottom_y, bottom_z, top_x, top_y, top_z = generate_cap_points(base_radius, height, taper_factor)
    ax.plot_surface(bottom_x, bottom_y, bottom_z, color=CAP_COLOR, alpha=CAP_ALPHA)
    ax.plot_surface(top_x, top_y, top_z, color=CAP_COLOR, alpha=CAP_ALPHA)
    
    # Generate and plot cylinder
    cylinder_x, cylinder_y, cylinder_z_grid = generate_cylinder_points(cylinder_radius, height)
    ax.plot_surface(cylinder_x, cylinder_y, cylinder_z_grid, color=CYLINDER_COLOR, alpha=CYLINDER_ALPHA)
    
    # Draw all connecting lines
    draw_guiding_lines(ax, base_radius, height, taper_factor, num_lines)
    draw_spiral_radial_guidelines(ax, base_radius, height, cylinder_radius, max_rotation_angle, taper_factor, num_lines) 