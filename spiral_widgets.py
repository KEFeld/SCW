import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import TextBox, Button
from spiral_constants import *
from spiral_utils import draw_complete_spiral, style_plot
import time

COLOR_INPUT_TEXT = 'white';
COLOR_TEXT_LABELS ='green';

# Create text input boxes - positioned higher and better organized
    # Left column - labels first, then input boxes below    
def create_label_and_input(x, y, label_text):
    """Helper function to create a label and input axes pair."""
    # Create label
    label_ax = plt.axes([x, y + 0.03, 0.12, 0.02])
    label_ax.text(0.5, 0.5, label_text, color=COLOR_TEXT_LABELS, 
                    ha='center', va='center', transform=label_ax.transAxes)
    label_ax.set_xticks([])
    label_ax.set_yticks([])
    
    # Create input box axes
    input_ax = plt.axes([x, y, 0.12, 0.03])
    
    return input_ax

    
def create_interactive_plot():
    """Create an interactive plot with text inputs for parameter control."""
    # Create figure with extra space for text boxes
    fig = plt.figure(figsize=(12, 10))
    
    # Create 3D subplot with adjusted position - move it up to make room for text boxes
    ax = fig.add_subplot(111, projection='3d')
    plt.subplots_adjust(bottom=0.4, top=0.95)  # Make room for inputs at bottom
    
    # Track last update time for debouncing
    last_update_time = 0
    update_delay = 0.5  # seconds
    
    def update(text):
        """Update function called when text inputs change."""
        nonlocal last_update_time
        
        # Debounce updates
        current_time = time.time()
        if current_time - last_update_time < update_delay:
            return
        
        try:
            # Get current values from text boxes
            base_radius = float(txt_base_radius.text)
            height = float(txt_height.text)
            max_angle = float(txt_max_angle.text) * np.pi
            cylinder_ratio = float(txt_cylinder_ratio.text)
            taper_factor = float(txt_taper_factor.text)
            num_lines = int(txt_num_lines.text)
            
            # Validate inputs
            if not (1 <= base_radius <= 10): return
            if not (5 <= height <= 20): return
            if not (1 <= max_angle/np.pi <= 16): return
            if not (0.1 <= cylinder_ratio <= 0.5): return
            if not (0.1 <= taper_factor <= 0.9): return
            if not (4 <= num_lines <= 16): return
            
            # Update last update time
            last_update_time = current_time
            
            # Clear the plot
            ax.clear()
            
            # Update global parameters
            global BASE_RADIUS, HEIGHT, MAX_ROTATION_ANGLE, CYLINDER_RADIUS, TAPER_FACTOR, NUM_LINES
            BASE_RADIUS = base_radius
            HEIGHT = height
            MAX_ROTATION_ANGLE = max_angle
            CYLINDER_RADIUS = base_radius * cylinder_ratio
            TAPER_FACTOR = taper_factor
            NUM_LINES = num_lines
            
            # Regenerate and plot everything
            draw_complete_spiral(ax)
            
            # Reapply styling
            style_plot(ax, fig)
            
            # Redraw
            fig.canvas.draw_idle()
            
        except ValueError:
            pass  # Ignore invalid number inputs
    
    # Left column inputs
    ax_base_radius = create_label_and_input(0.1, 0.25, 'Base Radius')
    ax_height = create_label_and_input(0.1, 0.20, 'Height')
    ax_max_angle = create_label_and_input(0.1, 0.15, 'Turns')
    
    # Right column inputs
    ax_cylinder_ratio = create_label_and_input(0.3, 0.25, 'Cylinder Ratio')
    ax_taper_factor = create_label_and_input(0.3, 0.20, 'Taper Factor')
    ax_num_lines = create_label_and_input(0.3, 0.15, 'Num Lines')
    
    txt_base_radius = TextBox(ax_base_radius, 'label', initial=str(BASE_RADIUS), color=COLOR_INPUT_TEXT)
    txt_height = TextBox(ax_height, '', initial=str(HEIGHT), color=COLOR_INPUT_TEXT)
    txt_max_angle = TextBox(ax_max_angle, '', initial=str(MAX_ROTATION_ANGLE/np.pi), color=COLOR_INPUT_TEXT)
    txt_cylinder_ratio = TextBox(ax_cylinder_ratio, '', initial=str(CYLINDER_RADIUS/BASE_RADIUS), color=COLOR_INPUT_TEXT)
    txt_taper_factor = TextBox(ax_taper_factor, '', initial=str(TAPER_FACTOR), color=COLOR_INPUT_TEXT)
    txt_num_lines = TextBox(ax_num_lines, '', initial=str(NUM_LINES), color=COLOR_INPUT_TEXT)
    
    # Connect text boxes to update function
    txt_base_radius.on_submit(update)
    txt_height.on_submit(update)
    txt_max_angle.on_submit(update)
    txt_cylinder_ratio.on_submit(update)
    txt_taper_factor.on_submit(update)
    txt_num_lines.on_submit(update)
    
    # Also connect to text change events for more responsive updates
    txt_base_radius.on_text_change(update)
    txt_height.on_text_change(update)
    txt_max_angle.on_text_change(update)
    txt_cylinder_ratio.on_text_change(update)
    txt_taper_factor.on_text_change(update)
    txt_num_lines.on_text_change(update)
    
    # Initial plot
    update(None)
    
    plt.show()

def create_static_plot():
    """Create a static plot without interactive controls."""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    draw_complete_spiral(ax)
    style_plot(ax, fig)
    plt.show() 