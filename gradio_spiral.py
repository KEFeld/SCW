import gradio as gr
import numpy as np
import matplotlib.pyplot as plt
from spiral_utils import draw_complete_spiral, style_plot
from spiral_constants import *

# Global variables to store the current figure and axes
current_fig = None
current_ax = None

def create_spiral_plot(base_radius, height, turns, cylinder_ratio, taper_factor, num_lines, 
                      elevation=20, azimuth=45):
    """Create a spiral plot with the given parameters for Gradio."""
    global current_fig, current_ax
    
    # Calculate derived parameters
    max_rotation_angle = turns * np.pi
    cylinder_radius = base_radius * cylinder_ratio
    
    # Create the plot
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    
    # Draw the complete spiral with current parameters
    draw_complete_spiral(ax, base_radius, height, max_rotation_angle, cylinder_radius, taper_factor, num_lines)
    
    # Style the plot
    style_plot(ax, fig, base_radius, height)
    
    # Set the view angle
    ax.view_init(elev=elevation, azim=azimuth)
    
    # Store the current figure and axes
    current_fig = fig
    current_ax = ax
    
    # Add a title to show current values for debugging
    ax.set_title(f"Base Radius: {base_radius}, Height: {height}, Turns: {turns}", 
                color='white', fontsize=10, pad=-30)
    
    return fig

def update_view_only(elevation, azimuth):
    """Update only the view angle without redrawing the geometry."""
    global current_fig, current_ax
    
    if current_ax is not None:
        # Just update the view angle
        current_ax.view_init(elev=elevation, azim=azimuth)
        return current_fig
    else:
        # Fallback to full redraw if no current plot exists
        return create_spiral_plot(5, 8, 8, 0.2, 0.7, 8, elevation, azimuth)

def create_gradio_interface():
    """Create the Gradio interface for spiral visualization."""
    with gr.Blocks(title="Spiral Visualization", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# Interactive Spiral Visualization")
        gr.Markdown("Adjust the parameters below to create different spiral patterns.")
        
        with gr.Row():
            with gr.Column(scale=1):
                # Parameter controls
                gr.Markdown("### Spiral Parameters")
                
                base_radius = gr.Slider(
                    minimum=1, maximum=10, value=5, step=0.2,
                    label="Base Radius", info="Radius at the base of the spiral"
                )
                
                height = gr.Slider(
                    minimum=5, maximum=20, value=8, step=1.0,
                    label="Height", info="Height of the spiral"
                )
                
                turns = gr.Slider(
                    minimum=1, maximum=16, value=8, step=1.0,
                    label="Number of Turns", info="How many complete rotations"
                )
                
                cylinder_ratio = gr.Slider(
                    minimum=0.1, maximum=0.5, value=0.2, step=0.02,
                    label="Cylinder Ratio", info="Ratio of inner cylinder to base radius"
                )
                
                taper_factor = gr.Slider(
                    minimum=0.1, maximum=0.9, value=0.7, step=0.05,
                    label="Taper Factor", info="How much the spiral tapers"
                )
                
                num_lines = gr.Slider(
                    minimum=4, maximum=16, value=8, step=1,
                    label="Number of Lines", info="Number of guiding lines"
                )
                
                gr.Markdown("### View Controls")
                
                elevation = gr.Slider(
                    minimum=-90, maximum=90, value=20, step=5,
                    label="Elevation", info="Vertical viewing angle (-90 to 90 degrees)"
                )
                
                azimuth = gr.Slider(
                    minimum=0, maximum=360, value=45, step=10,
                    label="Azimuth", info="Horizontal viewing angle (0 to 360 degrees)"
                )
                
                # Reset buttons
                with gr.Row():
                    reset_spiral_btn = gr.Button("Reset Spiral", variant="secondary")
                    reset_view_btn = gr.Button("Reset View", variant="secondary")
            
            with gr.Column(scale=2):
                # Plot output
                gr.Markdown("### 3D Spiral Visualization")
                plot_output = gr.Plot(label="Spiral Plot")
        
        # Connect spiral parameters to full redraw
        spiral_inputs = [base_radius, height, turns, cylinder_ratio, taper_factor, num_lines]
        for input_component in spiral_inputs:
            input_component.change(
                fn=create_spiral_plot,
                inputs=spiral_inputs + [elevation, azimuth],
                outputs=plot_output,
                show_progress=False
            )
        
        # Connect view parameters to view-only update
        view_inputs = [elevation, azimuth]
        for input_component in view_inputs:
            input_component.change(
                fn=update_view_only,
                inputs=view_inputs,
                outputs=plot_output,
                show_progress=False
            )
        
        # Reset functionality
        def reset_spiral_values():
            return [5, 8, 8, 0.2, 0.7, 8]
        
        def reset_view_values():
            return [20, 45]
        
        reset_spiral_btn.click(
            fn=reset_spiral_values,
            outputs=spiral_inputs
        )
        
        reset_view_btn.click(
            fn=reset_view_values,
            outputs=view_inputs
        )
        
        # Initial plot
        demo.load(
            fn=lambda: create_spiral_plot(5, 8, 8, 0.2, 0.7, 8, 20, 45),
            outputs=plot_output
        )
    
    return demo

def launch_gradio_app(share=False, show_error=True):
    """Launch the Gradio app."""
    demo = create_gradio_interface()
    demo.launch(share=share, show_error=show_error) 