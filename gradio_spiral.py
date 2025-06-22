import gradio as gr
import numpy as np
import matplotlib.pyplot as plt
from spiral_utils import draw_complete_spiral, style_plot
from spiral_constants import *

def create_spiral_plot(base_radius, height, turns, cylinder_ratio, taper_factor, num_lines):
    """Create a spiral plot with the given parameters for Gradio."""
    # Calculate derived parameters
    max_rotation_angle = turns * np.pi
    cylinder_radius = base_radius * cylinder_ratio
    
    # Create the plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Draw the complete spiral with current parameters
    draw_complete_spiral(ax, base_radius, height, max_rotation_angle, cylinder_radius, taper_factor, num_lines)
    
    # Style the plot
    style_plot(ax, fig, base_radius, height)
    
    # Add a title to show current values for debugging
    ax.set_title(f"Base Radius: {base_radius}, Height: {height}, Turns: {turns}", 
                color='white', fontsize=12, pad=-30)
    
    return fig

def create_gradio_interface():
    """Create the Gradio interface for spiral visualization."""
    with gr.Blocks(title="Spiral Visualization", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🌀 Interactive Spiral Visualization")
        gr.Markdown("Adjust the parameters below to create different spiral patterns.")
        
        with gr.Row():
            with gr.Column(scale=1):
                # Parameter controls
                gr.Markdown("### Parameters")
                
                base_radius = gr.Slider(
                    minimum=1, maximum=10, value=5, step=0.1,
                    label="Base Radius", info="Radius at the base of the spiral"
                )
                
                height = gr.Slider(
                    minimum=5, maximum=20, value=8, step=0.5,
                    label="Height", info="Height of the spiral"
                )
                
                turns = gr.Slider(
                    minimum=1, maximum=16, value=8, step=0.5,
                    label="Number of Turns", info="How many complete rotations"
                )
                
                cylinder_ratio = gr.Slider(
                    minimum=0.1, maximum=0.5, value=0.2, step=0.01,
                    label="Cylinder Ratio", info="Ratio of inner cylinder to base radius"
                )
                
                taper_factor = gr.Slider(
                    minimum=0.1, maximum=0.9, value=0.7, step=0.01,
                    label="Taper Factor", info="How much the spiral tapers"
                )
                
                num_lines = gr.Slider(
                    minimum=4, maximum=16, value=8, step=1,
                    label="Number of Lines", info="Number of guiding lines"
                )
                
                # Reset button
                reset_btn = gr.Button("Reset to Defaults", variant="secondary")
            
            with gr.Column(scale=2):
                # Plot output
                gr.Markdown("### 3D Spiral Visualization")
                plot_output = gr.Plot(label="Spiral Plot")
        
        # Connect inputs to output - FIXED: Connect each input to trigger the plot update
        inputs = [base_radius, height, turns, cylinder_ratio, taper_factor, num_lines]
        
        # Connect each input to update the plot
        for input_component in inputs:
            input_component.change(
                fn=create_spiral_plot,
                inputs=inputs,
                outputs=plot_output
            )
        
        # Reset functionality
        def reset_values():
            return [5, 8, 8, 0.2, 0.7, 8]
        
        reset_btn.click(
            fn=reset_values,
            outputs=inputs
        )
        
        # Initial plot
        demo.load(
            fn=lambda: create_spiral_plot(5, 8, 8, 0.2, 0.7, 8),
            outputs=plot_output
        )
    
    return demo

def launch_gradio_app(share=False, show_error=True):
    """Launch the Gradio app."""
    demo = create_gradio_interface()
    demo.launch(share=share, show_error=show_error) 