import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from spiral_constants import *
from spiral_utils import draw_complete_spiral, style_plot
from spiral_widgets import create_interactive_plot, create_static_plot

def main():
    """Main function to run the spiral visualization."""
    # Choose between interactive and static plot
    print("Spiral Visualization")
    print("1. Interactive plot (with sliders)")
    print("2. Static plot")
    
    choice = input("Enter your choice (1 or 2): ").strip()
    
    if choice == "1":
        print("Creating interactive plot...")
        create_interactive_plot()
    elif choice == "2":
        print("Creating static plot...")
        create_static_plot()
    else:
        print("Invalid choice. Creating interactive plot by default...")
        create_interactive_plot()

if __name__ == "__main__":
    main()
