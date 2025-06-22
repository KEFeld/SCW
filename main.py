from gradio_spiral import launch_gradio_app

def main():
    """Main function to run the Gradio spiral visualization."""
    print("Starting Spiral Visualization with Gradio...")
    launch_gradio_app(share=False, show_error=True)

if __name__ == "__main__":
    main()
