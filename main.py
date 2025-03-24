from vispy import app
from src.led_visualizer import LEDVisualizer

if __name__ == '__main__':
    visualizer = LEDVisualizer()
    app.run()