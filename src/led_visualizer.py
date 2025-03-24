from vispy import app, scene
from .led_matrix import LEDMatrix


class LEDVisualizer:
    def __init__(self):
        # Create the LED matrix
        self.matrix = LEDMatrix()
        
        # Visualization setup
        self.canvas = scene.SceneCanvas(keys='interactive', size=(800, 400), show=True)
        self.view = self.canvas.central_widget.add_view()
        self.view.camera = scene.cameras.TurntableCamera(up='z', distance=10)
        
        # Create LED scatter plot
        self.grid = scene.visuals.Markers()
        self.grid.set_data(self.matrix.positions, edge_color=None, face_color=(1, 1, 1, 1), size=10)
        self.view.add(self.grid)
        
        # Setup event handlers
        self.timer = app.Timer(interval=0.05, connect=self.update, start=True)
        self.canvas.events.key_press.connect(self.on_key)
    
    def update(self, ev):
        colors = self.matrix.update()
        self.grid.set_data(self.matrix.positions, face_color=colors, size=10)
    
    def on_key(self, event):
        self.matrix.handle_key(event.text)


__all__ = ['LEDVisualizer']