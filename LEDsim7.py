import math
import numpy as np
from vispy import app, scene

from src.utils import perlin_noise


# --- LED Visualization Setup ---

# Create a VisPy canvas
canvas = scene.SceneCanvas(keys='interactive', size=(800, 400), show=True)
view = canvas.central_widget.add_view()
view.camera = scene.cameras.TurntableCamera(up='z', distance=10)

# Initialize LED positions
led_positions = np.empty((0, 3))
num_leds = 0

# User-controlled swirl speed and pattern mode
swirl_speed = 0.03
pattern_mode = 1  # Default pattern

def add_led_line(start, end, spacing=0.5):
    global led_positions, num_leds
    distance = np.linalg.norm(np.array(end) - np.array(start))
    num_new_leds = int(distance / spacing) + 1
    t = np.linspace(0, 1, num_new_leds)[:, None]
    new_positions = (1 - t) * np.array(start) + t * np.array(end)
    led_positions = np.vstack([led_positions, new_positions])
    num_leds += num_new_leds
    return num_new_leds

def add_repeating_led_lines(start, end, n, spacing=0.5):
    global led_positions, num_leds
    angle_step = 2 * np.pi / n  
    for i in range(n):
        angle = i * angle_step
        rotation_matrix = np.array([
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle),  np.cos(angle), 0],
            [0, 0, 1]
        ])
        rotated_start = rotation_matrix @ np.array(start)
        rotated_end = rotation_matrix @ np.array(end)
        add_led_line(rotated_start, rotated_end, spacing)

def add_led_helix(radius, height_per_turn, turns, spacing=0.5):
    global led_positions, num_leds
    num_points = int((turns * 2 * np.pi * radius) / spacing)
    t = np.linspace(0, turns * 2 * np.pi, num_points)
    z = np.linspace(0, turns * height_per_turn, num_points)
    x = radius * np.cos(t)
    y = radius * np.sin(-t)
    new_positions = np.column_stack((x, y, z))
    led_positions = np.vstack([led_positions, new_positions])
    num_leds += num_points

# Create initial LED structure
a = add_repeating_led_lines([-10, 0, 0], [-10, 17.8, 9], 20)
a = num_leds  # Count of LEDs used in the repeating lines

add_led_helix(radius=5, height_per_turn=4, turns=2, spacing=0.5)
add_led_helix(radius=6, height_per_turn=4, turns=2, spacing=0.5)
add_led_helix(radius=7, height_per_turn=4, turns=2, spacing=0.5)
add_led_helix(radius=8, height_per_turn=4, turns=2, spacing=0.5)
b = num_leds  # Total count of LEDs

# Create LED scatter plot
grid = scene.visuals.Markers()
grid.set_data(led_positions, edge_color=None, face_color=(1, 1, 1, 1), size=10)
view.add(grid)

phase = 0  # Now a continuously increasing time parameter

def update(ev):
    global phase
    phase += swirl_speed  # Let phase increase indefinitely
    
    colors = np.zeros((num_leds, 4))  # RGBA colors
    colors[:, 3] = 1  # Full opacity
    
    if pattern_mode == 1:
        colors[:a, 0] = (np.sin(-phase + np.linspace(0, np.pi * 42, a)) + 1) / 2
        colors[:a, 2] = (np.cos(-phase + np.linspace(0, np.pi * 42, a)) + 1) / 2
    elif pattern_mode == 2:
        x_vals = np.floor(np.linspace(0, a - 1, a) / 40) / 20
        y_vals = 1 - (np.linspace(0, a - 1, a) % 40) / 40 
        colors[:a, 0] = np.clip((np.sin(x_vals * np.pi * 8 + phase) +
                                 np.cos(y_vals * np.pi * 3 - 2 * phase) + 
                                 np.sin((x_vals + y_vals) * np.pi * 8 + phase)) / 2, 0, 1)
        colors[:a, 2] = np.clip((np.sin(y_vals * np.pi * 3 + phase) *
                                 np.cos(x_vals * np.pi * 6 - 2 * phase) + 
                                 np.sin((x_vals - y_vals) * np.pi * 10 + 2 * phase)) / 2, 0, 1)
    elif pattern_mode == 3:
        x_vals = np.floor(np.linspace(0, a - 1, a) / 40) / 20
        y_vals = 1 - (np.linspace(0, a - 1, a) % 40) / 40 
        colors[:a, 0] = np.clip((np.sin(x_vals * np.pi * 8 + phase) +
                                 np.cos(y_vals * np.pi * 3 - 2 * phase)) / 2, 0, 1)
        colors[:a, 2] = np.clip((np.sin(y_vals * np.pi * 3 + phase) *
                                 np.cos(x_vals * np.pi * 6 - 2 * phase)) / 2, 0, 1)
    elif pattern_mode == 4:
        x_vals = np.floor(np.linspace(0, a - 1, a) / 40) / 20
        y_vals = 1 - (np.linspace(0, a - 1, a) % 40) / 40 
        # Instead of a periodic wrap, use continuously changing offsets.
        offset_x = 0.5 * phase
        offset_y = -0.3 * phase
        time_coord = 0.8 * phase
        for i in range(a):
            colors[i, 0] = np.clip(
                perlin_noise(x_vals[i]*2 + offset_x, y_vals[i]*2 + offset_y, time_coord, octaves=4)*2 - 1,
                0, 1)
            colors[i, 2] = np.clip(
                perlin_noise(y_vals[i]*2 + offset_y, x_vals[i]*2 + offset_x, time_coord, octaves=4)*2 - 1,
                0, 1)
    
    # Set the helix (spiral staircase) LEDs to a static brown color.
    colors[-(b - a):, 0] = 0.5  # R
    colors[-(b - a):, 1] = 0.2  # G
    grid.set_data(led_positions, face_color=colors, size=10)

def on_key(event):
    global swirl_speed, pattern_mode
    if event.text in '1234':
        pattern_mode = int(event.text)
    elif event.text == '+':
        swirl_speed += 0.01
    elif event.text == '-':
        swirl_speed = max(0.01, swirl_speed - 0.01)

timer = app.Timer(interval=0.05, connect=update, start=True)
canvas.events.key_press.connect(on_key)

if __name__ == '__main__':
    app.run()
