import math
import numpy as np
from vispy import app, scene

# --- Perlin Noise Implementation ---

# Permutation table
p = [151,160,137,91,90,15,
     131,13,201,95,96,53,194,233,7,225,
     140,36,103,30,69,142,8,99,37,240,21,10,23,
     190,6,148,247,120,234,75,0,26,197,62,94,252,219,203,117,
     35,11,32,57,177,33,88,237,149,56,87,174,20,125,136,171,
     168,68,175,74,165,71,134,139,48,27,166,77,146,158,231,83,
     111,229,122,60,211,133,230,220,105,92,41,55,46,245,40,244,
     102,143,54,65,25,63,161,1,216,80,73,209,76,132,187,208,
     89,18,169,200,196,135,130,116,188,159,86,164,100,109,198,173,
     186,3,64,52,217,226,250,124,123,5,202,38,147,118,126,255,
     82,85,212,207,206,59,227,47,16,58,17,182,189,28,42,223,
     183,170,213,119,248,152,2,44,154,163,70,221,153,101,155,167,
     43,172,9,129,22,39,253,19,98,108,110,79,113,224,232,178,
     185,112,104,218,246,97,228,251,34,242,193,238,210,144,12,191,
     179,162,241,81,51,145,235,249,14,239,107,49,192,214,31,181,
     199,106,157,184,84,204,176,115,121,50,45,127,4,150,254,
     138,236,205,93,222,114,67,29,24,72,243,141,128,195,78,66,
     215,61,156,180]
# Duplicate the permutation list
p = p * 2

def fade(t):
    return t * t * t * (t * (t * 6 - 15) + 10)

def lerp(t, a, b):
    return a + t * (b - a)

def grad(hash, x, y, z):
    h = hash & 15
    u = x if h < 8 else y
    if h < 4:
        v = y
    elif h == 12 or h == 14:
        v = x
    else:
        v = z
    return (u if (h & 1) == 0 else -u) + (v if (h & 2) == 0 else -v)

def perlin3d(x, y, z):
    X = int(math.floor(x)) & 255
    Y = int(math.floor(y)) & 255
    Z = int(math.floor(z)) & 255

    x -= math.floor(x)
    y -= math.floor(y)
    z -= math.floor(z)

    u = fade(x)
    v = fade(y)
    w = fade(z)

    A  = p[X] + Y
    AA = p[A] + Z
    AB = p[A + 1] + Z
    B  = p[X + 1] + Y
    BA = p[B] + Z
    BB = p[B + 1] + Z

    return lerp(w,
                lerp(v,
                     lerp(u, grad(p[AA], x, y, z),
                             grad(p[BA], x - 1, y, z)),
                     lerp(u, grad(p[AB], x, y - 1, z),
                             grad(p[BB], x - 1, y - 1, z))),
                lerp(v,
                     lerp(u, grad(p[AA + 1], x, y, z - 1),
                             grad(p[BA + 1], x - 1, y, z - 1)),
                     lerp(u, grad(p[AB + 1], x, y - 1, z - 1),
                             grad(p[BB + 1], x - 1, y - 1, z - 1))))

def perlin_noise(x, y, t, octaves=1, persistence=0.5, lacunarity=2.0):
    total = 0.0
    amplitude = 1.0
    frequency = 1.0
    max_amp = 0.0
    for _ in range(octaves):
        total += perlin3d(x * frequency, y * frequency, t * frequency) * amplitude
        max_amp += amplitude
        amplitude *= persistence
        frequency *= lacunarity
    # Normalize result to [0, 1]
    return (total / max_amp + 1) / 2

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
