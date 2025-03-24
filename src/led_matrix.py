import numpy as np
from .utils import perlin_noise


class LEDMatrix:
    def __init__(self):
        self.positions = np.empty((0, 3))
        self.num_leds = 0
        self.a = 0  # Number of LEDs in repeating lines
        self.b = 0  # Total number of LEDs
        
        # Animation parameters
        self.swirl_speed = 0.03
        self.pattern_mode = 1
        self.phase = 0
        
        # Build the LED structure
        self._build_led_structure()
    
    def _build_led_structure(self):
        # Create initial LED structure
        self.positions, self.num_leds, self.a = self._add_repeating_led_lines(
            [-10, 0, 0], [-10, 17.8, 9], 20)
        
        # Add helix structures
        for radius in [5, 6, 7, 8]:
            self.positions, self.num_leds, _ = self._add_led_helix(
                radius=radius, height_per_turn=4, turns=2)
        
        self.b = self.num_leds  # Total count of LEDs
    
    def _add_led_line(self, start, end, current_positions, current_num_leds, spacing=0.5):
        distance = np.linalg.norm(np.array(end) - np.array(start))
        num_new_leds = int(distance / spacing) + 1
        t = np.linspace(0, 1, num_new_leds)[:, None]
        new_positions = (1 - t) * np.array(start) + t * np.array(end)
        updated_positions = np.vstack([current_positions, new_positions])
        updated_num_leds = current_num_leds + num_new_leds
        return updated_positions, updated_num_leds, num_new_leds
    
    def _add_repeating_led_lines(self, start, end, n, current_positions=None, current_num_leds=None, spacing=0.5):
        if current_positions is None:
            current_positions = self.positions
        if current_num_leds is None:
            current_num_leds = self.num_leds
            
        angle_step = 2 * np.pi / n  
        total_new_leds = 0
        updated_positions = current_positions.copy()
        
        for i in range(n):
            angle = i * angle_step
            rotation_matrix = np.array([
                [np.cos(angle), -np.sin(angle), 0],
                [np.sin(angle),  np.cos(angle), 0],
                [0, 0, 1]
            ])
            rotated_start = rotation_matrix @ np.array(start)
            rotated_end = rotation_matrix @ np.array(end)
            updated_positions, updated_num_leds, new_leds = self._add_led_line(
                rotated_start, rotated_end, updated_positions, current_num_leds + total_new_leds, spacing)
            total_new_leds += new_leds
        
        return updated_positions, current_num_leds + total_new_leds, total_new_leds
    
    def _add_led_helix(self, radius, height_per_turn, turns, current_positions=None, current_num_leds=None, spacing=0.5):
        if current_positions is None:
            current_positions = self.positions
        if current_num_leds is None:
            current_num_leds = self.num_leds
            
        num_points = int((turns * 2 * np.pi * radius) / spacing)
        t = np.linspace(0, turns * 2 * np.pi, num_points)
        z = np.linspace(0, turns * height_per_turn, num_points)
        x = radius * np.cos(t)
        y = radius * np.sin(-t)
        new_positions = np.column_stack((x, y, z))
        updated_positions = np.vstack([current_positions, new_positions])
        updated_num_leds = current_num_leds + num_points
        return updated_positions, updated_num_leds, num_points
    
    def update(self):
        self.phase += self.swirl_speed
        colors = np.zeros((self.num_leds, 4))  # RGBA colors
        colors[:, 3] = 1  # Full opacity
        
        if self.pattern_mode == 1:
            colors[:self.a, 0] = (np.sin(-self.phase + np.linspace(0, np.pi * 42, self.a)) + 1) / 2
            colors[:self.a, 2] = (np.cos(-self.phase + np.linspace(0, np.pi * 42, self.a)) + 1) / 2
        elif self.pattern_mode == 2:
            x_vals = np.floor(np.linspace(0, self.a - 1, self.a) / 40) / 20
            y_vals = 1 - (np.linspace(0, self.a - 1, self.a) % 40) / 40 
            colors[:self.a, 0] = np.clip((np.sin(x_vals * np.pi * 8 + self.phase) +
                                         np.cos(y_vals * np.pi * 3 - 2 * self.phase) + 
                                         np.sin((x_vals + y_vals) * np.pi * 8 + self.phase)) / 2, 0, 1)
            colors[:self.a, 2] = np.clip((np.sin(y_vals * np.pi * 3 + self.phase) *
                                         np.cos(x_vals * np.pi * 6 - 2 * self.phase) + 
                                         np.sin((x_vals - y_vals) * np.pi * 10 + 2 * self.phase)) / 2, 0, 1)
        elif self.pattern_mode == 3:
            x_vals = np.floor(np.linspace(0, self.a - 1, self.a) / 40) / 20
            y_vals = 1 - (np.linspace(0, self.a - 1, self.a) % 40) / 40 
            colors[:self.a, 0] = np.clip((np.sin(x_vals * np.pi * 8 + self.phase) +
                                         np.cos(y_vals * np.pi * 3 - 2 * self.phase)) / 2, 0, 1)
            colors[:self.a, 2] = np.clip((np.sin(y_vals * np.pi * 3 + self.phase) *
                                         np.cos(x_vals * np.pi * 6 - 2 * self.phase)) / 2, 0, 1)
        elif self.pattern_mode == 4:
            x_vals = np.floor(np.linspace(0, self.a - 1, self.a) / 40) / 20
            y_vals = 1 - (np.linspace(0, self.a - 1, self.a) % 40) / 40 
            offset_x = 0.5 * self.phase
            offset_y = -0.3 * self.phase
            time_coord = 0.8 * self.phase
            for i in range(self.a):
                colors[i, 0] = np.clip(
                    perlin_noise(x_vals[i]*2 + offset_x, y_vals[i]*2 + offset_y, time_coord, octaves=4)*2 - 1,
                    0, 1)
                colors[i, 2] = np.clip(
                    perlin_noise(y_vals[i]*2 + offset_y, x_vals[i]*2 + offset_x, time_coord, octaves=4)*2 - 1,
                    0, 1)
        
        # Set the helix (spiral staircase) LEDs to a static brown color
        colors[-(self.b - self.a):, 0] = 0.5  # R
        colors[-(self.b - self.a):, 1] = 0.2  # G
        
        return colors
    
    def handle_key(self, key):
        if key in '1234':
            self.pattern_mode = int(key)
        elif key == '+':
            self.swirl_speed += 0.01
        elif key == '-':
            self.swirl_speed = max(0.01, self.swirl_speed - 0.01)


__all__ = ['LEDMatrix']