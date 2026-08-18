#!/usr/bin/env python3
# Copyright 2026 Ingo Dering
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Hand logic module — pure business logic with no UI dependencies.

Provides configuration management, servo math, data conversion, and
validation functions shared by the GUI and CLI tools.
"""
import os
import re
from pathlib import Path

import numpy as np
import yaml

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

APP_VERSION = "0.8"

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
CONFIG_FILE = DATA_DIR / "hand_config.yaml"   # Poses and sequences
APP_CONFIG_FILE = DATA_DIR / "config.yaml"     # Application settings

# Servo IDs per finger (matches GUI build order: Ring, Middle, Pointer, Thumb)
FINGER_NAMES = ["Ring", "Middle", "Pointer", "Thumb"]
SERVO_PAIRS = [(5, 6), (3, 4), (1, 2), (7, 8)]  # (servo1_id, servo2_id)

DEFAULT_PORT_LINUX = "/dev/ttyACM0"
DEFAULT_PORT_WINDOWS = "COM9"
DEFAULT_BAUDRATE = 1_000_000

KEYBOARD_HELP_TEXT = """\
        KEYBOARD CONTROLS
        ═════════════════════════════════════════════════════

        FINGER SELECTION:
            1, 2, 3, 4    Select finger (Ring, Middle, Pointer, Thumb)

        MOVEMENT CONTROLS:
            Up Arrow      Close finger (increase position)
            Down Arrow    Open finger (decrease position)
            Right Arrow   Move finger right
            Left Arrow    Move finger left

        QUICK ACTIONS:
            Q             Fully close selected finger
            E             Fully open selected finger
            C             Center left/right position

        PRECISION MODIFIERS:
            Normal        1° per keypress (precise, default)
            Shift + Key   5° per keypress (normal movement)
            Ctrl + Key    10° per keypress (fast movement)

        EXAMPLES:
            Press 1       → Select Ring finger
            Press ↑       → Close 1° (precise)
            Shift + ↑     → Close 5° (normal)
            Ctrl + ↑      → Close 10° (fast)
            Press Q       → Fully close to 110°
            Press E       → Fully open to 0°
            Press C       → Center to 0° left/right
"""


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def load_app_config():
    """Load application configuration from config.yaml.

    Returns:
        dict: Application configuration with serial ports, limits, speeds, paths
    """
    default_config = {
        'serial': {
            'port_windows': 'COM9',
            'port_linux': '/dev/ttyACM0',
            'baudrate': 1000000,
            'baudrate_options': [9600, 115200, 1000000]
        },
        'servos': {
            'pointer': [1, 2],
            'middle': [3, 4],
            'ring': [5, 6],
            'thumb': [7, 8],
            'all_ids': [1, 2, 3, 4, 5, 6, 7, 8]
        },
        'limits': {
            'servo_min': -40,
            'servo_max': 110,
            'base_min': 0,
            'base_max': 110,
            'side_min': -40,
            'side_max': 40
        },
        'speeds': {
            'default': 3,
            'min': 1,
            'max': 6
        },
        'auto_extremes': {
            'left_open': [25, -40],
            'right_open': [-40, 25],
            'left_closed': [110, 110],
            'right_closed': [110, 110],
            'center_open': [0, 0],
            'center_closed': [110, 110]
        },
        'paths': {
            'poses_sequences_file': 'data/hand_config.yaml'
        }
    }

    if not APP_CONFIG_FILE.exists():
        return default_config

    try:
        with APP_CONFIG_FILE.open('r') as f:
            config = yaml.safe_load(f) or {}
            # Merge with defaults
            for key in default_config:
                if key not in config:
                    config[key] = default_config[key]
                elif isinstance(default_config[key], dict):
                    for subkey in default_config[key]:
                        if subkey not in config[key]:
                            config[key][subkey] = default_config[key][subkey]
            return config
    except Exception as e:
        print(f"Error loading app config: {e}, using defaults")
        return default_config


def default_serial_port():
    """Return platform-specific default serial port from config."""
    config = load_app_config()
    if os.name == 'nt':
        return config['serial']['port_windows']
    return config['serial']['port_linux']


def ensure_data_dir():
    """Create the data directory if it does not already exist."""
    DATA_DIR.mkdir(exist_ok=True)


def load_config():
    """Load poses/sequences configuration from YAML file.

    Returns:
        dict: Configuration dictionary with 'poses' and 'sequences' keys.
    """
    if not CONFIG_FILE.exists():
        return {'poses': {}, 'sequences': {}}

    try:
        with CONFIG_FILE.open('r') as f:
            config = yaml.safe_load(f) or {}
            if 'poses' not in config:
                config['poses'] = {}
            if 'sequences' not in config:
                config['sequences'] = {}
            return config
    except Exception as e:
        print(f"Error loading config: {e}")
        return {'poses': {}, 'sequences': {}}


def save_config(config):
    """Save configuration to YAML file with inline array formatting.

    Args:
        config (dict): Configuration dictionary (see load_config for structure)

    Returns:
        bool: True if save successful, False on error
    """
    try:
        ensure_data_dir()

        yaml_str = yaml.dump(config, default_flow_style=False, sort_keys=False)

        # Replace positions lists with flow style
        def replace_positions(match):
            lines = match.group(0)
            values = re.findall(r'- (-?\d+)', lines)
            return f"    positions: [{', '.join(values)}]\n"

        yaml_str = re.sub(
            r'    positions:\n(?:    - -?\d+\n)+',
            replace_positions,
            yaml_str,
        )

        with CONFIG_FILE.open('w') as f:
            f.write(yaml_str)
        return True
    except Exception as e:
        print(f"Error saving config: {e}")
        import traceback
        traceback.print_exc()
        return False


def load_pose_definitions():
    """Load pose definitions from YAML config.

    Returns:
        list[dict]: List of ``{'name': str, 'positions': list[int]}`` dicts.
    """
    config = load_config()
    poses = []
    for name, data in config.get('poses', {}).items():
        poses.append({
            'name': name,
            'positions': data.get('positions', [0] * 8)
        })
    return poses


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_name(name):
    """Validate a pose/sequence name to prevent YAML corruption.

    Args:
        name (str): Name to validate

    Returns:
        tuple: ``(is_valid, error_message)``
    """
    if not name or not name.strip():
        return False, "Name cannot be empty"

    name = name.strip()

    if len(name) > 50:
        return False, "Name too long (max 50 characters)"

    forbidden_chars = [
        ':', '{', '}', '[', ']', ',', '&', '*', '#', '?', '|',
        '-', '<', '>', '=', '!', '%', '@', '`', '"', "'",
    ]

    for char in forbidden_chars:
        if char in name:
            return False, f"Name contains forbidden character: {char}"

    if name != name.strip():
        return False, "Name has leading/trailing spaces"

    if any(ord(c) < 32 for c in name):
        return False, "Name contains control characters"

    return True, ""


# ---------------------------------------------------------------------------
# Numeric / servo utilities
# ---------------------------------------------------------------------------

def clamp(value, min_value, max_value):
    """Clamp a numeric value between bounds."""
    return max(min_value, min(max_value, value))


def angle_rad(servo_id, degrees):
    """Convert degrees to radians, applying even-servo inversion."""
    if servo_id % 2 == 0:
        return np.deg2rad(-degrees)
    return np.deg2rad(degrees)


def coerce_numeric(value, default=0.0):
    """Convert controller return types (arrays, lists) to float."""
    if value is None:
        return default
    if isinstance(value, np.ndarray):
        try:
            return float(value.item())
        except Exception:
            data = value.tolist()
            return coerce_numeric(data[0], default) if data else default
    if isinstance(value, (list, tuple)):
        return coerce_numeric(value[0] if value else default, default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def coerce_angle_degrees(value, servo_id, default=0.0):
    """Convert a radian-based controller reading into GUI degrees."""
    radians = coerce_numeric(value, default)
    degrees = float(np.rad2deg(radians))
    if servo_id % 2 == 0:
        degrees = -degrees
    return degrees


def coerce_bool(value):
    """Coerce controller return types to boolean flags."""
    return bool(int(coerce_numeric(value, 0.0)))


def load_to_percent(load_value):
    """Convert the present-load reading into a percentage."""
    try:
        load_float = float(load_value)
    except (TypeError, ValueError):
        return 0.0
    magnitude = abs(load_float)
    if magnitude <= 1.5:
        percent = magnitude * 100.0
    else:
        percent = magnitude / 10.23
    percent = clamp(percent, 0.0, 150.0)
    return percent if load_float >= 0 else -percent


def estimate_current_from_load(load_value):
    """Rudimentary current estimate derived from torque percentage."""
    percent = abs(load_to_percent(load_value))
    if percent <= 0.1:
        return 0.0
    estimated_ma = clamp(percent / 100.0 * 1200.0, 0.0, 1500.0)
    return round(estimated_ma, 1)


def format_feedback_value(key, value):
    """Convert raw feedback values into user-friendly strings."""
    if value is None:
        return '—'

    if key in ('goal', 'position'):
        return f"{float(value):.2f}°"
    if key == 'speed':
        return f"{float(value):.1f}°/s"
    if key == 'voltage':
        return f"{float(value):.2f} V"
    if key == 'temperature':
        return f"{float(value):.1f} °C"
    if key == 'current':
        return f"{float(value):.0f} mA"
    if key == 'load':
        return f"{load_to_percent(value):.1f} %"
    if key == 'status':
        return f"0x{int(value) & 0xFF:02X}"
    if key == 'moving':
        return "Yes" if bool(value) else "No"
    return str(value)


def compute_auto_positions(base_pos, side_offset, limits, auto_extremes):
    """Compute servo positions from base/side values using bilinear interpolation.

    Args:
        base_pos (int): Close/open position value
        side_offset (int): Side offset value
        limits (dict): Keys: base_min, base_max, side_min, side_max, servo_min, servo_max
        auto_extremes (dict): Keys: left_open, right_open, left_closed, right_closed

    Returns:
        tuple[int, int]: Clamped (pos1, pos2) servo target values
    """
    base_min = limits['base_min']
    base_max = limits['base_max']
    side_min = limits['side_min']
    side_max = limits['side_max']
    servo_min = limits['servo_min']
    servo_max = limits['servo_max']

    base_pos = clamp(base_pos, base_min, base_max)
    side_offset = clamp(side_offset, side_min, side_max)

    left_open = auto_extremes['left_open']
    right_open = auto_extremes['right_open']
    left_closed = auto_extremes['left_closed']
    right_closed = auto_extremes['right_closed']

    t = base_pos / base_max if base_max != 0 else 0.0

    if side_offset < 0:  # moving left
        u = abs(side_offset) / abs(side_min) if side_min != 0 else 0.0
        center = (base_pos, base_pos)
        left_target = (
            left_open[0] + t * (left_closed[0] - left_open[0]),
            left_open[1] + t * (left_closed[1] - left_open[1]),
        )
        pos1 = center[0] + u * (left_target[0] - center[0])
        pos2 = center[1] + u * (left_target[1] - center[1])
    elif side_offset > 0:  # moving right
        u = side_offset / side_max if side_max != 0 else 0.0
        center = (base_pos, base_pos)
        right_target = (
            right_open[0] + t * (right_closed[0] - right_open[0]),
            right_open[1] + t * (right_closed[1] - right_open[1]),
        )
        pos1 = center[0] + u * (right_target[0] - center[0])
        pos2 = center[1] + u * (right_target[1] - center[1])
    else:
        pos1 = base_pos
        pos2 = base_pos

    return clamp(int(pos1), servo_min, servo_max), clamp(int(pos2), servo_min, servo_max)


def decompose_servo_positions(pos1, pos2, limits):
    """Decompose raw servo positions into base + side offset.

    Args:
        pos1 (int): First servo position
        pos2 (int): Second servo position
        limits (dict): Keys: servo_min, servo_max, side_min, side_max

    Returns:
        tuple[int, int]: (base_pos, side_offset)
    """
    servo_min = limits['servo_min']
    servo_max = limits['servo_max']
    side_min = limits['side_min']
    side_max = limits['side_max']

    pos1 = clamp(int(pos1), servo_min, servo_max)
    pos2 = clamp(int(pos2), servo_min, servo_max)
    base_pos = (pos1 + pos2) // 2
    side_offset = clamp(base_pos - pos1, side_min, side_max)
    return base_pos, side_offset


def get_time_window_indices(total, x_zoom, x_pan):
    """Compute start/end indices for a chart time window.

    Args:
        total (int): Total number of data points
        x_zoom (float): Zoom fraction (0.05–1.0)
        x_pan (float): Pan position (0.0–1.0)

    Returns:
        tuple[int, int]: (start_index, end_index)
    """
    if total == 0:
        return 0, 0
    window_fraction = clamp(x_zoom, 0.05, 1.0)
    window_size = max(2, int(total * window_fraction))
    window_size = min(window_size, total)
    max_start = total - window_size
    start = 0
    if max_start > 0:
        start = int(round(clamp(x_pan, 0.0, 1.0) * max_start))
    end = start + window_size
    return start, end
