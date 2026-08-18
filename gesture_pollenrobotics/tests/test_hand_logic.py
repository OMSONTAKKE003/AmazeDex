"""
Unit tests for hand_logic.py — pure business-logic functions.

Covers all functions in hand_logic.py that previously had no dedicated
test coverage: app config loading, serial port defaults, data-dir
creation, pose definitions, coercion helpers, feedback formatting,
auto-position interpolation, position decomposition, and chart windowing.
"""
import os
import pytest
import numpy as np
from pathlib import Path
from unittest.mock import patch

import hand_logic


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def logic_paths(tmp_path, monkeypatch):
    """Redirect CONFIG_FILE, DATA_DIR and APP_CONFIG_FILE to tmp_path."""
    cf = tmp_path / "hand_config.yaml"
    acf = tmp_path / "config.yaml"
    monkeypatch.setattr(hand_logic, "CONFIG_FILE", cf)
    monkeypatch.setattr(hand_logic, "DATA_DIR", tmp_path)
    monkeypatch.setattr(hand_logic, "APP_CONFIG_FILE", acf)
    return cf, acf, tmp_path


# ===================================================================
# FR-CFG-1: App Config Loading
# ===================================================================

class TestLoadAppConfig:
    """Tests for load_app_config() — AC 1.1, 1.2, 1.3."""

    def test_missing_file_returns_full_defaults(self, logic_paths):
        """AC 1.1: Missing file → full default config used."""
        cfg = hand_logic.load_app_config()
        assert 'serial' in cfg
        assert 'servos' in cfg
        assert 'limits' in cfg
        assert 'speeds' in cfg
        assert 'auto_extremes' in cfg

    def test_default_serial_ports(self, logic_paths):
        cfg = hand_logic.load_app_config()
        assert cfg['serial']['port_linux'] == '/dev/ttyACM0'
        assert cfg['serial']['port_windows'] == 'COM9'

    def test_default_baudrate_options(self, logic_paths):
        """AC 2.1: Default baudrate options from config."""
        cfg = hand_logic.load_app_config()
        assert cfg['serial']['baudrate_options'] == [9600, 115200, 1000000]

    def test_default_baudrate(self, logic_paths):
        """AC 2.2: Default baudrate is 1000000."""
        cfg = hand_logic.load_app_config()
        assert cfg['serial']['baudrate'] == 1000000

    def test_missing_keys_merged_from_defaults(self, logic_paths):
        """AC 1.2: Missing keys merged from defaults (two-level merge)."""
        _, acf, _ = logic_paths
        # Write partial config (only serial section, missing other sections)
        acf.write_text("serial:\n  baudrate: 9600\n")
        cfg = hand_logic.load_app_config()
        assert cfg['serial']['baudrate'] == 9600
        # Missing sub-key filled from default
        assert cfg['serial']['port_linux'] == '/dev/ttyACM0'
        # Missing top-level sections filled
        assert 'limits' in cfg
        assert 'speeds' in cfg

    def test_parse_failure_returns_defaults(self, logic_paths):
        """AC 1.3: Parse failure → defaults returned."""
        _, acf, _ = logic_paths
        acf.write_text("this: is: {malformed\n")
        cfg = hand_logic.load_app_config()
        assert cfg['serial']['baudrate'] == 1000000


# ===================================================================
# FR-CFG-2: Servo Mapping
# ===================================================================

class TestServoMapping:
    """Tests for servo constants — AC 2.1, 2.2."""

    def test_servo_pairs_match_config(self):
        """AC 2.1: pointer=[1,2], middle=[3,4], ring=[5,6], thumb=[7,8]."""
        assert hand_logic.SERVO_PAIRS == [(5, 6), (3, 4), (1, 2), (7, 8)]

    def test_all_servo_ids(self):
        """AC 2.2: all_ids = [1,2,3,4,5,6,7,8]."""
        all_ids = sorted(
            sid for pair in hand_logic.SERVO_PAIRS for sid in pair
        )
        assert all_ids == [1, 2, 3, 4, 5, 6, 7, 8]

    def test_finger_names_match_pairs(self):
        """AC 1.1: Exactly 4 fingers."""
        assert hand_logic.FINGER_NAMES == ["Ring", "Middle", "Pointer", "Thumb"]
        assert len(hand_logic.FINGER_NAMES) == len(hand_logic.SERVO_PAIRS)


# ===================================================================
# FR-CFG-3: Angle Limits
# ===================================================================

class TestAngleLimits:
    """AC 3.1: Default angle limits."""

    def test_defaults(self, logic_paths):
        cfg = hand_logic.load_app_config()
        lim = cfg['limits']
        assert lim['servo_min'] == -40
        assert lim['servo_max'] == 110
        assert lim['base_min'] == 0
        assert lim['base_max'] == 110
        assert lim['side_min'] == -40
        assert lim['side_max'] == 40


# ===================================================================
# FR-CFG-4: Auto Extremes
# ===================================================================

class TestAutoExtremes:
    """AC 4.1: Auto extreme values are configurable."""

    def test_defaults(self, logic_paths):
        cfg = hand_logic.load_app_config()
        ae = cfg['auto_extremes']
        assert 'left_open' in ae
        assert 'right_open' in ae
        assert 'left_closed' in ae
        assert 'right_closed' in ae
        assert 'center_open' in ae
        assert 'center_closed' in ae

    def test_custom_overrides(self, logic_paths):
        _, acf, _ = logic_paths
        acf.write_text(
            "auto_extremes:\n"
            "  left_open: [30, -30]\n"
            "  right_open: [-30, 30]\n"
        )
        cfg = hand_logic.load_app_config()
        assert cfg['auto_extremes']['left_open'] == [30, -30]
        assert cfg['auto_extremes']['right_open'] == [-30, 30]
        # Default sub-keys filled
        assert 'left_closed' in cfg['auto_extremes']


# ===================================================================
# FR-CFG-5: Speed Configuration
# ===================================================================

class TestSpeedConfig:
    """AC 5.1: speeds.default = 3, speeds.min = 1, speeds.max = 6."""

    def test_defaults(self, logic_paths):
        cfg = hand_logic.load_app_config()
        assert cfg['speeds']['default'] == 3
        assert cfg['speeds']['min'] == 1
        assert cfg['speeds']['max'] == 6


# ===================================================================
# FR-CONN-1 / default_serial_port
# ===================================================================

class TestDefaultSerialPort:
    """AC 1.3: Default matches platform-specific value."""

    def test_linux(self, logic_paths):
        with patch.object(os, 'name', 'posix'):
            port = hand_logic.default_serial_port()
        assert port == '/dev/ttyACM0'

    def test_windows(self, logic_paths):
        with patch.object(os, 'name', 'nt'):
            port = hand_logic.default_serial_port()
        assert port == 'COM9'


# ===================================================================
# FR-DATA-4: Data Directory Auto-Creation
# ===================================================================

class TestEnsureDataDir:
    """AC 4.1: data/ directory created if missing."""

    def test_creates_directory(self, tmp_path, monkeypatch):
        new_dir = tmp_path / "new_data"
        monkeypatch.setattr(hand_logic, "DATA_DIR", new_dir)
        assert not new_dir.exists()
        hand_logic.ensure_data_dir()
        assert new_dir.exists()

    def test_existing_directory_no_error(self, tmp_path, monkeypatch):
        monkeypatch.setattr(hand_logic, "DATA_DIR", tmp_path)
        hand_logic.ensure_data_dir()  # Should not raise


# ===================================================================
# load_pose_definitions
# ===================================================================

class TestLoadPoseDefinitions:

    def test_empty_config(self, logic_paths):
        poses = hand_logic.load_pose_definitions()
        assert poses == []

    def test_with_poses(self, logic_paths):
        cf, _, _ = logic_paths
        cf.write_text(
            "poses:\n"
            "  open:\n"
            "    positions: [0, 0, 0, 0, 0, 0, 0, 0]\n"
            "  close:\n"
            "    positions: [110, 110, 110, 110, 110, 110, 110, 110]\n"
        )
        poses = hand_logic.load_pose_definitions()
        assert len(poses) == 2
        names = {p['name'] for p in poses}
        assert names == {'open', 'close'}

    def test_missing_positions_defaults_to_zeros(self, logic_paths):
        cf, _, _ = logic_paths
        cf.write_text("poses:\n  empty:\n    description: no positions\n")
        poses = hand_logic.load_pose_definitions()
        assert poses[0]['positions'] == [0] * 8


# ===================================================================
# FR-FING-2 / FR-CFG-4: compute_auto_positions
# ===================================================================

class TestComputeAutoPositions:
    """AC 2.3: Auto-mode interpolation (compute_auto_positions)."""

    LIMITS = {
        'base_min': 0, 'base_max': 110,
        'side_min': -40, 'side_max': 40,
        'servo_min': -40, 'servo_max': 110,
    }
    EXTREMES = {
        'left_open': [25, -40],
        'right_open': [-40, 25],
        'left_closed': [110, 110],
        'right_closed': [110, 110],
        'center_open': [0, 0],
        'center_closed': [110, 110],
    }

    def test_center_open(self):
        """side=0, base=0 → both servos at 0."""
        p1, p2 = hand_logic.compute_auto_positions(0, 0, self.LIMITS, self.EXTREMES)
        assert p1 == 0
        assert p2 == 0

    def test_center_closed(self):
        """side=0, base=110 → both servos at 110."""
        p1, p2 = hand_logic.compute_auto_positions(110, 0, self.LIMITS, self.EXTREMES)
        assert p1 == 110
        assert p2 == 110

    def test_left_offset_open(self):
        """side=-40 (full left), base=0 → approaches left_open extremes."""
        p1, p2 = hand_logic.compute_auto_positions(0, -40, self.LIMITS, self.EXTREMES)
        assert p1 == 25   # left_open[0]
        assert p2 == -40  # left_open[1]

    def test_right_offset_open(self):
        """side=+40 (full right), base=0 → approaches right_open extremes."""
        p1, p2 = hand_logic.compute_auto_positions(0, 40, self.LIMITS, self.EXTREMES)
        assert p1 == -40  # right_open[0]
        assert p2 == 25   # right_open[1]

    def test_clamping_to_servo_limits(self):
        """Results are clamped to [servo_min, servo_max]."""
        p1, p2 = hand_logic.compute_auto_positions(55, -20, self.LIMITS, self.EXTREMES)
        assert -40 <= p1 <= 110
        assert -40 <= p2 <= 110

    def test_symmetry(self):
        """Left and right should be mirror images."""
        p1_l, p2_l = hand_logic.compute_auto_positions(0, -40, self.LIMITS, self.EXTREMES)
        p1_r, p2_r = hand_logic.compute_auto_positions(0, 40, self.LIMITS, self.EXTREMES)
        # With symmetric extremes, p1_left == p2_right and vice versa
        assert p1_l == p2_r
        assert p2_l == p1_r


# ===================================================================
# FR-FING-3: decompose_servo_positions
# ===================================================================

class TestDecomposeServoPositions:
    """AC 3.3: decompose_servo_positions for raw ↔ auto sync."""

    LIMITS = {
        'servo_min': -40, 'servo_max': 110,
        'side_min': -40, 'side_max': 40,
    }

    def test_equal_positions(self):
        """Equal positions → base=pos, side=0."""
        base, side = hand_logic.decompose_servo_positions(55, 55, self.LIMITS)
        assert base == 55
        assert side == 0

    def test_asymmetric_positions(self):
        """Different positions → averaged base with side offset."""
        base, side = hand_logic.decompose_servo_positions(70, 40, self.LIMITS)
        assert base == 55
        assert side == -15  # 55 - 70 (pos1 > pos2 means left → negative)

    def test_clamping_large_offset(self):
        """Very large offset clamped to side_max."""
        base, side = hand_logic.decompose_servo_positions(110, -40, self.LIMITS)
        assert -40 <= side <= 40

    def test_clamping_input_to_servo_range(self):
        """Input positions outside servo range are clamped."""
        base, side = hand_logic.decompose_servo_positions(200, -100, self.LIMITS)
        # 200 → 110, -100 → -40 → base = (110 + -40)//2 = 35
        assert base == 35

    def test_round_trip_with_compute(self):
        """Decompose should be roughly inverse of compute for center offset."""
        limits = {
            'base_min': 0, 'base_max': 110,
            'side_min': -40, 'side_max': 40,
            'servo_min': -40, 'servo_max': 110,
        }
        extremes = {
            'left_open': [25, -40], 'right_open': [-40, 25],
            'left_closed': [110, 110], 'right_closed': [110, 110],
            'center_open': [0, 0], 'center_closed': [110, 110],
        }
        # At center (side=0), compute gives (base, base), decompose gives (base, 0)
        p1, p2 = hand_logic.compute_auto_positions(60, 0, limits, extremes)
        base_back, side_back = hand_logic.decompose_servo_positions(p1, p2, limits)
        assert base_back == 60
        assert side_back == 0


# ===================================================================
# coerce_numeric
# ===================================================================

class TestCoerceNumeric:

    def test_none_returns_default(self):
        assert hand_logic.coerce_numeric(None) == 0.0

    def test_none_custom_default(self):
        assert hand_logic.coerce_numeric(None, 42.0) == 42.0

    def test_float_passthrough(self):
        assert hand_logic.coerce_numeric(3.14) == pytest.approx(3.14)

    def test_int_to_float(self):
        assert hand_logic.coerce_numeric(5) == 5.0

    def test_numpy_scalar(self):
        val = np.float64(2.5)
        assert hand_logic.coerce_numeric(val) == pytest.approx(2.5)

    def test_numpy_array_single_element(self):
        val = np.array([7.0])
        assert hand_logic.coerce_numeric(val) == pytest.approx(7.0)

    def test_list_single_element(self):
        assert hand_logic.coerce_numeric([9.0]) == pytest.approx(9.0)

    def test_tuple_single_element(self):
        assert hand_logic.coerce_numeric((4.0,)) == pytest.approx(4.0)

    def test_empty_list_returns_default(self):
        assert hand_logic.coerce_numeric([]) == 0.0

    def test_string_returns_default(self):
        assert hand_logic.coerce_numeric("bad") == 0.0

    def test_numeric_string(self):
        assert hand_logic.coerce_numeric("3.5") == pytest.approx(3.5)


# ===================================================================
# coerce_angle_degrees
# ===================================================================

class TestCoerceAngleDegrees:

    def test_odd_servo_positive_angle(self):
        rad = np.deg2rad(45)
        result = hand_logic.coerce_angle_degrees(rad, 1)
        assert result == pytest.approx(45.0)

    def test_even_servo_negated(self):
        rad = np.deg2rad(45)
        result = hand_logic.coerce_angle_degrees(rad, 2)
        assert result == pytest.approx(-45.0)

    def test_zero(self):
        result = hand_logic.coerce_angle_degrees(0.0, 1)
        assert result == pytest.approx(0.0)

    def test_none_returns_default_degrees(self):
        result = hand_logic.coerce_angle_degrees(None, 1)
        assert result == pytest.approx(0.0)

    def test_negative_radians_odd_servo(self):
        rad = np.deg2rad(-30)
        result = hand_logic.coerce_angle_degrees(rad, 3)
        assert result == pytest.approx(-30.0)

    def test_negative_radians_even_servo(self):
        rad = np.deg2rad(-30)
        result = hand_logic.coerce_angle_degrees(rad, 4)
        assert result == pytest.approx(30.0)


# ===================================================================
# coerce_bool
# ===================================================================

class TestCoerceBool:

    def test_zero_is_false(self):
        assert hand_logic.coerce_bool(0) is False

    def test_one_is_true(self):
        assert hand_logic.coerce_bool(1) is True

    def test_float_one_is_true(self):
        assert hand_logic.coerce_bool(1.0) is True

    def test_numpy_one(self):
        assert hand_logic.coerce_bool(np.float64(1.0)) is True

    def test_numpy_zero(self):
        assert hand_logic.coerce_bool(np.float64(0.0)) is False

    def test_none_is_false(self):
        assert hand_logic.coerce_bool(None) is False


# ===================================================================
# FR-MON-7: load_to_percent
# ===================================================================

class TestLoadToPercent:

    def test_zero_load(self):
        assert hand_logic.load_to_percent(0.0) == pytest.approx(0.0)

    def test_small_magnitude_times_100(self):
        """Magnitude <= 1.5 → percent = magnitude * 100."""
        assert hand_logic.load_to_percent(1.0) == pytest.approx(100.0)

    def test_large_magnitude_div_1023(self):
        """Magnitude > 1.5 → percent = magnitude / 10.23."""
        assert hand_logic.load_to_percent(102.3) == pytest.approx(10.0)

    def test_negative_load(self):
        result = hand_logic.load_to_percent(-1.0)
        assert result == pytest.approx(-100.0)

    def test_clamped_at_150(self):
        result = hand_logic.load_to_percent(2000.0)
        assert result == pytest.approx(150.0)

    def test_invalid_string_returns_zero(self):
        assert hand_logic.load_to_percent("bad") == pytest.approx(0.0)

    def test_none_returns_zero(self):
        assert hand_logic.load_to_percent(None) == pytest.approx(0.0)


# ===================================================================
# estimate_current_from_load
# ===================================================================

class TestEstimateCurrentFromLoad:

    def test_zero_load(self):
        assert hand_logic.estimate_current_from_load(0.0) == 0.0

    def test_full_load(self):
        """100% load → 1200 mA."""
        result = hand_logic.estimate_current_from_load(1.0)
        assert result == pytest.approx(1200.0)

    def test_negative_load_same_magnitude(self):
        """Negative load uses absolute value."""
        result = hand_logic.estimate_current_from_load(-1.0)
        assert result == pytest.approx(1200.0)

    def test_tiny_load_returns_zero(self):
        result = hand_logic.estimate_current_from_load(0.00001)
        assert result == 0.0

    def test_clamped_at_1500(self):
        result = hand_logic.estimate_current_from_load(2000.0)
        assert result <= 1500.0


# ===================================================================
# FR-MON-7: format_feedback_value (AC 7.2)
# ===================================================================

class TestFormatFeedbackValue:

    def test_none_returns_dash(self):
        assert hand_logic.format_feedback_value('position', None) == '—'

    def test_position_format(self):
        assert hand_logic.format_feedback_value('position', 45.0) == '45.00°'

    def test_goal_format(self):
        assert hand_logic.format_feedback_value('goal', 110.0) == '110.00°'

    def test_speed_format(self):
        assert hand_logic.format_feedback_value('speed', 5.0) == '5.0°/s'

    def test_voltage_format(self):
        assert hand_logic.format_feedback_value('voltage', 7.4) == '7.40 V'

    def test_temperature_format(self):
        assert hand_logic.format_feedback_value('temperature', 35.0) == '35.0 °C'

    def test_current_format(self):
        assert hand_logic.format_feedback_value('current', 500.0) == '500 mA'

    def test_load_format(self):
        result = hand_logic.format_feedback_value('load', 0.5)
        assert '50.0 %' in result

    def test_status_format(self):
        assert hand_logic.format_feedback_value('status', 0) == '0x00'
        assert hand_logic.format_feedback_value('status', 255) == '0xFF'

    def test_moving_yes(self):
        assert hand_logic.format_feedback_value('moving', True) == 'Yes'

    def test_moving_no(self):
        assert hand_logic.format_feedback_value('moving', False) == 'No'

    def test_unknown_key_str(self):
        assert hand_logic.format_feedback_value('unknown', 42) == '42'


# ===================================================================
# get_time_window_indices
# ===================================================================

class TestGetTimeWindowIndices:

    def test_empty_data(self):
        start, end = hand_logic.get_time_window_indices(0, 1.0, 0.0)
        assert start == 0
        assert end == 0

    def test_full_zoom_shows_all(self):
        start, end = hand_logic.get_time_window_indices(100, 1.0, 0.0)
        assert start == 0
        assert end == 100

    def test_half_zoom_start(self):
        start, end = hand_logic.get_time_window_indices(100, 0.5, 0.0)
        assert start == 0
        assert end == 50

    def test_half_zoom_end(self):
        start, end = hand_logic.get_time_window_indices(100, 0.5, 1.0)
        assert start == 50
        assert end == 100

    def test_half_zoom_middle(self):
        start, end = hand_logic.get_time_window_indices(100, 0.5, 0.5)
        assert start == 25
        assert end == 75

    def test_minimum_window_size(self):
        """Even with tiny zoom, window has at least 2 points."""
        start, end = hand_logic.get_time_window_indices(100, 0.01, 0.0)
        assert end - start >= 2

    def test_pan_clamped(self):
        """Pan > 1.0 clamped to end."""
        start, end = hand_logic.get_time_window_indices(100, 0.5, 5.0)
        assert end == 100

    def test_small_dataset(self):
        start, end = hand_logic.get_time_window_indices(3, 1.0, 0.0)
        assert start == 0
        assert end == 3


# ===================================================================
# angle_rad (additional coverage from hand_logic)
# ===================================================================

class TestAngleRadLogic:
    """Verify angle_rad from hand_logic module directly."""

    def test_odd_positive(self):
        assert hand_logic.angle_rad(1, 90) == pytest.approx(np.deg2rad(90))

    def test_even_negated(self):
        assert hand_logic.angle_rad(2, 90) == pytest.approx(np.deg2rad(-90))

    def test_zero(self):
        assert hand_logic.angle_rad(1, 0) == pytest.approx(0.0)
        assert hand_logic.angle_rad(2, 0) == pytest.approx(0.0)

    def test_odd_negative_degrees(self):
        """Negative degrees on odd servo stay negative."""
        assert hand_logic.angle_rad(3, -45) == pytest.approx(np.deg2rad(-45))

    def test_even_negative_degrees_flipped(self):
        """Negative degrees on even servo become positive radians."""
        assert hand_logic.angle_rad(4, -45) == pytest.approx(np.deg2rad(45))

    def test_higher_odd_servo_id(self):
        assert hand_logic.angle_rad(7, 60) == pytest.approx(np.deg2rad(60))

    def test_higher_even_servo_id(self):
        assert hand_logic.angle_rad(8, 60) == pytest.approx(np.deg2rad(-60))


# ===================================================================
# clamp
# ===================================================================

class TestClamp:

    def test_within_range(self):
        assert hand_logic.clamp(50, 0, 110) == 50

    def test_at_min(self):
        assert hand_logic.clamp(0, 0, 110) == 0

    def test_at_max(self):
        assert hand_logic.clamp(110, 0, 110) == 110

    def test_below_min(self):
        assert hand_logic.clamp(-10, 0, 110) == 0

    def test_above_max(self):
        assert hand_logic.clamp(200, 0, 110) == 110

    def test_negative_range(self):
        assert hand_logic.clamp(-50, -40, 40) == -40

    def test_float_values(self):
        assert hand_logic.clamp(0.5, 0.0, 1.0) == pytest.approx(0.5)


# ===================================================================
# validate_name
# ===================================================================

class TestValidateName:

    def test_valid_simple_name(self):
        ok, msg = hand_logic.validate_name("open")
        assert ok is True
        assert msg == ""

    def test_valid_with_spaces_inside(self):
        ok, _ = hand_logic.validate_name("my pose")
        assert ok is True

    def test_valid_with_numbers(self):
        ok, _ = hand_logic.validate_name("pose1")
        assert ok is True

    def test_empty_string(self):
        ok, msg = hand_logic.validate_name("")
        assert ok is False
        assert "empty" in msg.lower()

    def test_whitespace_only(self):
        ok, msg = hand_logic.validate_name("   ")
        assert ok is False
        assert "empty" in msg.lower()

    def test_too_long(self):
        ok, msg = hand_logic.validate_name("x" * 51)
        assert ok is False
        assert "long" in msg.lower()

    def test_exactly_50_chars_valid(self):
        ok, _ = hand_logic.validate_name("x" * 50)
        assert ok is True

    def test_forbidden_colon(self):
        ok, msg = hand_logic.validate_name("pose:1")
        assert ok is False
        assert ":" in msg

    def test_forbidden_brace(self):
        ok, _ = hand_logic.validate_name("pose{x}")
        assert ok is False

    def test_forbidden_bracket(self):
        ok, _ = hand_logic.validate_name("pose[0]")
        assert ok is False

    def test_forbidden_single_quote(self):
        ok, _ = hand_logic.validate_name("it's")
        assert ok is False

    def test_forbidden_double_quote(self):
        ok, _ = hand_logic.validate_name('say "hi"')
        assert ok is False

    def test_forbidden_hash(self):
        ok, _ = hand_logic.validate_name("pose#1")
        assert ok is False

    def test_forbidden_dash(self):
        ok, _ = hand_logic.validate_name("pose-1")
        assert ok is False

    def test_control_character(self):
        ok, msg = hand_logic.validate_name("bad\x01name")
        assert ok is False
        assert "control" in msg.lower()

    def test_leading_space_stripped_silently(self):
        """Leading spaces are stripped — name is accepted as valid."""
        ok, _ = hand_logic.validate_name(" leading")
        assert ok is True

    def test_trailing_space_stripped_silently(self):
        """Trailing spaces are stripped — name is accepted as valid."""
        ok, _ = hand_logic.validate_name("trailing ")
        assert ok is True


# ===================================================================
# load_config / save_config
# ===================================================================

class TestLoadConfig:

    def test_missing_file_returns_empty_structure(self, logic_paths):
        data = hand_logic.load_config()
        assert data == {'poses': {}, 'sequences': {}}

    def test_present_file_reads_poses(self, logic_paths):
        cf, _, _ = logic_paths
        cf.write_text(
            "poses:\n"
            "  fist:\n"
            "    positions: [110, 110, 110, 110, 110, 110, 110, 110]\n"
        )
        data = hand_logic.load_config()
        assert 'fist' in data['poses']

    def test_missing_poses_key_filled(self, logic_paths):
        cf, _, _ = logic_paths
        cf.write_text("sequences:\n  seq1:\n    steps: []\n")
        data = hand_logic.load_config()
        assert data['poses'] == {}

    def test_missing_sequences_key_filled(self, logic_paths):
        cf, _, _ = logic_paths
        cf.write_text("poses:\n  p1:\n    positions: [0,0,0,0,0,0,0,0]\n")
        data = hand_logic.load_config()
        assert data['sequences'] == {}

    def test_corrupt_yaml_returns_empty(self, logic_paths):
        cf, _, _ = logic_paths
        cf.write_text("this: is: {broken\n")
        data = hand_logic.load_config()
        assert data == {'poses': {}, 'sequences': {}}


class TestSaveConfig:

    def test_save_creates_file(self, logic_paths):
        cf, _, _ = logic_paths
        result = hand_logic.save_config({'poses': {}, 'sequences': {}})
        assert result is True
        assert cf.exists()

    def test_round_trip(self, logic_paths):
        """Data saved and loaded back is identical."""
        original = {
            'poses': {
                'open': {'positions': [0, 0, 0, 0, 0, 0, 0, 0]},
                'close': {'positions': [110, 110, 110, 110, 110, 110, 110, 110]},
            },
            'sequences': {},
        }
        hand_logic.save_config(original)
        loaded = hand_logic.load_config()
        assert loaded['poses']['open']['positions'] == [0, 0, 0, 0, 0, 0, 0, 0]
        assert loaded['poses']['close']['positions'] == [110] * 8

    def test_positions_formatted_inline(self, logic_paths):
        """Positions array must appear on one line, not block-style."""
        cfg = {'poses': {'p': {'positions': [0, 10, 20, 30, 40, 50, 60, 70]}}, 'sequences': {}}
        hand_logic.save_config(cfg)
        text = logic_paths[0].read_text()
        assert 'positions: [0, 10, 20, 30, 40, 50, 60, 70]' in text

    def test_save_returns_false_on_unwritable(self, logic_paths, tmp_path, monkeypatch):
        """Returns False (no exception) when file cannot be written."""
        readonly_dir = tmp_path / "readonly"
        readonly_dir.mkdir()
        readonly_dir.chmod(0o555)
        monkeypatch.setattr(hand_logic, "CONFIG_FILE", readonly_dir / "hand_config.yaml")
        monkeypatch.setattr(hand_logic, "DATA_DIR", readonly_dir)
        result = hand_logic.save_config({'poses': {}, 'sequences': {}})
        assert result is False
        readonly_dir.chmod(0o755)


# ===================================================================
# compute_auto_positions — extended edge cases
# ===================================================================

class TestComputeAutoPositionsEdgeCases:

    LIMITS = {
        'base_min': 0, 'base_max': 110,
        'side_min': -40, 'side_max': 40,
        'servo_min': -40, 'servo_max': 110,
    }
    EXTREMES = {
        'left_open': [25, -40],
        'right_open': [-40, 25],
        'left_closed': [110, 110],
        'right_closed': [110, 110],
        'center_open': [0, 0],
        'center_closed': [110, 110],
    }

    def test_full_left_closed(self):
        """side=-40, base=110 → left_closed = [110, 110]."""
        p1, p2 = hand_logic.compute_auto_positions(110, -40, self.LIMITS, self.EXTREMES)
        assert p1 == 110
        assert p2 == 110

    def test_full_right_closed(self):
        """side=+40, base=110 → right_closed = [110, 110]."""
        p1, p2 = hand_logic.compute_auto_positions(110, 40, self.LIMITS, self.EXTREMES)
        assert p1 == 110
        assert p2 == 110

    def test_base_clamped_below_zero(self):
        """base_pos < 0 clamped to base_min=0."""
        p1, p2 = hand_logic.compute_auto_positions(-10, 0, self.LIMITS, self.EXTREMES)
        assert p1 == 0
        assert p2 == 0

    def test_base_clamped_above_max(self):
        """base_pos > 110 clamped to base_max=110."""
        p1, p2 = hand_logic.compute_auto_positions(200, 0, self.LIMITS, self.EXTREMES)
        assert p1 == 110
        assert p2 == 110

    def test_side_clamped_below_min(self):
        """side_offset < -40 clamped to side_min."""
        p1_clamped, p2_clamped = hand_logic.compute_auto_positions(0, -100, self.LIMITS, self.EXTREMES)
        p1_exact, p2_exact = hand_logic.compute_auto_positions(0, -40, self.LIMITS, self.EXTREMES)
        assert p1_clamped == p1_exact
        assert p2_clamped == p2_exact

    def test_side_clamped_above_max(self):
        """side_offset > 40 clamped to side_max."""
        p1_clamped, p2_clamped = hand_logic.compute_auto_positions(0, 100, self.LIMITS, self.EXTREMES)
        p1_exact, p2_exact = hand_logic.compute_auto_positions(0, 40, self.LIMITS, self.EXTREMES)
        assert p1_clamped == p1_exact
        assert p2_clamped == p2_exact

    def test_half_left_mid_base(self):
        """Partial left, partial closed — within servo bounds."""
        p1, p2 = hand_logic.compute_auto_positions(55, -20, self.LIMITS, self.EXTREMES)
        assert -40 <= p1 <= 110
        assert -40 <= p2 <= 110

    def test_output_types_are_int(self):
        p1, p2 = hand_logic.compute_auto_positions(50, 10, self.LIMITS, self.EXTREMES)
        assert isinstance(p1, int)
        assert isinstance(p2, int)


# ===================================================================
# decompose_servo_positions — extended edge cases
# ===================================================================

class TestDecomposeServoPositionsEdgeCases:

    LIMITS = {
        'servo_min': -40, 'servo_max': 110,
        'side_min': -40, 'side_max': 40,
    }

    def test_pos2_greater_gives_positive_side(self):
        """pos2 > pos1 → right direction → positive side offset."""
        base, side = hand_logic.decompose_servo_positions(40, 70, self.LIMITS)
        assert base == 55
        assert side == 15   # 55 - 40 = 15 (positive = right)

    def test_left_direction_is_negative(self):
        """pos1 > pos2 → left direction → negative side offset."""
        base, side = hand_logic.decompose_servo_positions(70, 40, self.LIMITS)
        assert side < 0

    def test_right_direction_is_positive(self):
        """pos2 > pos1 → right direction → positive side offset."""
        base, side = hand_logic.decompose_servo_positions(40, 70, self.LIMITS)
        assert side > 0

    def test_sign_inverted_between_left_and_right(self):
        """Swapping pos1/pos2 should negate the side offset."""
        _, side_l = hand_logic.decompose_servo_positions(70, 40, self.LIMITS)
        _, side_r = hand_logic.decompose_servo_positions(40, 70, self.LIMITS)
        assert side_l == -side_r

    def test_positive_side_offset_clamped_to_side_max(self):
        """Very large positive offset (pos1 very low, pos2 very high) is clamped."""
        base, side = hand_logic.decompose_servo_positions(-40, 110, self.LIMITS)
        assert -40 <= side <= 40

    def test_base_is_midpoint(self):
        """Base is always the midpoint of pos1 and pos2."""
        base, _ = hand_logic.decompose_servo_positions(30, 80, self.LIMITS)
        assert base == (30 + 80) // 2

    def test_both_at_servo_min(self):
        base, side = hand_logic.decompose_servo_positions(-40, -40, self.LIMITS)
        assert base == -40
        assert side == 0

    def test_both_at_servo_max(self):
        base, side = hand_logic.decompose_servo_positions(110, 110, self.LIMITS)
        assert base == 110
        assert side == 0


# ===================================================================
# get_time_window_indices — extended edge cases
# ===================================================================

class TestGetTimeWindowIndicesEdgeCases:

    def test_zoom_below_min_clamped(self):
        """Zoom < 0.05 clamped — window is at least 2 points."""
        start, end = hand_logic.get_time_window_indices(100, 0.001, 0.0)
        assert end - start >= 2

    def test_pan_below_zero_clamped_to_start(self):
        start, end = hand_logic.get_time_window_indices(100, 0.5, -1.0)
        assert start == 0

    def test_total_equals_one(self):
        """Single data point — window is the whole dataset."""
        start, end = hand_logic.get_time_window_indices(1, 1.0, 0.0)
        assert start == 0
        assert end == 1

    def test_window_never_exceeds_total(self):
        """Window size cannot exceed total."""
        start, end = hand_logic.get_time_window_indices(10, 2.0, 0.0)
        assert end - start <= 10

    def test_full_pan_at_full_zoom_still_shows_all(self):
        start, end = hand_logic.get_time_window_indices(50, 1.0, 1.0)
        assert start == 0
        assert end == 50


# ===================================================================
# load_to_percent — extended edge cases
# ===================================================================

class TestLoadToPercentEdgeCases:

    def test_negative_large_clamped_at_minus_150(self):
        result = hand_logic.load_to_percent(-2000.0)
        assert result == pytest.approx(-150.0)

    def test_boundary_exactly_1_5(self):
        """Magnitude == 1.5 uses the *100 path → 150%."""
        result = hand_logic.load_to_percent(1.5)
        assert result == pytest.approx(150.0)

    def test_boundary_above_1_5(self):
        """Magnitude just above 1.5 uses /10.23 path."""
        result = hand_logic.load_to_percent(1.51)
        # 1.51 / 10.23 ≈ 0.1476 → well below 150
        assert 0 < result < 150.0

    def test_integer_input(self):
        result = hand_logic.load_to_percent(1)
        assert result == pytest.approx(100.0)


# ===================================================================
# format_feedback_value — extended edge cases
# ===================================================================

class TestFormatFeedbackValueEdgeCases:

    def test_status_high_byte_masked(self):
        """Status must mask to 0xFF."""
        assert hand_logic.format_feedback_value('status', 256) == '0x00'

    def test_status_lower_nibble(self):
        assert hand_logic.format_feedback_value('status', 16) == '0x10'

    def test_unknown_key_with_float(self):
        """Unknown key falls back to str()."""
        assert hand_logic.format_feedback_value('foo', 3.14) == '3.14'

    def test_load_negative_value(self):
        """Negative load returns negative percent string."""
        result = hand_logic.format_feedback_value('load', -1.0)
        assert result.startswith('-')
