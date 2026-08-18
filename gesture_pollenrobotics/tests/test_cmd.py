"""
Unit tests for pure functions in amazing_hand_cmd.py.

No hardware connection is needed; hardware calls are exercised
via a MagicMock controller object.
"""
import math
import io
import time
import pytest
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import numpy as np
import yaml

import amazing_hand_cmd as cmd


# ---------------------------------------------------------------------------
# angle_rad
# ---------------------------------------------------------------------------

class TestAngleRad:
    def test_odd_servo_id_positive_angle(self):
        assert cmd.angle_rad(1, 90) == pytest.approx(math.radians(90))

    def test_even_servo_id_negated(self):
        assert cmd.angle_rad(2, 90) == pytest.approx(math.radians(-90))

    def test_zero_degrees_odd(self):
        assert cmd.angle_rad(1, 0) == pytest.approx(0.0)

    def test_zero_degrees_even(self):
        assert cmd.angle_rad(2, 0) == pytest.approx(0.0)

    def test_negative_degrees_odd_preserved(self):
        assert cmd.angle_rad(3, -30) == pytest.approx(math.radians(-30))

    def test_negative_degrees_even_inverted(self):
        # Even ID negates the angle, so -30 → rad(30) = positive
        assert cmd.angle_rad(4, -30) == pytest.approx(math.radians(30))

    def test_large_angle_odd(self):
        assert cmd.angle_rad(7, 110) == pytest.approx(math.radians(110))

    def test_large_angle_even(self):
        assert cmd.angle_rad(8, 110) == pytest.approx(math.radians(-110))


# ---------------------------------------------------------------------------
# parse_step
# ---------------------------------------------------------------------------

class TestParseStep:
    # --- SLEEP steps ---

    def test_sleep_with_s_suffix(self):
        assert cmd.parse_step("SLEEP:2.0s") == ("sleep", 2.0)

    def test_sleep_without_suffix(self):
        assert cmd.parse_step("SLEEP:1.5") == ("sleep", 1.5)

    def test_sleep_uppercase_S_suffix(self):
        assert cmd.parse_step("SLEEP:3.0S") == ("sleep", 3.0)

    def test_sleep_lowercase(self):
        # case-insensitive startswith check
        assert cmd.parse_step("sleep:0.5s") == ("sleep", 0.5)

    # --- Pose steps ---

    def test_pose_with_speeds_and_delay(self):
        kind, name, speeds, delay = cmd.parse_step("open:3,3,3,3,3,3,3,3|2.0s")
        assert kind == "pose"
        assert name == "open"
        assert speeds == [3] * 8
        assert delay == pytest.approx(2.0)

    def test_pose_with_speeds_no_delay(self):
        kind, name, speeds, delay = cmd.parse_step("open:3,3,3,3,3,3,3,3")
        assert kind == "pose"
        assert name == "open"
        assert speeds == [3] * 8
        assert delay is None

    def test_pose_no_speeds_no_delay(self):
        """Old format: bare pose name."""
        kind, name, speeds, delay = cmd.parse_step("open")
        assert kind == "pose"
        assert name == "open"
        assert speeds == [3] * 8   # defaults
        assert delay is None

    def test_pose_no_speeds_with_delay(self):
        """Old format: pose_name|delay."""
        kind, name, speeds, delay = cmd.parse_step("open|1.5s")
        assert kind == "pose"
        assert name == "open"
        assert speeds == [3] * 8
        assert delay == pytest.approx(1.5)

    def test_speeds_padded_when_fewer_than_8(self):
        _, _, speeds, _ = cmd.parse_step("close:5,5|1.0s")
        assert speeds == [5, 5, 3, 3, 3, 3, 3, 3]

    def test_speeds_truncated_when_more_than_8(self):
        _, _, speeds, _ = cmd.parse_step("close:1,2,3,4,5,6,1,2,9,9")
        assert len(speeds) == 8
        assert speeds == [1, 2, 3, 4, 5, 6, 1, 2]

    def test_delay_without_s_suffix(self):
        _, _, _, delay = cmd.parse_step("open:3,3,3,3,3,3,3,3|2.0")
        assert delay == pytest.approx(2.0)

    def test_pose_name_stripped_of_whitespace(self):
        _, name, _, _ = cmd.parse_step("  open:3,3,3,3,3,3,3,3|1.0s  ")
        assert name == "open"

    def test_mixed_speeds(self):
        _, _, speeds, _ = cmd.parse_step("wave:1,1,1,1,6,6,3,3|0.5s")
        assert speeds == [1, 1, 1, 1, 6, 6, 3, 3]


# ---------------------------------------------------------------------------
# cmd_list output
# ---------------------------------------------------------------------------

class TestCmdList:
    def test_empty_config_prints_none_for_both(self, capsys):
        cmd.cmd_list({"poses": {}, "sequences": {}})
        out = capsys.readouterr().out
        assert out.count("(none)") >= 2

    def test_pose_names_appear_in_output(self, capsys):
        config = {
            "poses": {
                "open": {"positions": [0] * 8},
                "close": {"positions": [110] * 8},
            },
            "sequences": {},
        }
        cmd.cmd_list(config)
        out = capsys.readouterr().out
        assert "open" in out
        assert "close" in out

    def test_sequence_names_appear_in_output(self, capsys):
        config = {
            "poses": {"open": {"positions": [0] * 8}},
            "sequences": {"demo": {"steps": ["open:3,3,3,3,3,3,3,3|2.0s"]}},
        }
        cmd.cmd_list(config)
        out = capsys.readouterr().out
        assert "demo" in out

    def test_step_count_shown_for_sequence(self, capsys):
        config = {
            "poses": {"open": {"positions": [0] * 8}},
            "sequences": {
                "demo": {"steps": ["open:3,3,3,3,3,3,3,3|1.0s", "open:3,3,3,3,3,3,3,3|1.0s"]}
            },
        }
        cmd.cmd_list(config)
        out = capsys.readouterr().out
        assert "2 steps" in out


# ---------------------------------------------------------------------------
# load_config in amazing_hand_cmd
# ---------------------------------------------------------------------------

class TestCmdLoadConfig:
    def test_valid_yaml_loaded(self, tmp_path):
        cf = tmp_path / "test.yaml"
        cf.write_text(
            "poses:\n"
            "  open:\n"
            "    positions: [0, 0, 0, 0, 0, 0, 0, 0]\n"
        )
        config = cmd.load_config(cf)
        assert "open" in config["poses"]

    def test_missing_file_calls_sys_exit(self, tmp_path):
        with pytest.raises(SystemExit):
            cmd.load_config(tmp_path / "nonexistent.yaml")


# ---------------------------------------------------------------------------
# apply_pose (with mock controller)
# ---------------------------------------------------------------------------

class TestApplyPose:
    @pytest.fixture()
    def ctrl(self):
        return MagicMock()

    def test_write_goal_speed_called_for_all_8_servos(self, ctrl):
        cmd.apply_pose(ctrl, [0] * 8, [3] * 8)
        assert ctrl.write_goal_speed.call_count == 8

    def test_sync_write_goal_position_called_once(self, ctrl):
        cmd.apply_pose(ctrl, [0] * 8, [3] * 8)
        ctrl.sync_write_goal_position.assert_called_once()

    def test_all_8_servo_ids_passed_to_sync_write(self, ctrl):
        cmd.apply_pose(ctrl, [0] * 8, [3] * 8)
        ids, _ = ctrl.sync_write_goal_position.call_args[0]
        assert sorted(ids) == list(range(1, 9))

    def test_odd_servo_angles_positive(self, ctrl):
        # All positions = 90 degrees
        positions = [90] * 8
        cmd.apply_pose(ctrl, positions, [3] * 8)
        _, rads = ctrl.sync_write_goal_position.call_args[0]
        # rads order follows SERVO_PAIRS = [(5,6),(3,4),(1,2),(7,8)]
        # index 0 → servo 5 (odd) → +rad(90)
        assert rads[0] == pytest.approx(math.radians(90))

    def test_even_servo_angles_negated(self, ctrl):
        positions = [90] * 8
        cmd.apply_pose(ctrl, positions, [3] * 8)
        _, rads = ctrl.sync_write_goal_position.call_args[0]
        # index 1 → servo 6 (even) → -rad(90)
        assert rads[1] == pytest.approx(math.radians(-90))

    def test_speeds_applied_per_servo(self, ctrl):
        # speeds are in finger order matching SERVO_PAIRS: [(5,6),(3,4),(1,2),(7,8)]
        # speeds[0]→servo5, speeds[1]→servo6, speeds[2]→servo3, speeds[3]→servo4,
        # speeds[4]→servo1, speeds[5]→servo2, speeds[6]→servo7, speeds[7]→servo8
        speeds = [1, 2, 3, 4, 5, 6, 1, 2]
        cmd.apply_pose(ctrl, [0] * 8, speeds)
        speed_calls = [c.args for c in ctrl.write_goal_speed.call_args_list]
        servo_to_speed = {servo_id: spd for servo_id, spd in speed_calls}
        assert servo_to_speed[5] == 1   # speeds[0]
        assert servo_to_speed[6] == 2   # speeds[1]
        assert servo_to_speed[3] == 3   # speeds[2]
        assert servo_to_speed[4] == 4   # speeds[3]
        assert servo_to_speed[1] == 5   # speeds[4]
        assert servo_to_speed[2] == 6   # speeds[5]
        assert servo_to_speed[7] == 1   # speeds[6]
        assert servo_to_speed[8] == 2   # speeds[7]


# ---------------------------------------------------------------------------
# _interruptible_sleep (FR-SEQ-2 AC 2.4)
# ---------------------------------------------------------------------------

class TestInterruptibleSleep:
    def test_completes_on_duration(self):
        """AC 2.4: Sleep completes after requested duration."""
        stop_flag = [False]
        start = time.monotonic()
        cmd._interruptible_sleep(0.3, stop_flag)
        elapsed = time.monotonic() - start
        assert elapsed >= 0.3

    def test_stops_on_flag(self):
        """AC 2.4: Sleep terminates early when stop flag is set."""
        stop_flag = [True]
        start = time.monotonic()
        cmd._interruptible_sleep(10.0, stop_flag)
        elapsed = time.monotonic() - start
        assert elapsed < 1.0


# ---------------------------------------------------------------------------
# wait_for_motion (FR-SEQ-2 AC 2.3)
# ---------------------------------------------------------------------------

class TestWaitForMotion:
    """AC 2.3: wait_for_motion exits as soon as all servos stop moving."""

    @pytest.fixture()
    def ctrl(self):
        return MagicMock()

    def test_exits_when_all_servos_idle(self, ctrl):
        """Returns immediately once all moving flags are 0."""
        ctrl.read_moving.return_value = 0
        start = time.monotonic()
        cmd.wait_for_motion(ctrl, timeout=5.0)
        # Should finish well before the timeout
        assert time.monotonic() - start < 3.0

    def test_waits_while_any_servo_moving(self, ctrl):
        """Stays in loop while at least one servo reports moving=1."""
        call_count = [0]
        def _moving(sid):
            call_count[0] += 1
            # Report moving for the first few polls, then idle
            return 1 if call_count[0] < 5 else 0
        ctrl.read_moving.side_effect = _moving
        cmd.wait_for_motion(ctrl, timeout=5.0)
        # Must have polled more than once
        assert ctrl.read_moving.call_count > 1

    def test_timeout_respected(self, ctrl):
        """Hard timeout fires when servos never stop moving."""
        ctrl.read_moving.return_value = 1  # always moving
        start = time.monotonic()
        cmd.wait_for_motion(ctrl, timeout=0.5)
        elapsed = time.monotonic() - start
        # Should stop around 0.5 s, not hang
        assert elapsed < 2.0

    def test_read_exception_does_not_raise(self, ctrl):
        """A read failure is swallowed and we eventually hit the timeout."""
        ctrl.read_moving.side_effect = RuntimeError("serial error")
        # Should not propagate the exception
        cmd.wait_for_motion(ctrl, timeout=0.4)


# ---------------------------------------------------------------------------
# FR-CLI-5: Mutually exclusive actions
# ---------------------------------------------------------------------------

class TestCmdMutualExclusion:
    """AC 5.1, 5.2: Argument validation beyond what test_system covers."""

    def test_loop_without_sequence_error(self):
        """AC 5.2: --loop without --sequence → error."""
        import subprocess, sys
        result = subprocess.run(
            [sys.executable, str(Path(__file__).resolve().parent.parent / "amazing_hand_cmd.py"),
             "--list", "--loop"],
            capture_output=True, text=True,
        )
        assert result.returncode != 0


# ---------------------------------------------------------------------------
# FR-CLI-6: Config file override
# ---------------------------------------------------------------------------

class TestConfigFileOverride:
    """AC 6.1: Missing file → error + sys.exit(1)."""

    def test_missing_config_exits(self, tmp_path):
        missing = tmp_path / "no_such.yaml"
        with pytest.raises(SystemExit):
            cmd.load_config(missing)

    def test_valid_config_loads(self, tmp_path):
        cf = tmp_path / "cfg.yaml"
        cf.write_text("poses:\n  test:\n    positions: [0,0,0,0,0,0,0,0]\nsequences: {}\n")
        config = cmd.load_config(cf)
        assert "test" in config["poses"]
