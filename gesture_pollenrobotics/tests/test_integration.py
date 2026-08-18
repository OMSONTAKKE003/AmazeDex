"""
Integration tests: CLI commands with a mocked hardware controller.

These tests verify end-to-end behaviour of cmd_pose and cmd_sequence
without requiring physical servo hardware.  The Scs0009PyController is
replaced by a MagicMock so we can assert on the exact calls made.
"""
import pytest
from pathlib import Path
from unittest.mock import MagicMock

import yaml

import amazing_hand_cmd as cmd
import amazing_hand_gui as gui
import hand_logic


# ---------------------------------------------------------------------------
# Shared test data
# ---------------------------------------------------------------------------

SAMPLE_CONFIG = {
    "poses": {
        "open": {"positions": [0] * 8},
        "close": {"positions": [110] * 8},
        "half": {"positions": [55] * 8},
    },
    "sequences": {
        "demo": {
            "steps": [
                "open:3,3,3,3,3,3,3,3|0.1s",
                "close:3,3,3,3,3,3,3,3|0.1s",
            ]
        },
        "with_sleep": {
            "steps": [
                "open:3,3,3,3,3,3,3,3|0.1s",
                "SLEEP:0.1s",
                "close:3,3,3,3,3,3,3,3|0.1s",
            ]
        },
        "mixed_speeds": {
            "steps": [
                "open:1,1,1,1,6,6,3,3|0.1s",
                "close:6,6,6,6,1,1,3,3|0.1s",
            ]
        },
    },
}


@pytest.fixture()
def ctrl():
    return MagicMock()


# ---------------------------------------------------------------------------
# cmd_pose integration
# ---------------------------------------------------------------------------

class TestCmdPose:
    def test_applies_known_pose_open(self, ctrl):
        cmd.cmd_pose(ctrl, SAMPLE_CONFIG, "open")
        ctrl.sync_write_goal_position.assert_called_once()

    def test_applies_known_pose_close(self, ctrl):
        cmd.cmd_pose(ctrl, SAMPLE_CONFIG, "close")
        ctrl.sync_write_goal_position.assert_called_once()

    def test_speeds_sent_to_all_8_servos(self, ctrl):
        cmd.cmd_pose(ctrl, SAMPLE_CONFIG, "open")
        assert ctrl.write_goal_speed.call_count == 8

    def test_unknown_pose_exits_nonzero(self, ctrl):
        with pytest.raises(SystemExit):
            cmd.cmd_pose(ctrl, SAMPLE_CONFIG, "nonexistent_pose")

    def test_close_positions_passed_correctly(self, ctrl):
        """Verify that 110° positions are sent for the 'close' pose."""
        import math
        cmd.cmd_pose(ctrl, SAMPLE_CONFIG, "close")
        _, rads = ctrl.sync_write_goal_position.call_args[0]
        # Servo 1 (odd, first in SERVO_PAIRS) → rad(110)
        assert rads[0] == pytest.approx(math.radians(110))
        # Servo 2 (even) → rad(-110)
        assert rads[1] == pytest.approx(math.radians(-110))


# ---------------------------------------------------------------------------
# cmd_sequence integration
# ---------------------------------------------------------------------------

class TestCmdSequence:
    def test_all_pose_steps_executed(self, ctrl):
        cmd.cmd_sequence(ctrl, SAMPLE_CONFIG, "demo", loop=False)
        # 2 pose steps → 2 sync_write_goal_position calls
        assert ctrl.sync_write_goal_position.call_count == 2

    def test_sleep_step_does_not_trigger_hardware(self, ctrl):
        cmd.cmd_sequence(ctrl, SAMPLE_CONFIG, "with_sleep", loop=False)
        # 2 pose steps + 1 SLEEP → still only 2 hardware calls
        assert ctrl.sync_write_goal_position.call_count == 2

    def test_unknown_sequence_exits_nonzero(self, ctrl):
        with pytest.raises(SystemExit):
            cmd.cmd_sequence(ctrl, SAMPLE_CONFIG, "nonexistent_seq", loop=False)

    def test_unknown_pose_in_sequence_skipped(self, ctrl, capsys):
        config = {
            "poses": {"open": {"positions": [0] * 8}},
            "sequences": {
                "partial": {
                    "steps": [
                        "open:3,3,3,3,3,3,3,3|0.1s",
                        "ghost:3,3,3,3,3,3,3,3|0.1s",  # unknown pose
                    ]
                }
            },
        }
        cmd.cmd_sequence(ctrl, config, "partial", loop=False)
        # Only 1 valid pose step executed
        assert ctrl.sync_write_goal_position.call_count == 1
        out = capsys.readouterr().out
        assert "WARNING" in out or "ghost" in out

    def test_mixed_speeds_forwarded_to_controller(self, ctrl):
        cmd.cmd_sequence(ctrl, SAMPLE_CONFIG, "mixed_speeds", loop=False)
        # 2 pose steps × 8 servos = 16 write_goal_speed calls
        assert ctrl.write_goal_speed.call_count == 16

    def test_empty_sequence_exits_nonzero(self, ctrl):
        config = {
            "poses": {"open": {"positions": [0] * 8}},
            "sequences": {"empty_seq": {"steps": []}},
        }
        # The CLI rejects sequences with no steps
        with pytest.raises(SystemExit):
            cmd.cmd_sequence(ctrl, config, "empty_seq", loop=False)
        ctrl.sync_write_goal_position.assert_not_called()


# ---------------------------------------------------------------------------
# Config round-trip: GUI writes, CLI reads
# ---------------------------------------------------------------------------

class TestConfigRoundTrip:
    def test_gui_save_cmd_read(self, tmp_path, monkeypatch):
        """Pose saved by the GUI can be read back by the CLI's load_config."""
        cf = tmp_path / "hand_config.yaml"
        monkeypatch.setattr(hand_logic, "CONFIG_FILE", cf)
        monkeypatch.setattr(hand_logic, "DATA_DIR", tmp_path)

        original = {
            "poses": {
                "open": {"positions": [0] * 8},
                "close": {"positions": [110] * 8},
            },
            "sequences": {
                "demo": {"steps": ["open:3,3,3,3,3,3,3,3|1.0s"]}
            },
        }
        success = gui.save_config(original)
        assert success

        loaded = cmd.load_config(cf)
        assert set(loaded["poses"].keys()) == {"open", "close"}
        assert loaded["poses"]["open"]["positions"] == [0] * 8
        assert loaded["poses"]["close"]["positions"] == [110] * 8
        assert "demo" in loaded["sequences"]

    def test_negative_positions_survive_round_trip(self, tmp_path, monkeypatch):
        cf = tmp_path / "hand_config.yaml"
        monkeypatch.setattr(hand_logic, "CONFIG_FILE", cf)
        monkeypatch.setattr(hand_logic, "DATA_DIR", tmp_path)

        positions = [-40, -10, 0, 10, 20, 30, 40, 50]
        gui.save_config({"poses": {"neg": {"positions": positions}}, "sequences": {}})

        loaded = cmd.load_config(cf)
        assert loaded["poses"]["neg"]["positions"] == positions

    def test_sequence_steps_survive_round_trip(self, tmp_path, monkeypatch):
        cf = tmp_path / "hand_config.yaml"
        monkeypatch.setattr(hand_logic, "CONFIG_FILE", cf)
        monkeypatch.setattr(hand_logic, "DATA_DIR", tmp_path)

        steps = [
            "open:3,3,3,3,3,3,3,3|2.0s",
            "SLEEP:1.0s",
            "close:6,6,6,6,6,6,6,6|2.0s",
        ]
        gui.save_config({
            "poses": {"open": {"positions": [0] * 8}, "close": {"positions": [110] * 8}},
            "sequences": {"demo": {"steps": steps}},
        })

        loaded = cmd.load_config(cf)
        assert loaded["sequences"]["demo"]["steps"] == steps
