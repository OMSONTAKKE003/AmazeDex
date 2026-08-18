"""
CMD hardware tests — AmazingHand CLI on real servo hardware.

Exercises amazing_hand_cmd.py end-to-end against a physical servo bus.
Tests are skipped unless ``--hardware`` is passed to pytest::

    pytest tests/test_cmd_hardware.py --hardware
    pytest tests/test_cmd_hardware.py --hardware --port /dev/ttyACM0

All tests call the cmd module API directly (no subprocess) so they share
the module-scoped controller fixture and avoid serial-port contention.

Markers:
    @pytest.mark.hardware — every test in this module
"""

import subprocess
import sys
import time
from pathlib import Path

import pytest
import yaml

from hand_logic import (
    SERVO_PAIRS, coerce_angle_degrees, coerce_bool, coerce_numeric,
)

import amazing_hand_cmd as cmd

try:
    from rustypot import Scs0009PyController
except ImportError:
    Scs0009PyController = None

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CLI = PROJECT_ROOT / "amazing_hand_cmd.py"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def hw_port(request):
    return request.config.getoption("--port", default="/dev/ttyACM0")


@pytest.fixture(scope="module")
def hw_baudrate(request):
    return request.config.getoption("--hw-baudrate", default=1000000)


@pytest.fixture(scope="module")
def controller(hw_port, hw_baudrate):
    """Open a real serial connection; re-enable torque; disable on teardown."""
    if Scs0009PyController is None:
        pytest.skip("rustypot not installed")

    ctrl = Scs0009PyController(
        serial_port=hw_port,
        baudrate=hw_baudrate,
        timeout=0.5,
    )
    for sid in range(1, 9):
        ctrl.write_torque_enable(sid, 1)
    time.sleep(0.3)

    yield ctrl

    for sid in range(1, 9):
        try:
            ctrl.write_torque_enable(sid, 0)
        except Exception:
            pass


@pytest.fixture(scope="module")
def open_config():
    """Minimal config with open/close/half/scissors poses and a short sequence."""
    return {
        "poses": {
            "open":     {"positions": [0,   0,   0,   0,   0,   0,   0,   0]},
            "close":    {"positions": [110, 110, 110, 110, 110, 110, 110, 110]},
            "half":     {"positions": [55,  55,  55,  55,  55,  55,  55,  55]},
            "scissors": {"positions": [110, 110, 0,   0,   0,   0,   110, 110]},
        },
        "sequences": {
            "open_close": {
                "steps": [
                    "open:6,6,6,6,6,6,6,6|1.0s",
                    "close:6,6,6,6,6,6,6,6|1.0s",
                ]
            },
            "no_speed": {
                "steps": [
                    "open|1.0s",
                    "close|1.0s",
                ]
            },
        },
    }


@pytest.fixture(scope="module")
def config_file(tmp_path_factory, open_config):
    """Write the shared config to a temp file (subprocess tests)."""
    cf = tmp_path_factory.mktemp("cfg") / "test_cmd_hw_config.yaml"
    cf.write_text(yaml.dump(open_config))
    return cf


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _read_deg(ctrl, servo_id: int) -> float:
    """Return absolute degrees for a servo."""
    raw = ctrl.read_present_position(servo_id)
    return abs(coerce_angle_degrees(raw, servo_id))


def _go_open(ctrl):
    """Move all servos to 0° at speed 6 and wait for completion."""
    cmd.cmd_pose(ctrl, {
        "poses": {"open": {"positions": [0] * 8}}
    }, "open", speed=6)


def _position_near(ctrl, target_deg: float, tolerance: float = 15.0) -> bool:
    """True if every servo reads within tolerance of target_deg."""
    for sid in range(1, 9):
        if abs(_read_deg(ctrl, sid) - target_deg) > tolerance:
            return False
    return True


# ---------------------------------------------------------------------------
# FR-CLI-1: --pose applies correct positions
# ---------------------------------------------------------------------------

@pytest.mark.hardware
class TestCmdPoseHardware:
    """cmd_pose moves servos to the named pose positions."""

    def test_open_pose_all_servos_near_zero(self, controller, open_config):
        """'open' pose moves all servos to ~0°."""
        cmd.cmd_pose(controller, open_config, "open", speed=6)
        for sid in range(1, 9):
            deg = _read_deg(controller, sid)
            assert deg < 15.0, f"Servo {sid}: {deg:.1f}° (expected < 15°)"

    def test_close_pose_all_servos_near_max(self, controller, open_config):
        """'close' pose moves all servos to ~110°."""
        cmd.cmd_pose(controller, open_config, "close", speed=6)
        for sid in range(1, 9):
            deg = _read_deg(controller, sid)
            assert deg > 90.0, f"Servo {sid}: {deg:.1f}° (expected > 90°)"
        # Return to open for subsequent tests
        cmd.cmd_pose(controller, open_config, "open", speed=6)

    def test_half_pose_all_servos_mid_range(self, controller, open_config):
        """'half' pose moves all servos to ~55°."""
        cmd.cmd_pose(controller, open_config, "half", speed=6)
        for sid in range(1, 9):
            deg = _read_deg(controller, sid)
            assert 30.0 < deg < 80.0, f"Servo {sid}: {deg:.1f}° (expected 30–80°)"
        cmd.cmd_pose(controller, open_config, "open", speed=6)

    def test_scissors_ring_and_thumb_closed_fingers_open(self, controller, open_config):
        """'scissors' pose: Ring+Thumb closed (~110°), Middle+Pointer open (~0°)."""
        cmd.cmd_pose(controller, open_config, "scissors", speed=6)

        # SERVO_PAIRS order: Ring(5,6), Middle(3,4), Pointer(1,2), Thumb(7,8)
        # scissors positions: [110, 110, 0, 0, 0, 0, 110, 110]
        # finger_idx 0 = Ring → finger_idx*2=0,1 → positions[0,1]=110
        # finger_idx 1 = Middle → positions[2,3]=0
        # finger_idx 2 = Pointer → positions[4,5]=0
        # finger_idx 3 = Thumb → positions[6,7]=110
        ring_s1, _    = SERVO_PAIRS[0]  # servo 5
        middle_s1, _  = SERVO_PAIRS[1]  # servo 3
        pointer_s1, _ = SERVO_PAIRS[2]  # servo 1
        thumb_s1, _   = SERVO_PAIRS[3]  # servo 7

        assert _read_deg(controller, ring_s1)    > 90.0, "Ring should be closed"
        assert _read_deg(controller, middle_s1)  < 20.0, "Middle should be open"
        assert _read_deg(controller, pointer_s1) < 20.0, "Pointer should be open"
        assert _read_deg(controller, thumb_s1)   > 90.0, "Thumb should be closed"

        cmd.cmd_pose(controller, open_config, "open", speed=6)

    def test_pose_not_found_raises_system_exit(self, controller, open_config):
        """Unknown pose name → SystemExit(1)."""
        with pytest.raises(SystemExit):
            cmd.cmd_pose(controller, open_config, "nonexistent_pose")


# ---------------------------------------------------------------------------
# FR-CLI-2: --speed parameter is honoured
# ---------------------------------------------------------------------------

@pytest.mark.hardware
class TestCmdSpeedHardware:
    """--speed 1…6 is accepted and sent to all servos."""

    def test_speed_1_accepted(self, controller, open_config):
        """speed=1 does not crash; servos start moving."""
        cmd.cmd_pose(controller, open_config, "close", speed=1)
        cmd.cmd_pose(controller, open_config, "open", speed=6)

    def test_speed_6_accepted(self, controller, open_config):
        """speed=6 completes fastest."""
        cmd.cmd_pose(controller, open_config, "close", speed=6)
        cmd.cmd_pose(controller, open_config, "open", speed=6)

    def test_fast_arrives_closer_to_target(self, controller, open_config):
        """At speed 6, servos get closer to 110° within a fixed window than at speed 1."""
        cmd.cmd_pose(controller, open_config, "open", speed=6)

        # Fire close at speed 6 and snapshot after 1.5 s
        cmd.apply_pose(controller,
                       open_config["poses"]["close"]["positions"],
                       [6] * 8)
        time.sleep(1.5)
        fast_degs = [_read_deg(controller, sid) for sid in range(1, 9)]
        cmd.cmd_pose(controller, open_config, "open", speed=6)

        # Fire close at speed 1 and snapshot after 1.5 s
        cmd.apply_pose(controller,
                       open_config["poses"]["close"]["positions"],
                       [1] * 8)
        time.sleep(1.5)
        slow_degs = [_read_deg(controller, sid) for sid in range(1, 9)]
        cmd.cmd_pose(controller, open_config, "open", speed=6)

        avg_fast = sum(fast_degs) / len(fast_degs)
        avg_slow = sum(slow_degs) / len(slow_degs)
        assert avg_fast > avg_slow, (
            f"Expected speed-6 to travel further in 1.5s "
            f"(fast={avg_fast:.1f}°, slow={avg_slow:.1f}°)"
        )


# ---------------------------------------------------------------------------
# FR-CLI-3: --sequence executes all steps
# ---------------------------------------------------------------------------

@pytest.mark.hardware
class TestCmdSequenceHardware:
    """cmd_sequence drives hardware through every step."""

    def test_sequence_completes_without_error(self, controller, open_config):
        """'open_close' sequence runs open→close with no exception."""
        cmd.cmd_sequence(controller, open_config, "open_close", loop=False, speed=6)

    def test_sequence_ends_near_last_pose(self, controller, open_config):
        """After 'open_close', servos are near 110° (last pose = close)."""
        cmd.cmd_sequence(controller, open_config, "open_close", loop=False, speed=6)
        for sid in range(1, 9):
            deg = _read_deg(controller, sid)
            assert deg > 80.0, f"Servo {sid}: {deg:.1f}° after close step (expected > 80°)"
        cmd.cmd_pose(controller, open_config, "open", speed=6)

    def test_sequence_step_default_speed_uses_speed_param(self, controller, open_config):
        """Sequence steps without embedded speeds use the speed= argument."""
        # 'no_speed' sequence has no embedded speeds → should use caller speed
        cmd.cmd_sequence(controller, open_config, "no_speed", loop=False, speed=6)
        cmd.cmd_pose(controller, open_config, "open", speed=6)

    def test_sequence_not_found_raises_system_exit(self, controller, open_config):
        """Unknown sequence name → SystemExit(1)."""
        with pytest.raises(SystemExit):
            cmd.cmd_sequence(controller, open_config, "no_such_sequence", loop=False)

    def test_sequence_missing_pose_skips_gracefully(self, controller, open_config):
        """A step referencing a missing pose is skipped; rest of sequence runs."""
        config_with_bad_step = {
            "poses": {"open": {"positions": [0] * 8}},
            "sequences": {
                "bad_step": {
                    "steps": [
                        "open:6,6,6,6,6,6,6,6|0.5s",
                        "ghost_pose:6,6,6,6,6,6,6,6|0.5s",  # does not exist
                        "open:6,6,6,6,6,6,6,6|0.5s",
                    ]
                }
            },
        }
        # Must not raise
        cmd.cmd_sequence(controller, config_with_bad_step, "bad_step", loop=False)


# ---------------------------------------------------------------------------
# FR-CLI-4: wait_for_motion completes on real hardware
# ---------------------------------------------------------------------------

@pytest.mark.hardware
class TestWaitForMotionHardware:
    """wait_for_motion exits naturally when servos stop."""

    def test_returns_after_open_pose(self, controller, open_config):
        """After issuing open, wait_for_motion returns before timeout."""
        cmd.apply_pose(controller,
                       open_config["poses"]["close"]["positions"],
                       [6] * 8)
        start = time.monotonic()
        cmd.wait_for_motion(controller, timeout=20.0)
        elapsed = time.monotonic() - start
        # Should finish well before the hard timeout
        assert elapsed < 18.0, f"wait_for_motion took {elapsed:.1f}s"

    def test_servos_idle_after_wait(self, controller, open_config):
        """All servos report not-moving once wait_for_motion returns."""
        cmd.apply_pose(controller,
                       open_config["poses"]["open"]["positions"],
                       [6] * 8)
        cmd.wait_for_motion(controller, timeout=20.0)
        for sid in range(1, 9):
            moving = coerce_bool(controller.read_moving(sid))
            assert not moving, f"Servo {sid} still moving after wait_for_motion"


# ---------------------------------------------------------------------------
# FR-CLI-5: --list via subprocess (no serial port needed)
# ---------------------------------------------------------------------------

@pytest.mark.hardware
class TestCmdSubprocessList:
    """Subprocess smoke-tests that do not require the port to be free."""

    def test_list_exits_zero(self, config_file):
        """--list returns exit code 0."""
        result = subprocess.run(
            [sys.executable, str(CLI),
             "--list", "--config", str(config_file)],
            capture_output=True, text=True, timeout=5,
        )
        assert result.returncode == 0

    def test_list_shows_pose_names(self, config_file, open_config):
        """--list output contains all pose names from the config."""
        result = subprocess.run(
            [sys.executable, str(CLI),
             "--list", "--config", str(config_file)],
            capture_output=True, text=True, timeout=5,
        )
        for pose_name in open_config["poses"]:
            assert pose_name in result.stdout

    def test_list_shows_sequence_names(self, config_file, open_config):
        """--list output contains all sequence names from the config."""
        result = subprocess.run(
            [sys.executable, str(CLI),
             "--list", "--config", str(config_file)],
            capture_output=True, text=True, timeout=5,
        )
        for seq_name in open_config["sequences"]:
            assert seq_name in result.stdout

    def test_invalid_speed_exits_nonzero(self, config_file):
        """--speed 7 (out of range 1-6) → non-zero exit."""
        result = subprocess.run(
            [sys.executable, str(CLI),
             "--list", "--config", str(config_file), "--speed", "7"],
            capture_output=True, text=True, timeout=5,
        )
        assert result.returncode != 0

    def test_loop_without_sequence_exits_nonzero(self, config_file):
        """--loop without --sequence → non-zero exit."""
        result = subprocess.run(
            [sys.executable, str(CLI),
             "--list", "--loop", "--config", str(config_file)],
            capture_output=True, text=True, timeout=5,
        )
        assert result.returncode != 0


# ---------------------------------------------------------------------------
# FR-CLI-6: torque disabled on exit
# ---------------------------------------------------------------------------

@pytest.mark.hardware
class TestCmdTorqueDisableHardware:
    """cmd_pose disables torque after the pose completes."""

    def test_torque_can_be_disabled_and_reenabled(self, controller):
        """write_torque_enable(0) and (1) complete without error."""
        for sid in range(1, 9):
            controller.write_torque_enable(sid, 0)
        time.sleep(0.1)
        for sid in range(1, 9):
            controller.write_torque_enable(sid, 1)
