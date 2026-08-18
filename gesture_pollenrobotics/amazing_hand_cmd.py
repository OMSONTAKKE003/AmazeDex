#!/usr/bin/env python3
"""
amazing_hand_cmd.py – Command-line interface for AmazingHand servo controller.

Apply a saved pose or play a saved sequence from data/hand_config.yaml without
launching the full GUI.

Usage examples
--------------
# List all available poses and sequences:
    python amazing_hand_cmd.py --list

# Apply a single pose:
    python amazing_hand_cmd.py --pose open

# Play a sequence once:
    python amazing_hand_cmd.py --sequence demo

# Play a sequence in a loop (Ctrl+C to stop):
    python amazing_hand_cmd.py --sequence wave --loop

# Override serial port / baudrate:
    python amazing_hand_cmd.py --pose close --port /dev/ttyUSB0 --baudrate 1000000

# Set servo speed (1=slow … 6=fast, default 3):
    python amazing_hand_cmd.py --pose open --speed 6
    python amazing_hand_cmd.py --sequence demo --speed 2

# Use an alternative config file:
    python amazing_hand_cmd.py --list --config /path/to/hand_config.yaml
"""

import argparse
import signal
import sys
import time
from pathlib import Path

import numpy as np
import yaml

try:
    from rustypot import Scs0009PyController
except ImportError:
    print("ERROR: rustypot library not found. Install it with: pip install rustypot")
    sys.exit(1)

from hand_logic import (
    CONFIG_FILE, FINGER_NAMES, SERVO_PAIRS,
    DEFAULT_PORT_LINUX, DEFAULT_PORT_WINDOWS, DEFAULT_BAUDRATE,
    angle_rad, coerce_bool,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_CONFIG = CONFIG_FILE


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def load_config(config_path: Path) -> dict:
    """Load and return the YAML config."""
    if not config_path.exists():
        print(f"ERROR: Config file not found: {config_path}")
        sys.exit(1)
    with open(config_path, "r") as fh:
        return yaml.safe_load(fh) or {}


# ---------------------------------------------------------------------------
# Hardware helpers
# ---------------------------------------------------------------------------

def connect(port: str, baudrate: int) -> Scs0009PyController:
    """Open serial connection and enable torque on all 8 servos."""
    print(f"Connecting to {port} at {baudrate} baud …")
    try:
        ctrl = Scs0009PyController(
            serial_port=port,
            baudrate=baudrate,
            timeout=0.5,
        )
    except Exception as exc:
        print(f"ERROR: Could not open port {port}: {exc}")
        sys.exit(1)

    for servo_id in range(1, 9):
        ctrl.write_torque_enable(servo_id, 1)

    print("Connected.")
    return ctrl


def apply_pose(ctrl: Scs0009PyController, positions: list[int], speeds: list[int]) -> None:
    """
    Send speed + position commands to all 8 servos.

    Parameters
    ----------
    positions : list of 8 ints  – degrees for each servo (index 0→servo1 … 7→servo8)
    speeds    : list of 8 ints  – speed value (1-6) per servo
    """
    servo_ids = []
    positions_rad = []

    for finger_idx, (s1, s2) in enumerate(SERVO_PAIRS):
        pos1 = positions[finger_idx * 2]
        pos2 = positions[finger_idx * 2 + 1]
        spd1 = speeds[finger_idx * 2]
        spd2 = speeds[finger_idx * 2 + 1]

        ctrl.write_goal_speed(s1, spd1)
        ctrl.write_goal_speed(s2, spd2)

        servo_ids.append(s1)
        servo_ids.append(s2)
        positions_rad.append(angle_rad(s1, pos1))
        positions_rad.append(angle_rad(s2, pos2))

    ctrl.sync_write_goal_position(servo_ids, positions_rad)


def wait_for_motion(ctrl: Scs0009PyController, timeout: float = 30.0) -> None:
    """
    Block until all 8 servos stop moving or `timeout` seconds elapse.

    Polls each servo's moving flag every 0.1 s.  A 0.3 s startup delay
    is added so servos have time to actually begin moving before the first
    poll, avoiding a false "already idle" result.
    """
    servo_ids = list(range(1, 9))
    time.sleep(0.3)  # let motion begin
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            still_moving = any(
                coerce_bool(ctrl.read_moving(sid))
                for sid in servo_ids
            )
        except Exception:
            # If reading fails, fall back to waiting out the remaining time
            time.sleep(min(1.0, deadline - time.monotonic()))
            continue
        if not still_moving:
            return
        time.sleep(0.1)




def parse_step(step: str) -> tuple | None:
    """
    Parse a sequence step string.

    Returns one of:
      ('pose',  pose_name, speeds_list, delay_seconds)
      ('sleep', delay_seconds)
      None  – if parsing fails
    """
    step = step.strip()

    # --- SLEEP step ---
    if step.upper().startswith("SLEEP:"):
        raw = step.split(":", 1)[1].rstrip("sS")
        try:
            return ("sleep", float(raw))
        except ValueError:
            print(f"  WARNING: Cannot parse SLEEP duration in '{step}', skipping.")
            return None

    # --- Pose step: pose_name[:s1,s2,...,s8][|delay] ---
    delay = None
    if "|" in step:
        pose_part, delay_part = step.split("|", 1)
        try:
            delay = float(delay_part.rstrip("sS"))
        except ValueError:
            print(f"  WARNING: Cannot parse delay in '{step}', using no delay.")
    else:
        pose_part = step

    if ":" in pose_part:
        pose_name, speeds_str = pose_part.split(":", 1)
        try:
            speeds = [int(s) for s in speeds_str.split(",")]
        except ValueError:
            print(f"  WARNING: Cannot parse speeds in '{step}', using defaults.")
            speeds = [3] * 8
    else:
        pose_name = pose_part
        speeds = [3] * 8

    # Pad / truncate speeds to exactly 8 values
    if len(speeds) < 8:
        speeds = speeds + [3] * (8 - len(speeds))
    else:
        speeds = speeds[:8]

    return ("pose", pose_name.strip(), speeds, delay)


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def cmd_list(config: dict) -> None:
    """Print all available poses and sequences."""
    poses = config.get("poses", {})
    sequences = config.get("sequences", {})

    print(f"\nPoses ({len(poses)}):")
    if poses:
        for name, data in sorted(poses.items()):
            pos = data.get("positions", [])
            print(f"  {name:20s}  positions: {pos}")
    else:
        print("  (none)")

    print(f"\nSequences ({len(sequences)}):")
    if sequences:
        for seq_name, seq_data in sorted(sequences.items()):
            steps = seq_data.get("steps", [])
            print(f"  {seq_name:20s}  ({len(steps)} steps)")
            for step in steps:
                print(f"    - {step}")
    else:
        print("  (none)")
    print()


def cmd_pose(ctrl: Scs0009PyController, config: dict, pose_name: str, speed: int = 3) -> None:
    """Apply a single named pose."""
    poses = config.get("poses", {})
    if pose_name not in poses:
        print(f"ERROR: Pose '{pose_name}' not found.")
        print("Available poses:", ", ".join(sorted(poses.keys())) or "(none)")
        sys.exit(1)

    positions = poses[pose_name].get("positions", [0] * 8)
    speeds = [speed] * 8

    print(f"Applying pose '{pose_name}' at speed {speed}: {positions}")
    apply_pose(ctrl, positions, speeds)
    wait_for_motion(ctrl)


def cmd_sequence(
    ctrl: Scs0009PyController,
    config: dict,
    seq_name: str,
    loop: bool,
    speed: int = 3,
) -> None:
    """Play a named sequence, optionally looping."""
    sequences = config.get("sequences", {})
    if seq_name not in sequences:
        print(f"ERROR: Sequence '{seq_name}' not found.")
        print("Available sequences:", ", ".join(sorted(sequences.keys())) or "(none)")
        sys.exit(1)

    poses = config.get("poses", {})
    items = sequences[seq_name].get("steps", [])

    if not items:
        print(f"ERROR: Sequence '{seq_name}' has no steps.")
        sys.exit(1)

    stop_flag = [False]

    def _signal_handler(sig, frame):
        print("\nInterrupt received, stopping …", flush=True)
        stop_flag[0] = True

    signal.signal(signal.SIGINT, _signal_handler)

    iteration = 0
    while not stop_flag[0]:
        iteration += 1
        if loop:
            print(f"\n=== Loop iteration {iteration} ===")
        else:
            print("\n=== Starting sequence ===")

        for step in items:
            if stop_flag[0]:
                break

            parsed = parse_step(step)
            if parsed is None:
                continue

            if parsed[0] == "sleep":
                _, duration = parsed
                print(f"  SLEEP {duration}s …")
                _interruptible_sleep(duration, stop_flag)

            elif parsed[0] == "pose":
                _, pose_name, speeds, delay = parsed

                # If parse_step returned the default [3]*8 (no speeds embedded in
                # the step string), replace with the caller-supplied speed.
                if speeds == [3] * 8:
                    speeds = [speed] * 8

                if pose_name not in poses:
                    print(f"  WARNING: Pose '{pose_name}' not found, skipping.")
                    continue

                positions = poses[pose_name].get("positions", [0] * 8)
                speeds_display = f"{speeds[0]},{speeds[2]},{speeds[4]},{speeds[6]}"
                print(
                    f"  Pose '{pose_name}'  "
                    f"speeds=[{speeds_display},…]  "
                    f"delay={delay}s"
                )
                apply_pose(ctrl, positions, speeds)

                if delay is not None:
                    _interruptible_sleep(delay, stop_flag)
                else:
                    wait_for_motion(ctrl)

        if not loop:
            break

        if loop and not stop_flag[0]:
            time.sleep(0.5)

    print("=== Done ===")


def _interruptible_sleep(seconds: float, stop_flag: list) -> None:
    """Sleep for `seconds` in 0.1-s increments, honouring stop_flag."""
    elapsed = 0.0
    while elapsed < seconds and not stop_flag[0]:
        time.sleep(0.1)
        elapsed += 0.1


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _default_port() -> str:
    if sys.platform.startswith("win"):
        return DEFAULT_PORT_WINDOWS
    return DEFAULT_PORT_LINUX


def main() -> None:
    parser = argparse.ArgumentParser(
        description="AmazingHand command-line controller",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Action group – exactly one of these is required (except --list)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--list", action="store_true", help="List all poses and sequences")
    action.add_argument("--pose", metavar="NAME", help="Apply a named pose")
    action.add_argument("--sequence", metavar="NAME", help="Play a named sequence")

    parser.add_argument(
        "--loop",
        action="store_true",
        help="Loop the sequence until Ctrl+C (only with --sequence)",
    )
    parser.add_argument(
        "--port",
        default=_default_port(),
        help=f"Serial port (default: {_default_port()})",
    )
    parser.add_argument(
        "--baudrate",
        type=int,
        default=DEFAULT_BAUDRATE,
        help=f"Baud rate (default: {DEFAULT_BAUDRATE})",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help=f"Path to hand_config.yaml (default: {DEFAULT_CONFIG})",
    )
    parser.add_argument(
        "--speed",
        type=int,
        default=3,
        choices=range(1, 7),
        metavar="N",
        help="Servo speed 1 (slow) … 6 (fast), default 3. Overridden per-step in sequences that embed explicit speeds.",
    )

    args = parser.parse_args()

    if args.loop and not args.sequence:
        parser.error("--loop can only be used with --sequence")

    config = load_config(args.config)

    # --list does not need a hardware connection
    if args.list:
        cmd_list(config)
        return

    ctrl = connect(args.port, args.baudrate)

    try:
        if args.pose:
            cmd_pose(ctrl, config, args.pose, speed=args.speed)
        elif args.sequence:
            cmd_sequence(ctrl, config, args.sequence, loop=args.loop, speed=args.speed)
    finally:
        # Disable torque on exit so servos relax
        try:
            for servo_id in range(1, 9):
                ctrl.write_torque_enable(servo_id, 0)
        except Exception:
            pass


if __name__ == "__main__":
    main()
