from __future__ import annotations

import argparse
import json
import os
import time

import cv2
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.env_util import make_vec_env

from amazedex_cube_env import AmazeDexCubeEnv, FACENORMALS, FACENAMES
from aruco_pose import CubePoseTracker
import register_amazedex_env

# Control Parameters
CONTROL_DT = 0.02  # 50 Hz control loop
POLICY_STEPS_PER_TARGET = 250  # ~5 seconds execution per targeted face
ACTION_SMOOTHING = 0.2  # Action EMA filter parameter
SERVO_IDS = [1, 2, 3, 4, 5, 6, 7, 8]
GOAL_SPEED = 0
CALIBRATION_PATH = "hand_calibration.json"

# Motor control limits. MUST match the actuators' ctrlrange in robot.xml
# (all 8 use ctrlrange="-1.396263 1.396263"), NOT the policy's [-1, 1]
# action space - those are different scales. Previously this was left as
# [-1, 1], which meant the real hand only ever received ~1 rad of travel
# instead of the full ~1.4 rad range the policy was trained to command.
CTRL_LOW = np.array([-1.396263] * 8, dtype=np.float32)
CTRL_HIGH = np.array([1.396263] * 8, dtype=np.float32)

# Tolerances
DEADBAND_TOLERANCE_RAD = np.radians(0)  # ~3 degrees precision limit


def load_calibration(path: str = CALIBRATION_PATH) -> dict:
    if not os.path.exists(path):
        print("[WARN] No calibration file found. Using default offsets.")
        return {"offset_rad": [0.0] * 8, "sign": [1.0] * 8}
    with open(path) as f:
        return json.load(f)


class HandInterface:
    """Hardware interface to physical AmazeDex hardware via rustypot."""

    def __init__(
        self,
        serial_port: str = "COM14",
        baudrate: int = 1_000_000,
        timeout: float = 0.6,
        calibration: dict | None = None,
    ):
        from rustypot import Scs0009PyController

        self.controller = Scs0009PyController(serial_port=serial_port, baudrate=baudrate, timeout=timeout)
        for servo_id in SERVO_IDS:
            self.controller.write_torque_enable(servo_id, 1)

        self.controller.sync_write_goal_speed(SERVO_IDS, [GOAL_SPEED] * len(SERVO_IDS))

        calib = calibration or {"offset_rad": [0.0] * 8, "sign": [1.0] * 8}
        self.offset = np.asarray(calib["offset_rad"], dtype=np.float32)
        self.sign = np.asarray(calib["sign"], dtype=np.float32)

    def read_joint_positions(self) -> np.ndarray:
        raw = np.asarray(self.controller.sync_read_present_position(SERVO_IDS), dtype=np.float32)
        return self.sign * (raw - self.offset)

    def send_ctrl(self, ctrl_sim_frame: np.ndarray) -> None:
        raw = self.offset + self.sign * ctrl_sim_frame
        self.controller.sync_write_goal_position(SERVO_IDS, raw.tolist())

    def close(self) -> None:
        for servo_id in SERVO_IDS:
            self.controller.write_torque_enable(servo_id, 0)


def policy_action_to_ctrl(action: np.ndarray) -> np.ndarray:
    action = np.clip(action, -1.0, 1.0)
    return CTRL_LOW + (action + 1.0) * 0.5 * (CTRL_HIGH - CTRL_LOW)


def ctrl_to_policy_action(ctrl: np.ndarray) -> np.ndarray:
    action = 2.0 * (ctrl - CTRL_LOW) / (CTRL_HIGH - CTRL_LOW) - 1.0
    return np.clip(action, -1.0, 1.0)


def build_cube_obs(
    jpos: np.ndarray,
    jvel: np.ndarray,
    last_action: np.ndarray,
    target_face: int,
    R_cube: np.ndarray = np.eye(3),
) -> np.ndarray:
    """
    Constructs the 33-dimensional observation vector expected by the policy:
    [jpos(8), jvel(8), last_action(8), target_world_normal(3), target_onehot(6)]
    """
    target_local_normal = FACENORMALS[target_face]
    target_world_normal = R_cube @ target_local_normal

    one_hot = np.zeros(6, dtype=np.float32)
    one_hot[target_face] = 1.0

    return np.concatenate([jpos, jvel, last_action, target_world_normal, one_hot]).astype(np.float32)


def predict_action(model: PPO, vec_normalize: VecNormalize | None, obs: np.ndarray) -> np.ndarray:
    if vec_normalize is not None:
        obs = vec_normalize.normalize_obs(obs)
    action, _ = model.predict(obs, deterministic=True)
    return np.clip(action, -1.0, 1.0)


def execute_target_on_real_hand(
    model: PPO,
    vec_normalize: VecNormalize | None,
    hand: HandInterface,
    tracker: CubePoseTracker,
    target_face: int,
    cap: cv2.VideoCapture | None = None,
) -> bool:
    """Runs closed-loop policy execution on real hardware to orient the cube face up."""
    print(f"\n--- REAL HAND CUBE ROTATION START -> Target Face: {FACENAMES[target_face]} ---")

    joint_pos = hand.read_joint_positions()
    prev_joint_pos = joint_pos.copy()
    prev_action = ctrl_to_policy_action(joint_pos)
    last_sent_ctrl = policy_action_to_ctrl(prev_action)

    for _ in range(POLICY_STEPS_PER_TARGET):
        loop_start = time.perf_counter()

        # 1. Sense Real-time Kinematics
        joint_pos = hand.read_joint_positions()
        joint_vel = (joint_pos - prev_joint_pos) / CONTROL_DT

        # 2. Sense Real-time Cube Orientation via AprilTags (falls back to the
        #    last known pose if no tag is currently visible).
        frame = None
        if cap is not None and cap.isOpened():
            ok, raw_frame = cap.read()
            if ok:
                frame = cv2.flip(raw_frame, 1)
        R_cube = tracker.update(frame) if frame is not None else tracker.last_R

        # 3. Build Policy Observation
        obs = build_cube_obs(
            jpos=joint_pos,
            jvel=joint_vel,
            last_action=prev_action,
            target_face=target_face,
            R_cube=R_cube,
        )

        # 4. Predict Policy Action (with saved observation normalization, if any)
        action = predict_action(model, vec_normalize, obs)

        # 5. Smooth Action & Convert to Servo Space
        smoothed_action = ACTION_SMOOTHING * action + (1 - ACTION_SMOOTHING) * prev_action
        smoothed_action = np.clip(smoothed_action, -1.0, 1.0)
        target_ctrl = policy_action_to_ctrl(smoothed_action)

        # Deadband locking protection
        error = np.abs(target_ctrl - joint_pos)
        at_target = error < DEADBAND_TOLERANCE_RAD
        target_ctrl[at_target] = last_sent_ctrl[at_target]

        # 6. Execute Motor Commands
        hand.send_ctrl(target_ctrl)

        prev_joint_pos = joint_pos.copy()
        prev_action = smoothed_action.copy()
        last_sent_ctrl = target_ctrl.copy()

        if frame is not None:
            cv2.putText(
                frame,
                f"Targeting Face: {FACENAMES[target_face]}",
                (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )
            cv2.imshow("AmazeDex Cube Execution", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("--- EXECUTION ABORTED (quit) ---")
                return False

        while time.perf_counter() - loop_start < CONTROL_DT:
            pass

    print(f"--- TARGET REACHED / COMPLETED: Face {FACENAMES[target_face]} ---")
    return True


def run_sim_step(env, model: PPO, vec_normalize: VecNormalize | None, obs: np.ndarray, target_face: int):
    """One sim-mode step, actually driven by the trained policy (not random actions)."""
    env.unwrapped.targetface = target_face
    action = predict_action(model, vec_normalize, obs)
    return env.step(action)


def main() -> None:
    parser = argparse.ArgumentParser(description="Sim-to-Real execution script for AmazeDex Cube Target environment.")
    parser.add_argument("--model", default="models/best_model.zip", help="Path to trained PPO policy zip")
    parser.add_argument("--vecnorm", default="models/vec_normalize.pkl", help="Path to saved VecNormalize statistics")
    parser.add_argument("--mode", choices=["sim", "real"], default="sim")
    parser.add_argument("--port", default="COM14")
    parser.add_argument("--camera", type=int, default=0)
    args = parser.parse_args()

    model = PPO.load(args.model, device="cpu")
    vec_normalize = None
    if os.path.exists(args.vecnorm):
        vec_normalize = VecNormalize.load(args.vecnorm, make_vec_env("AmazeDex/CubeRotate-v0", n_envs=1))
        vec_normalize.training = False
        vec_normalize.norm_reward = False
        print("[INFO] Successfully loaded observation normalization statistics.")

    env = hand = tracker = None
    if args.mode == "sim":
        import gymnasium as gym

        env = gym.make("AmazeDex/CubeRotate-v0", render_mode="human")
        obs, _ = env.reset()
    else:
        hand = HandInterface(serial_port=args.port, calibration=load_calibration())
        tracker = CubePoseTracker()

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened() and args.mode == "real":
        print("[WARN] Could not open camera - real-hand cube-orientation sensing will be "
              "frozen at identity until a camera is available.")

    quit_requested = False
    target_face = 0  # Start with Face 1 (Top / Tag 0)

    print("\n=== Controls ===")
    print("Press '0' through '5' in terminal / GUI window to switch Target Face (1 to 6).")
    print("Press 'q' to Quit.")

    while not quit_requested:
        if args.mode == "sim":
            obs, reward, done, truncated, info = run_sim_step(env, model, vec_normalize, obs, target_face)
            if done or truncated:
                obs, _ = env.reset()
                target_face = (target_face + 1) % 6
                env.unwrapped.targetface = target_face

            env.render()
            time.sleep(CONTROL_DT)
        else:
            success = execute_target_on_real_hand(
                model=model,
                vec_normalize=vec_normalize,
                hand=hand,
                tracker=tracker,
                target_face=target_face,
                cap=cap,
            )
            if not success:
                break

            target_face = (target_face + 1) % 6

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            quit_requested = True
        elif ord("0") <= key <= ord("5"):
            target_face = key - ord("0")
            print(f"[USER ACTION] Switched Target Face to: {FACENAMES[target_face]}")

    cap.release()
    cv2.destroyAllWindows()
    if hand:
        hand.close()
    if env:
        env.close()


if __name__ == "__main__":
    main()