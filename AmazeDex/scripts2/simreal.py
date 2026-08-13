
import argparse
import dataclasses
import json
import os
import time

import cv2
import numpy as np
from stable_baselines3 import PPO

from amazedex_cube_env import (
    AmazeDexCubeEnv, FACE_NORMALS, FACE_NAMES, JOINTS, ACTUATORS,
    MODEL_PATH, JVEL_SCALE, CFG,
)
import register_amazedex_env
import mujoco

CONTROL_DT = 0.02
SERVO_IDS = [1, 2, 3, 4, 5, 6, 7, 8]

GOAL_SPEED = None
CALIBRATION_PATH = "hand_calibration.json"


class HandKinematics:
 

    def __init__(self, model_path: str = MODEL_PATH):
        model = mujoco.MjModel.from_xml_path(model_path)
        actids = np.array([model.actuator(n).id for n in ACTUATORS])
        lims = model.actuator_ctrlrange[actids]
        self.ctrl_lo, self.ctrl_hi = lims[:, 0], lims[:, 1]
        self.ctrl_mid = (self.ctrl_lo + self.ctrl_hi) / 2
        self.ctrl_half = (self.ctrl_hi - self.ctrl_lo) / 2


class ActionMapper:

    def __init__(self, kin: HandKinematics, cfg=CFG):
        self.kin = kin
        self.cfg = cfg
        self.grasp_frac = None
        self.filtered_ctrl = None

    def start(self, current_joint_pos_rad: np.ndarray) -> None:
        frac = np.clip((current_joint_pos_rad - self.kin.ctrl_mid) / self.kin.ctrl_half, -1.0, 1.0)
        self.grasp_frac = frac.copy()
        self.filtered_ctrl = frac.copy()

    def action_to_ctrl(self, action: np.ndarray) -> np.ndarray:
        action = np.clip(action, -1.0, 1.0).astype(np.float32)
        target_frac = np.clip(self.grasp_frac + action * self.cfg.grasp_band_frac, -1.0, 1.0)
        lpf_ctrl = (self.cfg.action_lpf * target_frac
                    + (1 - self.cfg.action_lpf) * self.filtered_ctrl)
        rate = self.cfg.max_ctrl_rate_frac
        step_delta = np.clip(lpf_ctrl - self.filtered_ctrl, -rate, rate)
        self.filtered_ctrl = self.filtered_ctrl + step_delta
        return self.kin.ctrl_mid + self.filtered_ctrl * self.kin.ctrl_half


def load_calibration(path: str = CALIBRATION_PATH) -> dict:
    if not os.path.exists(path):
        print("[WARN] No hand calibration file found. Using default offsets.")
        return {"offset_rad": [0.0] * 8, "sign": [1.0] * 8}
    with open(path) as f:
        return json.load(f)


class HandInterface:
    MAX_RAW_DELTA_PER_TICK = None  # set before real deployment -- see send_ctrl()

    def __init__(self, serial_port: str = "COM14", baudrate: int = 1_000_000,
                 timeout: float = 0.6, calibration: dict | None = None):
        from rustypot import Scs0009PyController

        if GOAL_SPEED is None or GOAL_SPEED <= 0:
            raise ValueError(
                "GOAL_SPEED is not configured (or is 0, which means 'unlimited speed' on "
                "Feetech/SCS-protocol servos, not 'off'). Set it from your SCS0009 datasheet "
                "before running on real hardware -- refusing to start rather than defaulting "
                "to max speed."
            )

        self.controller = Scs0009PyController(serial_port=serial_port, baudrate=baudrate, timeout=timeout)
        for servo_id in SERVO_IDS:
            self.controller.write_torque_enable(servo_id, 1)
        self.controller.sync_write_goal_speed(SERVO_IDS, [GOAL_SPEED] * len(SERVO_IDS))

        calib = calibration or {"offset_rad": [0.0] * 8, "sign": [1.0] * 8}
        self.offset = np.asarray(calib["offset_rad"], dtype=np.float32)
        self.sign = np.asarray(calib["sign"], dtype=np.float32)
        self._last_raw_cmd = None

    def read_joint_positions(self) -> np.ndarray:
        raw = np.asarray(self.controller.sync_read_present_position(SERVO_IDS), dtype=np.float32)
        return self.sign * (raw - self.offset)

    def send_ctrl(self, ctrl_sim_frame: np.ndarray) -> None:
        raw = self.offset + self.sign * ctrl_sim_frame
        if self.MAX_RAW_DELTA_PER_TICK is not None:
            if self._last_raw_cmd is None:
                self._last_raw_cmd = raw.copy()
            delta = np.clip(raw - self._last_raw_cmd, -self.MAX_RAW_DELTA_PER_TICK, self.MAX_RAW_DELTA_PER_TICK)
            raw = self._last_raw_cmd + delta
            self._last_raw_cmd = raw.copy()
        # else: no hardware-boundary clamp configured -- ActionMapper's
        # max_ctrl_rate_frac is the only thing protecting the servos here.
        # Set MAX_RAW_DELTA_PER_TICK above before real deployment.
        self.controller.sync_write_goal_position(SERVO_IDS, raw.tolist())

    def close(self) -> None:
        for servo_id in SERVO_IDS:
            self.controller.write_torque_enable(servo_id, 0)


def build_cube_obs(jpos_raw, jvel_raw, last_action, target_face, kin: HandKinematics,
                    R_cube: np.ndarray) -> np.ndarray:
    """Exactly mirrors AmazeDexCubeEnv._get_obs(): 27 dims, same order, same
    normalization. R_cube comes from the AprilTag tracker (cube-local ->
    world rotation matrix)."""
    jpos = np.clip((jpos_raw - kin.ctrl_mid) / kin.ctrl_half, -1.0, 1.0)
    jvel = np.clip(jvel_raw / JVEL_SCALE, -1.0, 1.0)
    cube_up_local = (R_cube.T @ np.array([0.0, 0.0, 1.0], dtype=np.float32)).astype(np.float32)
    onehot = np.zeros(6, dtype=np.float32)
    onehot[target_face] = 1.0

    obs = np.concatenate([jpos, jvel, last_action, cube_up_local, onehot]).astype(np.float32)
    if not np.all(np.isfinite(obs)):
        obs = np.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=-1.0)
    return obs


def theta_rad(R_cube: np.ndarray, target_face: int) -> float:
    """Angular distance from target face-normal to world +Z -- identical
    metric to AmazeDexCubeEnv._theta(), so success is judged the same way
    on hardware as it was scored in training."""
    n = R_cube @ FACE_NORMALS[target_face]
    return float(np.arccos(np.clip(n[2], -1.0, 1.0)))


def predict_action(model: PPO, obs: np.ndarray) -> np.ndarray:
    action, _ = model.predict(obs, deterministic=True)
    return np.clip(action, -1.0, 1.0)


def execute_target_on_real_hand(model, hand: HandInterface, tracker, target_face: int,
                                 kin: HandKinematics, cap: cv2.VideoCapture, cfg=CFG) -> bool:
    """Runs until the cube settles on target_face (success) or cfg.max_steps
    is hit (timeout). Returns True on success."""
    print(f"\n--- Target: {FACE_NAMES[target_face]} ---")

    mapper = ActionMapper(kin, cfg)
    joint_pos = hand.read_joint_positions()
    mapper.start(joint_pos)
    prev_joint_pos = joint_pos.copy()
    prev_action = np.zeros(8, dtype=np.float32)  # matches env reset: last_action starts at 0
    R_cube = tracker.last_R.copy()
    hold = 0

    for step in range(cfg.max_steps):
        loop_start = time.perf_counter()
        joint_pos = hand.read_joint_positions()
        joint_vel = (joint_pos - prev_joint_pos) / CONTROL_DT

        ok, raw_frame = cap.read()
        frame = cv2.flip(raw_frame, 1) if ok else None
        R_cube = tracker.update(frame) if frame is not None else R_cube

        obs = build_cube_obs(joint_pos, joint_vel, prev_action, target_face, kin, R_cube=R_cube)
        action = predict_action(model, obs)
        target_ctrl = mapper.action_to_ctrl(action)
        hand.send_ctrl(target_ctrl)

        prev_joint_pos = joint_pos.copy()
        prev_action = action.copy()

        theta = theta_rad(R_cube, target_face)
        angvel_proxy = np.linalg.norm(joint_vel)  # no direct cube-angvel sensor on the rig
        settled = theta < cfg.success_theta_rad and angvel_proxy < cfg.success_max_angvel
        hold = hold + 1 if settled else 0

        if frame is not None:
            cv2.putText(frame, f"Target: {FACE_NAMES[target_face]}  theta={np.degrees(theta):.1f}deg  hold={hold}",
                        (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.imshow("AmazeDex Cube Execution", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("--- ABORTED (quit) ---")
                return False

        if hold >= cfg.success_hold_steps:
            print(f"--- SUCCESS: {FACE_NAMES[target_face]} reached and held ({step + 1} steps) ---")
            return True

        elapsed = time.perf_counter() - loop_start
        if elapsed < CONTROL_DT:
            time.sleep(CONTROL_DT - elapsed)

    print(f"--- TIMEOUT: target {FACE_NAMES[target_face]} not reached in {cfg.max_steps} steps ---")
    return False


def execute_target_in_sim(env, model, target_face: int, cfg=CFG) -> bool:
    """Same run-until-success-or-timeout contract as the real-hand version,
    for apples-to-apples testing of a target-selection loop before deploying."""
    obs, _ = env.reset()
    env.unwrapped.targetface = target_face  # setter resets prev_theta/armed/hold
    obs = env.unwrapped._get_obs()
    print(f"\n--- Target: {FACE_NAMES[target_face]} ---")

    for step in range(cfg.max_steps):
        action = predict_action(model, obs)
        obs, reward, done, truncated, info = env.step(action)
        env.render()
        if info.get("success"):
            print(f"--- SUCCESS: {FACE_NAMES[target_face]} reached and held ({step + 1} steps) ---")
            return True
        if done or truncated:
            break
    print(f"--- TIMEOUT/DROPPED: target {FACE_NAMES[target_face]} not reached ---")
    return False


def prompt_target_face(default: int | None) -> int | None:
    names = ", ".join(f"{i}={n}" for i, n in enumerate(FACE_NAMES))
    raw = input(f"Target face [{names}] (blank to quit): ").strip()
    if raw == "":
        return None
    if raw.isdigit() and 0 <= int(raw) <= 5:
        return int(raw)
    print("Invalid input, expected 0-5.")
    return default


def main() -> None:
    parser = argparse.ArgumentParser(description="Sim-to-real execution for AmazeDex cube rotation.")
    parser.add_argument("--model", default="models/best_model")
    parser.add_argument("--mode", choices=["sim", "real"], default="sim")
    parser.add_argument("--port", default="COM14")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--calib", default="camera_calib.json", help="AprilTag camera+tag calibration (real mode only)")
    parser.add_argument("--target-face", type=int, default=None, choices=range(6),
                         help="Skip the prompt for the first attempt only; you're still asked after each attempt.")
    args = parser.parse_args()

    model = PPO.load(args.model, device="cpu")
    run_cfg = dataclasses.replace(CFG, max_steps=CFG.max_steps * 10)  # let each attempt run way longer before timing out
    env = hand = tracker = kin = cap = None

    if args.mode == "sim":
        import gymnasium as gym
        env = gym.make("AmazeDex/CubeRotate-v0", render_mode="human")
    else:
        from apriltag_pose import ApriltagCubeTracker, CameraCalib

        if not os.path.exists(args.calib):
            raise FileNotFoundError(
                f"{args.calib} not found. Real mode needs camera intrinsics, tag size, and the "
                "tag-id -> face-index map (see apriltag_pose.py docstring) -- there is no safe default."
            )
        cap = cv2.VideoCapture(args.camera)
        if not cap.isOpened():
            raise RuntimeError("Could not open camera -- real mode requires it for cube-orientation sensing.")

        hand = HandInterface(serial_port=args.port, calibration=load_calibration())
        kin = HandKinematics()
        tracker = ApriltagCubeTracker(CameraCalib.load(args.calib), FACE_NORMALS)
        print("[INFO] Physically pre-close the hand around the cube now (matching the training grasp), "
              "then press Enter to begin.")
        input()

    print("\n=== AmazeDex cube rotation ===")
    print("Enter a target face each round. The hand runs until it succeeds or times out, then asks again.")

    target_face = args.target_face
    try:
        while True:
            if target_face is None:
                target_face = prompt_target_face(None)
                if target_face is None:
                    break

            if args.mode == "sim":
                execute_target_in_sim(env, model, target_face, cfg=run_cfg)
            else:
                execute_target_on_real_hand(model, hand, tracker, target_face, kin, cap, cfg=run_cfg)

            target_face = None  # always re-prompt after an attempt -- no auto-cycling
    finally:
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()
        if hand:
            hand.close()
        if env:
            env.close()


if __name__ == "__main__":
    main()