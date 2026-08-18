import argparse
import dataclasses
import json
import os
import time
import cv2
import numpy as np
from sbx import SAC


from envs.amazedex_cube_env import (
    AmazeDexCubeEnv, FACE_NORMALS, JOINTS, TIP_SITES, ACTUATORS,
    MODEL_PATH, JVEL_SCALE, ANGVEL_SCALE, REACH_NORM, CFG, CUBE_LOCAL_CENTER,
    MAX_CTRL_RATE_FRAC,
)
import envs.register_amazedex_env
import mujoco

CONTROL_DT = 0.02
SERVO_IDS = [1, 2, 3, 4, 5, 6, 7, 8]

GOAL_SPEED = 0  
CALIBRATION_PATH = "hand_calibration.json"

# Physical-digit -> internal target_face (FACE_NORMALS index) map.
#
# The digit printed/AprilTag'd on each face of the real cube does NOT match
# FACE_NORMALS' array index for that face -- e.g. FACE_NORMALS[0] is +Z, but
# per scene.xml the tag glued to +Z is "tag_number_5", not "tag_number_0".
# Feeding a typed digit straight in as target_face (the old behavior here)
# silently commanded the hand to the wrong face -- this is the "asked for 0,
# hand went to the face showing 5" bug.
#
# Derived directly from scene.xml's cube body: each tag_number_<digit> geom's
# position relative to CUBE_LOCAL_CENTER, projected onto the dominant axis,
# gives the face normal, matched against FACE_NORMALS:
#   digit 5 -> +Z (FACE_NORMALS[0])   digit 0 -> -Z (FACE_NORMALS[1])
#   digit 4 -> +Y (FACE_NORMALS[2])   digit 2 -> -Y (FACE_NORMALS[3])
#   digit 1 -> +X (FACE_NORMALS[4])   digit 3 -> -X (FACE_NORMALS[5])
# Re-derive this if the cube mesh/tag placement in scene.xml ever changes.
DIGIT_TO_FACE_INDEX = {5: 0, 0: 1, 4: 2, 2: 3, 1: 4, 3: 5}
FACE_INDEX_TO_DIGIT = {v: k for k, v in DIGIT_TO_FACE_INDEX.items()}


class HandKinematics:


    def __init__(self, model_path: str = MODEL_PATH):
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        self.qposids = np.array([self.model.joint(n).qposadr[0] for n in JOINTS])
        self.cubeid = self.model.body("cube").id
        self.tip_sites = [self.model.site(n).id for n in TIP_SITES]

        actids = np.array([self.model.actuator(n).id for n in ACTUATORS])
        lims = self.model.actuator_ctrlrange[actids]
        self.ctrl_lo, self.ctrl_hi = lims[:, 0], lims[:, 1]
        self.ctrl_mid = (self.ctrl_lo + self.ctrl_hi) / 2
        self.ctrl_half = (self.ctrl_hi - self.ctrl_lo) / 2

        mujoco.mj_forward(self.model, self.data)
        cube_body_pos = np.array(self.model.body("cube").pos, dtype=np.float32)
        self.nominal_cube_pos = cube_body_pos + CUBE_LOCAL_CENTER

    def tip_to_cube(self, joint_pos_rad: np.ndarray) -> np.ndarray:
        self.data.qpos[self.qposids] = np.clip(joint_pos_rad, self.ctrl_lo, self.ctrl_hi)
        mujoco.mj_forward(self.model, self.data)
        cpos = self.nominal_cube_pos
        return np.array([cpos - self.data.site_xpos[s] for s in self.tip_sites], dtype=np.float32)


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
 
    MAX_RAW_DELTA_PER_TICK = None 

    def __init__(self, serial_port: str = "COM14", baudrate: int = 1_000_000,
                 timeout: float = 0.6, calibration: dict | None = None):
        from rustypot import Scs0009PyController

        if GOAL_SPEED is None or GOAL_SPEED <= 0:
            raise ValueError(
                "GOAL_SPEED is not configured (or is 0, which means 'unlimited speed' on "
                "Feetech/SCS-protocol servos, not 'off'). Set it from your SCS0009 datasheet "
                "before running on real hardware -- this was the primary cause of unsafe-fast "
                "moves. Refusing to start rather than defaulting to max speed."
            )

        self.controller = Scs0009PyController(serial_port=serial_port, baudrate=baudrate, timeout=timeout)
        for servo_id in SERVO_IDS:
            self.controller.write_torque_enable(servo_id, 1)
        self.controller.sync_write_goal_speed(SERVO_IDS, [GOAL_SPEED] * len(SERVO_IDS))

        calib = calibration or {"offset_rad": [0.0] * 8, "sign": [1.0] * 8}
        self.offset = np.asarray(calib["offset_rad"], dtype=np.float32)
        self.sign = np.asarray(calib["sign"], dtype=np.float32)
        self._last_raw_cmd = None  # lazily initialized to the first commanded position

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
        # max_ctrl_rate_frac is the only thing protecting the servos.
        # Set MAX_RAW_DELTA_PER_TICK above before real deployment.

        self.controller.sync_write_goal_position(SERVO_IDS, raw.tolist())

    def close(self) -> None:
        for servo_id in SERVO_IDS:
            self.controller.write_torque_enable(servo_id, 0)


def desired_rot_axis(R_cube: np.ndarray, target_face: int) -> np.ndarray | None:
    """Identical logic to AmazeDexCubeEnv._desired_rot_axis(), computed from
    R_cube instead of self.data.xmat -- same inputs (cube rotation + target
    face), same math, so axis_obs matches exactly what training saw."""
    n = R_cube @ FACE_NORMALS[target_face]
    axis = np.cross(n, np.array([0.0, 0.0, 1.0]))
    norm = np.linalg.norm(axis)
    if norm < 1e-6:
        if np.dot(n, np.array([0.0, 0.0, 1.0])) < 0:
            return np.array([1.0, 0.0, 0.0], dtype=np.float32)
        return None
    return (axis / norm).astype(np.float32)


def build_cube_obs(jpos_raw, jvel_raw, last_action, target_face, kin: HandKinematics,
                    R_cube=np.eye(3), cube_angvel_raw=None) -> np.ndarray:
    """Same normalization amazedex_cube_env.AmazeDexCubeEnv._get_obs() applies
    internally, so callers can't accidentally feed the policy raw units."""
    jpos = np.clip((jpos_raw - kin.ctrl_mid) / kin.ctrl_half, -1.0, 1.0)
    jvel = np.clip(jvel_raw / JVEL_SCALE, -1.0, 1.0)

    target_world_normal = R_cube @ FACE_NORMALS[target_face]
    cube_quat = _rotmat_to_quat_wxyz(R_cube)

    cube_angvel_raw = np.zeros(3, dtype=np.float32) if cube_angvel_raw is None else cube_angvel_raw
    cube_angvel = np.clip(cube_angvel_raw / ANGVEL_SCALE, -1.0, 1.0)

    one_hot = np.zeros(6, dtype=np.float32)
    one_hot[target_face] = 1.0

    tip_to_cube_n = np.clip(kin.tip_to_cube(jpos_raw) / REACH_NORM, -3.0, 3.0).flatten()
    # cube_pos_rel removed -- there is no translation sensor on the rig, so it
    # was always fed as zeros here while the env trained on true drift. Now
    # matches the 55-dim observation_space in amazedex_cube_env.py exactly.

    axis = desired_rot_axis(R_cube, target_face)
    axis_obs = axis if axis is not None else np.zeros(3, dtype=np.float32)

    return np.concatenate([jpos, jvel, last_action, cube_quat, cube_angvel,
                            target_world_normal, one_hot, tip_to_cube_n, axis_obs]).astype(np.float32)


def _rotmat_to_quat_wxyz(R: np.ndarray) -> np.ndarray:
    m = R
    tr = m[0, 0] + m[1, 1] + m[2, 2]
    if tr > 0:
        S = np.sqrt(tr + 1.0) * 2.0
        w, x, y, z = 0.25 * S, (m[2, 1] - m[1, 2]) / S, (m[0, 2] - m[2, 0]) / S, (m[1, 0] - m[0, 1]) / S
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        S = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        w, x, y, z = (m[2, 1] - m[1, 2]) / S, 0.25 * S, (m[0, 1] + m[1, 0]) / S, (m[0, 2] + m[2, 0]) / S
    elif m[1, 1] > m[2, 2]:
        S = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        w, x, y, z = (m[0, 2] - m[2, 0]) / S, (m[0, 1] + m[1, 0]) / S, 0.25 * S, (m[1, 2] + m[2, 1]) / S
    else:
        S = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        w, x, y, z = (m[1, 0] - m[0, 1]) / S, (m[0, 2] + m[2, 0]) / S, (m[1, 2] + m[2, 1]) / S, 0.25 * S
    return np.array([w, x, y, z], dtype=np.float32)


def theta_rad(R_cube: np.ndarray, target_face: int) -> float:
    """Angular distance from target face-normal to world +Z -- identical
    metric to AmazeDexCubeEnv._theta(), so success is judged the same way
    in real life as it was scored in training."""
    n = R_cube @ FACE_NORMALS[target_face]
    return float(np.arccos(np.clip(n[2], -1.0, 1.0)))


def predict_action(model: SAC, obs: np.ndarray) -> np.ndarray:
    action, _ = model.predict(obs, deterministic=True)
    return np.clip(action, -1.0, 1.0)


def execute_target_on_real_hand(model, hand: HandInterface, tracker, target_face: int,
                                 kin: HandKinematics, cap: cv2.VideoCapture, cfg=CFG) -> bool:
    """Runs until the cube settles on target_face (success) or cfg.max_steps
    is hit (terminate, report failure). Returns True on success."""
    print(f"\n--- Target: face {FACE_INDEX_TO_DIGIT[target_face]} ---")

    mapper = ActionMapper(kin, cfg)
    joint_pos = hand.read_joint_positions()
    mapper.start(joint_pos)
    prev_joint_pos = joint_pos.copy()
    prev_action = np.zeros(8, dtype=np.float32)   # matches env reset: last_action starts at 0
    prev_R_cube = tracker.last_R.copy()
    hold = 0

    for step in range(cfg.max_steps):
        loop_start = time.perf_counter()
        joint_pos = hand.read_joint_positions()
        joint_vel = (joint_pos - prev_joint_pos) / CONTROL_DT

        ok, raw_frame = cap.read()
        frame = cv2.flip(raw_frame, 1) if ok else None
        R_cube = tracker.update(frame) if frame is not None else tracker.last_R

        dR = R_cube @ prev_R_cube.T
        rvec_delta, _ = cv2.Rodrigues(dR)
        cube_angvel = (rvec_delta.flatten() / CONTROL_DT).astype(np.float32)
        prev_R_cube = R_cube.copy()

        obs = build_cube_obs(joint_pos, joint_vel, prev_action, target_face, kin,
                              R_cube=R_cube, cube_angvel_raw=cube_angvel)
        action = predict_action(model, obs)
        target_ctrl = mapper.action_to_ctrl(action)
        hand.send_ctrl(target_ctrl)

        prev_joint_pos = joint_pos.copy()
        prev_action = action.copy()

        theta = theta_rad(R_cube, target_face)
        settled = theta < cfg.success_theta_rad and np.linalg.norm(cube_angvel) < cfg.success_max_angvel
        hold = hold + 1 if settled else 0

        if frame is not None:
            cv2.putText(frame, f"Target: {FACE_INDEX_TO_DIGIT[target_face]}  theta={np.degrees(theta):.1f}deg  hold={hold}",
                        (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.imshow("AmazeDex Cube Execution", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("--- ABORTED (quit) ---")
                return False

        if hold >= cfg.success_hold_steps:
            print(f"--- SUCCESS: face {FACE_INDEX_TO_DIGIT[target_face]} reached and held ({step + 1} steps) ---")
            return True

        elapsed = time.perf_counter() - loop_start
        if elapsed < CONTROL_DT:
            time.sleep(CONTROL_DT - elapsed)

    print(f"--- TIMEOUT: target face {FACE_INDEX_TO_DIGIT[target_face]} not reached in {cfg.max_steps} steps ---")
    return False


def execute_target_in_sim(env, model, target_face: int, cfg=CFG) -> bool:
    """Same run-until-success-or-timeout contract as the real-hand version,
    for apples-to-apples testing of a target-selection loop before deploying.
    Paced to CONTROL_DT like the real-hand loop -- a MuJoCo step + policy
    forward pass takes a fraction of a millisecond, so without this the
    rollout finishes in a tiny fraction of the wall-clock time it represents
    (looks like a sped-up video, sometimes 40-200x)."""
    obs, _ = env.reset()
    env.unwrapped.targetface = target_face   # setter resets prev_theta/armed/hold
    obs = env.unwrapped._get_obs()
    print(f"\n--- Target: face {FACE_INDEX_TO_DIGIT[target_face]} ---")

    for step in range(cfg.max_steps):
        loop_start = time.perf_counter()

        action = predict_action(model, obs)
        obs, reward, done, truncated, info = env.step(action)
        env.render()
        if info.get("success"):
            print(f"--- SUCCESS: face {FACE_INDEX_TO_DIGIT[target_face]} reached and held ({step + 1} steps) ---")
            return True
        if done or truncated:
            break

        elapsed = time.perf_counter() - loop_start
        if elapsed < CONTROL_DT:
            time.sleep(CONTROL_DT - elapsed)
    print(f"--- TIMEOUT/DROPPED: target face {FACE_INDEX_TO_DIGIT[target_face]} not reached ---")
    return False


def prompt_target_face(default: int | None) -> int | None:
    raw = input("Target face -- digit as printed on the cube [0-5] (blank to quit): ").strip()
    if raw == "":
        return None
    if raw.isdigit() and 0 <= int(raw) <= 5:
        return DIGIT_TO_FACE_INDEX[int(raw)]  # digit -> internal FACE_NORMALS index
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
                         help="Digit as printed on the cube (0-5), NOT the internal FACE_NORMALS "
                              "index -- converted the same way the interactive prompt is. Skips "
                              "the prompt for the first attempt only; you're still asked after "
                              "each attempt.")
    args = parser.parse_args()

    model = SAC.load(args.model, device="cpu")
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

    target_face = DIGIT_TO_FACE_INDEX[args.target_face] if args.target_face is not None else None
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