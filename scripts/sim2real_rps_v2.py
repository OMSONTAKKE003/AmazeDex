"""
sim2real_rps_v2.py

Run a trained AmazeDex Rock-Paper-Scissors PPO policy in either:
  --mode sim   : MuJoCo simulation (AmazeDexRockPaperScissorsEnv, render_mode="human")
  --mode real  : Physical hand, Feetech SCS0009 servos via rustypot (Scs0009PyController)

Both modes share the same trigger: a webcam + MediaPipe hand-landmark classifier
watches for a stable human rock/paper/scissors gesture, then commands the
policy-driven hand to play the counter-gesture.

NOTE ON HARDWARE API: this uses Scs0009PyController with PER-SERVO calls
(read_present_position(i) / write_goal_position(i, val)) rather than the
batch sync_read_present_position/sync_write_goal_position API, because the
availability of the sync_ methods on Scs0009PyController (vs. Sts3215PyController,
where they're confirmed) was not verified. If you later confirm Scs0009PyController
does have sync_read_present_position / sync_write_goal_position, those calls are
faster (one serial transaction instead of 8) and can be swapped in directly in
HandInterface.read_joint_positions / HandInterface.send_ctrl.
"""
from __future__ import annotations
import argparse
import json
import os
import time
import csv
import cv2
import numpy as np
from collections import Counter
from stable_baselines3 import PPO

from amazedex_rps_env import (
    ACTUATOR_NAMES, GESTURE_NAMES, GESTURE_TARGETS, JOINT_NAMES,
    AmazeDexRockPaperScissorsEnv
)
from mediapipe_rps_gesture import classify_gesture
import mediapipe as mp

COUNTER_GESTURE = {
    "rock": "paper",
    "paper": "scissors",
    "scissors": "rock",
}

# Detection robustness: look at the last N frames, trigger only if a supermajority agree.
HISTORY_WINDOW = 15
MIN_VOTES = 10
REVEAL_COOLDOWN_S = 2.0

POLICY_STEPS_PER_REVEAL = 120
CONTROL_DT = 0.02

CTRL_LOW, CTRL_HIGH = -1.57, 1.57
SERVO_IDS = [1, 2, 3, 4, 5, 6, 7, 8]
GOAL_SPEED = 5

CALIBRATION_PATH = "hand_calibration.json"


def load_calibration(path: str = CALIBRATION_PATH) -> dict:
    if not os.path.exists(path):
        print(f"[WARN] no calibration file at {path}. Using default (zero offset, +1 sign).")
        return {"offset_rad": [0.0] * 8, "sign": [1.0] * 8}
    with open(path) as f:
        return json.load(f)


class HandInterface:
    """Thin wrapper around rustypot's Scs0009PyController, using per-servo
    read/write calls (see module docstring for why sync_ methods aren't used)."""

    def __init__(self, serial_port: str = "COM14", baudrate: int = 1_000_000, timeout: float = 0.6,
                 calibration: dict | None = None):
        from rustypot import Scs0009PyController
        self.controller = Scs0009PyController(serial_port=serial_port, baudrate=baudrate, timeout=timeout)
        for servo_id in SERVO_IDS:
            self.controller.write_torque_enable(servo_id, 1)
        calib = calibration or {"offset_rad": [0.0] * 8, "sign": [1.0] * 8}
        self.offset = np.asarray(calib["offset_rad"], dtype=np.float32)
        self.sign = np.asarray(calib["sign"], dtype=np.float32)

    def read_joint_positions(self) -> np.ndarray:
        """Returns joint positions in policy/sim frame (calibration applied), radians."""
        raw = np.array(
            [self.controller.read_present_position(i) for i in SERVO_IDS], dtype=np.float32
        ).ravel()
        return self.sign * (raw - self.offset)

    def send_ctrl(self, ctrl_sim_frame: np.ndarray) -> None:
        """Takes ctrl in policy/sim frame, converts to raw servo frame, and commands each servo."""
        raw = self.offset + self.sign * ctrl_sim_frame
        for servo_id, goal_pos in zip(SERVO_IDS, raw):
            self.controller.write_goal_speed(servo_id, GOAL_SPEED)
            self.controller.write_goal_position(servo_id, float(goal_pos))

    def close(self) -> None:
        for servo_id in SERVO_IDS:
            self.controller.write_torque_enable(servo_id, 0)


def policy_action_to_ctrl(action: np.ndarray) -> np.ndarray:
    action = np.clip(action, -1.0, 1.0)
    return CTRL_LOW + (action + 1.0) * 0.5 * (CTRL_HIGH - CTRL_LOW)


def ctrl_to_policy_action(ctrl: np.ndarray) -> np.ndarray:
    """Reverse maps physical angles back to policy action space [-1, 1] for safe initialization."""
    action = 2.0 * (ctrl - CTRL_LOW) / (CTRL_HIGH - CTRL_LOW) - 1.0
    return np.clip(action, -1.0, 1.0)


def build_obs(joint_pos: np.ndarray, prev_joint_pos: np.ndarray, prev_action: np.ndarray, target_idx: int) -> np.ndarray:
    one_hot = np.zeros(len(GESTURE_NAMES), dtype=np.float32)
    one_hot[target_idx] = 1.0
    return np.concatenate([
        np.ravel(joint_pos),
        np.ravel(prev_joint_pos),
        np.ravel(prev_action),
        one_hot
    ]).astype(np.float32)


def reveal_on_real_hand(model: PPO, hand: HandInterface, target_idx: int) -> None:
    # 1. READ CURRENT PHYSICAL STATE
    joint_pos = hand.read_joint_positions()
    prev_joint_pos = joint_pos.copy()
    prev_action = ctrl_to_policy_action(joint_pos)

    # --- EPISODE-START LOGGING ---
    print(f"\n--- REAL REVEAL START (SLOW & SAFE): {GESTURE_NAMES[target_idx].upper()} ---")
    print(f"[EP START] Motor positions (rad): {np.round(joint_pos, 3)}")

    os.makedirs("reveal_logs", exist_ok=True)
    log_file = f"reveal_logs/log_{int(time.time())}_{GESTURE_NAMES[target_idx]}.csv"

    # Generate the target joint configuration for the chosen gesture
    target_gesture_pose = GESTURE_TARGETS[target_idx]

    last_action = prev_action.copy()

    with open(log_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Timestamp", "Target_Angles", "Actual_Encoder", "Policy_Output"])

        # We use a longer sequence length (120 steps = 2.4 seconds) for a smooth glide
        total_steps = 120

        for step in range(total_steps):
            loop_start = time.perf_counter()

            joint_pos = hand.read_joint_positions()
            obs = build_obs(joint_pos, prev_joint_pos, prev_action, target_idx)
            action, _ = model.predict(obs, deterministic=True)
            last_action = action.copy()

            # Ultra-conservative smoothing (10% new action, 90% old action)
            smoothed_action = 0.10 * action + 0.90 * prev_action

            target_angles = policy_action_to_ctrl(smoothed_action)

            # Additional safety layer: Blend policy output progressively with the exact target gesture pose
            # to guarantee it doesn't drift or twitch midway through.
            progress = min(1.0, step / 60.0)  # Full blend by halfway through
            blended_ctrl = (1.0 - progress) * policy_action_to_ctrl(prev_action) + progress * target_gesture_pose

            hand.send_ctrl(blended_ctrl)

            writer.writerow([time.time(), blended_ctrl.tolist(), joint_pos.tolist(), action.tolist()])

            prev_joint_pos = joint_pos.copy()
            prev_action = smoothed_action.copy()

            while time.perf_counter() - loop_start < CONTROL_DT:
                pass

    # --- EPISODE-END LOGGING ---
    final_joint_pos = hand.read_joint_positions()
    print(f"[EP END]   Motor positions (rad): {np.round(final_joint_pos, 3)}")
    print(f"[EP END]   Final policy action output: {np.round(last_action, 3)}")
    print(f"--- REVEAL END | Log saved to {log_file} ---\n")


def reveal_in_sim(model: PPO, env: AmazeDexRockPaperScissorsEnv, target_idx: int) -> None:
    print(f"\n--- SIM REVEAL START: {GESTURE_NAMES[target_idx].upper()} ---")

    obs, _ = env.reset()
    env.unwrapped.target_gesture_idx = target_idx
    obs = env.unwrapped._get_obs()

    start_joint_pos = env.unwrapped.data.qpos[env.unwrapped.joint_qpos_adr].copy()
    print(f"[EP START] Joint qpos (rad): {np.round(start_joint_pos, 3)}")

    total_reward = 0.0
    final_dist = None
    info = {}
    action = None
    step_i = 0
    for step_i in range(POLICY_STEPS_PER_REVEAL):
        action, _ = model.predict(obs, deterministic=True)
        print(f"Sim Step {step_i:03d} | Policy Output (Action): {np.round(action, 3)}")

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        final_dist = info["dist_to_target"]
        if terminated or truncated:
            break

    end_joint_pos = env.unwrapped.data.qpos[env.unwrapped.joint_qpos_adr].copy()
    print(f"[EP END]   Joint qpos (rad): {np.round(end_joint_pos, 3)}")
    print(f"[EP END]   Final policy action output: {np.round(action, 3)}")
    print(
        f"--- SIM REVEAL END | steps={step_i + 1} "
        f"total_reward={total_reward:.2f} "
        f"final_dist={final_dist:.3f} "
        f"correct={info.get('correct_gesture')} ---\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/best_model.zip")
    parser.add_argument("--mode", choices=["sim", "real"], default="sim")
    parser.add_argument("--port", default="COM14")
    parser.add_argument("--camera", type=int, default=0)
    args = parser.parse_args()

    model = PPO.load(args.model, device='cpu')

    env = hand = None
    if args.mode == "sim":
        env = AmazeDexRockPaperScissorsEnv(render_mode="human")
        env.reset()
    else:
        hand = HandInterface(serial_port=args.port, calibration=load_calibration())

    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    cap = cv2.VideoCapture(args.camera)

    recent_labels: list[str] = []
    last_reveal_t = 0.0

    with mp_hands.Hands(max_num_hands=1) as hands:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)
            results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            human_label = "unknown"
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                raw_gesture = classify_gesture(results.multi_hand_landmarks[0].landmark)
                human_label = getattr(raw_gesture, 'label', raw_gesture) if raw_gesture else "unknown"

            # ROBUST DETECTION: majority-vote window
            recent_labels.append(human_label)
            if len(recent_labels) > HISTORY_WINDOW:
                recent_labels.pop(0)

            is_stable = False
            stable_label = None
            if len(recent_labels) == HISTORY_WINDOW:
                counts = Counter(recent_labels)
                top_label, top_count = counts.most_common(1)[0]
                if top_count >= MIN_VOTES and top_label != "unknown":
                    is_stable = True
                    stable_label = top_label

            now = time.time()
            status_color = (0, 255, 0) if is_stable else (0, 0, 255)
            display_label = stable_label if is_stable else human_label
            cv2.putText(frame, f"Detecting: {display_label}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)

            if is_stable and (now - last_reveal_t) > REVEAL_COOLDOWN_S:
                if stable_label in COUNTER_GESTURE:
                    robot_label = COUNTER_GESTURE[stable_label]
                    target_idx = GESTURE_NAMES.index(robot_label)

                    cv2.putText(frame, f"Countering with: {robot_label.upper()}!", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                    cv2.imshow("AmazeDex RPS", frame)
                    cv2.waitKey(1)

                    if args.mode == "sim":
                        reveal_in_sim(model, env, target_idx)
                    else:
                        reveal_on_real_hand(model, hand, target_idx)

                    last_reveal_t = time.time()
                    recent_labels = []

            cv2.imshow("AmazeDex RPS", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()
    if hand:
        hand.close()
    if env:
        env.close()


if __name__ == "__main__":
    main()