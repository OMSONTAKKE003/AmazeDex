from __future__ import annotations
import json
import os
import time
import cv2
import numpy as np
import mediapipe as mp
from rustypot import Scs0009PyController

from mediapipe_rps_gesture import classify_gesture
# Removed GESTURE_TARGETS and JOINT_NAMES if unused, kept GESTURE_NAMES just in case
from amazedex_rps_env import GESTURE_NAMES 

# --- must match amazedex_rps_env.py / robot.xml exactly ---
CTRL_LOW, CTRL_HIGH = -1.4, 1.4
SERVO_IDS = [1, 2, 3, 4, 5, 6, 7, 8]
GOAL_SPEED = 0
HARDCODED_GESTURES = {
    "paper": np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
    "rock": np.array([1.4, -1.4, 1.4, -1.4, 1.4, -1.4, 1.4, -1.4], dtype=np.float32),
    "scissors": np.array([0.0, 0.0, 0.0, 0.0, 1.4, -1.4, 1.4, -1.4], dtype=np.float32)
}
# Fixed syntax error here
COUNTER_GESTURE = {"rock": "paper", "paper": "scissors", "scissors": "rock"}

STEPS_PER_REVEAL = 120
CONTROL_DT = 0.02          # 50Hz, matches training frame_skip
DETECT_WINDOW = 5         # frames to agree on before triggering a reveal

class Hand:
    def __init__(self, port: str):
        self.ctrl = Scs0009PyController(serial_port=port, baudrate=1_000_000, timeout=2.99)
        for sid in SERVO_IDS:
            self.ctrl.write_torque_enable(sid, 1)

    def read(self) -> np.ndarray:
        raw = np.array([self.ctrl.read_present_position(i) for i in SERVO_IDS], dtype=np.float32).ravel()
        return 1 * (raw)

    def send(self, target_rad: np.ndarray) -> None:
        target_rad = np.clip(target_rad, CTRL_LOW, CTRL_HIGH)
        raw =  1 * target_rad
        for sid, pos in zip(SERVO_IDS, raw):
            self.ctrl.write_goal_speed(sid, GOAL_SPEED)
            self.ctrl.write_goal_position(sid, float(pos))

    def close(self) -> None:
        for sid in SERVO_IDS:
            self.ctrl.write_torque_enable(sid, 0)


def action_to_ctrl(action: np.ndarray) -> np.ndarray:
    action = np.clip(action, -1.0, 1.0)
    return CTRL_LOW + (action + 1.0) * 0.5 * (CTRL_HIGH - CTRL_LOW)


def ctrl_to_action(ctrl: np.ndarray) -> np.ndarray:
    return np.clip(2.0 * (ctrl - CTRL_LOW) / (CTRL_HIGH - CTRL_LOW) - 1.0, -1.0, 1.0)


def reveal(hand: Hand, target_name: str) -> None:
    """Run closed-loop on real servos until it settles into the hardcoded target gesture."""
    joint_pos = hand.read()
    prev_action = ctrl_to_action(joint_pos)
    
    # Fetch the hardcoded target action once
    target_action = HARDCODED_GESTURES[target_name]

    for _ in range(STEPS_PER_REVEAL):
        t0 = time.perf_counter()

        # EMA smoothing gently blends towards the hardcoded target action
        smoothed = 0.15 * target_action + 0.85 * prev_action
        hand.send(action_to_ctrl(smoothed))

        prev_action = smoothed

        while time.perf_counter() - t0 < CONTROL_DT:
            pass


def main(port: str = "COM14", camera: int = 0) -> None:
    hand = Hand(port)

    mp_hands = mp.solutions.hands
    cap = cv2.VideoCapture(camera)
    recent = []
    last_reveal_t = 0.0

    try:
        with mp_hands.Hands(max_num_hands=1) as hands:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                frame = cv2.flip(frame, 1)
                result = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

                label = "unknown"
                if result.multi_hand_landmarks:
                    gesture = classify_gesture(result.multi_hand_landmarks[0].landmark)
                    label = getattr(gesture, "label", gesture) or "unknown"

                recent.append(label)
                if len(recent) > DETECT_WINDOW:
                    recent.pop(0)

                stable = (len(recent) == DETECT_WINDOW and recent.count(label) == DETECT_WINDOW
                          and label in COUNTER_GESTURE)

                if stable and time.time() - last_reveal_t > 2.0:
                    target_name = COUNTER_GESTURE[label]
                    reveal(hand, target_name)
                    last_reveal_t = time.time()
                    recent = []

                cv2.putText(frame, label, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.imshow("RPS", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        hand.close()


if __name__ == "__main__":
    main()