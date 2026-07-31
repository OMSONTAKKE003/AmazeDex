from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass

import cv2
import numpy as np
import mediapipe as mp

GESTURE_NAMES = ("rock", "paper", "scissors")

CURLED = 1.0
EXTENDED = 0.0


FINGER_TO_JOINT_SLICE = {
    "index": (0, 2),
    "middle": (2, 4),
    "ring": (4, 6),
    "thumb": (6, 8),
}

GESTURE_FINGER_STATE = {
    # True = curled, False = extended
    "rock": {"index": True, "middle": True, "ring": True, "thumb": True},
    "paper": {"index": False, "middle": False, "ring": False, "thumb": False},
    "scissors": {"index": False, "middle": False, "ring": True, "thumb": True},
}

# MediaPipe landmark indices per finger: (MCP, PIP, DIP, TIP)
FINGER_LANDMARKS = {
    "index": (5, 6, 7, 8),
    "middle": (9, 10, 11, 12),
    "ring": (13, 14, 15, 16),
    "thumb": (17, 18, 19, 20),
}
WRIST = 0

CURL_THRESHOLD = 0.75


@dataclass
class GestureResult:
    label: str  # "rock" | "paper" | "scissors" | "unknown"
    confidence: float  # fraction of the 4 fingers matching the closest gesture
    finger_curled: dict  # {"index": bool, ...}
    joint_targets: np.ndarray  # shape (8,), matches JOINT_NAMES order
    one_hot: np.ndarray  # shape (3,), matches GESTURE_NAMES order


def _finger_curl_ratio(landmarks, finger: str) -> float:
    """Ratio in [~0, ~1]: 1.0 = straight/extended, lower = curled.

    Computed as (straight-line wrist->tip distance) / (sum of the three
    inter-joint segment lengths wrist->mcp->pip->tip). For a fully straight
    finger these are equal (ratio ~1). A curled finger's tip folds back
    toward the wrist, shrinking the straight-line distance while the segment
    lengths stay ~constant, so the ratio drops well below 1.
    """
    mcp_i, pip_i, dip_i, tip_i = FINGER_LANDMARKS[finger]
    wrist = np.array([landmarks[WRIST].x, landmarks[WRIST].y, landmarks[WRIST].z])
    mcp = np.array([landmarks[mcp_i].x, landmarks[mcp_i].y, landmarks[mcp_i].z])
    pip = np.array([landmarks[pip_i].x, landmarks[pip_i].y, landmarks[pip_i].z])
    tip = np.array([landmarks[tip_i].x, landmarks[tip_i].y, landmarks[tip_i].z])

    straight = np.linalg.norm(tip - wrist)
    segments = (
        np.linalg.norm(mcp - wrist)
        + np.linalg.norm(pip - mcp)
        + np.linalg.norm(tip - pip)
    )
    if segments < 1e-8:
        return 1.0
    return float(straight / segments)


def classify_gesture(landmarks) -> GestureResult:
    curl_ratio = {f: _finger_curl_ratio(landmarks, f) for f in FINGER_LANDMARKS}
    curled = {f: (curl_ratio[f] < CURL_THRESHOLD) for f in FINGER_LANDMARKS}

    best_label, best_score = "unknown", -1
    for name, expected in GESTURE_FINGER_STATE.items():
        score = sum(1 for f in expected if expected[f] == curled[f])
        if score > best_score:
            best_label, best_score = name, score

    confidence = best_score / 4.0
    label = best_label if confidence >= 0.75 else "unknown"

    joint_targets = np.zeros(8, dtype=np.float32)
    if label != "unknown":
        expected = GESTURE_FINGER_STATE[label]
        for finger, (lo, hi) in FINGER_TO_JOINT_SLICE.items():
            joint_targets[lo:hi] = CURLED if expected[finger] else EXTENDED
    else:
        # No confident label: fall back to the *measured* per-finger state
        # rather than a fixed gesture, so downstream code still gets a
        # usable (if noisier) 8-dim vector.
        for finger, (lo, hi) in FINGER_TO_JOINT_SLICE.items():
            joint_targets[lo:hi] = CURLED if curled[finger] else EXTENDED

    one_hot = np.zeros(3, dtype=np.float32)
    if label in GESTURE_NAMES:
        one_hot[GESTURE_NAMES.index(label)] = 1.0

    return GestureResult(label, confidence, curled, joint_targets, one_hot)


def main() -> None:
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_styles = mp.solutions.drawing_styles

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open webcam (index 0). Check camera permissions/index.")
        sys.exit(1)

    last_result: GestureResult | None = None

    with mp_hands.Hands(
        model_complexity=1,
        max_num_hands=1,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6,
    ) as hands:
        prev_t = time.time()
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Camera read failed.")
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            results = hands.process(rgb)
            rgb.flags.writeable = True

            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_styles.get_default_hand_landmarks_style(),
                    mp_styles.get_default_hand_connections_style(),
                )
                last_result = classify_gesture(hand_landmarks.landmark)

                text = f"{last_result.label} ({last_result.confidence*100:.0f}%)"
                cv2.putText(
                    frame, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.1,
                    (0, 255, 0) if last_result.label != "unknown" else (0, 165, 255),
                    2, cv2.LINE_AA,
                )
                fingers_txt = " ".join(
                    f"{f[:3]}:{'C' if v else 'E'}"
                    for f, v in last_result.finger_curled.items()
                )
                cv2.putText(
                    frame, fingers_txt, (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 255, 255), 1, cv2.LINE_AA,
                )
            else:
                last_result = None
                cv2.putText(
                    frame, "no hand detected", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2, cv2.LINE_AA,
                )

            now = time.time()
            fps = 1.0 / max(now - prev_t, 1e-6)
            prev_t = now
            cv2.putText(
                frame, f"{fps:.0f} fps", (20, frame.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA,
            )

            cv2.imshow("AmazeDex RPS gesture (q=quit, s=snapshot json)", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("s") and last_result is not None:
                payload = {
                    "label": last_result.label,
                    "confidence": last_result.confidence,
                    "joint_targets": last_result.joint_targets.tolist(),
                    "one_hot": last_result.one_hot.tolist(),
                }
                print(json.dumps(payload))

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
