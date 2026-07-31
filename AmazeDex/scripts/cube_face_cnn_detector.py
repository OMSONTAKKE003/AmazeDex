"""
cube_face_cnn_detector.py
==========================
Opens a webcam feed and predicts the die/number shown on the cube face in
real time using the checkpoint produced by train_face_cnn.py.
"""

from collections import Counter, deque
import os
import time
import cv2
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import torchvision.models as models

# ==========================================
# 0. HARDCODED SETTINGS
# ==========================================
MODEL_PATH             = r"C:\Users\luvja\Desktop\updatedwithstand\AmazeDex\face_cnn.pt"
CAMERA_INDEX           = 0
IMG_SIZE               = 224
CONF_THRESH             = 0.45   # Minimum softmax confidence to accept a prediction
ENTROPY_MAX            = 1.2    # Reject if the model is uncertain across classes
VOTE_WINDOW            = 3      # Frames kept for temporal smoothing
VOTE_MIN               = 3      # Min agreeing frames required to lock a prediction

USE_CENTER_SQUARE_CROP = True   # Crop a square from the frame center before resizing
CROP_MARGIN            = 0.05   # Slight margin expansion (5%) to prevent cutting off outer pips
USE_CLAHE              = False  # Set to False if training images were raw RGB without CLAHE


# ==========================================
# 1. FRAME PREPROCESSING HELPERS
# ==========================================

def center_square_crop(frame, output_size, margin=0.0):
    h, w, _ = frame.shape
    side = min(h, w)
    
    # Expand crop slightly if margin > 0
    margin_px = int(side * margin)
    side_cropped = max(10, side - (2 * margin_px))
    
    y1 = (h - side_cropped) // 2
    x1 = (w - side_cropped) // 2
    
    crop = frame[y1:y1 + side_cropped, x1:x1 + side_cropped]
    crop = cv2.resize(crop, (output_size, output_size))
    box = np.array([[x1, y1], [x1 + side_cropped, y1], [x1 + side_cropped, y1 + side_cropped], [x1, y1 + side_cropped]])
    return crop, box


def apply_clahe(bgr_frame):
    lab = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)


# ==========================================
# 2. MODEL LOADING
# ==========================================

def load_model(model_path, device):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Checkpoint not found at: {model_path}")

    checkpoint = torch.load(model_path, map_location=device)
    classes = [str(c) for c in checkpoint["classes"]]

    # Load torchvision MobileNetV3-Small
    model = models.mobilenet_v3_small(weights=None)
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, len(classes))

    raw_state = checkpoint["model_state"]
    clean_state = {k.replace("_orig_mod.", "").replace("module.", ""): v for k, v in raw_state.items()}

    model.load_state_dict(clean_state, strict=True)
    model.to(device)
    model.eval()
    
    return model, classes


# ==========================================
# 3. MAIN INFERENCE LOOP
# ==========================================

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, classes = load_model(MODEL_PATH, device)
    print(f"--> Successfully loaded model on {device}.")
    print(f"--> Target Classes Index Mapping: {dict(enumerate(classes))}")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW if os.name == 'nt' else cv2.CAP_ANY)
    if not cap.isOpened():
        print(f"Error: Could not access webcam feed at index {CAMERA_INDEX}.")
        return

    prediction_buffer = deque(maxlen=VOTE_WINDOW)
    last_printed_class = None
    prev_time = time.time()

    print("\n[DETECTOR READY] Press 'q' or 'ESC' to exit.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                break

            # 1. Square Crop
            if USE_CENTER_SQUARE_CROP:
                crop, box_pts = center_square_crop(frame, IMG_SIZE, margin=CROP_MARGIN)
            else:
                crop = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
                h, w, _ = frame.shape
                box_pts = np.array([[0, 0], [w, 0], [w, h], [0, h]])

            # 2. Preprocessing
            processed_crop = apply_clahe(crop) if USE_CLAHE else crop.copy()
            rgb_crop = cv2.cvtColor(processed_crop, cv2.COLOR_BGR2RGB)
            input_tensor = transform(rgb_crop).unsqueeze(0).to(device)

            # 3. Inference
            with torch.no_grad():
                outputs = model(input_tensor)
                probs = F.softmax(outputs, dim=1)[0]
                conf, pred_idx = torch.max(probs, dim=0)
                entropy = -torch.sum(probs * torch.log(probs + 1e-8)).item()

            raw_pred_class = classes[pred_idx.item()]
            raw_conf = conf.item()

            # 4. Filtering
            if raw_conf > CONF_THRESH and entropy < ENTROPY_MAX:
                prediction_buffer.append(raw_pred_class)
            else:
                prediction_buffer.append("None")

            counts = Counter(prediction_buffer)
            most_common, count = counts.most_common(1)[0]

            # 5. Lock-in detection
            if most_common != "None" and count >= VOTE_MIN:
                smoothed_class = most_common
                label_text = f"Number: {smoothed_class} ({raw_conf * 100:.1f}%)"
                box_color = (0, 255, 0)
                if smoothed_class != last_printed_class:
                    print(f"*** DETECTED: {smoothed_class} ***")
                    last_printed_class = smoothed_class
            else:
                label_text = f"Scanning... ({raw_pred_class}: {raw_conf * 100:.1f}%)"
                box_color = (0, 165, 255)
                last_printed_class = None

            # 6. Render bounding box & text
            cv2.polylines(frame, [box_pts], isClosed=True, color=box_color, thickness=3)
            x_txt, y_txt = box_pts[0]
            cv2.putText(frame, label_text, (int(x_txt), max(int(y_txt) - 10, 25)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, box_color, 2, cv2.LINE_AA)

            # 7. Render Probabilities HUD
            y_offset = 30
            cv2.putText(frame, "Class Probabilities:", (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
            for idx, cls_name in enumerate(classes):
                y_offset += 20
                p_val = probs[idx].item() * 100
                # Highlight active prediction in green or cyan
                text_color = (0, 255, 255) if idx == pred_idx.item() else (220, 220, 220)
                cv2.putText(frame, f"  {cls_name}: {p_val:5.1f}%", (10, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1, cv2.LINE_AA)

            # 8. FPS Calculation
            curr_time = time.time()
            fps = 1.0 / max(curr_time - prev_time, 1e-5)
            prev_time = curr_time
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, frame.shape[0] - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)

            cv2.imshow("Cube Face CNN Detector", frame)
            cv2.imshow("Model Input View", processed_crop)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()