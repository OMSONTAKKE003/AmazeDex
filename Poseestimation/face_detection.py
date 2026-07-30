"""
face_detection.py (Inference Script)
==========================
Opens a webcam feed and predicts the die/number shown on the cube face in
real time, using the checkpoint produced by train_face_cnn.py.
"""

from collections import Counter, deque
import os
import pathlib
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

# ==========================================
# 0. HARDCODED SETTINGS
# ==========================================
# Dynamically locate face_cnn.pt relative to where this script lives
SCRIPT_DIR = pathlib.Path(__file__).parent.resolve()
MODEL_PATH = str(SCRIPT_DIR / "face_cnn.pt")

CAMERA_INDEX = 0
IMG_SIZE = 224
CONF_THRESH = 0.70      # minimum softmax confidence to accept a prediction
ENTROPY_MAX = 1.2       # reject if the model is "unsure" across classes
VOTE_WINDOW = 5         # frames kept for temporal smoothing
VOTE_MIN = 3         # min agreeing frames in the window to lock in a result
USE_CENTER_SQUARE_CROP = True   # crop a square from the frame center before resizing


# ==========================================
# 1. Custom MobileNetV3-Small Architecture
# ==========================================

def _make_divisible(v, divisor=8, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < v:
        new_v += divisor
    return new_v


class HSwish(nn.Module):
    def forward(self, x):
        return x * F.relu6(x + 3, inplace=False) / 6.0


class HSigmoid(nn.Module):
    def forward(self, x):
        return F.relu6(x + 3, inplace=False) / 6.0


class SqueezeExcite(nn.Module):
    def __init__(self, in_chs, reduced_chs):
        super().__init__()
        self.fc1 = nn.Conv2d(in_chs, reduced_chs, 1)
        self.fc2 = nn.Conv2d(reduced_chs, in_chs, 1)

    def forward(self, x):
        scale = F.adaptive_avg_pool2d(x, 1)
        scale = self.fc1(scale)
        scale = F.relu(scale, inplace=True)
        scale = self.fc2(scale)
        scale = HSigmoid()(scale)
        return x * scale


class InvertedResidual(nn.Module):
    def __init__(self, in_chs, out_chs, kernal_size, exp_chs, use_se, use_hs, stride):
        super().__init__()
        self.stride = stride
        self.use_res_connect = self.stride == 1 and in_chs == out_chs

        layers = []
        if exp_chs != in_chs:
            layers.append(
                nn.Sequential(
                    nn.Conv2d(in_chs, exp_chs, 1, bias=False),
                    nn.BatchNorm2d(exp_chs),
                    HSwish() if use_hs else nn.ReLU6(inplace=True),
                )
            )

        layers.append(
            nn.Sequential(
                nn.Conv2d(
                    exp_chs, exp_chs, kernal_size,
                    stride=stride, padding=kernal_size // 2,
                    groups=exp_chs, bias=False,
                ),
                nn.BatchNorm2d(exp_chs),
                HSwish() if use_hs else nn.ReLU6(inplace=True),
            )
        )

        if use_se:
            reduced_chs = _make_divisible(exp_chs // 4, 8)
            layers.append(SqueezeExcite(exp_chs, reduced_chs))

        layers.append(
            nn.Sequential(
                nn.Conv2d(exp_chs, out_chs, 1, bias=False),
                nn.BatchNorm2d(out_chs),
            )
        )

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return x + self.block(x) if self.use_res_connect else self.block(x)


class MobileNetV3SmallCustom(nn.Module):
    def __init__(self, num_classes=6):
        super().__init__()
        self.features = nn.Sequential(
            nn.Sequential(
                nn.Conv2d(3, 16, 3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(16),
                HSwish(),
            ),
            InvertedResidual(16, 16, 3, 16, use_se=True, use_hs=False, stride=2),
            InvertedResidual(16, 24, 3, 72, use_se=False, use_hs=False, stride=2),
            InvertedResidual(24, 24, 3, 88, use_se=False, use_hs=False, stride=1),
            InvertedResidual(24, 40, 5, 96, use_se=True, use_hs=True, stride=2),
            InvertedResidual(40, 40, 5, 240, use_se=True, use_hs=True, stride=1),
            InvertedResidual(40, 40, 5, 240, use_se=True, use_hs=True, stride=1),
            InvertedResidual(40, 48, 5, 120, use_se=True, use_hs=True, stride=1),
            InvertedResidual(48, 48, 5, 144, use_se=True, use_hs=True, stride=1),
            InvertedResidual(48, 96, 5, 288, use_se=True, use_hs=True, stride=2),
            InvertedResidual(96, 96, 5, 576, use_se=True, use_hs=True, stride=1),
            InvertedResidual(96, 96, 5, 576, use_se=True, use_hs=True, stride=1),
            nn.Sequential(
                nn.Conv2d(96, 576, 1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(576),
                HSwish(),
            ),
        )
        self.classifier = nn.Sequential(
            nn.Linear(576, 1024),
            HSwish(),
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(1024, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        return self.classifier(x)


# ==========================================
# 2. Frame preprocessing helpers
# ==========================================

def center_square_crop(frame, output_size):
    h, w, _ = frame.shape
    side = min(h, w)
    y1 = (h - side) // 2
    x1 = (w - side) // 2
    crop = frame[y1:y1 + side, x1:x1 + side]
    crop = cv2.resize(crop, (output_size, output_size))
    box = np.array([[x1, y1], [x1 + side, y1], [x1 + side, y1 + side], [x1, y1 + side]])
    return crop, box


def apply_clahe(bgr_frame):
    lab = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)


# ==========================================
# 3. Model loading
# ==========================================

def load_model(model_path, device):
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"\n\n[ERROR] Model file not found at: {model_path}\n"
            f"Please ensure 'face_cnn.pt' is in the exact same directory as this script.\n"
        )

    checkpoint = torch.load(model_path, map_location=device)
    classes = [str(c) for c in checkpoint["classes"]]
    model = MobileNetV3SmallCustom(num_classes=len(classes))
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.to(device)
    model.eval()
    return model, classes


# ==========================================
# 4. Main real-time loop
# ==========================================

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, classes = load_model(MODEL_PATH, device)
    print(f"--> Loaded model on {device}. Classes: {classes}")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("Error: Could not access webcam feed.")
        return

    prediction_buffer = deque(maxlen=VOTE_WINDOW)
    last_printed_class = None

    print("\n[DETECTOR READY] Press 'q' to quit.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if USE_CENTER_SQUARE_CROP:
                crop, box_pts = center_square_crop(frame, IMG_SIZE)
            else:
                crop = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
                h, w, _ = frame.shape
                box_pts = np.array([[0, 0], [w, 0], [w, h], [0, h]])

            enhanced_crop = apply_clahe(crop)
            rgb_crop = cv2.cvtColor(enhanced_crop, cv2.COLOR_BGR2RGB)
            input_tensor = transform(rgb_crop).unsqueeze(0).to(device)

            with torch.no_grad():
                outputs = model(input_tensor)
                probs = F.softmax(outputs, dim=1)[0]
                conf, pred_idx = torch.max(probs, dim=0)
                entropy = -torch.sum(probs * torch.log(probs + 1e-8)).item()

            raw_pred_class = classes[pred_idx.item()]
            raw_conf = conf.item()

            if raw_conf > CONF_THRESH and entropy < ENTROPY_MAX:
                prediction_buffer.append(raw_pred_class)
            else:
                prediction_buffer.append("None")

            counts = Counter(prediction_buffer)
            most_common, count = counts.most_common(1)[0]

            if most_common != "None" and count >= VOTE_MIN:
                smoothed_class = most_common
                label_text = f"Number: {smoothed_class} ({raw_conf * 100:.1f}%)"
                box_color = (0, 255, 0)
                if smoothed_class != last_printed_class:
                    print(f"*** DETECTED: {smoothed_class} ***")
                    last_printed_class = smoothed_class
            else:
                label_text = f"Scanning... ({raw_pred_class}: {raw_conf * 100:.1f}%)"
                box_color = (0, 0, 255)
                last_printed_class = None

            cv2.polylines(frame, [box_pts], isClosed=True, color=box_color, thickness=3)
            x_txt, y_txt = box_pts[0]
            cv2.putText(frame, label_text, (int(x_txt), max(int(y_txt) - 10, 25)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, box_color, 2)

            y_offset = 30
            cv2.putText(frame, "Cube Number Detector", (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            for idx, cls_name in enumerate(classes):
                y_offset += 20
                p_val = probs[idx].item() * 100
                cv2.putText(frame, f"{cls_name}: {p_val:.1f}%", (10, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)

            cv2.imshow("Cube Face CNN Detector", frame)
            cv2.imshow("Model Input View", enhanced_crop)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()