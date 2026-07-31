import cv2
import numpy as np
import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision import transforms


FX = 600.0  
FY = 600.0 
CX = 320.0  
CY = 240.0 

CAMERA_MATRIX = np.array([
    [FX, 0, CX],
    [0, FY, CY],
    [0,  0,  1]
], dtype=np.float32)

DIST_COEFFS = np.zeros((4, 1), dtype=np.float32)  


TAG_SIZE = 0.04 
APRILTAG_DICT_TYPE = cv2.aruco.DICT_APRILTAG_36h11

TAG_ROTATION_OFFSETS = {
    0: 0,
    1: 0,
    2: 0,
    3: 0,
    4: 0,
    5: 90  
}

TAG_TO_FACE_MAP = {
    0: 6,
    1: 1,
    2: 3,
    3: 4,
    4: 2,
    5: 5
}

CNN_CONFIDENCE_THRESHOLD = 0.60

=

def load_checkpoint_model(model_weights_path, device):
    checkpoint = torch.load(model_weights_path, map_location=device)

    arch = checkpoint.get("arch", "mobilenet_v3_small")
    classes = checkpoint.get("classes", [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    img_size = checkpoint.get("img_size", 224)
    mean = checkpoint.get("mean", [0.485, 0.456, 0.406])
    std = checkpoint.get("std", [0.229, 0.224, 0.225])

    num_classes = len(classes)

    if hasattr(models, arch):
        model_fn = getattr(models, arch)
        model = model_fn(num_classes=num_classes)
    else:
        model = models.mobilenet_v3_small(num_classes=num_classes)

    # Load state dict key safely
    if isinstance(checkpoint, dict) and "model_state" in checkpoint:
        state_dict = checkpoint["model_state"]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # Handle mean / std formatting
    if isinstance(mean, (float, int)):
        mean = [mean] * 3
    if isinstance(std, (float, int)):
        std = [std] * 3

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((img_size, img_size) if isinstance(img_size, int) else img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    return model, transform, classes


def rotation_matrix_to_euler_angles(R):
    """Converts 3x3 Rotation Matrix to Euler angles (Roll, Pitch, Yaw) in degrees."""
    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    singular = sy < 1e-6

    if not singular:
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0

    return np.degrees([x, y, z])

def rotate_image_roi(image, angle):
    """Rotates an OpenCV image by 0, 90, 180, or 270 degrees clockwise."""
    if angle == 90:
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    elif angle == 180:
        return cv2.rotate(image, cv2.ROTATE_180)
    elif angle == 270:
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return image

def extract_tag_roi(frame, corners):
    """Perspective warp tag region into a normalized square image for the CNN."""
    pts_src = corners.reshape((4, 2)).astype(np.float32)
    side = 128
    pts_dst = np.array([
        [0, 0],
        [side - 1, 0],
        [side - 1, side - 1],
        [0, side - 1]
    ], dtype=np.float32)

    matrix = cv2.getPerspectiveTransform(pts_src, pts_dst)
    warped = cv2.warpPerspective(frame, matrix, (side, side))
    return warped



class Sim2RealCubeTracker:
    def __init__(self, model_weights_path=None):
        # Initialize ArUco Detector
        aruco_dict = cv2.aruco.getPredefinedDictionary(APRILTAG_DICT_TYPE)
        try:
            parameters = cv2.aruco.DetectorParameters()
            self.detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
            self.legacy_aruco = False
        except AttributeError:
            self.aruco_dict = aruco_dict
            self.parameters = cv2.aruco.DetectorParameters_create()
            self.legacy_aruco = True

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.cnn = None

        if model_weights_path and os.path.exists(model_weights_path):
            self.cnn, self.transform, self.classes = load_checkpoint_model(model_weights_path, self.device)
            print(f"[INFO] Successfully loaded model weights from {model_weights_path}")
        else:
            print(f"[WARN] Failed to load CNN weights from: {model_weights_path}")
            print("[WARN] Neural net predictions will be uncalibrated.")

    def detect_tags(self, frame):
        """Detect AprilTags in frame."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if not self.legacy_aruco:
            corners, ids, rejected = self.detector.detectMarkers(gray)
        else:
            corners, ids, rejected = cv2.aruco.detectMarkers(gray, self.aruco_dict, parameters=self.parameters)
        return corners, ids

    def predict_digit_cnn(self, roi_img):
        """Run CNN inference on ROI to predict number and confidence."""
        if self.cnn is None:
            return "Unknown", 0.0

        # Convert OpenCV BGR to RGB for standard torchvision models
        rgb_roi = cv2.cvtColor(roi_img, cv2.COLOR_BGR2RGB)
        tensor_img = self.transform(rgb_roi).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.cnn(tensor_img)
            probabilities = F.softmax(outputs, dim=1)
            conf, pred = torch.max(probabilities, dim=1)

        confidence = conf.item()
        predicted_idx = pred.item()
        predicted_digit = self.classes[predicted_idx] if predicted_idx < len(self.classes) else predicted_idx

        if confidence < CNN_CONFIDENCE_THRESHOLD:
            return "Unknown", confidence
        return predicted_digit, confidence

    def process_frame(self, frame):
        output_frame = frame.copy()
        corners, ids = self.detect_tags(frame)

        if ids is None or len(ids) == 0:
            cv2.putText(output_frame, "Status: No AprilTags Detected -> Number: Unknown",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            return output_frame

        cv2.aruco.drawDetectedMarkers(output_frame, corners, ids)

        flat_ids = np.ravel(ids)
        top_tag_idx = -1
        max_z_translation = float('inf')

        for i in range(len(flat_ids)):
            tag_corners = corners[i]

            obj_pts = np.array([
                [-TAG_SIZE/2,  TAG_SIZE/2, 0],
                [ TAG_SIZE/2,  TAG_SIZE/2, 0],
                [ TAG_SIZE/2, -TAG_SIZE/2, 0],
                [-TAG_SIZE/2, -TAG_SIZE/2, 0]
            ], dtype=np.float32)

            success, rvec, tvec = cv2.solvePnP(obj_pts, tag_corners[0], CAMERA_MATRIX, DIST_COEFFS)

            if success:
                z_dist = tvec[2][0]
                if z_dist < max_z_translation:
                    max_z_translation = z_dist
                    top_tag_idx = i

        if top_tag_idx != -1:
            best_id = int(flat_ids[top_tag_idx])
            best_corners = corners[top_tag_idx]

            obj_pts = np.array([
                [-TAG_SIZE/2,  TAG_SIZE/2, 0],
                [ TAG_SIZE/2,  TAG_SIZE/2, 0],
                [ TAG_SIZE/2, -TAG_SIZE/2, 0],
                [-TAG_SIZE/2, -TAG_SIZE/2, 0]
            ], dtype=np.float32)

            _, rvec, tvec = cv2.solvePnP(obj_pts, best_corners[0], CAMERA_MATRIX, DIST_COEFFS)
            
            R, _ = cv2.Rodrigues(rvec)
            roll, pitch, yaw = rotation_matrix_to_euler_angles(R)

            cv2.drawFrameAxes(output_frame, CAMERA_MATRIX, DIST_COEFFS, rvec, tvec, 0.03)

            roi = extract_tag_roi(frame, best_corners[0])
            rot_angle = TAG_ROTATION_OFFSETS.get(best_id, 0)
            calibrated_roi = rotate_image_roi(roi, rot_angle)

            digit_pred, confidence = self.predict_digit_cnn(calibrated_roi)

            # Prioritize mapping as absolute source of truth
            if best_id in TAG_TO_FACE_MAP:
                display_num = str(TAG_TO_FACE_MAP[best_id])
            else:
                # Last resort fallback if tag isn't mapped
                display_num = str(digit_pred)

            cv2.putText(output_frame, f"Top Face Tag ID: {best_id}", (20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            cv2.putText(output_frame, f"Cube Face Number: {display_num}", (20, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(output_frame, f"CNN Pred: {digit_pred} ({confidence*100:.1f}%)", (20, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.putText(output_frame, f"Orientation (R,P,Y): [{roll:.1f}deg, {pitch:.1f}deg, {yaw:.1f}deg]",
                        (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 100), 1)

            h, w, _ = output_frame.shape
            output_frame[10:110, w-110:w-10] = cv2.resize(calibrated_roi, (100, 100))
            cv2.rectangle(output_frame, (w-110, 10), (w-10, 110), (0, 255, 0), 1)
            cv2.putText(output_frame, "CNN ROI", (w-105, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

        return output_frame


if __name__ == "__main__":
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    MODEL_PATH = os.path.join(SCRIPT_DIR, "face_cnn.pt")
    
    tracker = Sim2RealCubeTracker(model_weights_path=MODEL_PATH)

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("[WARN] Camera feed not available. Generating a synthetic test frame...")
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        processed = tracker.process_frame(dummy_frame)
        cv2.imshow("Cube Face Detection & Orientation", processed)
        cv2.waitKey(0)
    else:
        print("[INFO] Starting perception stream. Press 'q' to quit.")
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            processed_frame = tracker.process_frame(frame)
            cv2.imshow("Cube Face Detection & Orientation", processed_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
    cv2.destroyAllWindows()