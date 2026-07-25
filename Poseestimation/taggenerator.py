"""
Generate tag36h11 AprilTag images for cube faces.
Requires: pip install opencv-python (opencv-contrib-python if aruco module missing)
"""

import cv2
import os

OUTPUT_DIR = "tags"
TAG_SIZE_PX = 600      # render resolution (not physical size — scale on print)
QUIET_ZONE_MODULES = 1 # white border, in tag-module units, added around the tag
NUM_TAGS = 6            # one per cube face

os.makedirs(OUTPUT_DIR, exist_ok=True)

aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)

for tag_id in range(NUM_TAGS):
    # generateImageMarker already includes a 1-module quiet zone by default
    img = cv2.aruco.generateImageMarker(aruco_dict, tag_id, TAG_SIZE_PX)

    out_path = os.path.join(OUTPUT_DIR, f"tag36h11_{tag_id:02d}.png")
    cv2.imwrite(out_path, img)
    print(f"saved {out_path}")

print(f"\nGenerated {NUM_TAGS} tags in tag36h11 family -> ./{OUTPUT_DIR}/")
print("Reminder before printing:")
print("  - print at 100% scale ('actual size'), not 'fit to page'")
print("  - verify physical printed size with calipers")
print("  - use matte paper/laminate, not glossy")