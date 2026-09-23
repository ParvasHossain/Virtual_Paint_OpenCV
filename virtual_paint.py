import os
import time
import math
import urllib.request
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ------------------------------------------------------------------
# 1. Automatic Model Check & Download
# ------------------------------------------------------------------
MODEL_FILE = "hand_landmarker.task"
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), MODEL_FILE)
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"

if not os.path.exists(MODEL_PATH):
    print(f"Model file '{MODEL_FILE}' not found. Downloading automatically...")
    try:
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Download completed successfully!")
    except Exception as e:
        print(f"Failed to download model file automatically: {e}")
        exit()

# ------------------------------------------------------------------
# 2. Initialize MediaPipe Hand Landmarker
# ------------------------------------------------------------------
base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.7,
    min_hand_presence_confidence=0.7,
    min_tracking_confidence=0.7
)
detector = vision.HandLandmarker.create_from_options(options)

# ------------------------------------------------------------------
# 3. Geometric Helper Functions
# ------------------------------------------------------------------
def calculate_distance(p1, p2):
    """Calculates Euclidean distance between two 2D points."""
    return math.hypot(p2[0] - p1[0], p2[1] - p1[1])

def get_palm_geometry(landmarks, w, h):
    """
    Calculates Palm Center Point and Palm Reference Size using 
    hand base keypoints (Wrist: 0, Index MCP: 5, Pinky MCP: 17).
    """
    wrist = (int(landmarks[0].x * w), int(landmarks[0].y * h))
    index_mcp = (int(landmarks[5].x * w), int(landmarks[5].y * h))
    pinky_mcp = (int(landmarks[17].x * w), int(landmarks[17].y * h))
    middle_mcp = (int(landmarks[9].x * w), int(landmarks[9].y * h))

    # Center of palm calculated as centroid of key palm anchors
    cx = int((wrist[0] + index_mcp[0] + pinky_mcp[0] + middle_mcp[0]) / 4)
    cy = int((wrist[1] + index_mcp[1] + pinky_mcp[1] + middle_mcp[1]) / 4)
    palm_center = (cx, cy)

    # Reference distance: Width across knuckles (index_mcp to pinky_mcp)
    palm_size = calculate_distance(index_mcp, pinky_mcp)
    
    return palm_center, max(palm_size, 1.0)

# ------------------------------------------------------------------
# 4. Camera Setup & Color Palette
# ------------------------------------------------------------------
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

canvas = None
draw_color = (255, 0, 255)
brush_thickness = 10
eraser_thickness = 50
px, py = 0, 0

colors = [
    {"color": (255, 0, 0),     "box": (20, 10, 120, 70),  "name": "BLUE"},
    {"color": (0, 255, 0),     "box": (140, 10, 240, 70), "name": "GREEN"},
    {"color": (0, 0, 255),     "box": (260, 10, 360, 70), "name": "RED"},
    {"color": (0, 255, 255),   "box": (380, 10, 480, 70), "name": "YELLOW"},
    {"color": (255, 0, 255),   "box": (500, 10, 600, 70), "name": "PURPLE"},
    {"color": (0, 0, 0),       "box": (620, 10, 720, 70), "name": "ERASER"},
    {"color": (255, 255, 255), "box": (740, 10, 840, 70), "name": "CLEAR"},
]

print("Starting Geometric Hand Tracking Paint Application... Press 'q' to quit.")

while cap.isOpened():
    success, img = cap.read()
    if not success:
        break

    img = cv2.flip(img, 1)
    if canvas is None:
        canvas = np.zeros_like(img)

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
    frame_timestamp_ms = int(time.time() * 1000)

    detection_result = detector.detect_for_video(mp_image, frame_timestamp_ms)

    # Render Header Navigation UI
    for col in colors:
        x1, y1, x2, y2 = col["box"]
        if col["name"] == "ERASER":
            cv2.rectangle(img, (x1, y1), (x2, y2), (200, 200, 200), -1)
        elif col["name"] == "CLEAR":
            cv2.rectangle(img, (x1, y1), (x2, y2), (50, 50, 50), -1)
        else:
            cv2.rectangle(img, (x1, y1), (x2, y2), col["color"], -1)

        if draw_color == col["color"]:
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 255), 3)

        cv2.putText(img, col["name"], (x1 + 10, y1 + 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # Hand Geometry Detection
    if detection_result.hand_landmarks:
        landmarks = detection_result.hand_landmarks[0]
        h, w, _ = img.shape

        # 1. Calculate Palm Center and Reference Size
        palm_center, palm_size = get_palm_geometry(landmarks, w, h)
        
        # Draw Palm Geometry Visualizers (Center Point + Bounding Circle)
        cv2.circle(img, palm_center, 6, (0, 255, 255), -1)
        cv2.circle(img, palm_center, int(palm_size * 1.2), (0, 255, 255), 1)

        # Key Tip Coordinates
        ix, iy = int(landmarks[8].x * w), int(landmarks[8].y * h)    # Index Tip
        mx, my = int(landmarks[12].x * w), int(landmarks[12].y * h)  # Middle Tip
        rx, ry = int(landmarks[16].x * w), int(landmarks[16].y * h)  # Ring Tip
        px_k, py_k = int(landmarks[20].x * w), int(landmarks[20].y * h) # Pinky Tip

        # 2. Compute Geometric Distance Metrics from Palm Center to Finger Tips
        dist_index = calculate_distance((ix, iy), palm_center) / palm_size
        dist_middle = calculate_distance((mx, my), palm_center) / palm_size
        dist_ring = calculate_distance((rx, ry), palm_center) / palm_size
        dist_pinky = calculate_distance((px_k, py_k), palm_center) / palm_size

        # A finger is considered extended if its tip-to-palm distance ratio > 1.25
        index_extended = dist_index > 1.25
        middle_extended = dist_middle > 1.25
        ring_extended = dist_ring > 1.25
        pinky_extended = dist_pinky > 1.25

        # Draw structural skeleton geometry lines
        cv2.line(img, palm_center, (ix, iy), (255, 255, 0), 1)
        if middle_extended:
            cv2.line(img, palm_center, (mx, my), (255, 255, 0), 1)

        # -------------------------------------------------------------
        # SELECTION MODE: Index & Middle Extended (Ring & Pinky Folded)
        # -------------------------------------------------------------
        if index_extended and middle_extended and not ring_extended:
            px, py = 0, 0  # Disconnect stroke
            cv2.rectangle(img, (ix - 10, iy - 10), (mx + 10, my + 10), draw_color, cv2.FILLED)

            if iy < 70:
                for col in colors:
                    x1, y1, x2, y2 = col["box"]
                    if x1 < ix < x2:
                        if col["name"] == "CLEAR":
                            canvas = np.zeros_like(img)
                        else:
                            draw_color = col["color"]

        # -------------------------------------------------------------
        # DRAWING MODE: Only Index Extended
        # -------------------------------------------------------------
        elif index_extended and not middle_extended:
            thickness = eraser_thickness if draw_color == (0, 0, 0) else brush_thickness
            cv2.circle(img, (ix, iy), thickness // 2, draw_color, -1)

            if px == 0 and py == 0:
                px, py = ix, iy

            cv2.line(canvas, (px, py), (ix, iy), draw_color, thickness)
            px, py = ix, iy
        else:
            px, py = 0, 0
    else:
        px, py = 0, 0

    # Overlay Canvas on Live Feed
    img_gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    _, img_inv = cv2.threshold(img_gray, 20, 255, cv2.THRESH_BINARY_INV)
    img_inv = cv2.cvtColor(img_inv, cv2.COLOR_GRAY2BGR)

    img = cv2.bitwise_and(img, img_inv)
    img = cv2.bitwise_or(img, canvas)

    cv2.imshow("Geometric Hand Paint Application", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()