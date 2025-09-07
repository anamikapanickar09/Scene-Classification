import cv2
import torch
import numpy as np
from ultralytics import YOLO
from collections import defaultdict, deque

# ==============================
# Load YOLOv8 model
# ==============================
yolo_model = YOLO("yolov8n.pt")  # You can use yolov8n, yolov8s, etc.

# ==============================
# Load MiDaS depth model
# ==============================
midas = torch.hub.load("intel-isl/MiDaS", "DPT_Hybrid")  # can also try "MiDaS_small"
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
transform = midas_transforms.dpt_transform
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
midas.to(device)
midas.eval()

# ==============================
# Object Tracking (simple)
# ==============================
# Store last N depth values for each object ID
object_depth_history = defaultdict(lambda: deque(maxlen=5))

# Parameters
NEAR_THRESHOLD = 0.2   # Below this depth value = near (0-1 normalized MiDaS)
APPROACH_RATE_THRESHOLD = -0.05  # Depth change per frame to trigger warning

# ==============================
# Video Capture
# ==============================
cap = cv2.VideoCapture(0)  # Change to video path or webcam index

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # YOLO object detection
    results = yolo_model(frame, verbose=False)[0]

    # MiDaS depth estimation
    input_batch = transform(frame).to(device)
    with torch.no_grad():
        prediction = midas(input_batch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=frame.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()
    depth_map = prediction.cpu().numpy()

    # Normalize depth for easier thresholding
    depth_norm = cv2.normalize(depth_map, None, 0, 1, cv2.NORM_MINMAX)

    # Process each detected object
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        cls_id = int(box.cls[0])
        label = yolo_model.names[cls_id]

        # Use center depth of bounding box
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        obj_depth = depth_norm[cy, cx]

        # Track depth changes (use object ID from YOLO's tracker if available)
        obj_id = f"{label}_{cx}_{cy}"  # Simple unique key
        object_depth_history[obj_id].append(obj_depth)

        # Determine approach rate
        if len(object_depth_history[obj_id]) >= 2:
            approach_rate = object_depth_history[obj_id][-1] - object_depth_history[obj_id][-2]

            # Check for collision risk
            collision_risk = obj_depth < NEAR_THRESHOLD or approach_rate < APPROACH_RATE_THRESHOLD

            # Draw bounding box
            color = (0, 255, 0)
            if collision_risk:
                color = (0, 0, 255)  # Red warning
                cv2.putText(frame, "⚠ COLLISION WARNING!", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{label} Depth:{obj_depth:.2f} AR:{approach_rate:.3f}",
                        (x1, y2 + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    cv2.imshow("Collision Detection", frame)
    cv2.imshow("Depth Map", depth_norm)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
        break

cap.release()
cv2.destroyAllWindows()
