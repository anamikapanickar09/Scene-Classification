from ultralytics import YOLO
import torch
import torch.nn as nn
from torchvision import models, transforms
import cv2
import numpy as np
from PIL import Image
from width import known_widths
from filterpy.kalman import KalmanFilter
from collections import defaultdict

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load scene classification model
scene_model = models.resnet18(pretrained=False)
scene_model.fc = nn.Linear(scene_model.fc.in_features, 3)
scene_model.load_state_dict(torch.load("resnet18_scene_classifier.pth", map_location=device))
scene_model.eval()
scene_model.to(device)

scene_transform = transforms.Compose([
    
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])
scene_classes = ['confined', 'normal', 'non_complex']

yolo_model = YOLO("yolov8n-seg.pt")
cap = cv2.VideoCapture("C:/Users/Babu/Downloads/walking-through-a-modern-american-suburban-home-entering-through-the-front-door-moving-through-the-living-room-and-into-the-kitchen_htm4pfwm__60dace838e0a4116929ba65a3b282bde__P360.mp4")

FOCAL_LENGTH = 450

# ✅ Initialize Kalman filters for each object label
kalman_filters = {}

def create_kalman_filter():
    kf = KalmanFilter(dim_x=4, dim_z=2)
    kf.F = np.array([[1, 0, 1, 0],
                     [0, 1, 0, 1],
                     [0, 0, 1, 0],
                     [0, 0, 0, 1]])  # state transition
    kf.H = np.array([[1, 0, 0, 0],
                     [0, 1, 0, 0]])  # measurement function
    kf.P *= 1000.  # covariance matrix
    kf.R = 10      # measurement noise
    kf.Q = np.eye(4)  # process noise
    return kf

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    img_height, img_width = frame.shape[:2]

    resized_frame = cv2.resize(frame, (224, 224))
    pil_frame = Image.fromarray(cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB))
    scene_input = scene_transform(pil_frame).unsqueeze(0).to(device)

    with torch.no_grad():
        scene_output = scene_model(scene_input)
        scene_pred = torch.argmax(scene_output, dim=1).item()
        scene_label = scene_classes[scene_pred]

    cv2.putText(frame, f"Scene: {scene_label}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    results = yolo_model(frame)[0]

    if results.masks is None or results.boxes is None:
        cv2.imshow("Scene + Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue

    for i in range(len(results.masks)):
        mask = results.masks[i].data[0].cpu().numpy()
        mask_resized = cv2.resize(mask, (img_width, img_height), interpolation=cv2.INTER_NEAREST)
        color_mask = np.zeros_like(frame, dtype=np.uint8)
        color_mask[mask_resized > 0.5] = (0, 255, 0)
        frame = cv2.addWeighted(frame, 1, color_mask, 0.4, 0)

        box = results.boxes[i].xyxy[0].cpu().numpy()
        x1, y1, x2, y2 = box
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

        label_idx = int(results.boxes[i].cls[0].item())
        label = results.names[label_idx]

        # ✅ Initialize Kalman filter if not already
        if label not in kalman_filters:
            kf = create_kalman_filter()
            kf.x[:2] = np.array([[cx], [cy]])
            kalman_filters[label] = kf

        # Update and predict
        kf = kalman_filters[label]
        kf.predict()
        kf.update([cx, cy])
        cx_filtered, cy_filtered = kf.x[0, 0], kf.x[1, 0]

        # Recalculate smoothed bounding box based on filtered center
        width = x2 - x1
        height = y2 - y1
        x1_smooth = int(cx_filtered - width / 2)
        y1_smooth = int(cy_filtered - height / 2)
        x2_smooth = int(cx_filtered + width / 2)
        y2_smooth = int(cy_filtered + height / 2)

        # Estimate distance
        object_width = known_widths.get(label, 10)
        pixel_width = x2_smooth - x1_smooth
        if pixel_width == 0:
            continue

        distance = (object_width * FOCAL_LENGTH) / pixel_width
        text = f"{label}: {distance:.2f} cm"

        cv2.rectangle(frame, (x1_smooth, y1_smooth), (x2_smooth, y2_smooth), (0, 0, 255), 2)
        cv2.putText(frame, text, (x1_smooth, y1_smooth - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    cv2.imshow("Scene + Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
