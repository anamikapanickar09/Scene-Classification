import cv2
import torch
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
import urllib.request
import os
import cv2
from ultralytics import YOLO


RELATIVE_THRESHOLD = 0.078
DETECTION_CROP_FACTOR = 6

YOLO_MODEL = "yolov8s.pt"
# yolov8n.pt (nano)
# yolov8s.pt (small)
# yolov8m.pt (medium)
# yolov8l.pt (large)
# yolov8x.pt (extra large)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
midas = torch.hub.load("isl-org/MiDaS", "MiDaS_small", pretrained=True)
# ['DPTDepthModel', 'DPT_BEiT_B_384', 'DPT_BEiT_L_384', 'DPT_BEiT_L_512', 'DPT_Hybrid', 'DPT_Large',
# 'DPT_LeViT_224', 'DPT_Next_ViT_L_384', 'DPT_SwinV2_B_384', 'DPT_SwinV2_L_384', 'DPT_SwinV2_T_256',
# 'DPT_Swin_L_384', 'MiDaS', 'MiDaS_small', 'MidasNet', 'MidasNet_small', 'transforms']

midas.to(device)
midas.eval()

midas_transforms = torch.hub.load("isl-org/MiDaS", "transforms")
transform = midas_transforms.small_transform

model = YOLO(YOLO_MODEL)

def estimate_depth(frame):
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    input_batch = transform(img).to(device)

    with torch.no_grad():
        prediction = midas(input_batch)
        depth = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze().cpu().numpy()

    return depth

def find_min_depth(depth_map, x1=None, y1=None, x2=None, y2=None):
    height, width = depth_map.shape
    if None in (x1, y1, x2, y2):
        x1 = width // DETECTION_CROP_FACTOR
        x2 = width - x1
        y1 = height // DETECTION_CROP_FACTOR
        y2 = height - y1
        print(x1, y1, x2, y2, width, height)
    else:
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
    
    region = depth_map[y1:y2, x1:x2]
    # print(region)
    # center_crop = depth[h//3:2*h//3, w//3:2*w//3]

    if region.size == 0:
        return None
    
    return np.min(region)



cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Webcam could not be opened.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    depth_map = estimate_depth(frame)
    max = np.max(depth_map)
    min = np.min(depth_map)
    print(f"min:{min}, max:{max}")
    detections = model(frame, stream=True, verbose=False)
    
    height, width = depth_map.shape
    bound_x1 = width // DETECTION_CROP_FACTOR
    bound_y1 = height // DETECTION_CROP_FACTOR
    bound_x2 = width - bound_x1
    bound_y2 = height - bound_y1

    can_move = find_min_depth(depth_map, bound_x1, bound_y1, bound_x2, bound_y2) >= RELATIVE_THRESHOLD
    if can_move:
        status = "move"
        color = (0, 255, 0)
    else:
        status = "dont move"
        color = (0, 0, 255)

    for detection in detections:
        for box in detection.boxes:
            conf = float(box.conf[0])
            if conf < 0.5:  # Only show if confidence >= 50%
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            relative_depth = find_min_depth(depth_map, x1, y1, x2, y2)

            cls = int(box.cls[0])
            is_near = relative_depth < RELATIVE_THRESHOLD
            if is_near:
                label = f"{model.names[cls]} near"
                color = (0, 0, 255)
            else:
                label = f"{model.names[cls]} far"
                color = (0, 255, 0)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 4)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 3)
    
    cv2.rectangle(frame, (bound_x1, bound_y1), (bound_x2, bound_y2), (0,0,0), 3)
    cv2.putText(frame, status, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    cv2.imshow("object detection + movment decision", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
