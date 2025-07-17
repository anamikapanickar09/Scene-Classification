from ultralytics import YOLO
import torch
import torch.nn as nn
from torchvision import models, transforms
import cv2
import numpy as np
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

scene_classes = ['confined', 'normal', 'open']  # Update if different
yolo_model = YOLO("yolov8n-seg.pt")

cap = cv2.VideoCapture("C:/Users\Babu\Downloads\walking-through-a-modern-american-suburban-home-entering-through-the-front-door-moving-through-the-living-room-and-into-the-kitchen_htm4pfwm__60dace838e0a4116929ba65a3b282bde__P360.mp4")

KNOWN_WIDTH = 4  # cm (adjust based on known object)
FOCAL_LENGTH = 450  # adjust after calibration

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
        x1, y1, x2, y2 = box.astype(int)
        label_idx = int(results.boxes[i].cls[0].item())
        label = results.names[label_idx]

        known_widths = {
            'person': 40,
            'bottle': 7,
            'chair': 45,
            'tv': 90,
            'cell phone': 7,
            # Add more YOLO classes as needed
        }

        object_width = known_widths.get(label, 10)  # default to 10 cm if unknown
        pixel_width = x2 - x1

        if pixel_width == 0:
            continue

        distance = (object_width * FOCAL_LENGTH) / pixel_width
        text = f"{label}: {distance:.2f} cm"


        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(frame, text, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    cv2.imshow("Scene + Detection", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
