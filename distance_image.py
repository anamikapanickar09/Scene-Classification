from ultralytics import YOLO
import cv2
import numpy as np

# Load YOLOv8 segmentation model
model = YOLO("yolov8n-seg.pt")

# Image path
img_path = "C:/Users/Babu/Downloads/pexels-pixabay-276724.jpg"
img = cv2.imread(img_path)
img_height, img_width = img.shape[:2]

# Run prediction
results = model(img_path)[0]

# Constants
KNOWN_WIDTH = 20   # cm (real object width)
FOCAL_LENGTH = 10  # needs calibration

# Loop through all detections
for i in range(len(results.masks)):
    # Get binary mask
    mask = results.masks[i].data[0].cpu().numpy()

    # Resize mask to match image size
    mask_resized = cv2.resize(mask, (img_width, img_height), interpolation=cv2.INTER_NEAREST)

    # Apply colored overlay where mask is True
    color_mask = np.zeros_like(img, dtype=np.uint8)
    color_mask[mask_resized > 0.5] = (0, 255, 0)
    img = cv2.addWeighted(img, 1, color_mask, 0.4, 0)

    # Get bounding box
    box = results.boxes[i].xyxy[0].cpu().numpy()
    x1, y1, x2, y2 = box.astype(int)
    pixel_width = x2 - x1

    # Estimate distance
    distance = (KNOWN_WIDTH * FOCAL_LENGTH) / pixel_width

    # Get label
    label_idx = int(results.boxes[i].cls[0].item())
    label = results.names[label_idx]
    text = f"{label}: {distance:.2f} cm"

    # Draw box and label
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
    cv2.putText(img, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (255, 255, 255), 2)

# Show and save the final image
cv2.imshow("Segmentation + Distance", img)
cv2.imwrite("output_segmented_distance.jpg", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

