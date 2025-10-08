from ultralytics import YOLO
import cv2
import numpy as np

# Load YOLOv8 segmentation model
model = YOLO("yolov8s-seg.pt")

# Run inference on an image
img_path = "C:/Users/Babu/Downloads/BASE-Object-Desk-Accessories-7.jpg"
img = cv2.imread(img_path)
results = model(img, verbose=False)

for r in results:
    if r.masks is not None:
        boxes = r.boxes.xyxy.cpu().numpy().astype(int)   # bounding boxes
        classes = r.boxes.cls.cpu().numpy().astype(int)  # class IDs
        class_names = model.names                        # class dictionary

        for i, mask in enumerate(r.masks.data):
            # Resize mask to match original image size
            m = mask.cpu().numpy()
            m = cv2.resize(m, (img.shape[1], img.shape[0]))
            m = (m > 0.5).astype(np.uint8) * 255

            # ---- Centroid calculation ----
            M = cv2.moments(m)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])

                # Draw centroid
                cv2.circle(img, (cx, cy), 6, (0, 255, 255), -1)
                cv2.putText(img, f"({cx},{cy})", (cx+10, cy-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            # ---- Bounding box ----
            x1, y1, x2, y2 = boxes[i]
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # ---- Class name ----
            cls_id = classes[i]
            cls_name = class_names[cls_id]
            cv2.putText(img, cls_name, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

            # ---- Overlay mask ----
            colored_mask = np.zeros_like(img)
            colored_mask[m > 0] = (0, 255, 0)  # Red mask
            img = cv2.addWeighted(img, 1, colored_mask, 0.5, 0)

# Show result
cv2.imshow("YOLOv8-Seg with Centroid + BBox + Class", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
