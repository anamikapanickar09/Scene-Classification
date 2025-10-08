import cv2
from ultralytics import YOLO

YOLO_MODEL_NAME = "yolov8s.pt"
model = YOLO(YOLO_MODEL_NAME)
# yolov8n.pt (nano)
# yolov8s.pt (small)
# yolov8m.pt (medium)
# yolov8l.pt (large)
# yolov8x.pt (extra large)


cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Webcam could not be opened.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, stream=True) # for better performance on video

    # Draw bounding boxes for each result
    for r in results:
        for box in r.boxes:
            conf = float(box.conf[0])                   # Confidence
            if conf <= 0.5:                             # Only show if confidence > 50%
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0])      # Box coordinates
            cls = int(box.cls[0])                       # Class ID
            label = f"{model.names[cls]} {conf:.2f}"    # Class label

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    cv2.imshow("YOLOv8 Video Detection", frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()