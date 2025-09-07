import torch
import urllib
import cv2
import numpy as np
from torchvision.transforms import Compose, Resize, ToTensor, Normalize

# Load the model
model_type = "DPT_Hybrid"  # DPT_Large, DPT_Hybrid, MiDaS_small
midas = torch.hub.load("intel-isl/MiDaS", model_type)
midas.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
midas.to(device)

# Load transforms
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
transform = midas_transforms.dpt_transform if model_type.startswith("DPT") else midas_transforms.small_transform

# Open camera
cap = cv2.VideoCapture("C:/Users/Babu/Downloads/walking-through-a-modern-american-suburban-home-entering-through-the-front-door-moving-through-the-living-room-and-into-the-kitchen_htm4pfwm__60dace838e0a4116929ba65a3b282bde__P360.mp4")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    input_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    input_tensor = transform(input_image).to(device)

    # Run inference
    with torch.no_grad():
        prediction = midas(input_tensor.unsqueeze(0))
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=input_image.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

    depth_map = prediction.cpu().numpy()
    
    # Normalize depth map for display
    depth_min = depth_map.min()
    depth_max = depth_map.max()
    depth_vis = 255 * (depth_map - depth_min) / (depth_max - depth_min)
    depth_vis = depth_vis.astype(np.uint8)

    # Show results
    cv2.imshow("Webcam", frame)
    cv2.imshow("Depth Map", depth_vis)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
