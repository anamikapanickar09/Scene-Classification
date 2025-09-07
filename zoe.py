import torch
import cv2
import numpy as np
from zoedepth.utils.config import get_config
from zoedepth.models.builder import build_model
from zoedepth.utils.misc import save_raw_16bit

# --------------------------
# 1. Setup device
# --------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# --------------------------
# 2. Load ZoeDepth model
# --------------------------
ZOE_VARIANT = "zoedepth_nk"   # ✅ correct variant
conf = get_config(ZOE_VARIANT, pretrained=True)  # <- only two args
model = build_model(conf)
model.to(device)
model.eval()

# --------------------------
# 3. Start video capture
# --------------------------
cap = cv2.VideoCapture(0)  # use 0 for webcam, or put video file path

prev_min_depth = None

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert BGR -> RGB for ZoeDepth
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Resize to ZoeDepth input size (optional: Zoe can handle different sizes)
    img_resized = cv2.resize(img_rgb, (640, 480))

    # Convert to tensor
    img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0

    # --------------------------
    # 4. Predict depth map
    # --------------------------
    with torch.no_grad():
        depth = model.infer(img_tensor)[0]  # (H, W)

    depth_np = depth.cpu().numpy()

    # --------------------------
    # 5. Find nearest object
    # --------------------------
    min_depth = np.min(depth_np)  # closest pixel
    mean_center_depth = np.mean(depth_np[200:280, 280:360])  # focus on center region

    # --------------------------
    # 6. Collision detection
    # --------------------------
    warning = ""
    if min_depth < 0.15:  # threshold (relative)
        warning = "⚠️ Object VERY close!"

    if prev_min_depth is not None:
        if min_depth < prev_min_depth - 0.05:  # object approaching fast
            warning = "⚠️ Object APPROACHING!"

    prev_min_depth = min_depth

    # --------------------------
    # 7. Display results
    # --------------------------
    depth_vis = cv2.normalize(depth_np, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_MAGMA)

    cv2.putText(depth_vis, f"Nearest depth: {min_depth:.3f}", (30, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    if warning:
        cv2.putText(depth_vis, warning, (30, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)

    cv2.imshow("Frame", frame)
    cv2.imshow("Depth", depth_vis)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
