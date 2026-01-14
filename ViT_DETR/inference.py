import torch
import cv2
import numpy as np
from my_model import EdgeDETR

# --- CONFIG ---
weights_path = "weights.pth"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 1. Load Model
model = EdgeDETR(num_classes=1).to(device)
model.load_state_dict(torch.load(weights_path, map_location=device))
model.eval()

# 2. Camera Loop
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret: break
    
    # Preprocess
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(img).permute(2, 0, 1).float().unsqueeze(0).to(device) / 255.0

    # Inference
    with torch.no_grad():
        out = model(tensor)

    # Filter (Confidence > 70%)
    probs = out['pred_logits'].softmax(-1)[0, :, :-1]
    keep = probs.max(-1).values > 0.7
    
    if keep.any():
        # Get best box
        idx = probs.max(-1).values.argmax()
        box = out['pred_boxes'][0, idx].cpu().numpy()
        h, w, _ = frame.shape
        
        # Convert 0-1 to Pixels
        cx, cy, bw, bh = box * [w, h, w, h]
        
        # --- HERE IS YOUR COORDINATE HANDSHAKE ---
        # print(f"Found Object at Pixels: {cx}, {cy}")
        # Call your conversion function here:
        # real_x, real_y = pixel_to_cm(cx, cy)
        # move_robot(real_x, real_y)
        
        # Visual
        cv2.rectangle(frame, (int(cx-bw/2), int(cy-bh/2)), (int(cx+bw/2), int(cy+bh/2)), (0,255,0), 2)

    cv2.imshow("Jetson View", frame)
    if cv2.waitKey(1) == ord('q'): break

cap.release()
cv2.destroyAllWindows()