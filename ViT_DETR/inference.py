import torch
import cv2
import numpy as np
from my_model import EdgeDETR

# --- CONFIG ---
NUM_CLASSES = 5
CONF_THRESH = 0.7
FOCAL_LENGTH_PX = 800  # <--- YOU NEED THIS (Calibrate it!)
REAL_WIDTH_CM = 5.0    # <--- YOU NEED THIS (Width of the object)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- LOAD MODEL ---
model = EdgeDETR(num_classes=NUM_CLASSES).to(device)
model.load_state_dict(torch.load("edge_detr_weights.pth", map_location=device))
model.eval()

def detect_object(frame, target_class=0):
    h, w, _ = frame.shape
    
    # Preprocess
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
    img = img.unsqueeze(0).to(device)

    with torch.no_grad():
        out = model(img)

    logits = out['pred_logits'][0]
    boxes = out['pred_boxes'][0]

    probs = logits.softmax(-1)
    scores, labels = probs[:, :-1].max(-1)

    mask = (labels == target_class) & (scores > CONF_THRESH)

    if not mask.any():
        return None

    best = scores[mask].argmax()
    idx = torch.where(mask)[0][best]

    cx, cy, bw, bh = boxes[idx].cpu().numpy()
    
    # Convert relative (0-1) to Pixels
    cx_px, cy_px = cx * w, cy * h
    width_px = bw * w  # Width in pixels
    
    # --- ADD THIS MATH TO GET DISTANCE ---
    # Z = (Real Width * Focal Length) / Pixel Width
    distance_cm = (REAL_WIDTH_CM * FOCAL_LENGTH_PX) / width_px

    return {
        "cx": cx_px,
        "cy": cy_px,
        "w": width_px,
        "h": bh * h,
        "confidence": scores[idx].item(),
        "class": target_class,
        "distance_z": distance_cm  # <--- NOW IT FINDS DISTANCE
    }