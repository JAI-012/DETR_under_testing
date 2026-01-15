"""
Final Inference Script for Eye-in-Hand Object Detection
- Accurate object detection with confidence thresholding
- Distance estimation using pinhole camera model
- Real-time visualization
- Per-class object size lookup
"""

import torch
import cv2
import numpy as np
from my_model import EdgeDETR

# ========================
# CONFIGURATION
# ========================

NUM_CLASSES = 5
CONF_THRESH = 0.7

# --- CAMERA CALIBRATION (YOU MUST CALIBRATE THESE!) ---
# Method: Place object at known distance, measure pixel width, calculate focal length
# Formula: focal_length = (pixel_width * distance) / real_width
FOCAL_LENGTH_PX = 800  # Placeholder - CALIBRATE THIS!

# Object real-world dimensions (in cm)
# Measure each object precisely!
OBJECT_DIMENSIONS = {
    0: {'name': 'Pen', 'width': 1.0, 'length': 14.0},
    1: {'name': 'Screwdriver', 'width': 2.5, 'length': 15.0},
    2: {'name': 'Object3', 'width': 3.0, 'length': 10.0},
    3: {'name': 'Object4', 'width': 2.0, 'length': 8.0},
    4: {'name': 'Object5', 'width': 4.0, 'length': 12.0},
}

# Visualization colors (BGR format for OpenCV)
CLASS_COLORS = {
    0: (255, 0, 0),    # Blue
    1: (0, 255, 0),    # Green
    2: (0, 0, 255),    # Red
    3: (255, 255, 0),  # Cyan
    4: (255, 0, 255),  # Magenta
}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ========================
# MODEL LOADING
# ========================

print("Loading EdgeDETR model...")
model = EdgeDETR(num_classes=NUM_CLASSES).to(device)
model.load_state_dict(torch.load("edge_detr_weights.pth", map_location=device))
model.eval()
print("Model loaded successfully!")

# ========================
# DETECTION FUNCTION
# ========================

def detect_objects(frame, min_confidence=CONF_THRESH):
    """
    Detect all objects in frame with confidence above threshold
    
    Args:
        frame: Input image (H, W, 3) BGR format
        min_confidence: Minimum confidence threshold
        
    Returns:
        list: List of detection dicts with keys:
              ['class', 'name', 'cx', 'cy', 'w', 'h', 'confidence', 'distance_cm']
    """
    h, w, _ = frame.shape
    
    # Preprocess
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
    img_tensor = img_tensor.unsqueeze(0).to(device)

    # Inference
    with torch.no_grad():
        outputs = model(img_tensor)

    # Parse outputs
    logits = outputs['pred_logits'][0]  # [num_queries, num_classes+1]
    boxes = outputs['pred_boxes'][0]    # [num_queries, 4]

    # Get class predictions (excluding background class)
    probs = logits.softmax(-1)[:, :-1]  # [num_queries, num_classes]
    scores, pred_classes = probs.max(-1)

    # Filter by confidence
    valid_mask = scores > min_confidence
    
    if not valid_mask.any():
        return []

    # Extract valid detections
    valid_scores = scores[valid_mask]
    valid_classes = pred_classes[valid_mask]
    valid_boxes = boxes[valid_mask]

    detections = []
    
    for score, cls, box in zip(valid_scores, valid_classes, valid_boxes):
        cls_id = cls.item()
        
        # Convert normalized coords to pixels
        cx, cy, bw, bh = box.cpu().numpy()
        cx_px = cx * w
        cy_px = cy * h
        width_px = bw * w
        height_px = bh * h
        
        # Calculate distance using pinhole camera model
        # Z = (Real_Width * Focal_Length) / Pixel_Width
        if cls_id in OBJECT_DIMENSIONS:
            real_width = OBJECT_DIMENSIONS[cls_id]['width']
            distance_cm = (real_width * FOCAL_LENGTH_PX) / width_px if width_px > 0 else 0
            obj_name = OBJECT_DIMENSIONS[cls_id]['name']
        else:
            distance_cm = 0
            obj_name = f"Unknown_{cls_id}"
        
        detections.append({
            'class': cls_id,
            'name': obj_name,
            'cx': cx_px,
            'cy': cy_px,
            'w': width_px,
            'h': height_px,
            'confidence': score.item(),
            'distance_cm': distance_cm
        })
    
    return detections

# ========================
# VISUALIZATION
# ========================

def draw_detections(frame, detections):
    """
    Draw bounding boxes and labels on frame
    
    Args:
        frame: Input image (modified in-place)
        detections: List of detection dicts from detect_objects()
    """
    for det in detections:
        cls_id = det['class']
        cx, cy = det['cx'], det['cy']
        w, h = det['w'], det['h']
        conf = det['confidence']
        dist = det['distance_cm']
        name = det['name']
        
        # Calculate box corners
        x1 = int(cx - w/2)
        y1 = int(cy - h/2)
        x2 = int(cx + w/2)
        y2 = int(cy + h/2)
        
        # Get color
        color = CLASS_COLORS.get(cls_id, (255, 255, 255))
        
        # Draw box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # Draw center point
        cv2.circle(frame, (int(cx), int(cy)), 5, color, -1)
        
        # Prepare label text
        label = f"{name}: {conf:.2f}"
        if dist > 0:
            label += f" | {dist:.1f}cm"
        
        # Draw label background
        (label_w, label_h), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
        )
        cv2.rectangle(
            frame, 
            (x1, y1 - label_h - 10), 
            (x1 + label_w, y1), 
            color, 
            -1
        )
        
        # Draw label text
        cv2.putText(
            frame, 
            label, 
            (x1, y1 - 5), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.6, 
            (0, 0, 0),  # Black text
            2
        )

# ========================
# CALIBRATION HELPER
# ========================

def calibrate_focal_length(known_distance_cm, known_width_cm, measured_pixel_width):
    """
    Helper function to calibrate FOCAL_LENGTH_PX
    
    Usage:
        1. Place object at known distance (e.g., 30cm)
        2. Measure its width in pixels from a saved frame
        3. Run this function
        4. Update FOCAL_LENGTH_PX at top of script
    
    Args:
        known_distance_cm: Measured distance to object
        known_width_cm: Real width of object
        measured_pixel_width: Width in pixels from image
    
    Returns:
        float: Calibrated focal length
    """
    focal_length = (measured_pixel_width * known_distance_cm) / known_width_cm
    print(f"\n=== CALIBRATION RESULT ===")
    print(f"Focal Length: {focal_length:.2f} pixels")
    print(f"Update line 20: FOCAL_LENGTH_PX = {focal_length:.0f}")
    return focal_length

# ========================
# MAIN LOOP
# ========================

def main():
    print("\n=== Starting Real-Time Detection ===")
    print(f"Device: {device}")
    print(f"Confidence Threshold: {CONF_THRESH}")
    print(f"Focal Length: {FOCAL_LENGTH_PX} px (calibrated: {'No' if FOCAL_LENGTH_PX == 800 else 'Yes'})")
    print("\nControls:")
    print("  'q' - Quit")
    print("  's' - Save current frame")
    print("  'c' - Toggle confidence threshold (0.5 / 0.7)")
    print("-" * 50)
    
    # Open camera
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("ERROR: Could not open camera")
        return
    
    # Set camera resolution (optional)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    frame_count = 0
    current_threshold = CONF_THRESH
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("ERROR: Failed to grab frame")
            break
        
        frame_count += 1
        
        # Detect objects
        detections = detect_objects(frame, min_confidence=current_threshold)
        
        # Draw detections
        draw_detections(frame, detections)
        
        # Draw info overlay
        info_text = f"Frame: {frame_count} | Detections: {len(detections)} | Conf: {current_threshold:.1f}"
        cv2.putText(
            frame, info_text, (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
        )
        
        # Show frame
        cv2.imshow("EdgeDETR - Eye-in-Hand Detection", frame)
        
        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            print("\nQuitting...")
            break
        elif key == ord('s'):
            filename = f"capture_{frame_count}.jpg"
            cv2.imwrite(filename, frame)
            print(f"Saved: {filename}")
        elif key == ord('c'):
            current_threshold = 0.5 if current_threshold == 0.7 else 0.7
            print(f"Confidence threshold changed to: {current_threshold}")
    
    cap.release()
    cv2.destroyAllWindows()
    print("Done!")

# ========================
# CALIBRATION MODE
# ========================

def calibration_mode():
    """
    Interactive calibration mode
    Run this once to calibrate FOCAL_LENGTH_PX
    """
    print("\n=== CALIBRATION MODE ===")
    print("1. Place a known object (e.g., pen) at a measured distance")
    print("2. Measure the distance from camera to object (cm)")
    print("3. Press 's' to capture frame")
    print("4. Manually measure pixel width in saved image")
    print("5. Enter measurements below\n")
    
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        cv2.imshow("Calibration - Press 's' to capture", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('s'):
            cv2.imwrite("calibration_capture.jpg", frame)
            print("Saved: calibration_capture.jpg")
            print("\nNow measure the object width in pixels using an image viewer")
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    # Get measurements
    try:
        distance = float(input("Enter measured distance (cm): "))
        real_width = float(input("Enter real object width (cm): "))
        pixel_width = float(input("Enter measured pixel width: "))
        
        calibrate_focal_length(distance, real_width, pixel_width)
    except ValueError:
        print("Invalid input")

# ========================
# ENTRY POINT
# ========================

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == '--calibrate':
        calibration_mode()
    else:
        main()
