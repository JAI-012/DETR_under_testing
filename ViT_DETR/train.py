import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from my_model import EdgeDETR
from dataset import RobotDataset
from utils.matcher import HungarianMatcher
from utils.box_ops import box_cxcywh_to_xyxy, generalized_box_iou

# --- CONFIG ---
NUM_CLASSES = 5       # 0=Pen, 1=Screwdriver, etc.
BATCH_SIZE = 4
LEARNING_RATE = 1e-4
EPOCHS = 50

def train():


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 1. Init Model
    # DETR requires N+1 outputs (Classes + Background)
    # EdgeDETR logic: linear_class output dim is num_classes + 1
    model = EdgeDETR(num_classes=NUM_CLASSES).to(device)
    
    # Sanity Check
    with torch.no_grad():  # <--- Added wrapper
        dummy_in = torch.randn(1, 3, 480, 640).to(device)
        dummy_out = model(dummy_in)
    assert dummy_out['pred_logits'].shape[-1] == NUM_CLASSES + 1

    # 2. Data Loader
    ds = RobotDataset('dataset/images', 'dataset/labels')
    loader = DataLoader(
        ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=lambda x: tuple(zip(*x))
    )

    # 3. Optimizer & Matcher
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    
    # CRITICAL FIX (Point 5): Added cost_class
    # Without this, the matcher ignores class predictions!
    matcher = HungarianMatcher(cost_class=1, cost_bbox=5, cost_giou=2)

    model.train()
    print("Starting Strictly Corrected EdgeDETR Training...")

    for epoch in range(EPOCHS):
        total_loss = 0.0

        for batch_idx, (imgs, targets) in enumerate(loader):
            imgs = torch.stack(imgs).to(device)
            targets = [
                {'boxes': t['boxes'].to(device), 'labels': t['labels'].to(device)}
                for t in targets
            ]

            # --- Forward Pass ---
            outputs = model(imgs)
            
            # CRITICAL FIX (Point 10): No grad for matcher
            # The matcher is just a calculation, doesn't need gradients
            with torch.no_grad():
                indices = matcher(outputs, targets)

            # Initialize loss accumulators for this BATCH
            # CRITICAL FIX (Point 3): Use safe tensors, not floats
            batch_loss_ce = torch.tensor(0.0, device=device)
            batch_loss_bbox = torch.tensor(0.0, device=device)
            batch_loss_giou = torch.tensor(0.0, device=device)
            
            total_matched_boxes = 0

            # --- Calculate Loss per Image in Batch ---
            for i, (src_idx, tgt_idx) in enumerate(indices):
                assert targets[i]['labels'].dtype == torch.long, \
                    f"Labels must be torch.long (int64), got {targets[i]['labels'].dtype}"
                # 1. Classification Target
                # CRITICAL FIX (Point 4): Background index is NUM_CLASSES
                # If we have 5 objects (0-4), Background is 5.
                target_classes = torch.full(
                    (outputs['pred_logits'].shape[1],), 
                    NUM_CLASSES, 
                    dtype=torch.long, 
                    device=device
                )
                
                # CRITICAL FIX (Point 2): Correct indentation
                if len(tgt_idx) == 0:
                    # Even if no objects, we still calculate background loss!
                    batch_loss_ce += F.cross_entropy(outputs['pred_logits'][i], target_classes)
                    continue

                # Assign object classes to matched indices
                target_classes[src_idx] = targets[i]['labels'][tgt_idx]
                
                # Accumulate Classification Loss
                batch_loss_ce += F.cross_entropy(outputs['pred_logits'][i], target_classes)

                # 2. Box Regression Loss (Only for matched objects)
                src_boxes = outputs['pred_boxes'][i][src_idx]
                tgt_boxes = targets[i]['boxes'][tgt_idx]

                batch_loss_bbox += F.l1_loss(src_boxes, tgt_boxes, reduction='sum')
                
                iou_loss = (1 - torch.diag(
                    generalized_box_iou(
                        box_cxcywh_to_xyxy(src_boxes), 
                        box_cxcywh_to_xyxy(tgt_boxes)
                    )
                ))
                batch_loss_giou += iou_loss.sum()
                
                total_matched_boxes += len(tgt_idx)

            # --- Normalize & Combine (CRITICAL FIX: Point 1 - OUTSIDE LOOP) ---
            
            # Normalize CE loss by batch size
            loss_ce = batch_loss_ce / len(imgs)

            # Normalize Box losses by total number of matched boxes
            if total_matched_boxes > 0:
                loss_bbox = batch_loss_bbox / total_matched_boxes
                loss_giou = batch_loss_giou / total_matched_boxes
            else:
                # CRITICAL FIX (Point 7): Don't break graph
                loss_bbox = batch_loss_bbox * 0.0
                loss_giou = batch_loss_giou * 0.0

            # Final Weighted Loss
            loss = 1.0 * loss_ce + 5.0 * loss_bbox + 2.0 * loss_giou

            # --- Backward Pass ---
            optimizer.zero_grad()
            loss.backward()
            
            # CRITICAL FIX (Point 9): Gradient Clipping
            # Prevents exploding gradients, crucial for Transformers
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
            
            optimizer.step()

            total_loss += loss.item()

        # CRITICAL FIX (Point 8): Log Average Loss
        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch}: Average Loss = {avg_loss:.4f}")

    torch.save(model.state_dict(), "edge_detr_weights.pth")
    print("Training complete. Weights saved.")

if __name__ == "__main__":
    train()